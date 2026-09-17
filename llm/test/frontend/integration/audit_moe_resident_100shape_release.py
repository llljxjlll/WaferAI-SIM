"""Audit one source-bound, two-fresh MoE resident 1..10 release matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if type(result) is not dict:
        raise ValueError(f"JSON object required: {path}")
    return result


def load_frozen_driver(source_root: Path):
    path = source_root / "llm/test/frontend/integration/run_moe_full_model_native_mesh_matrix.py"
    if not path.is_file():
        raise ValueError(f"frozen driver missing: {path}")
    spec = importlib.util.spec_from_file_location("frozen_moe_native_matrix", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load frozen driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module._ROOT.resolve() != source_root.resolve():
        raise ValueError("frozen driver resolved the wrong source root")
    return module


def exact_case_paths(plan: dict, shard_roots: list[Path], replacement_root: Path,
                     driver) -> dict[str, Path]:
    if plan.get("schema_version") != "moe-resident-uncovered-release-plan-v1":
        raise ValueError("unexpected release plan schema")
    canonical = set(driver.RELEASE_SHAPES)
    if len(canonical) != 100 or plan.get("canonical_shape_count") != 100:
        raise ValueError("release shape count is not 100")
    covered = plan.get("covered_case_receipts")
    uncovered = plan.get("uncovered_shapes")
    shards = plan.get("uncovered_shards")
    if type(covered) is not dict or type(uncovered) is not list or type(shards) is not dict:
        raise ValueError("release plan is malformed")
    if len(shard_roots) != 7 or plan.get("shard_count") != 7:
        raise ValueError("exactly seven shard roots are required")
    if len(covered) != 35 or len(uncovered) != 65 or len(set(uncovered)) != 65:
        raise ValueError("35/65 release partition drifted")
    if set(covered) & set(uncovered) or set(covered) | set(uncovered) != canonical:
        raise ValueError("release partition has missing, duplicate, or extra shapes")
    result: dict[str, Path] = {}
    for shape, row in covered.items():
        if type(row) is not dict or set(row) != {"case_evidence", "sha256"}:
            raise ValueError(f"malformed documented case: {shape}")
        path = Path(row["case_evidence"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"documented case changed: {shape}")
        result[shape] = path.parent
    for index, root in enumerate(shard_roots):
        expected = list(driver.select_shapes(tuple(uncovered), shard_index=index, shard_count=7))
        if shards.get(str(index)) != expected:
            raise ValueError(f"release shard {index} selection drifted")
        binding = read_json(root / "matrix_binding.json")
        receipt = read_json(root / "matrix_receipt.json")
        if (binding.get("schema_version") != "moe-full-model-native-matrix-binding-v1"
                or binding.get("shapes") != expected
                or binding.get("shard_index") != index
                or binding.get("shard_count") != 7
                or receipt.get("schema_version") != "moe-full-model-native-matrix-receipt-v1"
                or receipt.get("status") != "verified"
                or receipt.get("shapes") != expected
                or receipt.get("completed_shapes") != expected
                or receipt.get("cases") != [f"{shape}/case_evidence.json" for shape in expected]
                or receipt.get("binding_sha256") != sha256(root / "matrix_binding.json")):
            raise ValueError(f"release shard {index} receipt or binding drifted")
        for shape in expected:
            result[shape] = root / shape
    replacements = ("1x3", "1x4", "2x2")
    replacement_binding = read_json(replacement_root / "matrix_binding.json")
    replacement_receipt = read_json(replacement_root / "matrix_receipt.json")
    if (replacement_binding.get("shapes") != list(replacements)
            or replacement_binding.get("shard_index") != 0
            or replacement_binding.get("shard_count") != 1
            or replacement_receipt.get("status") != "verified"
            or replacement_receipt.get("shapes") != list(replacements)
            or replacement_receipt.get("completed_shapes") != list(replacements)
            or replacement_receipt.get("cases")
                != [f"{shape}/case_evidence.json" for shape in replacements]
            or replacement_receipt.get("binding_sha256")
                != sha256(replacement_root / "matrix_binding.json")):
        raise ValueError("replacement matrix receipt or binding drifted")
    for shape in replacements:
        result[shape] = replacement_root / shape
    if set(result) != canonical or len(result) != 100:
        raise ValueError("release case map is not exactly canonical 100")
    return result


def audit(args: argparse.Namespace) -> dict:
    source_root = args.source_root.resolve()
    driver = load_frozen_driver(source_root)
    plan_path = args.plan.resolve()
    plan = read_json(plan_path)
    if Path(plan.get("source_driver", "")).resolve() != Path(driver.__file__).resolve():
        raise ValueError("release plan points at a different frozen driver")
    cases = exact_case_paths(plan, args.shard_roots, args.replacement_root, driver)
    driver_digest = sha256(Path(driver.__file__))
    runner = source_root / "llm/test/frontend/integration/run_moe_full_model_sequence_runtime_canary.py"
    runner_digest = sha256(runner)
    tool_paths = {name: getattr(args, name).resolve()
                  for name in ("finalizer", "resolver", "npusim", "simulation")}
    if any(not path.is_file() for path in tool_paths.values()):
        raise ValueError("a frozen native tool is missing")
    tool_digests = {name: sha256(path) for name, path in tool_paths.items()}
    for root in [*args.shard_roots, args.replacement_root]:
        binding = read_json(root / "matrix_binding.json")
        if (binding.get("driver_sha256") != driver_digest
                or binding.get("runner_sha256") != runner_digest
                or binding.get("tool_sha256") != tool_digests):
            raise ValueError(f"matrix source/tool binding drifted: {root}")
    signatures = set()
    receipts: dict[str, str] = {}
    cycles: list[int] = []
    compile_seconds: list[float] = []
    native_seconds: list[float] = []
    python_rss_kib: list[int] = []
    children_rss_kib: list[int] = []
    for shape in driver.RELEASE_SHAPES:
        root = cases[shape]
        evidence = driver.audit_cached_case(root, shape)
        receipts[shape] = sha256(root / "case_evidence.json")
        rows, columns = map(int, shape.split("x"))
        for index, fresh in enumerate(evidence["fresh"]):
            hardware = read_json(root / f"fresh{index}" / "hardware.json")
            stacks = hardware["memory_system"]["hbm_stacks"]
            if (len(stacks) != rows * columns
                    or any(stack.get("capacity_bytes") != 1 << 30 for stack in stacks)
                    or hardware["memory"].get("sram_size") != 131072
                    or hardware["memory"]["sram"].get("capacity_bytes") != 131072):
                raise ValueError(f"{shape} is not the frozen resident hardware profile")
            entry = fresh["source_tool_at_entry"]
            if entry["imported_python_sha256"].get(
                "llm/test/frontend/integration/run_moe_full_model_sequence_runtime_canary.py"
            ) != runner_digest:
                raise ValueError(f"{shape} was produced by a different runner")
            if entry.get("tool_sha256") != tool_digests:
                raise ValueError(f"{shape} native tool bytes drifted")
            signatures.add(json.dumps(entry, sort_keys=True))
            cycles.append(fresh["makespan_cycles"])
            compile_seconds.append(fresh["phase_wall_seconds"]["production_compile"])
            native_seconds.append(fresh["phase_wall_seconds"]["native_npusim"])
            python_rss_kib.append(fresh["python_peak_rss_kib"])
            children_rss_kib.append(fresh["children_max_rss_kib"])
        print(f"MoE resident release AUDIT {shape} two fresh", flush=True)
    if len(signatures) != 1 or len(receipts) != 100 or len(cycles) != 200:
        raise ValueError("release source/tool binding or execution count drifted")
    return {
        "schema_version": "moe-resident-100shape-release-audit-v1",
        "status": "verified",
        "scope": "shape-scaled full-active resident inference; Prefill+2 Decode; timing only",
        "resident_hbm_bytes_per_die": 1073741824,
        "resident_sram_bytes_per_die": 131072,
        "source_root": str(source_root),
        "frozen_driver_sha256": driver_digest,
        "frozen_runner_sha256": runner_digest,
        "native_tool_sha256": tool_digests,
        "plan_sha256": sha256(plan_path),
        "source_tool_entry_sha256": hashlib.sha256(next(iter(signatures)).encode()).hexdigest(),
        "case_count": 100,
        "independent_native_executions": 200,
        "three_segment_finalizer_and_resolver_calls_each": 600,
        "cycles_min": min(cycles),
        "cycles_max": max(cycles),
        "production_compile_wall_seconds_min_max": [min(compile_seconds), max(compile_seconds)],
        "native_npusim_wall_seconds_min_max": [min(native_seconds), max(native_seconds)],
        "frontend_peak_rss_kib_min_max": [min(python_rss_kib), max(python_rss_kib)],
        "native_children_peak_rss_kib_min_max": [min(children_rss_kib), max(children_rss_kib)],
        "case_evidence_sha256": receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--shard-root", dest="shard_roots", type=Path, action="append", required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MoE resident release PASS {result['case_count']} cases, "
          f"{result['independent_native_executions']} fresh executions", flush=True)


if __name__ == "__main__":
    main()
