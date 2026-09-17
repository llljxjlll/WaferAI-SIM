"""Reopen and aggregate the four strict Dense resident 100-shape shards."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


_SHAPES = tuple(f"{r}x{c}" for r in range(1, 11) for c in range(1, 11))
_DRIVER = Path("llm/test/frontend/integration/run_dense_native_mesh_matrix.py")
_RUNNER = Path("llm/test/frontend/integration/run_dense_sequence_runtime_canary.py")
_DRAM = Path("DRAMSys/configs/hbm2-example.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if type(data) is not dict:
        raise ValueError(f"expected object: {path}")
    return data


def check_release_partition(bindings: tuple[dict[str, Any], ...]) -> None:
    """Reject missing, repeated, reordered, or extra canonical cases."""
    if len(bindings) != 4 or {b.get("shard_index") for b in bindings} != set(range(4)):
        raise ValueError("release requires exactly four distinct shard indices")
    if any(b.get("shard_count") != 4 for b in bindings):
        raise ValueError("release shard count drifted")
    observed: list[str] = []
    for binding in bindings:
        index = binding["shard_index"]
        expected = list(_SHAPES[index::4])
        if binding.get("shapes") != expected:
            raise ValueError(f"shard {index} is missing, reordered, or has extra cases")
        observed.extend(expected)
    if len(observed) != 100 or set(observed) != set(_SHAPES):
        raise ValueError("release shape union is not exactly 100 unique cases")


def _frozen_driver(path: Path):
    spec = importlib.util.spec_from_file_location("_frozen_dense_matrix_release", path)
    if spec is None or spec.loader is None:
        raise ValueError("frozen Dense matrix driver cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_release(
    shard_roots: tuple[Path, ...], source_root: Path,
    tools: dict[str, Path],
) -> dict[str, Any]:
    source_root = source_root.resolve()
    driver_path = source_root / _DRIVER
    runner_path = source_root / _RUNNER
    dram_path = source_root / _DRAM
    for path in (driver_path, runner_path, dram_path, *tools.values()):
        if not path.is_file():
            raise ValueError(f"bound release input is missing: {path}")
    tool_sha = {name: _sha(path.resolve()) for name, path in sorted(tools.items())}
    driver_sha = _sha(driver_path)
    runner_sha = _sha(runner_path)
    dram_sha = _sha(dram_path)
    driver = _frozen_driver(driver_path)
    if tuple(driver.RELEASE_SHAPES) != _SHAPES:
        raise ValueError("frozen driver does not define canonical 100-shape order")

    roots = tuple(path.resolve() for path in shard_roots)
    if len(roots) != 4 or len(set(roots)) != 4:
        raise ValueError("release shard roots must be four distinct directories")
    bindings = tuple(_json(root / "matrix_binding.json") for root in roots)
    check_release_partition(bindings)
    receipts: dict[str, str] = {}
    case_sha: dict[str, str] = {}
    for root, binding in zip(roots, bindings):
        index = binding["shard_index"]
        if (binding.get("schema_version") != "dense-native-mesh-matrix-binding-v3"
                or binding.get("driver_sha256") != driver_sha
                or binding.get("runner_sha256") != runner_sha
                or binding.get("dram_config_sha256") != dram_sha
                or binding.get("tool_sha256") != tool_sha):
            raise ValueError(f"shard {index} source/tool bytes drifted")
        receipt_path = root / "matrix_receipt.json"
        receipt = _json(receipt_path)
        shapes = binding["shapes"]
        if (receipt.get("status") != "verified"
                or receipt.get("binding_sha256") != _sha(root / "matrix_binding.json")
                or receipt.get("shapes") != shapes
                or receipt.get("completed_shapes") != shapes
                or receipt.get("cases") != [f"{shape}/case_evidence.json" for shape in shapes]):
            raise ValueError(f"shard {index} receipt is incomplete or drifted")
        actual_dirs = {part.name for part in root.iterdir() if part.is_dir()}
        if actual_dirs != set(shapes):
            raise ValueError(f"shard {index} has missing or extra case directories")
        for shape in shapes:
            driver.audit_cached_case(root / shape, shape)
            case_sha[shape] = _sha(root / shape / "case_evidence.json")
        receipts[str(index)] = _sha(receipt_path)
    if set(case_sha) != set(_SHAPES):
        raise ValueError("case audit did not cover the complete release order")
    return {
        "schema_version": "dense-native-mesh-release-audit-v1",
        "status": "verified",
        "source_root": str(source_root),
        "driver_sha256": driver_sha,
        "runner_sha256": runner_sha,
        "dram_config_sha256": dram_sha,
        "tool_sha256": tool_sha,
        "shard_receipt_sha256": receipts,
        "case_evidence_sha256": {shape: case_sha[shape] for shape in _SHAPES},
        "case_count": 100,
        "native_fresh_count": 200,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-roots", nargs=4, type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    for name in ("finalizer", "resolver", "npusim", "simulation"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_release(
        tuple(args.shard_roots), args.source_root,
        {name: getattr(args, name) for name in
         ("finalizer", "resolver", "npusim", "simulation")},
    )
    if args.output.exists():
        raise ValueError("release audit output already exists")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("Dense native 100-shape resident release audit PASS 100 cases / 200 Fresh")


if __name__ == "__main__":
    main()
