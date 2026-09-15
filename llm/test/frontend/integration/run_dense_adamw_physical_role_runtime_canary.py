"""Two independent full-chain H16/L2 AdamW physical m/v external-state canaries.

Proof scope: 1x1 two AdamW update steps with true LSU-gated paging, functional=0.
No claim of numeric AdamW update or any full-model/four-family M2 acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import signal
import time

from llm.frontend.wafer_frontend.passes.dense_adamw_compile_sequence import compile_dense_adamw_step
from llm.frontend.wafer_frontend.passes.dense_adamw_mid_program_residency import (
    derive_dense_adamw_mid_program_residency, assign_dense_adamw_bounded_slots,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_paged_compile_sequence import (
    compile_dense_adamw_paged_step,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_paged_runtime import (
    build_dense_adamw_paged_runtime,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_physical_authority import (
    prove_physical_adamw_authority_and_capacity,
)
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json

from .dense_adamw_native_role_observer import observe_native_adamw_role_values
from .flexible_mesh_release_hardware import specialize_p5_large_release_hardware
from .run_dense_adamw_dma_component_canary import build_source_dma_program
from .run_dense_adamw_paged_offload_runtime_canary import (
    _linked_oracle, _observe, _source_oracle,
)
from .tp4_external_sgd_stage import _DRAM, _sha, _source_files, _stage, _verify_sources


_ROOT = Path(__file__).resolve().parents[4]


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _verify(sources: dict[str, str], tools: Path,
            binaries: dict[str, str], resources: dict[str, str]) -> None:
    _verify_sources(sources)
    if any(_sha(tools / name) != digest for name, digest in binaries.items()):
        raise RuntimeError("immutable run-local AdamW native/finalizer binary drift")
    if any(not Path(path).is_file() or _sha(Path(path)) != digest
           for path, digest in resources.items()):
        raise RuntimeError("signed behavioral DRAMSys/simulation resource drift")


def _hardware(root: Path, *, hbm_bytes: int) -> Path:
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 1 << 20
    hardware["memory"]["sram"]["capacity_bytes"] = 1 << 20
    hardware["memory"]["sram"]["regions"] = [{
        "name": "sram", "base_bytes": 0, "size_bytes": 1 << 20,
        "allocator": "block", "spillable": False,
        "access": ["compute", "dte", "lsu", "legacy", "noc_rx"],
    }]
    system = hardware["memory_system"]
    if (len(system["hbm_stacks"]) != 1
            or len(system["address_policy"]["home_ranges"]) != 1
            or system["hbm_stacks"][0]["backend"] != "behavioral"):
        raise RuntimeError("native real 1x1 HBM hardware topology drift")
    system["hbm_stacks"][0]["capacity_bytes"] = hbm_bytes
    system["address_policy"]["home_ranges"][0]["size_bytes"] = hbm_bytes
    system["address_policy"]["stack_interleave_bytes"] = 64
    system["address_policy"]["channel_interleave_bytes"] = 64
    physical_fabric_from_data(hardware, path="physical_adamw.hardware")
    path = root / "hardware.json"
    path.write_text(json.dumps(hardware, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8")
    return path


def _tools(args: argparse.Namespace, root: Path) -> tuple[Path, dict[str, str],
                                                          dict[str, str]]:
    tools = root / "tools"
    tools.mkdir()
    binaries = {}
    for name, src in (("npusim", args.npusim), ("npusim_program_finalizer",
                                               args.finalizer)):
        dest = tools / name
        shutil.copy2(src.resolve(), dest)
        binaries[name] = _sha(dest)
    dram = root / "DRAMSys"
    dram.mkdir()
    (dram / "configs").symlink_to(_DRAM, target_is_directory=True)
    if (dram / "configs").resolve() != _DRAM.resolve():
        raise RuntimeError("real DRAMSys configs symlink does not match source")
    resources = {str(path.resolve()): _sha(path) for path in (
        _DRAM / "hbm2-example.json",
        _DRAM / "addressmapping/am_hbm2_8Gb_pc_brc.json",
        _DRAM / "mcconfig/fr_fcfs.json",
        _DRAM / "memspec/HBM2.json",
        _DRAM / "simconfig/example.json",
        args.simulation.resolve(),
    )}
    return tools, binaries, resources


def _materialize(index: int, args: argparse.Namespace, root: Path,
                 tools: Path, binaries: dict[str, str], resources: dict[str, str],
                 hardware: Path, sources: dict[str, str]) -> dict[str, object]:
    material = root / f"materialization_{index}"
    material.mkdir()
    artifacts = material / "artifacts"
    artifacts.mkdir()
    start = time.monotonic()
    def compile_timeout(_signal_number: int, _frame: object) -> None:
        raise TimeoutError("AdamW materialization exceeded fixed compile budget")
    prev = signal.signal(signal.SIGALRM, compile_timeout)
    signal.alarm(args.compile_timeout)
    try:
        resident, physical, window, program, _ = build_source_dma_program()
        _source_oracle(resident, physical)
        original = tuple(compile_dense_adamw_step(window.materialization, physical, s)
                         for s in (0, 1))
        schedule = derive_dense_adamw_mid_program_residency(window, original)
        slots = assign_dense_adamw_bounded_slots(window, schedule)
        linked = tuple(compile_dense_adamw_paged_step(window, physical, s, slots)
                       for s in (0, 1))
        for step, item in enumerate(linked):
            _linked_oracle(item, window.materialization, physical, step)
        contract = build_dense_adamw_paged_runtime(
            window, schedule, slots, linked, program,
        )
        witness = prove_physical_adamw_authority_and_capacity(
            resident, window, original[0], contract, program,
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)
    _verify(sources, tools, binaries, resources)
    imported_now = _source_files()
    added_sources = {path: digest for path, digest in imported_now.items()
                     if path not in sources}
    sources.update(added_sources)
    _write(material / "source_postcompile.json", sources)
    compile_receipt = {
        "materialization_index": index,
        "compile_wall_seconds": round(time.monotonic() - start, 3),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "offload_request_digest": window.materialization.request_digest,
        "original_linked_manifest_digest": canonical_digest(original[0].manifest),
        "paged_linked_manifest_digests": list(contract.linked_manifest_digests),
        "physical_authority_digest": witness.digest,
        "physical_resident_capacity_failure": witness.resident_rejection_code,
        "physical_resident_capacity_error": witness.resident_rejection,
        "external_m_seed_digest": next(item.seed_digest for item in witness.roles
                                       if item.kind == "optimizer_moment1"),
        "external_v_seed_digest": next(item.seed_digest for item in witness.roles
                                       if item.kind == "optimizer_moment2"),
        "source_precompile_sha256": _sha(root / "source_precompile.json"),
        "source_postcompile_sha256": _sha(material / "source_postcompile.json"),
        "added_imported_sources": added_sources,
        "source_count": len(sources), "tools": binaries,
    }
    _write(material / "compiled_receipt.json", compile_receipt)
    _verify(sources, tools, binaries, resources)
    (artifacts / "external_dma_program.json").write_text(canonical_json(program))
    (material / "dense_adamw_paged_runtime.json").write_text(canonical_json(contract))
    (material / "physical_adamw_authority.json").write_text(canonical_json(witness))
    linked_paths = []
    npup_paths = []
    finalized = []
    for step, item in enumerate(linked):
        _verify(sources, tools, binaries, resources)
        source_path = material / f"step_{step}.linked.json"
        program_path = material / f"step_{step}.npup"
        final_report = material / f"step_{step}.finalizer.json"
        source_path.write_text(canonical_json(item.manifest))
        status = _stage((
            str(tools / "npusim_program_finalizer"), "--input", str(source_path),
            "--output", str(program_path), "--report", str(final_report),
        ), tools, material / f"step_{step}.finalizer.stdout.txt",
            timeout=args.stage_timeout)
        report = json.loads(final_report.read_text())
        if (report["artifact_sha256"] != _sha(program_path)
                or report["linked_manifest_id"] != item.manifest.id
                or report["linked_manifest_digest"] != canonical_digest(item.manifest)):
            raise RuntimeError("native typed finalizer or source binding digest mismatch")
        finalized.append({"npup_sha256": _sha(program_path),
                          "linked_sha256": _sha(source_path), "report": report,
                          "stage": status})
        linked_paths.append(source_path)
        npup_paths.append(program_path)
        _verify(sources, tools, binaries, resources)
    mapping = material / "mapping.spec"
    mapping.write_text("0:0\n")
    observations = []
    for fresh in (0, 1):
        _verify(sources, tools, binaries, resources)
        stdout_path = material / f"fresh_{fresh}.npusim.stdout.txt"
        native = _stage((
            str(tools / "npusim"), "--program-sequence",
            ",".join(str(p) for p in npup_paths),
            "--linked-manifest-sequence", ",".join(str(p) for p in linked_paths),
            "--dense-adamw-paged-runtime",
            str(material / "dense_adamw_paged_runtime.json"),
            "--hardware-config", str(hardware),
            "--simulation-config", str(args.simulation.resolve()),
            "--mapping-config", str(mapping), "--trace-window", "1000000",
        ), tools, stdout_path, timeout=args.stage_timeout)
        _verify(sources, tools, binaries, resources)
        stdout = stdout_path.read_text()
        old = _observe(stdout, contract)
        role = observe_native_adamw_role_values(stdout, witness)
        observation = {"native": old, "role_probes": role, "stage": native}
        observations.append(observation)
        _write(material / f"fresh_{fresh}.observation.json", observation)
    if observations[0]["native"] != observations[1]["native"] or (
        observations[0]["role_probes"] != observations[1]["role_probes"]
    ):
        raise RuntimeError("two independent native fresh optimizer instances drifted")
    return {"compiled": compile_receipt, "finalizers": finalized,
            "native_observation": observations[0],
            "native_instances": len(observations), "source_witness_digest": witness.digest,
            "request_case_id": window.materialization.request.case_id,
            "source_dma_program_id": program.id, "physical_role_values": role["roles"]}


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    tools, binaries, resources = _tools(args, root)
    hardware = _hardware(root, hbm_bytes=36864)
    sources = _source_files()
    sources[str(Path(__file__).resolve())] = _sha(Path(__file__).resolve())
    _write(root / "source_precompile.json", sources)
    _verify(sources, tools, binaries, resources)
    results = []
    try:
        for index in (0, 1):
            _verify(sources, tools, binaries, resources)
            results.append(_materialize(index, args, root, tools, binaries,
                                        resources, hardware, sources))
    except Exception as error:
        changed = {
            path: {"frozen_digest": digest,
                   "observed_digest": (_sha(Path(path)) if Path(path).is_file()
                                       else None)}
            for path, digest in sources.items()
            if not Path(path).is_file() or _sha(Path(path)) != digest
        }
        _write(root / "adamw_physical_blocked_receipt.json", {
            "status": "BLOCKED", "reason": str(error), "completed_materializations":
            len(results), "repeatability_verified": False,
            "source_drift": changed, "source_binding_sha256":
            _sha(root / "source_precompile.json"),
            "tool_sha256": binaries, "hardware_sha256": _sha(hardware),
            "wall_seconds": round(time.monotonic() - started, 3),
        })
        raise
    first, second = results
    stable = {
        "request_case_id", "source_dma_program_id", "source_witness_digest",
        "physical_role_values",
    }
    if any(first[key] != second[key] for key in stable) or (
        tuple(item["npup_sha256"] for item in first["finalizers"])
        != tuple(item["npup_sha256"] for item in second["finalizers"])
    ) or first["native_observation"]["native"] != second["native_observation"]["native"]:
        raise RuntimeError("independent AdamW materialize→finalize→native runtime drift")
    _verify(sources, tools, binaries, resources)
    report = {
        "scope": "1x1 H16/L2 AdamW 83-state physical m/v two-step timing only",
        "independent_materializations": 2, "fresh_native_instances": 4,
        "repeatability_verified": True,
        "numeric_adamw_correctness_verified": False,
        "four_family_full_model_m2_verified": False,
        "total_wall_seconds": round(time.monotonic() - started, 3),
        "source_binding_sha256": _sha(root / "source_precompile.json"),
        "postcompile_source_binding_sha256": _sha(
            root / "materialization_1/source_postcompile.json"),
        "source_count": len(sources), "source_drift": [],
        "hardware_sha256": _sha(hardware), "resources": resources,
        "run_local_binary_sha256": binaries,
        "chains": results,
    }
    _write(root / "adamw_physical_two_materializations_receipt.json", report)
    print("AdamW 1x1 83-StateABI actual m/v external native two-step "
          "two-independent-materialization PASS functional=0 numeric_correctness=false")
    return report


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npusim", type=Path,
                        default=_ROOT / "build-debug-adamw-role-values/npusim")
    parser.add_argument("--finalizer", type=Path,
                        default=_ROOT / "build-debug-final/npusim_program_finalizer")
    parser.add_argument("--simulation", type=Path,
                        default=_ROOT / "llm/test/simulation_config/default_spec.json")
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument("--stage-timeout", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    run(_args())
