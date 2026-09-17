"""One independent complete L2/two-step Dense AdamW low-HBM native Fresh.

The finite external tier holds all 75 initial StateABI payloads.  A genuine
LSU-gated DMA restores each load into one 4 KiB HBM die and writes every
store back before the next record.  Native AdamW remains timing-only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_adamw_offload_preflight import (
    derive_full_dense_adamw_offload_window,
    require_full_dense_adamw_resident_capacity,
)
from llm.frontend.wafer_frontend.passes.full_dense_adamw_paged import (
    rebase_full_dense_adamw_to_blocking_slot,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    _deterministic_timing_state_overrides, _resolved_state_abis,
)
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.test.frontend.unit.test_flexible_dense_train import _spec

from .flexible_mesh_release_hardware import specialize_p5_large_release_hardware
from .run_full_dense_adamw_two_step_native_fresh import _compile


_ROOT = Path(__file__).resolve().parents[4]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(command: list[str], cwd: Path, timeout: int, log: Path,
             *, expect_success: bool = True) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout, check=False)
    log.write_text(result.stdout, encoding="utf-8")
    if expect_success and result.returncode:
        raise RuntimeError(f"native command exit={result.returncode}; log={log}; "
                           f"tail={result.stdout[-3000:]}")
    if not expect_success and result.returncode == 0:
        raise RuntimeError("low-HBM resident-only native baseline unexpectedly succeeded")
    return result.stdout


@builder_validation_session()
def run(*, output: Path, npusim: Path, finalizer: Path, simulation: Path,
        timeout: int = 600) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("full AdamW offload Fresh requires an empty output root")
    output.mkdir(parents=True, exist_ok=True)
    tools = {name: path.resolve() for name, path in (
        ("npusim", npusim), ("finalizer", finalizer),
        ("simulation", simulation))}
    if any(not path.is_file() for path in tools.values()):
        raise RuntimeError("one native tool or simulation config is missing")
    source, physical = _compile()
    plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
    window = derive_full_dense_adamw_offload_window(
        source, plan, hbm_capacity_bytes=4096,
        external_capacity_bytes=8192, sram_capacity_bytes=1 << 20)
    try:
        require_full_dense_adamw_resident_capacity(window)
    except SchemaError as error:
        if error.code != "memory_capacity_exceeded":
            raise
    else:
        raise RuntimeError("same-model low-HBM resident-only preflight did not reject")
    # The capacity preflight above already deep-validated this exact source.
    seeds, _expected = _deterministic_timing_state_overrides(
        _resolved_state_abis(source))
    paged = rebase_full_dense_adamw_to_blocking_slot(
        source, physical, window, seeds)
    if (paged.state_payload_bytes != 5716 or paged.slot_bytes != 256 or
            len(paged.states) != 75 or len(paged.events) != 350):
        raise RuntimeError("full physical AdamW offload inventory drifted")
    source_manifest = output / "source.linked.json"
    paged_manifest = output / "paged.linked.json"
    physical_path = output / "physical_dag.json"
    physical_path.write_text(canonical_json(physical), encoding="utf-8")
    source_manifest.write_text(canonical_json(source.manifest), encoding="utf-8")
    paged_manifest.write_text(canonical_json(paged.paged_manifest), encoding="utf-8")
    sidecar = {
        "schema_version": "wafer_frontend.full_dense_adamw_paged_runtime/v1alpha1",
        "producer_pass": "full_dense_adamw_paged",
        "source_manifest_relative_path": source_manifest.name,
        "paged_manifest_relative_path": paged_manifest.name,
        "source_manifest_id": paged.source_manifest_id,
        "source_manifest_digest": paged.source_manifest_digest,
        "paged_manifest_id": paged.paged_manifest.id,
        "paged_manifest_digest": canonical_digest(paged.paged_manifest),
        "physical_dag_digest": paged.physical_dag_digest,
        "source_ir1_id": paged.source_ir1_id,
        "hbm_capacity_bytes": paged.hbm_capacity_bytes,
        "external_capacity_bytes": paged.external_capacity_bytes,
        "slot_address": paged.slot_address,
        "slot_bytes": paged.slot_bytes,
        "state_payload_bytes": paged.state_payload_bytes,
        "link_bytes_per_cycle": 32,
        "link_latency_cycles": 8,
        "queue_depth": 4,
        "max_outstanding": 4,
        "states": [asdict(item) for item in paged.states],
        "events": [asdict(item) for item in paged.events],
    }
    sidecar_path = output / "full_adamw_paged_runtime.json"
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True,
                                       separators=(",", ":")), encoding="utf-8")
    paged_npup = output / "paged.npup"
    source_npup = output / "source.npup"
    for name, manifest, program in (
        ("paged", paged_manifest, paged_npup),
        ("source", source_manifest, source_npup),
    ):
        report = output / f"{name}.finalizer.json"
        _command([str(tools["finalizer"]), "--input", str(manifest),
                  "--output", str(program), "--report", str(report)],
                 output, 180, output / f"{name}.finalizer.stdout.txt")
        data = json.loads(report.read_text(encoding="utf-8"))
        if data.get("artifact_sha256") != _sha(program):
            raise RuntimeError("native finalizer NPUP digest mismatch")
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 1 << 20
    hardware["memory"]["sram"]["capacity_bytes"] = 1 << 20
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = 1 << 20
    memory = hardware["memory_system"]
    if (len(memory["hbm_stacks"]) != 1 or
            memory["hbm_stacks"][0]["backend"] != "behavioral"):
        raise RuntimeError("expected one real behavioral HBM backend")
    memory["hbm_stacks"][0]["capacity_bytes"] = 4096
    memory["address_policy"]["home_ranges"][0]["size_bytes"] = 4096
    memory["address_policy"]["stack_interleave_bytes"] = 4096
    (output / "DRAMSys").symlink_to(_ROOT / "DRAMSys",
                                    target_is_directory=True)
    hardware_path = output / "hardware.json"
    hardware_path.write_text(json.dumps(hardware, sort_keys=True,
                                        separators=(",", ":")), encoding="utf-8")
    mapping = output / "mapping.spec"
    mapping.write_text("0:0\n", encoding="utf-8")
    resident_cwd = output / "resident_reject"
    resident_cwd.mkdir()
    rejected = _command([str(tools["npusim"]), "--program", str(source_npup),
        "--hardware-config", str(hardware_path),
        "--simulation-config", str(tools["simulation"]),
        "--mapping-config", str(mapping), "--trace-window", "1000000",
        "--program-one-shot"], resident_cwd, timeout,
        output / "resident_reject.stdout.txt", expect_success=False)
    if ("[SIM_RESULT] makespan_cycles=" in rejected or
            "DecodeAddress: address not covered by any home range"
            not in rejected):
        raise RuntimeError("resident-only native rejection was not the signed HBM home range")
    native_cwd = output / "offload_run"
    native_cwd.mkdir()
    stdout = _command([str(tools["npusim"]), "--program", str(paged_npup),
        "--full-dense-adamw-paged-runtime", str(sidecar_path),
        "--hardware-config", str(hardware_path),
        "--simulation-config", str(tools["simulation"]),
        "--mapping-config", str(mapping), "--trace-window", "1000000",
        "--program-one-shot"], native_cwd, timeout,
        output / "offload.stdout.txt")
    events = re.findall(
        r"\[FULL_DENSE_ADAMW_DMA_EVENT\] index=(\d+) step=(\d+) "
        r"program_record=(\d+) direction=(\w+) state_ref=(\S+) "
        r"bytes=(\d+) issue_cycle=(\d+) completed_at_ticks=(\d+) "
        r"lsu_dependency_complete=1 pass=1", stdout)
    if len(events) != 350 or any(
        int(actual[0]) != expected.index or
        int(actual[1]) != expected.step or
        int(actual[2]) != expected.program_record_index or
        actual[3] != expected.kind or actual[4] != expected.state_ref or
        int(actual[5]) != expected.size_bytes
        for actual, expected in zip(events, paged.events)
    ):
        raise RuntimeError("350 real LSU-gated DMA completions changed signed source order")
    authority = re.findall(
        r"\[FULL_DENSE_ADAMW_EXTERNAL_STATE\] version=(\d+) "
        r"states=(\d+) bytes=(\d+) digest=([0-9a-f]{64}) "
        r"pending=(\d+) functional=(\d+) pass=(\d+)", stdout)
    if (len(authority) != 3 or
            [tuple(item[:3]) for item in authority] !=
            [("0", "75", "5716"), ("1", "75", "5716"),
             ("2", "75", "5716")] or
            any(item[4:] != ("0", "0", "1") for item in authority)):
        raise RuntimeError("external authority did not reach version 0→1→2")
    drain = re.findall(
        r"\[FULL_DENSE_ADAMW_PAGED_DMA_DRAIN\] events=(\d+) "
        r"submitted=(\d+) completed=(\d+) external_read_bytes=(\d+) "
        r"external_write_bytes=(\d+) hbm_read_bytes=(\d+) "
        r"hbm_write_bytes=(\d+) pending=(\d+) pass=(\d+)", stdout)
    if drain != [("350", "350", "350", "14584", "11432",
                  "11432", "14584", "0", "1")]:
        raise RuntimeError(f"full native external/HBM byte/drain oracle failed: {drain}")
    cycles = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", stdout)
    if (len(cycles) != 1 or stdout.count("[TRAIN_ADAMW] core=") != 30 or
            stdout.count("[TRAIN_CE] core=") != 2 or
            stdout.count("[TRAIN_CE_BACKWARD] core=") != 2 or
            "[DRAIN] router_residual=0" not in stdout):
        raise RuntimeError("full offloaded CE/backward/AdamW native work incomplete")
    receipt = {
        "schema_version": "full_dense_adamw_paged_native_fresh/v1",
        "mesh": [1, 1], "layers": 2, "steps": 2,
        "physical_actions": len(physical.actions),
        "state_version_edges": len(physical.state_version_edges),
        "states": len(paged.states), "state_payload_bytes": paged.state_payload_bytes,
        "resident_hbm_highwater_bytes": window.resident_hbm_highwater_bytes,
        "hbm_capacity_bytes": 4096, "external_capacity_bytes": 8192,
        "slot_bytes": paged.slot_bytes,
        "resident_only_preflight_rejected": True,
        "resident_only_native_rejected": True,
        "offload_dma_events": len(events),
        "external_read_bytes": 14584,
        "external_write_bytes": 11432,
        "authority_digests": [item[3] for item in authority],
        "adamw_updates": 30, "makespan_cycles": int(cycles[0]),
        "numeric_mode": "timing", "model_functional_verified": False,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip(),
        "files_sha256": {name: _sha(path) for name, path in {
            "physical_dag": physical_path,
            "source_linked": source_manifest,
            "paged_linked": paged_manifest,
            "sidecar": sidecar_path, "source_npup": source_npup,
            "paged_npup": paged_npup, "hardware": hardware_path,
            "native_stdout": output / "offload.stdout.txt",
            "resident_reject_stdout": output / "resident_reject.stdout.txt",
            **tools,
        }.items()},
    }
    (output / "evidence.json").write_text(json.dumps(receipt,
        ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "npusim", "finalizer", "simulation"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    run(output=args.output, npusim=args.npusim,
        finalizer=args.finalizer, simulation=args.simulation,
        timeout=args.timeout)


if __name__ == "__main__":
    main()
