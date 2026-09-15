"""Bounded L2 Dense activation checkpoint native timing canary.

One invocation uses a fresh empty directory and materializes both variants.
Run twice in separate Python processes for independent full-chain receipts.
The extra actions are a scoped timing overlay, not validated full training IR1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_checkpoint_eviction_graft import (
    graft_dense_checkpoint_eviction,
)
from llm.frontend.wafer_frontend.passes.dense_checkpoint_nonzero_program_io import (
    build_dense_checkpoint_nonzero_program_io,
)
from llm.frontend.wafer_frontend.passes.dense_checkpoint_physical_source import (
    derive_dense_checkpoint_activation_tape,
    derive_dense_checkpoint_physical_cut,
)
from llm.frontend.wafer_frontend.passes.dense_checkpoint_timing_overlay import (
    build_dense_checkpoint_timing_overlay,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.test.frontend.integration.flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)
from llm.test.frontend.integration.run_bounded_dense_seeded_ce_canary import (
    build_case,
)

_ROOT = Path(__file__).resolve().parents[4]
_SIM = _ROOT / "llm/test/program/p5_behavioral_simulation.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_REPLAY_ACTION = "global_action_bc57cb5c877160c6"
_BACKWARD_ACTION = "global_action_85f0379642a64773"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot() -> dict[str, str]:
    paths = {Path(__file__).resolve()}
    for module in sys.modules.values():
        value = getattr(module, "__file__", None)
        if value is None:
            continue
        path = Path(value).resolve()
        if path.suffix == ".py" and path.is_relative_to(_ROOT / "llm"):
            paths.add(path)
    return {str(path.relative_to(_ROOT)): _sha(path)
            for path in sorted(paths) if path.exists()}


def _assert_source(snapshot: dict[str, str]) -> None:
    for relative, digest in snapshot.items():
        if _sha(_ROOT / relative) != digest:
            raise RuntimeError(f"frozen imported source drift: {relative}")


def _assert_tools(tools: dict[str, tuple[Path, str]]) -> None:
    for role, (path, digest) in tools.items():
        if _sha(path) != digest:
            raise RuntimeError(f"frozen {role} binary drift")


def _hardware(capacity: int) -> dict[str, object]:
    if capacity not in (1408, 1472):
        raise RuntimeError("bounded L2 test capacity profile changed")
    h = json.loads(specialize_p5_large_release_hardware(1, 1))
    h["memory"]["sram_size"] = 65536
    h["memory"]["sram"]["capacity_bytes"] = 65536
    h["memory"]["sram"]["regions"] = [{
        "name": "sram", "base_bytes": 0, "size_bytes": 65536,
        "allocator": "block", "spillable": False,
        "access": ["compute", "dte", "lsu", "legacy", "noc_rx"],
    }]
    policy = h["memory_system"]["address_policy"]
    policy["home_ranges"] = [{"base": 0, "die_id": 0, "size_bytes": capacity}]
    policy["stack_interleave_bytes"] = capacity
    stacks = h["memory_system"]["hbm_stacks"]
    if len(stacks) != 1 or stacks[0]["compute_die_id"] != 0:
        raise RuntimeError("one-die HBM stack inventory changed")
    stacks[0]["capacity_bytes"] = capacity
    if (policy["channel_interleave_bytes"] != 64 or capacity % 64 or
            h["die"] != {"x": 1, "y": 1}):
        raise RuntimeError("illegal checkpoint physical HBM geometry")
    return h


def _runtime_evidence(stdout: str, *, checkpoint: bool,
                      activation_bytes: int) -> dict[str, object]:
    phases = re.findall(r"\[PROGRAM_IO\] phase=(\w+) mode=timing "
                        r"initializations=(\d+) probes=(\d+) "
                        r"checksum=([0-9a-f]{64}) pass=(\d+)", stdout)
    if ([x[0] for x in phases] != ["resolved", "applied", "verify"] or
            any((x[1], x[2], x[4]) != ("59", "2", "1") for x in phases)):
        raise RuntimeError("ProgramIO native three-phase proof incomplete")
    memory = re.findall(r"\[PROGRAM_MEMORY\] core=(\d+) lsu_issued=(\d+) "
                        r"lsu_completed=(\d+) lsu_hbm_read_bytes=(\d+) "
                        r"lsu_hbm_write_bytes=(\d+).* "
                        r"lsu_residual=(\d+) dte_residual=(\d+)", stdout)
    if len(memory) != 1 or memory[0][:3] != ("0", "17", "17") or \
            memory[0][4] != str(activation_bytes) or memory[0][5:] != ("0", "0"):
        raise RuntimeError("native LSU activation traffic/drain mismatch")
    hbm_probe = re.findall(r"\[PROGRAM_IO_PROBE\].*die=0 address=1344 "
                           r"bytes=(\d+).*valid=(\d+) exact=(\d+) pass=(\d+)",
                           stdout)
    if hbm_probe != [(str(activation_bytes), "1", "1", "1")]:
        raise RuntimeError("actual nonzero activation HBM probe failed")
    ce = re.findall(r"\[TRAIN_CE_BACKWARD\] core=(\d+) invocations=(\d+) "
                    r"rank_rows=(\d+).* logits_read_bytes=(\d+)", stdout)
    if ce != [("0", "1", "4", "128")]:
        raise RuntimeError("native backward consumer missing")
    matmul = len(re.findall(r"start compute primitive Matmul_f", stdout))
    if matmul != (10 if checkpoint else 9):
        raise RuntimeError("native replay MATMUL count mismatch")
    makespan = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", stdout)
    if len(makespan) != 1 or int(makespan[0]) <= 0 or \
            "[DRAIN] router_residual=0" not in stdout or \
            "[DRAIN] d2d_link_residual=0" not in stdout:
        raise RuntimeError("NpuSim result/drain incomplete")
    return {"program_io_phases": phases, "memory": memory[0],
            "activation_hbm_probe": hbm_probe[0],
            "matmul_f_invocations": matmul,
            "makespan_cycles": int(makespan[0]), "drain_zero": True}


def run_fresh(output: Path, finalizer: Path, npusim: Path) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("independent fresh output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    tool_dir = output / "tools"
    tool_dir.mkdir()
    local_tools = {}
    for role, original in (("finalizer", finalizer), ("npusim", npusim)):
        original = original.resolve()
        path = tool_dir / original.name
        shutil.copy2(original, path)
        local_tools[role] = (path, _sha(path))
    _assert_tools(local_tools)
    clock = time.monotonic()
    pre_materialization_snapshot = _source_snapshot()
    case = build_case()   # independent source carrier materialization
    cut = derive_dense_checkpoint_physical_cut(
        case.manifest, replay_action_id=_REPLAY_ACTION,
        backward_action_id=_BACKWARD_ACTION)
    if (cut.highest_parameter_end_bytes != 1344 or
            cut.checkpoint_saved_bytes != 32 or
            cut.no_checkpoint_saved_bytes != 128 or
            len(cut.persistent_parameter_states) != 15):
        raise RuntimeError("fixed L2 physical source inventory changed")
    low_rejection = None
    try:
        derive_dense_checkpoint_activation_tape(
            cut, checkpoint_enabled=False, hbm_capacity_bytes=1408)
    except SchemaError as error:
        low_rejection = str(error)
    if not low_rejection or "activation tape exceeds the real HBM home capacity" not in low_rejection:
        raise RuntimeError("same-model low-HBM resident negative did not reject")
    _assert_source(pre_materialization_snapshot)
    snapshot = _source_snapshot()
    results = {}
    for name, checkpoint, capacity in (("full_tape_no_recompute", False, 1472),
                                       ("checkpoint_replay", True, 1408)):
        _assert_source(snapshot)
        _assert_tools(local_tools)
        segment = output / name
        segment.mkdir()
        tape = derive_dense_checkpoint_activation_tape(
            cut, checkpoint_enabled=checkpoint, hbm_capacity_bytes=capacity)
        manifest = (graft_dense_checkpoint_eviction(case.manifest, cut, tape)
                    if checkpoint else
                    build_dense_checkpoint_timing_overlay(case.manifest, cut, tape).manifest)
        linked = segment / "linked.json"
        linked.write_text(canonical_json(manifest), encoding="utf-8")
        artifact = segment / "program.npup"
        report = segment / "finalizer_report.json"
        step = time.monotonic()
        fin = subprocess.run([
            str(local_tools["finalizer"][0]), "--input", str(linked),
            "--output", str(artifact), "--report", str(report)],
            capture_output=True, text=True, timeout=120, check=False)
        (segment / "finalizer.stdout").write_text(fin.stdout + fin.stderr)
        if fin.returncode or not artifact.exists():
            raise RuntimeError(f"{name} native finalizer failed: {fin.stdout} {fin.stderr}")
        finalizer_wall = time.monotonic() - step
        _assert_source(snapshot)
        _assert_tools(local_tools)
        activation = [state for fragment in manifest.fragments
                      for state in fragment.state_abi
                      if state.kind is StateKind.ACTIVATION]
        if len(activation) != 1 or activation[0].size_bytes != tape.size_bytes:
            raise RuntimeError("typed activation StateABI closure incomplete")
        contract = build_dense_checkpoint_nonzero_program_io(
            case.profile, manifest, case.graft, cut, tape, activation[0],
            _sha(artifact))
        sidecar = segment / "program_io.json"
        sidecar.write_text(canonical_json(contract), encoding="utf-8")
        seed = next(entry for entry in contract.initializations
                    if getattr(entry.target, "state_abi_id", None) ==
                       cut.replay_parameter_state.id)
        weight = next(blob.payload() for blob in contract.blobs if blob.id == seed.blob_ref)
        if not any(weight):
            raise RuntimeError("native replay W StateABI HBM seed is zero")
        hardware = segment / "hardware.json"
        hardware.write_text(json.dumps(_hardware(capacity), sort_keys=True,
                                       separators=(",", ":")), encoding="utf-8")
        low_native_negative = None
        if not checkpoint:
            low_hw = segment / "low_hbm_negative.hardware.json"
            low_hw.write_text(json.dumps(_hardware(1408), sort_keys=True,
                                         separators=(",", ":")), encoding="utf-8")
            negative = subprocess.run([
                str(local_tools["npusim"][0]), "--program", str(artifact),
                "--linked-manifest", str(linked), "--program-io", str(sidecar),
                "--hardware-config", str(low_hw),
                "--simulation-config", str(_SIM), "--mapping-config", str(_MAPPING),
                "--trace-window", "1000000"], cwd=_ROOT / "llm",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=60, text=True, check=False)
            (segment / "low_hbm_native_negative.stdout").write_text(
                negative.stdout, encoding="utf-8")
            if (negative.returncode == 0 or
                    "DecodeAddress: address not covered by any home range" not in negative.stdout or
                    "[SIM_RESULT]" in negative.stdout):
                raise RuntimeError("native low-HBM full-tape resident negative failed")
            low_native_negative = {
                "exit": negative.returncode,
                "hardware_sha256": _sha(low_hw),
                "stdout_sha256": _sha(segment / "low_hbm_native_negative.stdout"),
                "stdout_bytes": len(negative.stdout.encode()),
                "reason": "DecodeAddress: address not covered by any home range"}
        _assert_source(snapshot)
        _assert_tools(local_tools)
        step = time.monotonic()
        with (segment / "npusim.stdout").open("w", encoding="utf-8") as out:
            run = subprocess.run([
                str(local_tools["npusim"][0]), "--program", str(artifact),
                "--linked-manifest", str(linked), "--program-io", str(sidecar),
                "--hardware-config", str(hardware),
                "--simulation-config", str(_SIM), "--mapping-config", str(_MAPPING),
                "--trace-window", "1000000"],
                cwd=_ROOT / "llm", stdout=out, stderr=subprocess.STDOUT,
                timeout=600, check=False)
        runtime_wall = time.monotonic() - step
        stdout = (segment / "npusim.stdout").read_text(encoding="utf-8")
        if run.returncode or len(stdout) < 1000:
            raise RuntimeError(f"{name} native runtime failed exit={run.returncode}")
        native = _runtime_evidence(stdout, checkpoint=checkpoint,
                                   activation_bytes=tape.size_bytes)
        _assert_source(snapshot)
        _assert_tools(local_tools)
        results[name] = {
            "manifest_id": manifest.id, "manifest_sha256": _sha(linked),
            "activation_state_abi_id": activation[0].id,
            "activation_hbm_capacity_bytes": capacity,
            "activation_tape_bytes": tape.size_bytes,
            "artifact_sha256": _sha(artifact), "artifact_bytes": artifact.stat().st_size,
            "program_io_id": contract.id, "program_io_sha256": _sha(sidecar),
            "program_io_bytes": sidecar.stat().st_size,
            "hardware_sha256": _sha(hardware),
            "native_stdout_sha256": _sha(segment / "npusim.stdout"),
            "native_stdout_bytes": len(stdout.encode()),
            "finalizer_wall_seconds": round(finalizer_wall, 6),
            "runtime_wall_seconds": round(runtime_wall, 6),
            "low_hbm_native_negative": low_native_negative,
            **native,
        }
    _assert_source(snapshot)
    _assert_tools(local_tools)
    result = {
        "status": "PASS", "scope": "L2 one-core Dense forward plus native seeded CE backward timing carrier",
        "deep_source_status": "REJECTED", "deep_source_reason": case.source_gate,
        "numeric_gradient_equivalence_verified": False,
        "functional_matmul": False,
        "full_model_training_verified": False,
        "source_manifest_id": case.manifest.id,
        "source_manifest_sha256": hashlib.sha256(case.manifest_json.encode()).hexdigest(),
        "source_cut_id": cut.id, "source_cut_digest": cut.manifest_digest,
        "low_hbm_no_checkpoint_rejection": low_rejection,
        "imported_source_sha256": snapshot, "imported_source_count": len(snapshot),
        "tool_sha256": {role: digest for role, (_, digest) in local_tools.items()},
        "dram_config_sha256": _sha(_ROOT / "DRAMSys/configs/hbm2-example.json"),
        "simulation_sha256": _sha(_SIM), "mapping_sha256": _sha(_MAPPING),
        "wall_seconds": round(time.monotonic() - clock, 6),
        "results": results,
    }
    (output / "receipt.json").write_text(json.dumps(result, sort_keys=True,
                                                  indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_fresh(args.output, args.finalizer, args.npusim)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAIL", "reason": str(error),
                   "type": type(error).__name__,
                   "source_current_sha256": _source_snapshot(),
                   "tool_current_sha256": {
                       "finalizer": _sha(args.finalizer.resolve()),
                       "npusim": _sha(args.npusim.resolve())}}
        (args.output / "failure.json").write_text(
            json.dumps(failure, sort_keys=True, indent=2), encoding="utf-8")
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
