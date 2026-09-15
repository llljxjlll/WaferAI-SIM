"""Two fresh one-die two-step SGD runs requiring actual external state restore."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

from llm.frontend.wafer_frontend.passes.external_state_authority import (
    defer_external_state_initializations,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryObjectKind
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json

from .flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)
from .run_dense_external_offload_runtime_canary import _build_offload_case
from .run_dense_training_sequence_runtime_canary import (
    _missing_full_model_opcodes,
    _offload_sequence,
    _validate_static_bindings,
    observe_runtime,
)


_ROOT = Path(__file__).resolve().parents[4]
_DRAM = _ROOT / "DRAMSys/configs"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(
    command: tuple[str, ...], cwd: Path, stdout: Path, *, timeout: int,
) -> dict[str, object]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=False, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout.write_text(str(error.stdout or "") + "\nTIMEOUT\n")
        receipt = {"wall_seconds": round(time.monotonic() - started, 3),
                   "exit_code": None, "reason": f"stage exceeded {timeout}s",
                   "command": command}
        stdout.with_suffix(".stage.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(receipt["reason"]) from error
    stdout.write_text(result.stdout, encoding="utf-8")
    receipt = {
        "wall_seconds": round(time.monotonic() - started, 3),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "exit_code": result.returncode,
        "command": command,
    }
    stdout.with_suffix(".stage.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"stage exit={result.returncode} stdout={stdout}: "
            + result.stdout[-1200:]
        )
    return receipt


def _source_files() -> dict[str, str]:
    root = (_ROOT / "llm").resolve()
    sources = {
        str(path.resolve()): _sha(path)
        for module in tuple(sys.modules.values())
        for value in (getattr(module, "__file__", None),)
        if value is not None
        for path in (Path(value),)
        if path.suffix == ".py" and path.resolve().is_relative_to(root)
    }
    sources[str(Path(__file__).resolve())] = _sha(Path(__file__).resolve())
    return dict(sorted(sources.items()))


def _verify_sources(sources: dict[str, str]) -> None:
    drift = [
        name for name, digest in sources.items()
        if not Path(name).is_file() or _sha(Path(name)) != digest
    ]
    if drift:
        raise RuntimeError(f"source bytes drifted during authoritative SGD: {drift[:3]}")


def _observe(stdout: str, *, hbm_bytes: int, state_count: int,
             matmul_records: int) -> dict[str, object]:
    seen = observe_runtime(
        stdout, state_count=state_count, hbm_bytes=hbm_bytes,
        matmul_records=matmul_records,
    )
    ready = re.findall(
        r"\[EXTERNAL_DMA_READY\].*completed=(\d+).*"
        r"external_read_bytes=(\d+).*hbm_write_bytes=(\d+).*pending=0",
        stdout,
    )
    drain = re.findall(
        r"\[EXTERNAL_DMA_DRAIN\] probes=(\d+).*"
        r"external_read_bytes=(\d+).*external_write_bytes=(\d+).*"
        r"hbm_read_bytes=(\d+).*hbm_write_bytes=(\d+).*pending=0",
        stdout,
    )
    absent = re.findall(
        r"\[EXTERNAL_AUTHORITY_PRELOAD\] state_abis=(\d+) "
        r"hbm_initializations=0 present_bytes=0 source_bytes=(\d+) pass=1",
        stdout,
    )
    restored = re.findall(
        r"\[EXTERNAL_AUTHORITY_RESTORED\] state_abis=(\d+) "
        r"payload_bytes=(\d+) matched=1 pending=0 pass=1",
        stdout,
    )
    expected = str(hbm_bytes)
    if (ready != [("1", expected, expected)] or
        drain != [("1", expected, expected, expected, expected)] or
        absent != [(str(state_count), expected)] or
        restored != [(str(state_count), expected)]):
        raise RuntimeError(
            f"real external absence/restore/final writeback drifted: "
            f"{absent=} {ready=} {restored=} {drain=}"
        )
    tokens = (
        "[EXTERNAL_AUTHORITY_PRELOAD]",
        "[EXTERNAL_DMA_READY]",
        "[EXTERNAL_AUTHORITY_RESTORED]",
        "[DENSE_TRAINING_SEQUENCE_STATE] version=0",
        "[DENSE_TRAINING_SEQUENCE_STEP] index=0",
        "[DENSE_TRAINING_SEQUENCE_STEP] index=1",
        "[EXTERNAL_DMA_DRAIN]",
        "[SIM_RESULT]",
    )
    positions = [stdout.find(token) for token in tokens]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise RuntimeError(f"external restore was not before SGD compute: {positions}")
    return {
        "state_versions": list(seen.versions),
        "state_digests": list(seen.hbm_digests),
        "sgd_invocations": seen.sgd_invocations,
        "makespan_cycles": seen.makespan_cycles,
        "state_abis": state_count,
        "authoritative_bytes": hbm_bytes,
        "completed_dma": 2,
        "external_probes": 1,
        "pending_requests": 0,
        "functional": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    frozen = {}
    for name in ("npusim", "npusim_program_finalizer"):
        src = args.npusim if name == "npusim" else args.finalizer
        dst = tools / name
        shutil.copy2(src.resolve(), dst)
        frozen[name] = _sha(dst)
    dram = root / "DRAMSys"
    dram.mkdir(exist_ok=True)
    dram_link = dram / "configs"
    if not dram_link.exists():
        dram_link.symlink_to(_DRAM, target_is_directory=True)
    if dram_link.resolve() != _DRAM.resolve():
        raise RuntimeError("run-local DRAMSys configs symlink drifted")
    config_sha = {
        str(path.resolve()): _sha(path)
        for path in (_DRAM / "hbm2-example.json",
                     _DRAM / "addressmapping/am_hbm2_8Gb_pc_brc.json",
                     _DRAM / "mcconfig/fr_fcfs.json",
                     _DRAM / "memspec/HBM2.json",
                     _DRAM / "simconfig/example.json",
                     args.simulation.resolve())
    }

    sequence = _offload_sequence()
    sequence.validate()
    state_count, matmul_records = _validate_static_bindings(sequence)
    linked = sequence.segments[0].linked_program
    seeds, expected = build_deterministic_timing_state_overrides(linked)
    if expected or len(seeds) != state_count:
        raise RuntimeError("linked load-before-store StateABI seed coverage drifted")
    state_abis = linked.manifest.fragments[0].fragment.state_abi if hasattr(
        linked.manifest.fragments[0], "fragment",
    ) else linked.manifest.fragments[0].state_abi
    ordered = tuple(sorted(state_abis, key=lambda item: item.address))
    payload = bytearray()
    for abi in ordered:
        if abi.address != len(payload) or len(seeds[abi.state_ref]) != abi.size_bytes:
            raise RuntimeError("signed SGD StateABI does not form packed image")
        payload.extend(seeds[abi.state_ref])
    hbm_bytes = len(payload)
    peak = max(item.peak_bytes
               for item in sequence.materialization.memory_plan.peaks)
    parameter = sum(item.size_bytes
                    for item in sequence.materialization.state_inventory
                    if item.object_kind is MemoryObjectKind.PARAMETER)
    capacity = peak - parameter
    if parameter != hbm_bytes or capacity != 34880:
        raise RuntimeError("original one-die bounded physical fixture drifted")
    manifest, plan, program, action_graph, binding, hbm, state_bytes, rejection, _, resident_peak = (
        _build_offload_case(
            sequence.materialization, hbm_bytes_override=capacity,
            dirty_writeback=True, payload_override=bytes(payload),
        )
    )
    if (hbm, state_bytes, resident_peak) != (capacity, hbm_bytes, peak) or (
        "memory_capacity_exceeded" not in rejection
    ):
        raise RuntimeError("same-model same-low-HBM resident failure was not proven")
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (root / "workload_manifest.json").write_text(canonical_json(manifest))
    (root / "blocking_offload_plan.json").write_text(canonical_json(plan))
    (artifacts / "external_dma_program.json").write_text(canonical_json(program))
    (artifacts / "external_dma_action_graph.json").write_text(
        canonical_json(action_graph)
    )
    binding_path = root / "external_dma_runtime_binding.json"
    binding_path.write_text(canonical_json(binding))
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 1 << 20
    hardware["memory"]["sram"]["capacity_bytes"] = 1 << 20
    hardware["memory"]["sram"]["regions"] = [{
        "name": "sram", "base_bytes": 0, "size_bytes": 1 << 20,
        "allocator": "block", "spillable": False,
        "access": ["compute", "dte", "lsu", "legacy", "noc_rx"],
    }]
    hardware["memory_system"]["hbm_stacks"][0]["capacity_bytes"] = capacity
    hardware["memory_system"]["address_policy"]["home_ranges"][0][
        "size_bytes"
    ] = capacity
    hardware["memory_system"]["address_policy"]["stack_interleave_bytes"] = capacity
    hardware_path = root / "hardware.json"
    hardware_path.write_text(
        json.dumps(hardware, sort_keys=True, separators=(",", ":"))
    )
    mapping_path = root / "mapping.spec"
    mapping_path.write_text("0:0\n")
    sources = _source_files()
    (root / "source_binding.json").write_text(
        json.dumps(sources, indent=2, sort_keys=True) + "\n"
    )

    executions = []
    preloaded_rejection = None
    started = time.monotonic()
    for index in (0, 1):
        _verify_sources(sources)
        fresh = root / f"execution_{index}"
        fresh.mkdir(exist_ok=True)
        stage_rows = []
        for step, segment in enumerate(sequence.segments):
            _verify_sources(sources)
            if any(_sha(tools / name) != digest
                   for name, digest in frozen.items()):
                raise RuntimeError("run-local binary SHA changed before finalization")
            linked_path = fresh / f"step_{step}.linked.json"
            artifact = fresh / f"step_{step}.npup"
            report_path = fresh / f"step_{step}.finalizer.json"
            sidecar_path = fresh / f"step_{step}.program_io.json"
            linked_path.write_text(canonical_json(segment.linked_program.manifest))
            _stage((
                str(tools / "npusim_program_finalizer"),
                "--input", str(linked_path), "--output", str(artifact),
                "--report", str(report_path),
            ), tools, fresh / f"step_{step}.finalizer.stdout.txt",
                timeout=args.timeout)
            finalizer = json.loads(report_path.read_text())
            if (finalizer["artifact_sha256"] != _sha(artifact) or
                finalizer["linked_manifest_digest"] !=
                    canonical_digest(segment.linked_program.manifest)):
                raise RuntimeError("real SGD finalizer linked/artifact binding drifted")
            contract = build_timing_program_io(
                segment.linked_program, _sha(artifact),
                state_seed_overrides=seeds,
            )
            deferred = defer_external_state_initializations(
                contract, segment.linked_program.manifest, program, seeds,
            )
            deferred.validate_against(segment.linked_program.manifest)
            if any(item.target.kind.value == "hbm"
                   for item in deferred.initializations):
                raise RuntimeError("external StateABI still preloaded from ProgramIO")
            sidecar_path.write_text(canonical_json(deferred))
            stage_rows.append({
                "step": step,
                "linked_sha256": _sha(linked_path),
                "linked_bytes": linked_path.stat().st_size,
                "artifact_sha256": _sha(artifact),
                "artifact_bytes": artifact.stat().st_size,
                "program_io_sha256": _sha(sidecar_path),
                "program_io_bytes": sidecar_path.stat().st_size,
                "hbm_host_initializations": 0,
            })
        if index == 0:
            negative = root / "negative_preloaded_hbm"
            negative.mkdir(exist_ok=True)
            sidecars = []
            for step, segment in enumerate(sequence.segments):
                ordinary = build_timing_program_io(
                    segment.linked_program, stage_rows[step]["artifact_sha256"],
                    state_seed_overrides=seeds,
                )
                ordinary.validate_against(segment.linked_program.manifest)
                original = negative / f"step_{step}.host_preload.program_io.json"
                original.write_text(canonical_json(ordinary))
                sidecars.append(original)
            wrong_command = (
                str(tools / "npusim"),
                "--external-dma-binding", str(binding_path),
                "--program-sequence", ",".join(
                    str(fresh / f"step_{step}.npup") for step in (0, 1)
                ),
                "--linked-manifest-sequence", ",".join(
                    str(fresh / f"step_{step}.linked.json") for step in (0, 1)
                ),
                "--program-io-sequence", ",".join(map(str, sidecars)),
                "--hardware-config", str(hardware_path),
                "--simulation-config", str(args.simulation.resolve()),
                "--mapping-config", str(mapping_path),
                "--trace-window", "1000000",
            )
            candidate = subprocess.run(
                wrong_command, cwd=tools, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                timeout=args.timeout,
            )
            (negative / "npusim.stdout.txt").write_text(candidate.stdout)
            preloaded_rejection = {
                "exit_code": candidate.returncode,
                "old_hbm_host_initializations": sum(
                    item.target.kind.value == "hbm"
                    for item in ordinary.initializations
                ),
                "no_external_ready_or_compute": all(
                    token not in candidate.stdout for token in (
                        "[EXTERNAL_DMA_READY]",
                        "[DENSE_TRAINING_SEQUENCE_STEP]",
                        "[SIM_RESULT]",
                    )
                ),
                "reason": "external training ProgramIO preloaded HBM StateABI",
            }
            (negative / "negative_receipt.json").write_text(
                json.dumps(preloaded_rejection, indent=2, sort_keys=True) + "\n"
            )
            if (candidate.returncode != 2 or
                preloaded_rejection["old_hbm_host_initializations"] !=
                    state_count or
                preloaded_rejection["reason"] not in candidate.stdout or
                not preloaded_rejection["no_external_ready_or_compute"]):
                raise RuntimeError(
                    "ordinary host-preloaded StateABI negative was not "
                    "rejected before actual external DMA/compute"
                )
        if any(_sha(tools / name) != digest for name, digest in frozen.items()):
            raise RuntimeError("run-local binary SHA changed before simulation")
        _verify_sources(sources)
        npusim_stdout = fresh / "npusim.stdout.txt"
        runtime = _stage((
            str(tools / "npusim"),
            "--external-dma-binding", str(binding_path),
            "--program-sequence", ",".join(
                str(fresh / f"step_{step}.npup") for step in (0, 1)
            ),
            "--linked-manifest-sequence", ",".join(
                str(fresh / f"step_{step}.linked.json") for step in (0, 1)
            ),
            "--program-io-sequence", ",".join(
                str(fresh / f"step_{step}.program_io.json") for step in (0, 1)
            ),
            "--hardware-config", str(hardware_path),
            "--simulation-config", str(args.simulation.resolve()),
            "--mapping-config", str(mapping_path),
            "--trace-window", "1000000",
        ), tools, npusim_stdout, timeout=args.timeout)
        _verify_sources(sources)
        if any(_sha(tools / name) != digest for name, digest in frozen.items()):
            raise RuntimeError("run-local binary SHA changed after simulation")
        executions.append({
            "index": index, "stages": stage_rows, "runtime": runtime,
            "observed": _observe(
                npusim_stdout.read_text(), hbm_bytes=hbm_bytes,
                state_count=state_count, matmul_records=matmul_records,
            ),
        })
    left, right = executions
    if left["observed"] != right["observed"] or [
        (row["linked_sha256"], row["artifact_sha256"], row["program_io_sha256"])
        for row in left["stages"]
    ] != [
        (row["linked_sha256"], row["artifact_sha256"], row["program_io_sha256"])
        for row in right["stages"]
    ]:
        raise RuntimeError("two independent external SGD executions drifted")
    receipt = {
        "case_id": program.case_digest,
        "mesh": "1x1", "active_dies": [0], "optimizer": "sgd",
        "state_versions": [0, 1, 2], "functional": False,
        "model_scope": (
            "full-model" if not _missing_full_model_opcodes(linked)
            else "gradient-matmul-motif"
        ),
        "hbm_capacity_bytes": capacity, "resident_peak_bytes": peak,
        "resident_rejection": rejection, "external_authority_bytes": hbm_bytes,
        "source_binding_sha256": _sha(root / "source_binding.json"),
        "tool_sha256": frozen, "hardware_sha256": _sha(hardware_path),
        "preloaded_program_io_rejection": preloaded_rejection,
        "dram_resources_sha256": config_sha, "executions": executions,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    (root / "evidence.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"External SGD AUTHORITATIVE PASS mesh=1x1 active=1 "
        f"state_abis={state_count} bytes={hbm_bytes} two_fresh=2 "
        f"makespan={left['observed']['makespan_cycles']} "
        f"scope={receipt['model_scope']} functional=0"
    )
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/dense-sgd-external-authority-canary"))
    parser.add_argument("--npusim", type=Path,
                        default=_ROOT / "build-debug-sgd-authority/npusim")
    parser.add_argument("--finalizer", type=Path,
                        default=_ROOT / "build-debug-final/npusim_program_finalizer")
    parser.add_argument("--simulation", type=Path,
                        default=_ROOT / "llm/test/program/p5_behavioral_simulation.json")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.timeout <= 0 or not all(
        getattr(args, name).is_file() for name in ("npusim", "finalizer", "simulation")
    ):
        parser.error("positive timeout and real npusim/finalizer/simulation files required")
    return args


if __name__ == "__main__":
    run(_parse_args())
