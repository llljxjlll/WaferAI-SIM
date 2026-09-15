"""True 1×4/TP4 two-step SGD motif with 60 authoritative external states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import time

from llm.frontend.wafer_frontend.passes.dense_training_physical_offload_program import (
    build_dense_training_physical_offload_program,
)
from llm.frontend.wafer_frontend.passes.dense_training_physical_offload_source import (
    build_dense_training_physical_offload_source,
)
from llm.frontend.wafer_frontend.passes.dense_training_physical_resident_capacity import (
    reject_physically_resident_dense_training,
)
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.external_state_authority import (
    defer_external_state_initializations,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides, build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RegionManifest
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json

from ..unit.test_dense_training_compile_sequence import _sequence
from .flexible_mesh_release_hardware import specialize_p5_large_release_hardware
from .tp4_external_sgd_stage import (
    _DRAM, _ROOT, _sha, _source_files, _stage, _verify_sources,
)
from .tp4_external_sgd_observer import (
    _validate_static_bindings, observe_runtime,
)


_MESH = (1, 4)
_DIES = (0, 1, 2, 3)
_PER_DIE_HBM = 9600
_EXTERNAL_CAPACITY = 32768


def _observe_tp4(stdout: str, *, hbm_bytes: int,
                 state_count: int, matmul_records: int) -> dict[str, object]:
    seen = observe_runtime(stdout, state_count=state_count,
                           hbm_bytes=hbm_bytes, matmul_records=matmul_records)
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
    n = str(state_count)
    b = str(hbm_bytes)
    if (ready != [(n, b, b)] or
            drain != [(n, b, b, b, b)] or
            absent != [(n, b)] or restored != [(n, b)]):
        raise RuntimeError(f"TP4 external/HBM exact restore and probe failed: "
                           f"{absent=} {ready=} {restored=} {drain=}")
    order = (
        "[EXTERNAL_AUTHORITY_PRELOAD]",
        "[EXTERNAL_DMA_READY]",
        "[EXTERNAL_AUTHORITY_RESTORED]",
        "[DENSE_TRAINING_SEQUENCE_STATE] version=0",
        "[DENSE_TRAINING_SEQUENCE_STEP] index=0",
        "[DENSE_TRAINING_SEQUENCE_STEP] index=1",
        "[EXTERNAL_DMA_DRAIN]",
        "[SIM_RESULT]",
    )
    positions = [stdout.find(token) for token in order]
    if any(index < 0 for index in positions) or positions != sorted(positions):
        raise RuntimeError(f"TP4 training ran before authoritative restore: {positions}")
    if seen.sgd_invocations != 120 or seen.versions != (0, 1, 2):
        raise RuntimeError("four true Die/two-step SGD/state versions drifted")
    return {
        "versions": seen.versions,
        "state_digests": seen.hbm_digests,
        "sgd_invocations": seen.sgd_invocations,
        "makespan_cycles": seen.makespan_cycles,
        "state_abis": state_count,
        "authoritative_bytes": hbm_bytes,
        "completed_dma": 2 * state_count,
        "external_probes": state_count,
        "pending_requests": 0,
        "functional": False,
    }


def _hardware(output: Path, *, capacity: int) -> Path:
    hardware = json.loads(specialize_p5_large_release_hardware(*_MESH))
    hardware["memory"]["sram_size"] = 1 << 20
    hardware["memory"]["sram"]["capacity_bytes"] = 1 << 20
    hardware["memory"]["sram"]["regions"] = [{
        "name": "sram", "base_bytes": 0, "size_bytes": 1 << 20,
        "allocator": "block", "spillable": False,
        "access": ["compute", "dte", "lsu", "legacy", "noc_rx"],
    }]
    stacks = hardware["memory_system"]["hbm_stacks"]
    homes = hardware["memory_system"]["address_policy"]["home_ranges"]
    if (len(stacks) != 4 or len(homes) != 4 or
            any(item["compute_die_id"] != index or
                homes[index]["die_id"] != index or
                homes[index]["base"] != index * (1 << 20)
                for index, item in enumerate(stacks))):
        raise RuntimeError("actual four-Die hardware home and stack changed")
    for stack, home in zip(stacks, homes):
        stack["capacity_bytes"] = capacity
        home["size_bytes"] = capacity
    hardware["memory_system"]["address_policy"]["allow_gaps"] = True
    # 0, 1, 2, 3 MiB linked homes align to a real 64-byte HBM stripe,
    # while the 9600-byte per-Die home is too small to be a base stripe.
    policy = hardware["memory_system"]["address_policy"]
    policy["stack_interleave_bytes"] = 64
    policy["channel_interleave_bytes"] = 64
    if any(
        home["base"] % policy["stack_interleave_bytes"] or
        home["size_bytes"] % policy["stack_interleave_bytes"] or
        home["base"] % policy["channel_interleave_bytes"] or
        home["size_bytes"] % policy["channel_interleave_bytes"] or
        home["size_bytes"] != stack["capacity_bytes"]
        for stack, home in zip(stacks, homes)
    ):
        raise RuntimeError("native address policy home/stack/channel stripe is invalid")
    physical_fabric_from_data(hardware, path="tp4_external_sgd.hardware")
    path = output / "hardware.json"
    path.write_text(json.dumps(hardware, sort_keys=True, separators=(",", ":")))
    return path


def _freeze_tools(args: argparse.Namespace, root: Path) -> tuple[Path, dict[str, str],
                                                               dict[str, str]]:
    tools = root / "tools"
    tools.mkdir(exist_ok=True)
    binaries = {}
    for name in ("npusim", "npusim_program_finalizer"):
        source = args.npusim if name == "npusim" else args.finalizer
        dest = tools / name
        shutil.copy2(source.resolve(), dest)
        binaries[name] = _sha(dest)
    dram = root / "DRAMSys"
    dram.mkdir(exist_ok=True)
    configs = dram / "configs"
    if not configs.exists():
        configs.symlink_to(_DRAM, target_is_directory=True)
    if configs.resolve() != _DRAM.resolve():
        raise RuntimeError("behavioral DRAMSys config moved after tool binding")
    configs_sha = {str(path.resolve()): _sha(path) for path in (
        _DRAM / "hbm2-example.json",
        _DRAM / "addressmapping/am_hbm2_8Gb_pc_brc.json",
        _DRAM / "mcconfig/fr_fcfs.json",
        _DRAM / "memspec/HBM2.json",
        _DRAM / "simconfig/example.json",
        args.simulation.resolve(),
    )}
    return tools, binaries, configs_sha


def _verify_frozen(sources: dict[str, str], tools: Path,
                   binaries: dict[str, str], resources: dict[str, str]) -> None:
    _verify_sources(sources)
    if any(_sha(tools / name) != digest for name, digest in binaries.items()):
        raise RuntimeError("immutable run-local TP4 tool binary changed")
    if any(not Path(path).is_file() or _sha(Path(path)) != digest
           for path, digest in resources.items()):
        raise RuntimeError("behavioral DRAMSys/simulation config moved")


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tools, binaries, resources = _freeze_tools(args, root)
    hardware_path = _hardware(root, capacity=_PER_DIE_HBM)
    sources_before = _source_files()
    (root / "source_precompile.json").write_text(
        json.dumps(sources_before, indent=2, sort_keys=True) + "\n"
    )
    compile_started = time.monotonic()

    def timeout_compile(_signum: int, _frame: object) -> None:
        raise TimeoutError("real TP4 production compile exceeded fixed stage budget")

    previous = signal.signal(signal.SIGALRM, timeout_compile)
    signal.alarm(args.compile_timeout)
    try:
        sequence = _sequence(*_MESH)
        sequence.validate()
    except Exception as error:
        (root / "compile_blocked.json").write_text(json.dumps({
            "status": "BLOCKED", "reason": str(error),
            "compile_wall_seconds": round(time.monotonic() - compile_started, 3),
            "compile_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "source_precompile_sha256": _sha(root / "source_precompile.json"),
            "binaries": binaries,
        }, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    try:
        _verify_sources(sources_before)
    except RuntimeError as error:
        changed = {
            path: {"precompile_sha256": digest,
                   "current_sha256": (_sha(Path(path)) if Path(path).is_file()
                                      else None)}
            for path, digest in sources_before.items()
            if not Path(path).is_file() or _sha(Path(path)) != digest
        }
        (root / "source_drift_blocked.json").write_text(json.dumps({
            "status": "BLOCKED_SOURCE_DRIFT",
            "stage": "postcompile_pre_finalizer",
            "reason": str(error), "changed_source_bytes": changed,
            "compile_wall_seconds": round(time.monotonic() - compile_started, 3),
            "compile_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "runtime_started": False, "compiled_object_discarded": True,
            "runlocal_tool_sha256": binaries,
        }, indent=2, sort_keys=True) + "\n")
        raise
    compile_stats = {
        "wall_seconds": round(time.monotonic() - compile_started, 3),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    state_count, matmul_records = _validate_static_bindings(sequence)
    linked = sequence.segments[0].linked_program
    seeds, expected = build_deterministic_timing_state_overrides(linked)
    if expected or state_count != 60 or len(seeds) != 60:
        raise RuntimeError("linked four-Die StateABI source coverage changed")
    hbm_bytes = sum(len(value) for value in seeds.values())
    if hbm_bytes != 19072:
        raise RuntimeError("physical 1×4 linked payload differs from canonical case")
    source = build_dense_training_physical_offload_source(
        sequence, hbm_capacity_bytes=_PER_DIE_HBM,
        external_capacity_bytes=_EXTERNAL_CAPACITY,
    )
    physical_rejection = reject_physically_resident_dense_training(
        sequence, source,
    )
    if (physical_rejection.linked_state_abi_count != state_count or
            physical_rejection.linked_state_logical_bytes != hbm_bytes or
            physical_rejection.linked_state_padding_bytes != 160 * len(_DIES) or
            physical_rejection.hbm_capacity_bytes_per_die != _PER_DIE_HBM or
            physical_rejection.rejection_code != "memory_capacity_exceeded"):
        raise RuntimeError("native 60-StateABI resident HBM rejection was not physical")
    (root / "physical_resident_rejection.json").write_text(
        canonical_json(physical_rejection)
    )
    case = build_dense_training_physical_offload_program(source, seeds)
    if (len(case.program.descriptors) != 120 or
            len(case.program.external_probes) != 60 or
            tuple(item.location_ref for item in source.hbm_capacities) !=
                tuple(f"die:{die}" for die in _DIES) or
            any(sum(item.logical_bytes for item in source.declarations
                    if item.die_id == die) != 4768 for die in _DIES)):
        raise RuntimeError("60 per-Die StateABI signed offload changed")
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    inputs = {
        "workload_manifest": source.external_manifest,
        "blocking_offload_plan": case.plan,
        "external_dma_action_graph": case.action_graph,
        "external_dma_program": case.program,
    }
    files = {}
    for name, value in inputs.items():
        path = (artifacts if name.startswith("external_dma") else root) / (name + ".json")
        path.write_text(canonical_json(value))
        files[name] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    binding_path = root / "external_dma_runtime_binding.json"
    binding_path.write_text(canonical_json(case.runtime_binding))
    mapping_path = root / "mapping.spec"
    mapping_path.write_text("0:0\n")
    sources = _source_files()
    (root / "source_binding.json").write_text(json.dumps(sources, indent=2,
                                                       sort_keys=True) + "\n")
    (root / "compiled_receipt.json").write_text(json.dumps({
        "status": "COMPILED", "mesh": "1x4", "active_dies": _DIES,
        "physical_state_abis": 60,
        "physical_state_bytes": hbm_bytes,
        "source_memory_plan_sha256": canonical_digest(source.external_manifest.memory_plan),
        "physical_source_digest": source.digest,
        "external_capacity_bytes": _EXTERNAL_CAPACITY,
        "per_die_hbm_capacity_bytes": _PER_DIE_HBM,
        "physical_resident_rejection": {
            "sha256": _sha(root / "physical_resident_rejection.json"),
            "digest": physical_rejection.digest,
            "state_abis": physical_rejection.linked_state_abi_count,
            "linked_state_logical_bytes": physical_rejection.linked_state_logical_bytes,
            "linked_state_padding_bytes": physical_rejection.linked_state_padding_bytes,
            "hbm_requests_digest": physical_rejection.hbm_requests_digest,
            "code": physical_rejection.rejection_code,
            "reason": physical_rejection.rejection,
        },
        "generic_p3_resident_rejection": source.resident_rejection,
        "compiled": compile_stats, "artifact_inputs": files,
        "source_binding_sha256": _sha(root / "source_binding.json"),
        "hardware_sha256": _sha(hardware_path),
        "immutable_binaries": binaries, "dram_resources_sha256": resources,
    }, indent=2, sort_keys=True) + "\n")
    _verify_frozen(sources, tools, binaries, resources)

    executions = []
    negative = None
    for index in (0, 1):
        fresh = root / f"execution_{index}"
        fresh.mkdir(exist_ok=True)
        stages = []
        for step, segment in enumerate(sequence.segments):
            _verify_frozen(sources, tools, binaries, resources)
            linked_path = fresh / f"step_{step}.linked.json"
            npup = fresh / f"step_{step}.npup"
            report_path = fresh / f"step_{step}.finalizer.json"
            program_io = fresh / f"step_{step}.program_io.json"
            linked_path.write_text(canonical_json(segment.linked_program.manifest))
            finalizer = _stage((
                str(tools / "npusim_program_finalizer"),
                "--input", str(linked_path), "--output", str(npup),
                "--report", str(report_path),
            ), tools, fresh / f"step_{step}.finalizer.stdout.txt",
                timeout=args.stage_timeout)
            report = json.loads(report_path.read_text())
            if (report.get("artifact_sha256") != _sha(npup) or
                    report.get("linked_manifest_digest") !=
                        canonical_digest(segment.linked_program.manifest)):
                raise RuntimeError("TP4 finalizer linked/real artifact binding failed")
            ordinary = build_timing_program_io(
                segment.linked_program, _sha(npup),
                state_seed_overrides=seeds,
            )
            deferred = defer_external_state_initializations(
                ordinary, segment.linked_program.manifest,
                case.program, seeds,
            )
            deferred.validate_against(segment.linked_program.manifest)
            if any(item.target.kind.value == "hbm"
                   for item in deferred.initializations):
                raise RuntimeError("physical linked training state still host-preloaded")
            program_io.write_text(canonical_json(deferred))
            if index == 0:
                negative_dir = root / "negative_preloaded_hbm"
                negative_dir.mkdir(exist_ok=True)
                (negative_dir / f"step_{step}.host_preload.program_io.json").write_text(
                    canonical_json(ordinary)
                )
            stages.append({
                "step": step,
                "linked_sha256": _sha(linked_path), "linked_bytes": linked_path.stat().st_size,
                "artifact_sha256": _sha(npup), "artifact_bytes": npup.stat().st_size,
                "program_io_sha256": _sha(program_io),
                "program_io_bytes": program_io.stat().st_size,
                "hbm_host_initializations": 0,
                "finalizer": finalizer,
            })
        _verify_frozen(sources, tools, binaries, resources)
        command_prefix = (
            str(tools / "npusim"),
            "--external-dma-binding", str(binding_path),
            "--program-sequence", ",".join(
                str(fresh / f"step_{step}.npup") for step in (0, 1)
            ),
            "--linked-manifest-sequence", ",".join(
                str(fresh / f"step_{step}.linked.json") for step in (0, 1)
            ),
        )
        command_suffix = (
            "--hardware-config", str(hardware_path),
            "--simulation-config", str(args.simulation.resolve()),
            "--mapping-config", str(mapping_path),
            "--trace-window", "1000000",
        )
        if index == 0:
            negative_dir = root / "negative_preloaded_hbm"
            host = tuple(negative_dir / f"step_{step}.host_preload.program_io.json"
                         for step in (0, 1))
            wrong = subprocess.run(
                (*command_prefix, "--program-io-sequence", ",".join(map(str, host)),
                 *command_suffix),
                cwd=tools, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False, text=True, timeout=args.stage_timeout,
            )
            (negative_dir / "npusim.stdout.txt").write_text(wrong.stdout)
            negative = {
                "exit_code": wrong.returncode,
                "hbm_host_initializations": 60,
                "no_external_ready_or_compute": not any(token in wrong.stdout for token in (
                    "[EXTERNAL_DMA_READY]", "[DENSE_TRAINING_SEQUENCE_STEP]",
                    "[SIM_RESULT]",
                )),
                "reason": "external training ProgramIO preloaded HBM StateABI",
            }
            (negative_dir / "negative_receipt.json").write_text(json.dumps(
                negative, indent=2, sort_keys=True
            ) + "\n")
            if (wrong.returncode != 2 or negative["reason"] not in wrong.stdout or
                    not negative["no_external_ready_or_compute"]):
                raise RuntimeError("old 60-HBM-host-preload case failed to fail-fast")
        stdout = fresh / "npusim.stdout.txt"
        runtime = _stage((*command_prefix,
                          "--program-io-sequence", ",".join(
                              str(fresh / f"step_{step}.program_io.json")
                              for step in (0, 1)
                          ), *command_suffix),
                         tools, stdout, timeout=args.stage_timeout)
        _verify_frozen(sources, tools, binaries, resources)
        observed = _observe_tp4(stdout.read_text(), hbm_bytes=hbm_bytes,
                                state_count=state_count,
                                matmul_records=matmul_records)
        executions.append({
            "index": index, "stages": stages, "runtime": runtime,
            "observed": observed,
        })
    a, b = executions
    if (a["observed"] != b["observed"] or
            [(item["linked_sha256"], item["artifact_sha256"],
              item["program_io_sha256"]) for item in a["stages"]] !=
            [(item["linked_sha256"], item["artifact_sha256"],
              item["program_io_sha256"]) for item in b["stages"]]):
        raise RuntimeError("two fresh physical four-Die SGD executions drifted")
    result = {
        "status": "PASS", "mesh": "1x4", "active_dies": _DIES,
        "case_id": case.program.case_digest, "optimizer": "sgd",
        "scope": "gradient-matmul-motif", "functional": False,
        "physical_state_abis": 60, "state_bytes_total": hbm_bytes,
        "per_die_state_bytes": [4768] * 4,
        "per_die_linked_span_bytes": [4928] * 4,
        "per_die_hbm_capacity_bytes": _PER_DIE_HBM,
        "external_capacity_bytes": _EXTERNAL_CAPACITY,
        "physical_resident_rejection": {
            "sha256": _sha(root / "physical_resident_rejection.json"),
            "digest": physical_rejection.digest,
            "state_abis": physical_rejection.linked_state_abi_count,
            "linked_state_logical_bytes": physical_rejection.linked_state_logical_bytes,
            "linked_state_padding_bytes": physical_rejection.linked_state_padding_bytes,
            "hbm_requests_digest": physical_rejection.hbm_requests_digest,
            "code": physical_rejection.rejection_code,
            "reason": physical_rejection.rejection,
        },
        "generic_p3_resident_rejection": source.resident_rejection,
        "preloaded_negative": negative,
        "compiled": compile_stats,
        "signed_input_receipt_sha256": _sha(root / "compiled_receipt.json"),
        "source_binding_sha256": _sha(root / "source_binding.json"),
        "source_count": len(sources),
        "source_drift": [],
        "tool_sha256": binaries,
        "hardware_sha256": _sha(hardware_path),
        "dram_resources_sha256": resources,
        "artifact_inputs": files,
        "executions": executions,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    (root / "evidence.json").write_text(json.dumps(result, indent=2,
                                                  sort_keys=True) + "\n")
    print("TP4 external AUTHORITATIVE PASS active=4 state_abis=60 "
          f"bytes={hbm_bytes} restore=60 writeback=60 two_fresh=2 "
          f"makespan={a['observed']['makespan_cycles']} "
          "scope=gradient-matmul-motif functional=0")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/dense-sgd-tp4-external-authority-canary"))
    parser.add_argument("--npusim", type=Path,
                        default=_ROOT / "build-debug-sgd-authority/npusim")
    parser.add_argument("--finalizer", type=Path,
                        default=_ROOT / "build-debug-final/npusim_program_finalizer")
    parser.add_argument("--simulation", type=Path,
                        default=_ROOT / "llm/test/program/p5_behavioral_simulation.json")
    parser.add_argument("--compile-timeout", type=int, default=1200)
    parser.add_argument("--stage-timeout", type=int, default=600)
    args = parser.parse_args()
    if (args.compile_timeout <= 0 or args.stage_timeout <= 0 or
            not all(getattr(args, name).is_file() for name in (
                "npusim", "finalizer", "simulation",
            ))):
        parser.error("fixed positive compile/runtime budgets and real tools required")
    return args


if __name__ == "__main__":
    run(_parse_args())
