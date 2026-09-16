"""Run measured TP16 Dense inference on an implementation-only rectangular mesh.

Each shape is compiled once into three immutable linked manifests, then each
independent execution finalizes, resolves, and runs all three segments afresh.
The 1..10 release hardware specializer and published workload matrix are never
used as evidence for this extended experimental scope.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
    compile_dense_e2e_sequence_runtime_profiles,
)
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.program_io import _build_timing_program_io_prevalidated
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RegionManifest
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.ir2 import StateUseAccess
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTier, MemoryTierCapacity
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadInferenceSteps,
    WorkloadMeshSpec,
    WorkloadParallelSpec,
    WorkloadRunCapability,
    WorkloadRunRequest,
    WorkloadStepSpec,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_legacy_dense_backend import (
    _capability,
    _legacy_spec,
    _request,
)

from .flexible_mesh_release_hardware import p5_large_hardware_template_json
from .run_dense_sequence_runtime_canary import _bind_native_hardware_to_fabric


_ROOT = Path(__file__).resolve().parents[4]
_TP = 16
_SRAM_BYTES = 1 << 20
_HBM_BYTES_PER_DIE = 1 << 20
_SHAPES = {"1x16": (1, 16), "16x1": (16, 1)}
_KV_BYTES = (4_096, 8_192, 12_288)


def build_case(rows: int, columns: int):
    """Construct an independently dimensioned, full-participation TP16 case."""

    mesh = RectMeshSpec(rows, columns)
    mesh.validate()
    if mesh.within_release_envelope or mesh.rank_count != _TP:
        raise ValueError("extended Dense TP16 canary requires 16 dies outside release")
    base = _request(layers=2, prefill=1, decode=2)
    model = replace(
        base.model, num_attention_heads=16, num_kv_heads=16, head_dim=2,
    )
    request = WorkloadRunRequest.create(
        family=base.family,
        model=model,
        steps=WorkloadStepSpec(
            inference=WorkloadInferenceSteps(1, 2, _TP),
        ),
        mesh=WorkloadMeshSpec(rows, columns),
        parallel=WorkloadParallelSpec(
            tp=_TP, active_die_ids=tuple(range(_TP)),
        ),
        memory=base.memory,
        execution=base.execution,
    )
    # This request-local capability is an input to materialization, not a
    # published family/runtime declaration for all shapes up to 16x16.
    capability = WorkloadRunCapability.create(
        max_mesh_rows=rows,
        max_mesh_columns=columns,
        max_mesh_ranks=_TP,
        families=_capability().families,
    )
    capacities = tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{rank}",
            base_address=0,
            capacity_bytes=_HBM_BYTES_PER_DIE,
            alignment_bytes=64,
        )
        for rank in range(_TP)
    )
    manifest = materialize_workload_preflight(
        request, capability, capacities=capacities,
    )
    if manifest.placement.active_die_ids != tuple(range(_TP)):
        raise RuntimeError("materialization does not use every physical die")
    fabric = physical_fabric_from_data(
        minimal_hardware(columns, rows, sram_bytes=_SRAM_BYTES)
    )
    template = _legacy_spec(layers=2, prefill=16, decode=0)
    template = replace(
        template,
        model=replace(
            template.model, NH=16, KVH=16, DH=2, rotary_dim=2,
        ),
        parallel=replace(
            template.parallel,
            instances=(
                replace(template.parallel.instances[0], tp=_TP, sp=True),
            ),
        ),
    )
    template.validate()
    spaces = valid_hbm_address_spaces(
        fabric, size_bytes_per_die=_HBM_BYTES_PER_DIE,
    )
    return manifest, template, fabric, spaces


def extended_hardware(rows: int, columns: int, spaces, fabric=None) -> str:
    """Specialize a private runtime document bound to the actual Fabric grid."""

    if (rows, columns) not in _SHAPES.values():
        raise ValueError("experimental hardware only supports measured TP16 shapes")
    if fabric is None:
        fabric = physical_fabric_from_data(
            minimal_hardware(columns, rows, sram_bytes=_SRAM_BYTES)
        )
    hardware = json.loads(p5_large_hardware_template_json())
    hardware["die"] = {"x": columns, "y": rows}
    hardware["die_ports"]["overrides"] = [
        {"side": side, "idx": 0, "role": "c2c", "dir": side}
        for side in ("N", "E", "S", "W")
        if (side in ("N", "S") and rows > 1)
        or (side in ("E", "W") and columns > 1)
    ]
    _bind_native_hardware_to_fabric(hardware, fabric)
    memory = hardware["memory"]
    memory["sram_size"] = _SRAM_BYTES
    memory["sram"]["capacity_bytes"] = _SRAM_BYTES
    # The three linked programs define exactly one 1MiB SRAM region named
    # "sram".  Template double_b/input/comm/etc would overlap a full-span
    # region; match its name, physical partition, and all initiator access.
    memory["sram"]["regions"] = [{
        "name": "sram", "base_bytes": 0, "size_bytes": _SRAM_BYTES,
        "allocator": "block", "spillable": False,
        "access": ["compute", "dte", "lsu", "legacy", "noc_rx"],
    }]
    system = hardware["memory_system"]
    base_stack = system["hbm_stacks"][0]
    system["hbm_stacks"] = [
        {
            **base_stack,
            "stack_id": space.die_id,
            "compute_die_id": space.die_id,
            "capacity_bytes": space.size_bytes,
        }
        for space in spaces
    ]
    address = system["address_policy"]
    address["home_ranges"] = [
        {
            "die_id": space.die_id,
            "base": space.base_address,
            "size_bytes": space.size_bytes,
        }
        for space in spaces
    ]
    address["stack_interleave_bytes"] = max(
        space.size_bytes for space in spaces
    )
    return json.dumps(hardware, sort_keys=True, separators=(",", ":"))


def validate_physical_hbm(hardware: str, spaces, rows: int, columns: int) -> dict[str, object]:
    """Bind the materialized HBM address spaces to the actual NpuSim hardware."""

    document = json.loads(hardware)
    if document["die"] != {"x": columns, "y": rows}:
        raise RuntimeError("physical HBM mesh differs from materialized mesh")
    stacks = document["memory_system"]["hbm_stacks"]
    ranges = document["memory_system"]["address_policy"]["home_ranges"]
    if len(spaces) != rows * columns or len(stacks) != len(spaces) or len(ranges) != len(spaces):
        raise RuntimeError("physical HBM stack/home-range count differs from materialization")
    for space, stack, home in zip(spaces, stacks, ranges):
        if (
            stack["stack_id"] != space.die_id
            or stack["compute_die_id"] != space.die_id
            or stack["capacity_bytes"] != space.size_bytes
            or home["die_id"] != space.die_id
            or home["base"] != space.base_address
            or home["size_bytes"] != space.size_bytes
        ):
            raise RuntimeError(f"physical HBM capacity/home-range differs on die {space.die_id}")
    return {
        "hardware_sha256": _sha(hardware.encode("utf-8")),
        "hbm_bytes_per_die": [space.size_bytes for space in spaces],
        "hbm_total_bytes": sum(space.size_bytes for space in spaces),
    }


def bind_dram_resources(
    hardware: str, simulation: Path, simulator_cwd: Path,
) -> dict[str, object]:
    """Close the real monitor and all HBM channels against cwd DRAMSys files."""

    cwd = simulator_cwd.resolve()
    resource_root = (cwd / "../DRAMSys/configs").resolve()
    if not resource_root.is_dir():
        raise RuntimeError(f"behavioral DRAMSys resources are absent from {cwd}")
    checked_in = (_ROOT / "DRAMSys/configs/hbm2-example.json").resolve()
    document = json.loads(hardware)
    sim_bytes = simulation.resolve().read_bytes()
    monitor = json.loads(sim_bytes)["gpu"]["dram_config_file"]
    channels = [
        stack["channel_dram_config"]
        for stack in document["memory_system"]["hbm_stacks"]
    ]
    references = [monitor, *channels]
    if len(channels) != 16 or any(type(item) is not str or not item for item in references):
        raise RuntimeError("behavioral monitor/channel DRAM config is incomplete")
    resolved = [
        (cwd / name).resolve() if not Path(name).is_absolute()
        else Path(name).resolve()
        for name in references
    ]
    if any(path != checked_in or not path.is_file() for path in resolved):
        raise RuntimeError("behavioral monitor/channel DRAM config path differs")
    config = json.loads(checked_in.read_text(encoding="utf-8"))["simulation"]
    dependencies = {
        kind: resource_root / directory / config[kind]
        for kind, directory in (
            ("addressmapping", "addressmapping"),
            ("mcconfig", "mcconfig"),
            ("memspec", "memspec"),
            ("simconfig", "simconfig"),
        )
    }
    if any(not path.is_file() for path in dependencies.values()):
        raise RuntimeError("behavioral DRAMSys dependency is absent")
    return {
        "simulator_cwd": str(cwd),
        "resource_root": str(resource_root),
        "simulation_sha256": _sha(sim_bytes),
        "channel_reference_count": len(resolved),
        "hbm2_config": {
            "path": str(checked_in), "sha256": _sha(checked_in.read_bytes()),
        },
        "dependencies": {
            kind: {"path": str(path.resolve()), "sha256": _sha(path.read_bytes())}
            for kind, path in dependencies.items()
        },
    }


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stage(command: tuple[str, ...], stdout: Path, *, cwd: Path, timeout: int):
    """Capture stage wall time and live peak RSS from the real subprocess."""

    started = time.monotonic()
    with stdout.open("wb") as output:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
        )
        rss_kib = 0
        while process.poll() is None:
            try:
                for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss_kib = max(rss_kib, int(line.split()[1]))
                        break
            except FileNotFoundError:
                pass
            if time.monotonic() - started > timeout:
                process.kill()
                exit_code = process.wait()
                stdout.with_suffix(".stage.json").write_text(
                    json.dumps({
                        "wall_seconds": round(time.monotonic() - started, 3),
                        "peak_rss_kib": rss_kib,
                        "exit_code": exit_code,
                        "reason": f"stage exceeded {timeout}s",
                        "command": list(command),
                    }, indent=2, sort_keys=True), encoding="utf-8",
                )
                raise TimeoutError(f"stage exceeded {timeout}s: {command[0]}")
            time.sleep(0.05)
        exit_code = process.wait()
    result = {
        "wall_seconds": round(time.monotonic() - started, 3),
        "peak_rss_kib": rss_kib,
        "exit_code": exit_code,
        "command": list(command),
    }
    stdout.with_suffix(".stage.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8",
    )
    if exit_code != 0:
        raise RuntimeError(
            f"stage failed: {command[0]} exit={exit_code}\n"
            + stdout.read_text(encoding="utf-8", errors="replace")[-12_000:]
        )
    return result


def observe_runtime(
    output: str, *, rows: int = 1, columns: int = 16,
    core_grid: tuple[int, int] = (2, 2),
) -> dict[str, object]:
    """Require sequence closure and all sixteen actual physical Die links."""

    segments = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)",
        output,
    )
    kv = re.findall(
        r"\[DENSE_SEQUENCE_KV\] index=(\d+) bytes=(\d+) "
        r"digest=([0-9a-f]{64}) pass=1",
        output,
    )
    drain = re.findall(
        r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)",
        output,
    )
    results = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", output)
    p2p = re.findall(
        r"\[P5 P2P DRAIN\] core=(\d+) residual=(\d+)", output,
    )
    timing = re.findall(r"\[P5 P2P TIMING DRAIN\] residual=(\d+)", output)
    if segments != [("0", "0"), ("1", "0"), ("2", "1")]:
        raise RuntimeError(f"Dense sequence segment closure failed: {segments}")
    if [(int(index), int(size)) for index, size, _ in kv] != list(
        enumerate(_KV_BYTES)
    ):
        raise RuntimeError(f"Dense sequence KV closure failed: {kv}")
    if drain != [("3", "1")] or len(results) != 1 or int(results[0]) <= 0:
        raise RuntimeError("Dense sequence drain or unique SIM_RESULT failed")
    if len(p2p) != _TP or {int(core) for core, _ in p2p} != {
        4 * rank for rank in range(_TP)
    } or any(residual != "0" for _, residual in p2p):
        raise RuntimeError(f"16 active P2P endpoint drain closure failed: {p2p}")
    if timing != ["0"]:
        raise RuntimeError(f"shared P2P timing drain failed: {timing}")
    if (rows, columns) not in _SHAPES.values() or core_grid[0] <= 0 or core_grid[1] <= 0:
        raise RuntimeError("TP16 physical mesh or per-Die core grid is invalid")
    stride = core_grid[0] * core_grid[1]
    memory = re.findall(r"\[PROGRAM_MEMORY\] core=(\d+) lsu_issued=(\d+)", output)
    physical_dies = sorted({int(core) // stride for core, issued in memory if int(issued) > 0})
    if physical_dies != list(range(_TP)):
        raise RuntimeError(f"actual NpuSim physical Die coverage is incomplete: {physical_dies}")
    link_rows = re.findall(
        r"\[D2D_LINK\][^\n]*?die(\d+)->die(\d+) dir=([EWNS])"
        r"[^\n]*?data_in=(\d+) data_out=(\d+)", output,
    )
    forward, reverse = (("E", "W") if rows == 1 else ("N", "S"))
    expected_links = {
        link for die in range(_TP - 1)
        for link in ((die, die + 1, forward), (die + 1, die, reverse))
    }
    actual_links = {(int(source), int(target), direction)
                    for source, target, direction, _, _ in link_rows}
    if actual_links != expected_links or len(link_rows) != len(expected_links) or any(
        int(data_in) <= 0 or int(data_in) != int(data_out)
        for _, _, _, data_in, data_out in link_rows
    ):
        raise RuntimeError("real TP16 physical end-to-end neighbor link traffic is incomplete")
    data = re.findall(r"\[D2D_DATA\] in_pkts=(\d+) out_pkts=(\d+)", output)
    if len(data) != 1 or int(data[0][0]) <= 0 or data[0][0] != data[0][1]:
        raise RuntimeError("real TP16 D2D data packets are missing or unbalanced")
    for marker in (
        "[DRAIN] router_residual=0",
        "[DRAIN] d2d_link_residual=0",
        "[CREDIT] data_balanced=1 ctrl_balanced=1",
    ):
        if marker not in output:
            raise RuntimeError(f"missing runtime drain marker {marker}")
    return {
        "segments": segments,
        "kv_bytes": [int(size) for _, size, _ in kv],
        "kv_digests": [digest for _, _, digest in kv],
        "one_shot_drain": drain,
        "p2p_drained_cores": sorted(int(core) for core, _ in p2p),
        "physical_die_ids": physical_dies,
        "nonzero_physical_link_count": len(link_rows),
        "d2d_data_packets": [int(item) for item in data[0]],
        "makespan_cycles": int(results[0]),
    }


def finalize_runtime_receipts(
    root: Path, evidence: dict[str, object],
) -> None:
    """Publish completion only after both native executions are available."""

    executions = evidence.get("executions")
    if not isinstance(executions, list) or len(executions) != 2:
        raise RuntimeError("TP16 completion needs exactly two native executions")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("npusim"), dict)
        or item["npusim"].get("exit_code") != 0
        or not item.get("observed")
        for item in executions
    ):
        raise RuntimeError("TP16 completion cannot publish a failed native execution")
    receipt_path = root / "compiled_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("runtime_status") != "not_measured":
        raise RuntimeError("TP16 preflight receipt status changed before completion")
    evidence["runtime_status"] = "verified"
    evidence_path = root / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8",
    )
    receipt.update({
        "runtime_status": "verified",
        "execution_count": 2,
        "evidence_sha256": _sha(evidence_path.read_bytes()),
    })
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8",
    )


def _seeds(profile) -> dict[str, bytes]:
    abi_by_binding = {
        abi.hbm_binding_ref: abi
        for fragment in profile.manifest.fragments
        for abi in (
            fragment.fragment.state_abi
            if isinstance(fragment, RegionManifest)
            else fragment.state_abi
        )
    }
    first_access: dict[str, StateUseAccess] = {}
    for action in profile.lowering_context.global_dag.actions:
        for use in action.state_uses:
            first_access.setdefault(use.hbm_binding_ref, use.access)
    return {
        abi_by_binding[binding].state_ref: bytes(
            abi_by_binding[binding].size_bytes,
        )
        for binding, access in first_access.items()
        if access is StateUseAccess.READ
    }


def bound_frontend_sources() -> dict[str, str]:
    """Bind the actual imported frontend/fixture Python bytes for this case."""
    frontend_root = (_ROOT / "llm/frontend/wafer_frontend").resolve()
    support = {
        Path(__file__).resolve(),
        (_ROOT / "llm/test/frontend/flexible_mesh_fixtures.py").resolve(),
        (_ROOT / "llm/test/frontend/unit/_fixtures.py").resolve(),
        (_ROOT / "llm/test/frontend/unit/test_legacy_dense_backend.py").resolve(),
        (_ROOT / "llm/test/frontend/integration/flexible_mesh_release_hardware.py").resolve(),
        (_ROOT / "llm/test/frontend/integration/run_dense_sequence_runtime_canary.py").resolve(),
    }
    paths = set(support)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(frontend_root):
                paths.add(path)
    return {str(path): _sha(path.read_bytes()) for path in sorted(paths)}


def run(args: argparse.Namespace) -> dict[str, object]:
    rows, columns = _SHAPES[args.mesh_size]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tool_binding = {
        kind: _sha(path.resolve().read_bytes())
        for kind, path in (
            ("finalizer", args.finalizer),
            ("resolver", args.resolver),
            ("npusim", args.npusim),
            ("simulation", args.simulation),
        )
    }
    frontend_sources = bound_frontend_sources()
    (root / "frontend_source_binding.json").write_text(
        json.dumps({
            "file_count": len(frontend_sources),
            "files": frontend_sources,
            "sha256": _sha(canonical_json(frontend_sources).encode("utf-8")),
        }, indent=2, sort_keys=True), encoding="utf-8",
    )
    materialize_started = time.monotonic()
    manifest, template, fabric, spaces = build_case(rows, columns)
    materialize_wall = round(time.monotonic() - materialize_started, 3)
    hardware = extended_hardware(rows, columns, spaces, fabric)
    physical_hbm = validate_physical_hbm(hardware, spaces, rows, columns)
    dram_resources = bind_dram_resources(
        hardware, args.simulation, args.npusim.resolve().parent,
    )
    (root / "preflight.json").write_text(
        json.dumps({
            "mesh": {"rows": rows, "columns": columns},
            "active_dies": list(manifest.placement.active_die_ids),
            "materialize_wall_seconds": materialize_wall,
            "frontend_peak_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF,
            ).ru_maxrss,
            "compile_budget_seconds": args.compile_timeout,
            "tool_binding_sha256": tool_binding,
            "physical_hbm": physical_hbm,
            "dram_resources": dram_resources,
            "runtime_status": "not_measured",
        }, indent=2, sort_keys=True), encoding="utf-8",
    )
    hardware_preflight_path = root / "hardware.preflight.json"
    hardware_preflight_path.write_text(hardware, encoding="utf-8")
    if _sha(hardware_preflight_path.read_bytes()) != physical_hbm["hardware_sha256"]:
        raise RuntimeError("TP16 hardware preflight SHA differs from physical HBM")
    hardware_sram_preflight = _stage(
        (
            str(args.resolver.resolve()), "--validate-hardware-sram",
            str(hardware_preflight_path),
        ),
        root / "hardware_sram_preflight.stdout.txt",
        cwd=args.resolver.resolve().parent, timeout=min(args.timeout, 60),
    )
    if (
        "[HARDWARE_SRAM_PREFLIGHT] "
        "region=sram capacity_bytes=1048576 region_count=1"
        not in (root / "hardware_sram_preflight.stdout.txt").read_text(
            encoding="utf-8",
        )
    ):
        raise RuntimeError("production hardware SRAM preflight lacks TP16 ABI")
    (root / "hardware_sram_preflight.json").write_text(
        json.dumps({
            "hardware_sha256": physical_hbm["hardware_sha256"],
            "resolver_sha256": tool_binding["resolver"],
            "stage": hardware_sram_preflight,
            "runtime_status": "not_measured",
        }, indent=2, sort_keys=True), encoding="utf-8",
    )
    codec_preflight = _stage(
        (str(args.resolver.resolve()), "--multi-request-attention-preflight"),
        root / "resolver_codec_preflight.stdout.txt",
        cwd=args.resolver.resolve().parent, timeout=min(args.timeout, 60),
    )
    if (
        "[PROGRAM_IO_CODEC_PREFLIGHT] "
        "multi_request_prefill=1 multi_request_decode=1 "
        "unequal_context_rejected=1 invalid_decode_context_rejected=1"
        not in (root / "resolver_codec_preflight.stdout.txt").read_text(
            encoding="utf-8",
        )
    ):
        raise RuntimeError("resolver binary lacks multi-request attention codec")
    (root / "resolver_codec_preflight.json").write_text(
        json.dumps({
            "codec": "real_16_prefill_plus_16_decode_requests_with_invalid_context_negatives",
            "resolver_sha256": tool_binding["resolver"],
            "stage": codec_preflight,
            "runtime_status": "not_measured",
        }, indent=2, sort_keys=True), encoding="utf-8",
    )
    compile_started = time.monotonic()
    def _budget_expired(_signal, _frame):
        raise TimeoutError(f"TP16 frontend compile exceeded {args.compile_timeout}s")

    old_handler = signal.signal(signal.SIGALRM, _budget_expired)
    signal.alarm(args.compile_timeout)
    try:
        with builder_validation_session():
            sequence, profiles = compile_dense_e2e_sequence_runtime_profiles(
                manifest, template, fabric, hbm_address_spaces=spaces,
            )
    except Exception as error:
        (root / "compile_failure.json").write_text(
            json.dumps({
                "runtime_status": "not_measured",
                "wall_seconds": round(time.monotonic() - compile_started, 3),
                "frontend_peak_rss_kib": resource.getrusage(
                    resource.RUSAGE_SELF,
                ).ru_maxrss,
                "reason": str(error),
            }, indent=2, sort_keys=True), encoding="utf-8",
        )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    if any(_sha(Path(path).read_bytes()) != sha
           for path, sha in frontend_sources.items()):
        raise RuntimeError("bound Dense frontend source bytes changed during compile")
    sequence_digest = sequence.digest
    compile_wall = round(time.monotonic() - compile_started, 3)
    common = root / "compiled"
    common.mkdir(exist_ok=True)
    manifest_bytes = []
    for index, segment in enumerate(sequence.segments):
        raw = canonical_json(segment.linked_manifest).encode("utf-8")
        (common / f"segment_{index}.linked.json").write_bytes(raw)
        manifest_bytes.append(raw)
    (common / "hardware.json").write_text(hardware, encoding="utf-8")
    if _sha((common / "hardware.json").read_bytes()) != physical_hbm["hardware_sha256"]:
        raise RuntimeError("compiled physical HBM hardware SHA drifted")
    (common / "mapping.spec").write_text("0:0\n", encoding="utf-8")

    evidence = {
        "scope": "experimental_dense_inference_resident_hbm_timing_tp16",
        "published_release_matrix": False,
        "mesh": {"rows": rows, "columns": columns},
        "active_dies": list(manifest.placement.active_die_ids),
        "physical_hbm": physical_hbm,
        "dram_resources": dram_resources,
        "model": {
            "layers": 2, "hidden": 32, "intermediate": 64,
            "attention_heads": 16, "kv_heads": 16, "head_dim": 2,
            "prefill_per_request": 1, "requests": 16, "decode_steps": 2,
        },
        "sequence_digest": sequence_digest,
        "case_id": manifest.request.case_id,
        "request_digest": manifest.request.digest,
        "tool_binding_sha256": tool_binding,
        "materialize_wall_seconds": materialize_wall,
        "compile_wall_seconds": compile_wall,
        "frontend_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "linked": [
            {
                "manifest_bytes": len(raw),
                "manifest_sha256": _sha(raw),
                "symbolic_records": sum(
                    len(stream.records)
                    for fragment in sequence.segments[index].linked_manifest.fragments
                    for stream in (
                        fragment.fragment.core_streams
                        if isinstance(fragment, RegionManifest)
                        else fragment.core_streams
                    )
                ),
                "core_count": len(sequence.segments[index].linked_manifest.core_streams),
            }
            for index, raw in enumerate(manifest_bytes)
        ],
        "executions": [],
    }
    (root / "compiled_receipt.json").write_text(
        json.dumps({
            "runtime_status": "not_measured",
            "case_id": manifest.request.case_id,
            "request_digest": manifest.request.digest,
            "tool_binding_sha256": tool_binding,
            "physical_hbm": physical_hbm,
            "dram_resources": dram_resources,
            "active_dies": evidence["active_dies"],
            "sequence_digest": sequence_digest,
            "materialize_wall_seconds": materialize_wall,
            "compile_wall_seconds": compile_wall,
            "frontend_peak_rss_kib": evidence["frontend_peak_rss_kib"],
            "linked": evidence["linked"],
        }, indent=2, sort_keys=True), encoding="utf-8",
    )
    def verify_tool_binding() -> None:
        if any(_sha(Path(path).read_bytes()) != sha
               for path, sha in frontend_sources.items()):
            raise RuntimeError("bound Dense frontend source bytes changed during case")
        for kind, path in (
            ("finalizer", args.finalizer),
            ("resolver", args.resolver),
            ("npusim", args.npusim),
            ("simulation", args.simulation),
        ):
            if _sha(path.resolve().read_bytes()) != tool_binding[kind]:
                raise RuntimeError(f"bound {kind} bytes changed during this case")
        if bind_dram_resources(
            hardware, args.simulation, args.npusim.resolve().parent,
        ) != dram_resources:
            raise RuntimeError("bound behavioral DRAMSys resources changed during case")

    for execution_index in range(2):
        verify_tool_binding()
        directory = root / f"execution_{execution_index}"
        directory.mkdir(exist_ok=True)
        finalized = []
        artifacts = []
        sidecars = []
        for index, segment in enumerate(sequence.segments):
            linked = directory / f"segment_{index}.linked.json"
            linked.write_bytes(manifest_bytes[index])
            artifact = directory / f"segment_{index}.npup"
            report_path = directory / f"segment_{index}.finalizer.json"
            verify_tool_binding()
            finalizer = _stage(
                (
                    str(args.finalizer.resolve()), "--input", str(linked),
                    "--output", str(artifact), "--report", str(report_path),
                ),
                directory / f"segment_{index}.finalizer.stdout.txt",
                cwd=args.finalizer.resolve().parent, timeout=args.timeout,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            artifact_sha = _sha(artifact.read_bytes())
            if (
                report.get("artifact_sha256") != artifact_sha
                or report.get("artifact_bytes") != artifact.stat().st_size
                or report.get("linked_manifest_id") != segment.linked_manifest.id
                or report.get("linked_manifest_digest")
                != canonical_digest(segment.linked_manifest)
                or report.get("record_count")
                != evidence["linked"][index]["symbolic_records"]
            ):
                raise RuntimeError(f"finalizer closure failed segment={index}")
            # LinkedProgramProfile was already validated by the production
            # compiler and sequence validator; the private builder still checks
            # every semantic ABI/use and validates the final ProgramIO contract.
            io_started = time.monotonic()
            def _io_budget_expired(_signal, _frame):
                raise TimeoutError(
                    f"TP16 ProgramIO segment={index} exceeded "
                    f"{args.program_io_timeout}s"
                )

            old_io_handler = signal.signal(signal.SIGALRM, _io_budget_expired)
            signal.alarm(args.program_io_timeout)
            try:
                contract = _build_timing_program_io_prevalidated(
                    profiles[index], artifact_sha,
                    state_seed_overrides=_seeds(profiles[index]),
                )
                contract.validate_against(segment.linked_manifest)
                sidecar = directory / f"segment_{index}.program_io.json"
                sidecar.write_text(canonical_json(contract), encoding="utf-8")
            except Exception as error:
                (directory / f"segment_{index}.program_io_failure.json").write_text(
                    json.dumps({
                        "runtime_status": "not_measured",
                        "wall_seconds": round(time.monotonic() - io_started, 3),
                        "frontend_peak_rss_kib": resource.getrusage(
                            resource.RUSAGE_SELF,
                        ).ru_maxrss,
                        "reason": str(error),
                    }, indent=2, sort_keys=True), encoding="utf-8",
                )
                raise
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_io_handler)
            program_io_phase = {
                "wall_seconds": round(time.monotonic() - io_started, 3),
                "frontend_peak_rss_kib": resource.getrusage(
                    resource.RUSAGE_SELF,
                ).ru_maxrss,
            }
            verify_tool_binding()
            resolver = _stage(
                (
                    str(args.resolver.resolve()), "--resolve", str(linked),
                    str(artifact), str(sidecar),
                ),
                directory / f"segment_{index}.resolver.stdout.txt",
                cwd=args.resolver.resolve().parent, timeout=args.timeout,
            )
            resolver_output = (directory / f"segment_{index}.resolver.stdout.txt").read_text()
            if (
                f"initializations={len(contract.initializations)}"
                not in resolver_output
                or f"probes={len(contract.output_probes)}" not in resolver_output
            ):
                raise RuntimeError(f"resolver counts failed segment={index}")
            finalized.append({
                "segment": index,
                "artifact_bytes": artifact.stat().st_size,
                "artifact_sha256": artifact_sha,
                "exact_records": report["record_count"],
                "program_io_bytes": sidecar.stat().st_size,
                "program_io_sha256": _sha(sidecar.read_bytes()),
                "program_io_build": program_io_phase,
                "finalizer": finalizer,
                "resolver": resolver,
            })
            artifacts.append(artifact)
            sidecars.append(sidecar)
        hardware_path = directory / "hardware.json"
        hardware_path.write_text(hardware, encoding="utf-8")
        if _sha(hardware_path.read_bytes()) != physical_hbm["hardware_sha256"]:
            raise RuntimeError("execution physical HBM hardware SHA drifted")
        mapping_path = directory / "mapping.spec"
        mapping_path.write_text("0:0\n", encoding="utf-8")
        runtime_stdout = directory / "npusim.stdout.txt"
        verify_tool_binding()
        if _sha(hardware_path.read_bytes()) != physical_hbm["hardware_sha256"]:
            raise RuntimeError("execution physical HBM hardware SHA changed before runtime")
        runtime = _stage(
            (
                str(args.npusim.resolve()),
                "--program-sequence", ",".join(map(str, artifacts)),
                "--linked-manifest-sequence", ",".join(
                    str(directory / f"segment_{index}.linked.json")
                    for index in range(3)
                ),
                "--program-io-sequence", ",".join(map(str, sidecars)),
                "--hardware-config", str(hardware_path),
                "--simulation-config", str(args.simulation.resolve()),
                "--mapping-config", str(mapping_path),
                "--trace-window", "1000000",
            ),
            runtime_stdout, cwd=args.npusim.resolve().parent,
            timeout=args.timeout,
        )
        verify_tool_binding()
        observed = observe_runtime(
            runtime_stdout.read_text(encoding="utf-8"),
            rows=rows, columns=columns, core_grid=fabric.dies[0].noc_grid,
        )
        execution = {
            "index": execution_index,
            "stages": finalized,
            "npusim": runtime,
            "observed": observed,
        }
        evidence["executions"].append(execution)
        (directory / "execution.json").write_text(
            json.dumps(execution, indent=2, sort_keys=True), encoding="utf-8",
        )
    first, second = evidence["executions"]
    if first["observed"] != second["observed"] or [
        (stage["artifact_sha256"], stage["program_io_sha256"])
        for stage in first["stages"]
    ] != [
        (stage["artifact_sha256"], stage["program_io_sha256"])
        for stage in second["stages"]
    ]:
        raise RuntimeError("independent TP16 executions drifted")
    evidence["total_wall_seconds"] = round(time.monotonic() - started, 3)
    evidence["frontend_peak_rss_kib"] = resource.getrusage(
        resource.RUSAGE_SELF,
    ).ru_maxrss
    finalize_runtime_receipts(root, evidence)
    print(
        f"Extended Dense TP16 PASS shape={args.mesh_size} "
        f"active_dies={len(evidence['active_dies'])} "
        f"makespan={first['observed']['makespan_cycles']} "
        f"wall={evidence['total_wall_seconds']}s "
        f"rss={evidence['frontend_peak_rss_kib']}KiB"
    )
    return evidence


def _parse_args() -> argparse.Namespace:
    build = _ROOT / "build-debug-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-size", choices=tuple(_SHAPES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--finalizer", type=Path, default=build / "npusim_program_finalizer",
    )
    parser.add_argument(
        "--resolver", type=Path, default=build / "npusim_program_io_selftest",
    )
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument(
        "--simulation", type=Path,
        default=_ROOT / "llm/test/program/p5_behavioral_simulation.json",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--compile-timeout", type=int, default=1800)
    parser.add_argument("--program-io-timeout", type=int, default=900)
    args = parser.parse_args()
    if args.timeout <= 0 or args.compile_timeout <= 0 or args.program_io_timeout <= 0:
        parser.error("--timeout, --compile-timeout and --program-io-timeout must be positive")
    for field in ("finalizer", "resolver", "npusim", "simulation"):
        if not getattr(args, field).is_file():
            parser.error(f"--{field} must name an existing file")
    return args


if __name__ == "__main__":
    run(_parse_args())
