"""Run two Dense SGD steps in one 1x1 NpuSim/HBM instance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import subprocess

from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.passes.dense_training_compile_sequence import (
    compile_dense_training_sequence,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
)
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryObjectKind,
    MemoryTier,
    MemoryTierCapacity,
)
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadFamily,
    WorkloadMeshSpec,
    WorkloadModelArchitecture,
    WorkloadModelSpec,
    WorkloadOptimizerKind,
    WorkloadOptimizerSpec,
    WorkloadParallelSpec,
    WorkloadRunRequest,
    WorkloadStepSpec,
    WorkloadTrainingSteps,
)
from llm.test.frontend.unit.test_dense_training_compile_sequence import (
    _sequence,
)
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec
from llm.test.frontend.unit.test_workload_materialization import _capability
from .flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)
from .run_dense_external_offload_runtime_canary import _build_offload_case


_ROOT = Path(__file__).resolve().parents[4]


def _offload_sequence():
    """Build a 64-byte packed parameter image for exact aggregate DMA."""

    raw = _hardware(1, 1)
    fabric = physical_fabric_from_data(raw)
    spaces = hbm_address_spaces_from_data(raw)
    legacy = _spec(1, 1)
    legacy = replace(
        legacy,
        model=replace(
            legacy.model,
            V=2,
            H=32,
            I=1,
            NH=1,
            KVH=1,
            DH=32,
            rotary_dim=32,
        ),
    )
    request = WorkloadRunRequest.create(
        family=WorkloadFamily.DENSE_TRAINING,
        model=WorkloadModelSpec(
            architecture=WorkloadModelArchitecture.LLAMA_DENSE,
            vocabulary_size=2,
            hidden_size=32,
            intermediate_size=1,
            num_layers=2,
            num_attention_heads=1,
            num_kv_heads=1,
            head_dim=32,
            max_sequence_length=64,
            dtype=DType.FP16,
        ),
        steps=WorkloadStepSpec(
            training=WorkloadTrainingSteps(
                step_count=2,
                global_batch_size=1,
                micro_batch_size=1,
                micro_batch_count=1,
                sequence_length=1,
            )
        ),
        mesh=WorkloadMeshSpec(1, 1),
        parallel=WorkloadParallelSpec(tp=1, dp=1, active_die_ids=(0,)),
        optimizer=WorkloadOptimizerSpec(WorkloadOptimizerKind.SGD, 0.001),
    )
    capacities = tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{space.die_id}",
            base_address=space.base_address,
            capacity_bytes=space.size_bytes,
            alignment_bytes=space.alignment_bytes,
        )
        for space in spaces
    )
    materialization = materialize_workload_preflight(
        request,
        _capability(supported=True),
        capacities=capacities,
    )
    return compile_dense_training_sequence(
        materialization,
        legacy,
        fabric,
        hbm_address_spaces=spaces,
    )


@dataclass(frozen=True, slots=True)
class DenseTrainingSequenceRuntimeObservation:
    versions: tuple[int, ...]
    hbm_bytes: int
    hbm_digests: tuple[str, ...]
    step_count: int
    sgd_invocations: int
    makespan_cycles: int
    functional: bool = False


def _run(command: tuple[str, ...], *, cwd: Path, timeout: int) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"returncode={completed.returncode}: {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def _records(linked):
    return tuple(
        record
        for item in linked.manifest.fragments
        for record in (
            item.fragment.core_streams[0].records
            if type(item) is RegionManifest
            else item.core_streams[0].records
        )
    )


def _validate_static_bindings(sequence) -> tuple[int, int]:
    linked = sequence.segments[0].linked_program
    records = _records(linked)
    by_action: dict[str, set[RecordOpcode]] = {}
    for record in records:
        by_action.setdefault(record.source_global_action_id, set()).add(
            record.opcode
        )
    wgrad_refs: set[str] = set()
    sgd_refs: set[str] = set()
    store_refs: set[str] = set()
    for binding in sequence.segments[0].parameter_bindings:
        for shard in binding.legacy_shards:
            wgrad_refs.update(shard.wgrad_action_refs)
            sgd_refs.update(shard.sgd_action_refs)
            store_refs.update(shard.store_action_refs)
    expected = (
        (wgrad_refs, RecordOpcode.MATMUL, "WGRAD"),
        (sgd_refs, RecordOpcode.SGD_UPDATE, "SGD"),
        (store_refs, RecordOpcode.LSU_STORE, "store"),
    )
    for refs, opcode, name in expected:
        if not refs or any(opcode not in by_action.get(ref, set()) for ref in refs):
            raise RuntimeError(f"static {name} action-to-record binding failed")
    if len(sgd_refs) != 15 or len(store_refs) != 15 or len(wgrad_refs) != 15:
        raise RuntimeError("1x1 sequence must bind 15 legacy parameter carriers")
    matmul_count = sum(record.opcode is RecordOpcode.MATMUL for record in records)
    return len(wgrad_refs), matmul_count


def observe_runtime(
    output: str,
    *,
    state_count: int,
    hbm_bytes: int,
    matmul_records: int,
) -> DenseTrainingSequenceRuntimeObservation:
    segments = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)",
        output,
    )
    states = re.findall(
        r"\[DENSE_TRAINING_SEQUENCE_STATE\] version=(\d+) bytes=(\d+) "
        r"digest=([0-9a-f]{64}) content_changed=(\d+) "
        r"functional=(\d+) pass=(\d+)",
        output,
    )
    steps = re.findall(
        r"\[DENSE_TRAINING_SEQUENCE_STEP\] index=(\d+) "
        r"input_version=(\d+) output_version=(\d+) "
        r"trainable_states=(\d+) matmul_records=(\d+) "
        r"sgd_records=(\d+) store_records=(\d+) functional=(\d+) pass=(\d+)",
        output,
    )
    drain = re.findall(
        r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)",
        output,
    )
    makespan = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", output)
    if segments != [("0", "0"), ("1", "1")]:
        raise RuntimeError(f"two-step pause/drain closure failed: {segments}")
    if len(states) != 3:
        raise RuntimeError(f"requires state versions 0,1,2: {states}")
    if tuple(int(item[0]) for item in states) != (0, 1, 2):
        raise RuntimeError("state versions are not continuous")
    if any(
        int(item[1]) != hbm_bytes
        or item[4:] != ("0", "1")
        for item in states
    ):
        raise RuntimeError("HBM state marker bytes/functional/pass drifted")
    expected_steps = (
        ("0", "0", "1"),
        ("1", "1", "2"),
    )
    if tuple(item[:3] for item in steps) != expected_steps or any(
        int(item[3]) != state_count
        or int(item[4]) != matmul_records
        or int(item[5]) != state_count
        or int(item[6]) != state_count
        or item[7:] != ("0", "1")
        for item in steps
    ):
        raise RuntimeError(f"WGRAD/SGD/store step closure failed: {steps}")
    if drain != [("2", "1")] or len(makespan) != 1:
        raise RuntimeError("final one-shot drain or SIM_RESULT is not exact")
    sgd_invocations = output.count("[TRAIN_SGD]")
    if sgd_invocations != 2 * state_count:
        raise RuntimeError("runtime SGD invocation count is not exact")
    if output.count("[DENSE_SEQUENCE_PROGRAM_IO]") != 2:
        raise RuntimeError("ProgramIO did not verify after both steps")
    return DenseTrainingSequenceRuntimeObservation(
        versions=(0, 1, 2),
        hbm_bytes=hbm_bytes,
        hbm_digests=tuple(item[2] for item in states),
        step_count=2,
        sgd_invocations=sgd_invocations,
        makespan_cycles=int(makespan[0]),
    )


def run(args: argparse.Namespace) -> DenseTrainingSequenceRuntimeObservation:
    external_offload = bool(getattr(args, "external_offload", False))
    sequence = _offload_sequence() if external_offload else _sequence(1, 1)
    sequence.validate()
    state_count, matmul_records = _validate_static_bindings(sequence)
    linked = sequence.segments[0].linked_program
    state_seeds, state_expected = build_deterministic_timing_state_overrides(
        linked
    )
    if len(state_seeds) != state_count or state_expected:
        raise RuntimeError("deterministic trainable-state seed coverage changed")
    hbm_bytes = sum(map(len, state_seeds.values()))
    runtime_hbm_capacity = None
    external_binding_path: Path | None = None

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if external_offload:
        state_abis = {
            abi.id: abi
            for item in linked.manifest.fragments
            for abi in (
                item.fragment.state_abi
                if isinstance(item, RegionManifest)
                else item.state_abi
            )
        }
        ordered_abis = tuple(
            sorted(state_abis.values(), key=lambda item: item.address)
        )
        cursor = 0
        payload = bytearray()
        for abi in ordered_abis:
            if abi.address != cursor or abi.state_ref not in state_seeds:
                raise RuntimeError(
                    "offload parameter image must exactly match contiguous HBM ABI"
                )
            seed = state_seeds[abi.state_ref]
            if len(seed) != abi.size_bytes:
                raise RuntimeError("offload seed size differs from HBM ABI")
            payload.extend(seed)
            cursor += abi.size_bytes
        if cursor != hbm_bytes:
            raise RuntimeError("offload parameter image extent changed")
        resident_peak = max(
            item.peak_bytes
            for item in sequence.materialization.memory_plan.peaks
        )
        parameter_bytes = sum(
            item.size_bytes
            for item in sequence.materialization.state_inventory
            if item.object_kind is MemoryObjectKind.PARAMETER
        )
        if parameter_bytes != hbm_bytes:
            raise RuntimeError(
                "P3 parameter bytes differ from linked training HBM ABI"
            )
        runtime_hbm_capacity = resident_peak - parameter_bytes
        if runtime_hbm_capacity % 64 or runtime_hbm_capacity <= hbm_bytes:
            raise RuntimeError("bounded offload HBM capacity is not canonical")
        (
            offload_manifest,
            offload_plan,
            dma_program,
            action_graph,
            runtime_binding,
            planned_hbm_bytes,
            planned_parameter_bytes,
            resident_oom,
            _,
            planned_resident_peak,
        ) = _build_offload_case(
            sequence.materialization,
            hbm_bytes_override=runtime_hbm_capacity,
            dirty_writeback=True,
            payload_override=bytes(payload),
        )
        if (
            planned_hbm_bytes != runtime_hbm_capacity
            or planned_parameter_bytes != parameter_bytes
            or planned_resident_peak != resident_peak
            or not resident_oom
            or len(dma_program.descriptors) != 2
        ):
            raise RuntimeError("training offload plan closure failed")
        artifacts_dir = output / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        (output / "workload_manifest.json").write_text(
            canonical_json(offload_manifest), encoding="utf-8"
        )
        (output / "blocking_offload_plan.json").write_text(
            canonical_json(offload_plan), encoding="utf-8"
        )
        (artifacts_dir / "external_dma_action_graph.json").write_text(
            canonical_json(action_graph), encoding="utf-8"
        )
        (artifacts_dir / "external_dma_program.json").write_text(
            canonical_json(dma_program), encoding="utf-8"
        )
        external_binding_path = output / "external_dma_runtime_binding.json"
        external_binding_path.write_text(
            canonical_json(runtime_binding), encoding="utf-8"
        )
    manifests: list[Path] = []
    programs: list[Path] = []
    sidecars: list[Path] = []
    artifacts: list[str] = []
    for index, segment in enumerate(sequence.segments):
        manifest_path = output / f"step_{index}.linked.json"
        artifact_path = output / f"step_{index}.npup"
        report_path = output / f"step_{index}.finalizer.json"
        manifest_path.write_text(
            canonical_json(segment.linked_program.manifest), encoding="utf-8"
        )
        _run(
            (
                str(args.finalizer.resolve()),
                "--input",
                str(manifest_path),
                "--output",
                str(artifact_path),
                "--report",
                str(report_path),
            ),
            cwd=output,
            timeout=120,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if (
            report.get("artifact_sha256") != artifact_digest
            or report.get("linked_manifest_id")
            != segment.linked_program.manifest.id
            or report.get("linked_manifest_digest")
            != canonical_digest(segment.linked_program.manifest)
        ):
            raise RuntimeError(f"step {index} finalizer closure failed")
        contract = build_timing_program_io(
            linked,
            artifact_digest,
            state_seed_overrides=state_seeds,
        )
        contract.validate_against(linked.manifest)
        sidecar_path = output / f"step_{index}.program_io.json"
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        manifests.append(manifest_path)
        programs.append(artifact_path)
        sidecars.append(sidecar_path)
        artifacts.append(artifact_digest)

    hardware_path = output / "hardware.json"
    mapping_path = output / "mapping.spec"
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 1 << 20
    hardware["memory"]["sram"]["capacity_bytes"] = 1 << 20
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = 1 << 20
    if runtime_hbm_capacity is not None:
        hardware["memory_system"]["hbm_stacks"][0][
            "capacity_bytes"
        ] = runtime_hbm_capacity
        hardware["memory_system"]["address_policy"]["home_ranges"][0][
            "size_bytes"
        ] = runtime_hbm_capacity
        hardware["memory_system"]["address_policy"][
            "stack_interleave_bytes"
        ] = runtime_hbm_capacity
    hardware_path.write_text(
        json.dumps(hardware, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    mapping_path.write_text("0:0\n", encoding="utf-8")
    command = [
            str(args.npusim.resolve()),
            "--program-sequence",
            ",".join(map(str, programs)),
            "--linked-manifest-sequence",
            ",".join(map(str, manifests)),
            "--program-io-sequence",
            ",".join(map(str, sidecars)),
            "--hardware-config",
            str(hardware_path),
            "--simulation-config",
            str(args.simulation.resolve()),
            "--mapping-config",
            str(mapping_path),
            "--trace-window",
            "1000000",
    ]
    if external_binding_path is not None:
        command[1:1] = [
            "--external-dma-binding",
            str(external_binding_path),
        ]
    runtime_output = _run(
        tuple(command),
        cwd=output,
        timeout=args.timeout,
    )
    (output / "npusim.stdout.txt").write_text(
        runtime_output, encoding="utf-8"
    )
    observation = observe_runtime(
        runtime_output,
        state_count=state_count,
        hbm_bytes=hbm_bytes,
        matmul_records=matmul_records,
    )
    if external_offload:
        ready = re.findall(
            r"\[EXTERNAL_DMA_READY\].*completed=(\d+).*"
            r"external_read_bytes=(\d+).*hbm_write_bytes=(\d+).*pending=0",
            runtime_output,
        )
        drain = re.findall(
            r"\[EXTERNAL_DMA_DRAIN\] probes=(\d+).*"
            r"external_read_bytes=(\d+).*external_write_bytes=(\d+).*"
            r"hbm_read_bytes=(\d+).*hbm_write_bytes=(\d+).*pending=0",
            runtime_output,
        )
        expected_bytes = str(hbm_bytes)
        if ready != [("1", expected_bytes, expected_bytes)]:
            raise RuntimeError(f"training bring-in closure failed: {ready}")
        if drain != [
            ("1", expected_bytes, expected_bytes, expected_bytes, expected_bytes)
        ]:
            raise RuntimeError(f"training dirty writeback closure failed: {drain}")
        ready_at = runtime_output.find("[EXTERNAL_DMA_READY]")
        step0_at = runtime_output.find("[DENSE_TRAINING_SEQUENCE_STEP] index=0")
        step1_at = runtime_output.find("[DENSE_TRAINING_SEQUENCE_STEP] index=1")
        drain_at = runtime_output.find("[EXTERNAL_DMA_DRAIN]")
        if not (0 <= ready_at < step0_at < step1_at < drain_at):
            raise RuntimeError("bring-in/compute/writeback runtime order failed")
        evidence = {
            "schema_version": (
                "npusim.dense_training_external_offload_canary/v1alpha1"
            ),
            "mesh": "1x1",
            "optimizer": "sgd",
            "steps": 2,
            "functional": False,
            "parameter_bytes": hbm_bytes,
            "hbm_capacity_bytes": runtime_hbm_capacity,
            "resident_peak_bytes": resident_peak,
            "versions": [0, 1, 2],
            "external_dma_descriptors": 2,
            "pending_requests": 0,
            "sim_result_count": 1,
        }
        (output / "external_offload_evidence.json").write_text(
            json.dumps(evidence, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        "Dense training sequence runtime canary PASS "
        f"external_offload={int(external_offload)} "
        f"sequence={sequence.digest} artifacts={','.join(artifacts)} "
        f"versions=0,1,2 hbm={','.join(observation.hbm_digests)} "
        "functional=0"
    )
    return observation


def _parse_args() -> argparse.Namespace:
    build = _ROOT / "build-debug-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=build / "dense-training-sequence-runtime-canary",
    )
    parser.add_argument(
        "--finalizer", type=Path, default=build / "npusim_program_finalizer"
    )
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument(
        "--simulation",
        type=Path,
        default=_ROOT / "llm/test/program/p5_behavioral_simulation.json",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--external-offload",
        action="store_true",
        help="gate two SGD steps between parameter bring-in and dirty writeback",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    for name in ("finalizer", "npusim", "simulation"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name} must name an existing file")
    return args


if __name__ == "__main__":
    run(_parse_args())
