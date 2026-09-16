"""Production artifact adapter for the fail-closed Flexible-MoE subset.

The flexible planner is intentionally independent from the frozen EP4
pipeline.  This module keeps the proven 1x1 path and dispatches multi-die
plans to an independent strict DTE adapter over the same public artifact ABIs.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import struct

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    AddressOperandBinding,
    AddressRelocation,
    BufferABI,
    CommandFragment,
    CoreFragmentStream,
    CoreRuntimeBinding,
    EmptyCoreAckPolicy,
    FragmentInterface,
    FragmentKind,
    LinkedCoreStream,
    LinkedProgramManifest,
    LinkedRecordRef,
    LogicalStartEvent,
    ManifestInputDigest,
    ManifestInputKind,
    ProgramControlEnvelope,
    ProgramFailurePolicy,
    ProgramSymbol,
    ProgramSymbolDefinition,
    ProgramSymbolKind,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    RuntimeSymbol,
    RuntimeSymbolDefinition,
    RuntimeSymbolKind,
    SemanticOperandId,
    StateABI,
    StateOperandBinding,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.flexible_moe import (
    FlexibleMoeExecutablePlan,
    FlexibleMoeMode,
    FlexibleMoeSpec,
    MoeRectActionKind,
    MoeRectStateRole,
)
from ..schema.flexible_moe_standard import FlexibleMoeStandardLoweringPlan
from ..schema.global_action import LogicalCoreRef
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateLifetime,
    StateKind,
)
from ..schema.program_io import (
    ProgramBlob,
    ProgramHbmTarget,
    ProgramIoContract,
    ProgramIoMode,
    ProgramIoPurpose,
    ProgramIoTargetKind,
    ProgramOutputCapture,
    ProgramOutputComparison,
    ProgramOutputProbe,
    ProgramSramInitialization,
    ProgramSramTarget,
)
from ..schema.serde import canonical_digest, canonical_json
from .flexible_moe_standard import plan_flexible_moe_standard_mapping


_LOWERING_PASS = "flexible_moe_production_lowering"
_LINKER_PASS = "flexible_moe_production_linker"
_PROGRAM_IO_PASS = "build_flexible_moe_production_program_io"
_REGION_REF = "flexible_moe.sram.core0"
_REGION_NAME = "input"
_REGION_BASE_BYTES = 4096
_REGION_SIZE_BYTES = 36864
_SCHEDULE_ID = "flexible_moe.production.schedule.core0"
_LEARNING_RATE_BITS = struct.unpack("<Q", struct.pack("<d", 1.0e-3))[0]


@dataclass(frozen=True, slots=True)
class FlexibleMoeProductionArtifacts:
    """A real standard artifact result; runtime execution is not implied."""

    standard_ir: FlexibleMoeStandardLoweringPlan
    fragments: tuple[CommandFragment, ...]
    manifest: LinkedProgramManifest
    lower_link_verified: bool
    runtime_verified: bool

    def validate_against(
        self,
        plan: FlexibleMoeExecutablePlan,
        spec: FlexibleMoeSpec,
        path: str = "flexible_moe_production_artifacts",
        *,
        allow_zero_work_omission: bool = False,
    ) -> None:
        plan.validate_against(spec, f"{path}.plan")
        self.standard_ir.validate_against(plan, spec, f"{path}.standard_ir")
        if self.fragments != self.manifest.fragments:
            raise SchemaError("fragment tuple must equal the linked manifest input", path=f"{path}.fragments")
        self.manifest.validate(f"{path}.manifest")
        plan_input = next(
            digest
            for digest in self.manifest.input_digests
            if digest.kind is ManifestInputKind.FLEXIBLE_MOE_PLAN
        )
        mapping_input = next(
            digest
            for digest in self.manifest.input_digests
            if digest.kind is ManifestInputKind.FLEXIBLE_MOE_STANDARD_MAPPING
        )
        if (
            self.manifest.source_ir1_id != spec.id
            or self.manifest.source_projection_id != self.standard_ir.id
            or self.manifest.source_schedule_set_id != _SCHEDULE_ID
            or self.manifest.source_global_dag_id != plan.id
            or plan_input.artifact_id != plan.id
            or mapping_input.artifact_id != self.standard_ir.id
        ):
            raise SchemaError(
                "production manifest source lineage is not exact",
                path=f"{path}.manifest",
            )
        if (
            plan_input.digest != canonical_digest(plan)
            or mapping_input.digest != canonical_digest(self.standard_ir)
        ):
            raise SchemaError(
                "production manifest input digest does not match its typed source",
                path=f"{path}.manifest.input_digests",
            )
        fragment_inputs = {
            digest.artifact_id: digest.digest
            for digest in self.manifest.input_digests
            if digest.kind is ManifestInputKind.COMMAND_FRAGMENT
        }
        expected_fragment_inputs = {
            fragment.id: canonical_digest(fragment) for fragment in self.fragments
        }
        if fragment_inputs != expected_fragment_inputs:
            raise SchemaError(
                "production manifest fragment digest closure is not exact",
                path=f"{path}.manifest.input_digests",
            )
        action_ids = {action.id for action in plan.actions}
        zero_work_source_actions = {
            action.id for action in plan.actions
            if action.kind in (
                MoeRectActionKind.GATE,
                MoeRectActionKind.PACK,
                MoeRectActionKind.WEIGHTED_COMBINE,
                MoeRectActionKind.EXPERT_FORWARD,
                MoeRectActionKind.EXPERT_DGRAD,
                MoeRectActionKind.EXPERT_WGRAD,
                MoeRectActionKind.GATE_WGRAD,
                MoeRectActionKind.COMBINE_BACKWARD,
            ) and not action.assignment_refs
            and action.flops == 0 and action.logical_bytes == 0
        }
        claimed = {
            action_id
            for fragment in self.fragments
            for action_id in fragment.claimed_action_ids
        }
        expected = action_ids - zero_work_source_actions if allow_zero_work_omission else action_ids
        if allow_zero_work_omission:
            from .flexible_moe_multi_production import (
                expert_projection_action_ids, expert_wgrad_action_ids,
                gate_wgrad_cast_action_id, expert_dgrad_action_ids,
            )
            expected |= {
                child_id
                for action in plan.actions
                if action.kind is MoeRectActionKind.EXPERT_FORWARD and action.assignment_refs
                for child_id in expert_projection_action_ids(plan.id, action.id)
            }
            if spec.mode is FlexibleMoeMode.TRAIN:
                expected |= {
                    child_id
                    for action in plan.actions
                    if action.kind is MoeRectActionKind.EXPERT_DGRAD and action.assignment_refs
                    for child_id in expert_dgrad_action_ids(plan.id, action.id)
                }
                expected |= {
                    child_id
                    for action in plan.actions
                    if action.kind is MoeRectActionKind.EXPERT_WGRAD and action.assignment_refs
                    # The explicit full-model-dataflow profile writes three
                    # native FP32 dW projections; its source action is gate
                    # and the only named children are up/down.  Original
                    # default profiles keep their five legacy children.
                    for child_id in expert_wgrad_action_ids(plan.id, action.id)[:2]
                }
                expected |= {
                    gate_wgrad_cast_action_id(plan.id, action.id)
                    for action in plan.actions
                    if action.kind is MoeRectActionKind.GATE_WGRAD and action.assignment_refs
                }
        if claimed != expected:
            raise SchemaError("production fragments must cover exactly the required plan actions", path=f"{path}.fragments")
        if self.manifest.source_global_dag_id != plan.id:
            raise SchemaError("manifest must retain exact plan provenance", path=f"{path}.manifest")
        if self.lower_link_verified is not True or self.runtime_verified is not False:
            raise SchemaError("validated artifacts are linked but not runtime evidence", path=path)


@dataclass(frozen=True, slots=True)
class FlexibleMoeRuntimeEvidence:
    """Typed proof for one actual finalizer/resolver/npusim execution."""

    mode: FlexibleMoeMode
    manifest_id: str
    manifest_digest: str
    program_artifact_sha256: str
    initialization_count: int
    probe_count: int
    makespan_cycles: int
    lsu_issued: int
    lsu_completed: int
    residual_count: int
    timing_execution: bool
    functional_execution: bool
    runtime_verified: bool

    def validate_against(
        self,
        artifacts: FlexibleMoeProductionArtifacts,
        spec: FlexibleMoeSpec,
        program_io: ProgramIoContract,
        path: str = "flexible_moe_runtime_evidence",
    ) -> None:
        if self.mode is not spec.mode:
            raise SchemaError("mode disagrees with source spec", path=f"{path}.mode")
        if (
            self.manifest_id != artifacts.manifest.id
            or self.manifest_digest != canonical_digest(artifacts.manifest)
        ):
            raise SchemaError("manifest closure is not exact", path=path)
        if self.program_artifact_sha256 != program_io.program_artifact_sha256:
            raise SchemaError("artifact SHA closure is not exact", path=path)
        if (
            self.initialization_count != len(program_io.initializations)
            or self.probe_count != len(program_io.output_probes)
            or self.probe_count == 0
        ):
            raise SchemaError("ProgramIO counts are not exact", path=path)
        if self.makespan_cycles <= 0 or self.lsu_issued <= 0:
            raise SchemaError("runtime work must be positive", path=path)
        if self.lsu_issued != self.lsu_completed or self.residual_count != 0:
            raise SchemaError("runtime did not drain exactly", path=path)
        if (
            self.timing_execution is not True
            or self.functional_execution is not False
            or self.runtime_verified is not True
        ):
            raise SchemaError("evidence boundary changed", path=path)


def _one_number(output: str, pattern: str, label: str) -> int:
    values = re.findall(pattern, output)
    if len(values) != 1:
        raise SchemaError(f"requires one exact {label} marker", path="npusim_output")
    return int(values[0])


def observe_flexible_moe_production_runtime(
    artifacts: FlexibleMoeProductionArtifacts,
    spec: FlexibleMoeSpec,
    program_io: ProgramIoContract,
    *,
    finalizer_report: dict[str, object],
    resolver_output: str,
    npusim_output: str,
    finalizer_exit_code: int,
    resolver_exit_code: int,
    npusim_exit_code: int,
) -> FlexibleMoeRuntimeEvidence:
    """Close actual tool outputs over the typed manifest and ProgramIO ABI."""

    if (finalizer_exit_code, resolver_exit_code, npusim_exit_code) != (0, 0, 0):
        raise SchemaError("production toolchain exit code is non-zero", path="runtime")
    expected_sha = program_io.program_artifact_sha256
    if (
        finalizer_report.get("artifact_sha256") != expected_sha
        or finalizer_report.get("linked_manifest_id") != artifacts.manifest.id
        or finalizer_report.get("linked_manifest_digest")
        != canonical_digest(artifacts.manifest)
    ):
        raise SchemaError("finalizer report closure is not exact", path="finalizer_report")
    initializations = len(program_io.initializations)
    probes = len(program_io.output_probes)
    if f"initializations={initializations} probes={probes}" not in resolver_output:
        raise SchemaError("resolver lost exact ProgramIO counts", path="resolver_output")
    verify = re.findall(
        r"\[PROGRAM_IO\] phase=verify mode=timing initializations=(\d+) "
        r"probes=(\d+)[^\n]*\bpass=(\d+)",
        npusim_output,
    )
    if verify != [(str(initializations), str(probes), "1")]:
        raise SchemaError("ProgramIO verify marker is not exact", path="npusim_output")
    probe_passes = re.findall(r"\[PROGRAM_IO_PROBE\][^\n]*\bpass=(\d+)", npusim_output)
    if probe_passes != ["1"] * probes:
        raise SchemaError("not every ProgramIO probe passed", path="npusim_output")
    makespan = _one_number(
        npusim_output, r"\[SIM_RESULT\] makespan_cycles=(\d+)", "makespan",
    )
    memory = re.findall(
        r"\[PROGRAM_MEMORY\][^\n]*lsu_issued=(\d+) lsu_completed=(\d+)"
        r"[^\n]*lsu_residual=(\d+) dte_residual=(\d+)",
        npusim_output,
    )
    if len(memory) != spec.mesh.rank_count:
        raise SchemaError(
            "requires one exact memory marker per active rank",
            path="npusim_output",
        )
    memory_values = tuple(tuple(map(int, item)) for item in memory)
    issued = sum(item[0] for item in memory_values)
    completed = sum(item[1] for item in memory_values)
    lsu_residual = sum(item[2] for item in memory_values)
    dte_residual = sum(item[3] for item in memory_values)
    residuals = (
        lsu_residual,
        dte_residual,
        _one_number(npusim_output, r"P5 P2P TIMING DRAIN\] residual=(\d+)", "P2P drain"),
        _one_number(npusim_output, r"\[DRAIN\] router_residual=(\d+)", "router drain"),
        _one_number(npusim_output, r"\[DRAIN\] d2d_link_residual=(\d+)", "D2D drain"),
    )
    result = FlexibleMoeRuntimeEvidence(
        spec.mode,
        artifacts.manifest.id,
        canonical_digest(artifacts.manifest),
        expected_sha,
        initializations,
        probes,
        makespan,
        issued,
        completed,
        sum(residuals),
        True,
        False,
        True,
    )
    result.validate_against(artifacts, spec, program_io)
    return result


def _align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def _symbol(kind: ProgramSymbolKind, source_ref: str, role: str) -> ProgramSymbol:
    return ProgramSymbol(
        stable_artifact_id(
            "flexible_moe_program_symbol",
            {"kind": int(kind), "source_ref": source_ref, "role": role},
            schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
        ),
        kind,
        source_ref,
    )


def _buffer(
    *,
    name: str,
    core: LogicalCoreRef,
    offset: int,
    size_bytes: int,
    dtype: DType,
    ownership: BufferOwnership,
    action_count: int,
    region_ref: str = _REGION_REF,
) -> BufferABI:
    element_bytes = 4 if dtype in (DType.FP32, DType.INT32) else 2
    if size_bytes % element_bytes:
        raise SchemaError("buffer size must be dense in its dtype", path=name)
    value_id = f"flexible_moe.value.{name}"
    binding_id = f"flexible_moe.binding.{name}"
    storage_id = f"flexible_moe.storage.{name}"
    semantic = {
        "schedule_id": _SCHEDULE_ID,
        "binding_id": binding_id,
        "value_id": value_id,
        "logical_core": core,
        "region_ref": region_ref,
        "region_offset_bytes": offset,
        "size_bytes": size_bytes,
        "dtype": dtype,
        "ownership": ownership,
    }
    return BufferABI(
        id=stable_artifact_id(
            "flexible_moe_buffer_abi",
            semantic,
            schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
        ),
        schedule_id=_SCHEDULE_ID,
        binding_id=binding_id,
        value_id=value_id,
        logical_core=core,
        tensor_slice=TensorSlice(value_id, (0,), (size_bytes // element_bytes,)),
        region_ref=region_ref,
        region_offset_bytes=offset,
        size_bytes=size_bytes,
        alignment_bytes=64,
        banks=(),
        storage_id=storage_id,
        alias_of=None,
        lifetime_start=0,
        lifetime_end_exclusive=max(1, action_count + 1),
        dtype=dtype,
        layout="flexible_moe_dense/v1",
        ownership=ownership,
    )


def _state_abi(plan, spec) -> tuple[StateABI, ...]:
    result = []
    address_by_rank: dict[int, int] = {}
    stored_state_refs = {
        state_ref
        for action in plan.actions
        if action.kind is MoeRectActionKind.STATE_STORE
        for state_ref in action.state_refs
    }
    for state in plan.state_bindings:
        if state.role not in (
            MoeRectStateRole.EXPERT_PARAMETER,
            MoeRectStateRole.GATE_PARAMETER,
        ):
            continue
        address = _align(address_by_rank.get(state.owner_rank, 0), 4096)
        dtype_bytes = 4 if state.dtype is DType.FP32 else 2
        result.append(StateABI.create(
            state_ref=state.id,
            hbm_binding_ref=f"flexible_moe.hbm.{state.id}",
            kind=(
                StateKind.TRAINABLE_PARAMETER
                if state.id in stored_state_refs
                else StateKind.PARAMETER
            ),
            lifetime=PersistentStateLifetime.PERSISTENT,
            access=(
                PersistentStateAccess.READ_WRITE
                if state.id in stored_state_refs
                else PersistentStateAccess.READ_ONLY
            ),
            shape=(state.size_bytes // dtype_bytes,),
            dtype=state.dtype,
            layout="flexible_moe_parameter/v1",
            die_id=state.owner_rank,
            address=address,
            size_bytes=state.size_bytes,
            alignment_bytes=4096,
        ))
        address_by_rank[state.owner_rank] = address + state.size_bytes
    return tuple(sorted(result, key=lambda item: item.id))


def lower_link_flexible_moe_production(
    plan: FlexibleMoeExecutablePlan,
    spec: FlexibleMoeSpec,
    *,
    physical_region_name: str | None = None,
    full_model_dataflow: bool = False,
    runtime_core_ids: tuple[int, ...] | None = None,
) -> FlexibleMoeProductionArtifacts:
    """Lower/link the exact timing subset into one real public manifest."""

    plan.validate_against(spec)
    release_region = physical_region_name is not None
    region_name = _REGION_NAME if physical_region_name is None else physical_region_name
    if type(region_name) is not str or not region_name:
        raise SchemaError("physical SRAM region name is empty", path="physical_region_name")
    if (spec.mesh.rank_count != 1
            or (full_model_dataflow and spec.mode is FlexibleMoeMode.TRAIN)):
        from .flexible_moe_multi_production import lower_link_flexible_moe_multi

        return lower_link_flexible_moe_multi(
            plan, spec, physical_region_name=physical_region_name,
            full_model_dataflow=full_model_dataflow,
            runtime_core_ids=runtime_core_ids,
        )
    if runtime_core_ids not in (None, (0,)):
        raise SchemaError(
            "single-rank runtime_core_ids must be absent or (0,)",
            path="runtime_core_ids",
        )
    if plan.flows:
        raise SchemaError("1x1 production plan must not contain remote flows", path="plan.flows")
    standard_ir = plan_flexible_moe_standard_mapping(plan, spec)
    core = LogicalCoreRef(0, 0)
    state_abis = _state_abi(plan, spec)
    state_by_ref = {item.state_ref: item for item in state_abis}

    offset = 0
    buffers = []
    activation_bytes = max(64, spec.trace.token_count * spec.hidden_size * 2)
    workspace_bytes = max(
        1024,
        activation_bytes,
        spec.trace.token_count * spec.intermediate_size * 2,
        spec.trace.token_count * spec.expert_count * 2,
    )
    for name, size, dtype, ownership in (
        ("activation", activation_bytes, DType.FP16, BufferOwnership.BORROWED),
        ("output", workspace_bytes, DType.FP16, BufferOwnership.OWNED),
    ):
        offset = _align(offset)
        buffers.append(_buffer(
            name=name, core=core,
            offset=offset + (_REGION_BASE_BYTES if release_region else 0),
            size_bytes=size, dtype=dtype,
            ownership=ownership, action_count=len(plan.actions),
        ))
        offset += size
    for state in plan.state_bindings:
        offset = _align(offset)
        buffers.append(_buffer(
            name=f"state.{state.id}", core=core,
            offset=offset + (_REGION_BASE_BYTES if release_region else 0),
            size_bytes=state.size_bytes, dtype=state.dtype,
            ownership=BufferOwnership.OWNED, action_count=len(plan.actions),
        ))
        offset += state.size_bytes
    buffers = tuple(sorted(buffers, key=lambda item: item.id))
    if release_region:
        spans = sorted(
            (item.region_offset_bytes, item.region_offset_bytes + item.size_bytes)
            for item in buffers
        )
        if (
            spans[-1][1] > (1 << 20)
            or any(left[1] > right[0] for left, right in zip(spans, spans[1:]))
        ):
            raise SchemaError(
                "release SRAM subspans overlap or exceed 1 MiB", path="buffers",
            )
    buffer_by_name = {
        item.value_id.removeprefix("flexible_moe.value."): item for item in buffers
    }
    activation = buffer_by_name["activation"]
    output = buffer_by_name["output"]
    buffer_by_state = {
        name.removeprefix("state."): item
        for name, item in buffer_by_name.items()
        if name.startswith("state.")
    }

    region_symbol = _symbol(ProgramSymbolKind.SRAM_REGION, _REGION_REF, "region")
    label_by_buffer = {
        item.id: _symbol(ProgramSymbolKind.SRAM_LABEL, item.storage_id, "label")
        for item in buffers
    }
    absolute_by_buffer = {
        item.id: _symbol(ProgramSymbolKind.ABSOLUTE_ADDRESS, item.binding_id, "absolute")
        for item in buffers
    }
    hbm_by_state = {
        item.id: _symbol(ProgramSymbolKind.ABSOLUTE_ADDRESS, item.hbm_binding_ref, "hbm")
        for item in state_abis
    }

    state_records: list[RelocatableRecord] = []
    state_relocations: list[AddressRelocation] = []
    compute_records: list[RelocatableRecord] = []
    compute_relocations: list[AddressRelocation] = []
    refs_by_action: dict[str, list[tuple[str, int]]] = {}

    def add_record(
        target: str,
        record: RelocatableRecord,
        relocations: tuple[tuple[SemanticOperandId, ProgramSymbol, int], ...],
    ) -> None:
        records = state_records if target == "state" else compute_records
        address = state_relocations if target == "state" else compute_relocations
        index = len(records)
        records.append(record)
        refs_by_action.setdefault(record.source_global_action_id, []).append((target, index))
        for operand_id, symbol, addend in relocations:
            address.append(AddressRelocation(index, operand_id, symbol.kind, symbol.id, addend))

    first_action = plan.actions[0].id
    for abi in buffers:
        label = label_by_buffer[abi.id]
        add_record("state", RelocatableRecord(first_action, RecordOpcode.SRAM_ALLOC_AT, (
            RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region_symbol.id),
            RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label.id),
            RecordOperand.literal("region_offset_bytes", abi.region_offset_bytes),
            RecordOperand.literal("size_bytes", abi.size_bytes),
            RecordOperand.literal("alignment_bytes", abi.alignment_bytes),
            RecordOperand.literal("lifetime", 0),
            RecordOperand.literal("spillable", not release_region),
        )), (
            (SemanticOperandId.REGION_NAME, region_symbol, 0),
            (SemanticOperandId.LABEL_SYMBOL, label, 0),
        ))

    def absolute(abi: BufferABI) -> ProgramSymbol:
        return absolute_by_buffer[abi.id]

    def add_bind(
        action_id: str,
        inputs: tuple[BufferABI, ...],
        destination: BufferABI,
    ) -> None:
        if not 1 <= len(inputs) <= 16:
            raise SchemaError("SRAM_BIND input count must lie in [1, 16]", path="plan.actions")
        operands = [RecordOperand.literal("input_count", len(inputs))]
        relocations = []
        for index in range(16):
            operand_id = SemanticOperandId(
                int(SemanticOperandId.SRAM_BIND_INPUT_0) + index
            )
            if index < len(inputs):
                label = label_by_buffer[inputs[index].id]
                operands.append(RecordOperand.address(
                    f"input_label_{index}", operand_id, label.id,
                ))
                relocations.append((operand_id, label, 0))
            else:
                operands.append(RecordOperand.literal(f"input_label_{index}", 0))
        output_label = label_by_buffer[destination.id]
        operands.append(RecordOperand.address(
            "output_label", SemanticOperandId.SRAM_BIND_OUTPUT, output_label.id,
        ))
        relocations.append((SemanticOperandId.SRAM_BIND_OUTPUT, output_label, 0))
        add_record(
            "compute",
            RelocatableRecord(action_id, RecordOpcode.SRAM_BIND, tuple(operands)),
            tuple(relocations),
        )

    for action in plan.actions:
        if action.kind is MoeRectActionKind.STATE_LOAD:
            for state_ref in action.state_refs:
                state = state_by_ref.get(state_ref)
                if state is None:
                    continue
                destination = buffer_by_state[state_ref]
                hbm = hbm_by_state[state.id]
                add_record("state", RelocatableRecord(action.id, RecordOpcode.LSU_LOAD, (
                    RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS, hbm.id),
                    RecordOperand.literal("size_bytes", state.size_bytes),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, absolute(destination).id),
                )), (
                    (SemanticOperandId.HBM_ADDRESS, hbm, 0),
                    (SemanticOperandId.DESTINATION_ADDRESS, absolute(destination), 0),
                ))
            continue
        if action.kind is MoeRectActionKind.STATE_STORE:
            for state_ref in action.state_refs:
                state = state_by_ref.get(state_ref)
                if state is None:
                    continue
                source = buffer_by_state[state_ref]
                hbm = hbm_by_state[state.id]
                add_record("state", RelocatableRecord(action.id, RecordOpcode.LSU_STORE, (
                    RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS, hbm.id),
                    RecordOperand.literal("size_bytes", state.size_bytes),
                    RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, absolute(source).id),
                )), (
                    (SemanticOperandId.HBM_ADDRESS, hbm, 0),
                    (SemanticOperandId.SOURCE_ADDRESS, absolute(source), 0),
                ))
            continue

        state_buffers = tuple(
            buffer_by_state[ref] for ref in action.state_refs if ref in buffer_by_state
        )
        if action.kind in (MoeRectActionKind.EXPERT_SGD, MoeRectActionKind.GATE_SGD):
            if len(state_buffers) != 2:
                raise SchemaError("SGD requires parameter and gradient buffers", path="plan.actions")
            weight, gradient = state_buffers
            weight_symbol = absolute(weight)
            add_bind(action.id, (weight, gradient), weight)
            add_record("compute", RelocatableRecord(action.id, RecordOpcode.SGD_UPDATE, (
                RecordOperand.literal("weight_datatype", 1),
                RecordOperand.literal("gradient_datatype", 3),
                RecordOperand.literal("output_datatype", 1),
                RecordOperand.literal("rounding", 0),
                RecordOperand.address("weight_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, weight_symbol.id),
                RecordOperand.address("gradient_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(gradient).id),
                RecordOperand.address("updated_weight_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, weight_symbol.id),
                RecordOperand.literal("element_count", weight.size_bytes // 2),
                RecordOperand.literal("learning_rate_f64_bits", _LEARNING_RATE_BITS),
                RecordOperand.literal("momentum_f64_bits", 0),
            )), (
                (SemanticOperandId.COMPUTE_INPUT_ADDRESS, weight_symbol, 0),
                (SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(gradient), 0),
                (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, weight_symbol, 0),
            ))
        elif action.kind in (
            MoeRectActionKind.PACK,
            MoeRectActionKind.WEIGHTED_COMBINE,
            MoeRectActionKind.COMBINE_BACKWARD,
            MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE,
            MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE,
        ):
            add_record("compute", RelocatableRecord(action.id, RecordOpcode.LOCAL_REDUCE, (
                RecordOperand.literal("input_dtype", 0),
                RecordOperand.literal("accumulator_dtype", 1),
                RecordOperand.literal("output_dtype", 0),
                RecordOperand.literal("reduce_op", 1),
                RecordOperand.literal("rounding", 0),
                RecordOperand.literal("order", 0),
                RecordOperand.literal("input_count", 1),
                RecordOperand.literal("element_count", activation.size_bytes // 2),
                RecordOperand.literal("input_stride_bytes", activation.size_bytes),
                RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, absolute(activation).id),
                RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, absolute(output).id),
            )), (
                (SemanticOperandId.SOURCE_ADDRESS, absolute(activation), 0),
                (SemanticOperandId.DESTINATION_ADDRESS, absolute(output), 0),
            ))
        else:
            wgrad = action.kind in (
                MoeRectActionKind.EXPERT_WGRAD, MoeRectActionKind.GATE_WGRAD,
            )
            data = activation if wgrad else (
                state_buffers[0] if state_buffers else activation
            )
            destination = output
            output_width = (
                spec.expert_count
                if action.kind in (MoeRectActionKind.GATE, MoeRectActionKind.GATE_WGRAD)
                else spec.intermediate_size
            )
            parameters = (
                (
                    1,
                    len(action.assignment_refs),
                    spec.hidden_size,
                    spec.expert_count,
                )
                if action.kind is MoeRectActionKind.GATE
                else (
                    (1, 32, 1, 16)
                    if wgrad
                    else (1, 1, max(1, spec.hidden_size), max(1, output_width))
                )
            )
            add_bind(action.id, (activation,), destination)
            add_record("compute", RelocatableRecord(action.id, RecordOpcode.MATMUL, (
                RecordOperand.literal("datatype", 1),
                RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(activation).id),
                RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(data).id),
                RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(destination).id),
                RecordOperand.literal(
                    "parameters",
                    parameters,
                ),
            )), (
                (SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute(activation), 0),
                (SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute(data), 0),
                (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute(destination), 0),
            ))

    terminal = plan.actions[-1]
    terminal_target = (
        "state"
        if terminal.kind in (MoeRectActionKind.STATE_LOAD, MoeRectActionKind.STATE_STORE)
        else "compute"
    )
    for abi in buffers:
        label = label_by_buffer[abi.id]
        add_record(terminal_target, RelocatableRecord(
            terminal.id,
            RecordOpcode.SRAM_FREE,
            (RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label.id),),
        ), ((SemanticOperandId.SYMBOL, label, 0),))

    state_program_symbols = tuple(sorted({
        symbol.id: symbol for symbol in (
            region_symbol,
            *label_by_buffer.values(),
            *absolute_by_buffer.values(),
            *hbm_by_state.values(),
        )
    }.values(), key=lambda item: item.id))
    compute_symbol_ids = {relocation.symbol_ref for relocation in compute_relocations}
    compute_program_symbols = tuple(
        item for item in state_program_symbols if item.id in compute_symbol_ids
    )
    state_fragment = CommandFragment.create(
        producer_pass=_LOWERING_PASS,
        source_global_dag_id=plan.id,
        kind=FragmentKind.STATE_IO,
        claimed_action_ids=tuple(sorted(
            action.id for action in plan.actions
            if action.kind in (MoeRectActionKind.STATE_LOAD, MoeRectActionKind.STATE_STORE)
        )),
        core_streams=(CoreFragmentStream(
            core, tuple(state_records), (),
            tuple(sorted(state_relocations, key=lambda item: (item.record_index, int(item.operand_id)))),
        ),),
        runtime_symbols=(),
        program_symbols=state_program_symbols,
        buffer_abi=buffers,
        state_abi=state_abis,
    )
    compute_fragment = CommandFragment.create(
        producer_pass=_LOWERING_PASS,
        source_global_dag_id=plan.id,
        kind=FragmentKind.COARSE,
        claimed_action_ids=tuple(sorted(
            action.id for action in plan.actions
            if action.kind not in (MoeRectActionKind.STATE_LOAD, MoeRectActionKind.STATE_STORE)
        )),
        core_streams=(CoreFragmentStream(
            core, tuple(compute_records), (),
            tuple(sorted(compute_relocations, key=lambda item: (item.record_index, int(item.operand_id)))),
        ),),
        runtime_symbols=(),
        program_symbols=compute_program_symbols,
        buffer_abi=buffers,
        state_abi=(),
    )
    state_fragment.validate("state_fragment")
    compute_fragment.validate("compute_fragment")
    fragments = tuple(sorted((state_fragment, compute_fragment), key=lambda item: item.id))

    buffer_by_symbol = {
        **{symbol.id: next(item for item in buffers if item.id == abi_id) for abi_id, symbol in label_by_buffer.items()},
        **{symbol.id: next(item for item in buffers if item.id == abi_id) for abi_id, symbol in absolute_by_buffer.items()},
    }
    state_by_symbol = {symbol.id: state for state_id, symbol in hbm_by_state.items() for state in state_abis if state.id == state_id}
    address_bindings = []
    state_bindings = []
    for fragment in fragments:
        for stream in fragment.core_streams:
            for relocation in stream.address_relocations:
                if relocation.operand_id is SemanticOperandId.HBM_ADDRESS:
                    state_bindings.append(StateOperandBinding(
                        fragment.id, core, relocation.record_index,
                        relocation.operand_id, state_by_symbol[relocation.symbol_ref].id,
                    ))
                    continue
                abi = buffer_by_symbol.get(relocation.symbol_ref)
                if relocation.operand_id is SemanticOperandId.REGION_NAME:
                    record = stream.records[relocation.record_index]
                    label_ref = record.operands[1].symbol_ref
                    abi = buffer_by_symbol[label_ref]
                if abi is None:
                    raise SchemaError("address relocation has no BufferABI", path="fragments")
                address_bindings.append(AddressOperandBinding(
                    fragment.id, core, relocation.record_index,
                    relocation.operand_id, (abi.id,), (abi.tensor_slice,),
                ))

    definitions = [ProgramSymbolDefinition(
        region_symbol, region_name,
        0 if release_region else _REGION_BASE_BYTES,
        (1 << 20) if release_region else _REGION_SIZE_BYTES,
        (core,),
    )]
    for index, abi in enumerate(buffers):
        definitions.append(ProgramSymbolDefinition(
            label_by_buffer[abi.id], f"flexible_moe_label_{index}", 0, 0, (core,),
        ))
        definitions.append(ProgramSymbolDefinition(
            absolute_by_buffer[abi.id], f"flexible_moe_abs_{index}",
            (0 if release_region else _REGION_BASE_BYTES) + abi.region_offset_bytes,
            abi.size_bytes, (core,),
        ))
    for index, state in enumerate(state_abis):
        definitions.append(ProgramSymbolDefinition(
            hbm_by_state[state.id], f"flexible_moe_hbm_{index}",
            state.address, state.size_bytes, (core,),
        ))
    definitions = tuple(sorted(definitions, key=lambda item: item.symbol.id))

    symbol_fragments = {
        symbol.id: tuple(fragment.id for fragment in fragments if symbol in fragment.program_symbols)
        for symbol in state_program_symbols
    }
    interfaces = []
    for fragment in fragments:
        local = tuple(symbol.id for symbol in fragment.program_symbols)
        exports = tuple(sorted(symbol for symbol in local if fragment.id == min(symbol_fragments[symbol])))
        interfaces.append(FragmentInterface(
            fragment.id, (), (), tuple(sorted(set(local).difference(exports))), exports, (), (),
        ))

    linked_refs = []
    fragment_by_role = {"state": state_fragment, "compute": compute_fragment}
    for action in plan.actions:
        for role, record_index in refs_by_action[action.id]:
            fragment = fragment_by_role[role]
            linked_refs.append(LinkedRecordRef(fragment.id, record_index, action.id))
    start = RuntimeSymbol(
        stable_artifact_id(
            "flexible_moe_start_tag", {"plan_id": plan.id, "core": core},
            schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
        ),
        RuntimeSymbolKind.START_TAG,
        first_action,
    )
    artifacts = (
        ManifestInputDigest(
            ManifestInputKind.FLEXIBLE_MOE_PLAN,
            plan.id,
            plan.schema_version,
            canonical_digest(plan),
        ),
        ManifestInputDigest(
            ManifestInputKind.FLEXIBLE_MOE_STANDARD_MAPPING,
            standard_ir.id,
            standard_ir.schema_version,
            canonical_digest(standard_ir),
        ),
        *tuple(
            ManifestInputDigest(
                ManifestInputKind.COMMAND_FRAGMENT, fragment.id,
                fragment.schema_version, canonical_digest(fragment),
            )
            for fragment in fragments
        ),
    )
    manifest = LinkedProgramManifest.create(
        producer_pass=_LINKER_PASS,
        capabilities=0,
        source_ir1_id=spec.id,
        source_projection_id=standard_ir.id,
        source_schedule_set_id=_SCHEDULE_ID,
        source_global_dag_id=plan.id,
        input_digests=tuple(sorted(artifacts, key=lambda item: (item.kind.value, item.artifact_id))),
        fragments=fragments,
        fragment_interfaces=tuple(sorted(interfaces, key=lambda item: item.fragment_id)),
        core_bindings=(CoreRuntimeBinding(core, "flexible_moe.core0", 0, "flexible_moe.sram_profile0"),),
        core_streams=(LinkedCoreStream(core, 0, tuple(linked_refs)),),
        runtime_symbol_definitions=(RuntimeSymbolDefinition(start, (core,), None, None),),
        program_symbol_definitions=definitions,
        address_operand_bindings=tuple(sorted(address_bindings, key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id),
        ))),
        state_operand_bindings=tuple(sorted(state_bindings, key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id),
        ))),
        core_groups=(),
        envelope=ProgramControlEnvelope(
            (core,), (LogicalStartEvent(core, start.id, 1),),
            (core,), (core,), (core,), EmptyCoreAckPolicy.INCLUDE_EMPTY,
            ProgramFailurePolicy.ABORT_ALL,
        ),
    )
    manifest.validate("flexible_moe_production_manifest")
    if len(canonical_json(manifest).encode("utf-8")) > spec.limits.max_artifact_file_bytes:
        raise SchemaError(
            "linked manifest file capacity exceeded",
            path="flexible_moe_production_manifest",
        )
    result = FlexibleMoeProductionArtifacts(
        standard_ir, fragments, manifest, True, False,
    )
    result.validate_against(plan, spec)
    return result


def build_flexible_moe_production_program_io(
    artifacts: FlexibleMoeProductionArtifacts,
    plan: FlexibleMoeExecutablePlan,
    spec: FlexibleMoeSpec,
    program_artifact_sha256: str,
) -> ProgramIoContract:
    """Build the real timing ProgramIo sidecar for an actual artifact SHA."""

    artifacts.validate_against(plan, spec)
    if program_artifact_sha256 == "0" * 64:
        raise SchemaError(
            "production ProgramIO requires an actual non-placeholder artifact SHA",
            path="program_artifact_sha256",
        )
    manifest = artifacts.manifest
    definitions = {
        item.symbol.id: (index, item)
        for index, item in enumerate(manifest.program_symbol_definitions)
    }
    buffers = {
        abi.id: abi
        for fragment in artifacts.fragments for abi in fragment.buffer_abi
    }
    states = {
        abi.id: abi
        for fragment in artifacts.fragments for abi in fragment.state_abi
    }
    labels = {
        item.symbol.source_ref: (symbol_id, index, item)
        for symbol_id, (index, item) in definitions.items()
        if item.symbol.kind is ProgramSymbolKind.SRAM_LABEL
    }
    hbms = {
        item.symbol.source_ref: (symbol_id, index, item)
        for symbol_id, (index, item) in definitions.items()
        if item.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
    }
    blobs_by_payload = {}
    initializations = []
    probes = []
    runtime_core_by_logical = {
        item.logical_core: item.runtime_core_id for item in manifest.core_bindings
    }

    def blob(size: int) -> ProgramBlob:
        return blobs_by_payload.setdefault(size, ProgramBlob.create(bytes(size)))

    activations = tuple(sorted(
        (item for item in buffers.values() if item.value_id.endswith(".activation")),
        key=lambda item: (item.logical_core.die_id, item.logical_core.local_core_id),
    ))
    if not activations:
        activations = (next(
            item for item in buffers.values()
            if item.value_id == "flexible_moe.value.activation"
        ),)

    def sram_target(abi: BufferABI) -> ProgramSramTarget:
        symbol_id, index, definition = labels[abi.storage_id]
        return ProgramSramTarget(
            ProgramIoTargetKind.SRAM, runtime_core_by_logical[abi.logical_core],
            symbol_id, index, definition.name,
            abi.id, abi.storage_id, abi.value_id, abi.tensor_slice, abi.dtype, abi.layout,
        )

    for activation in activations:
        seed = blob(activation.size_bytes)
        initializations.append(ProgramSramInitialization.create(
            target=sram_target(activation), offset_bytes=0,
            length_bytes=activation.size_bytes, blob_ref=seed.id,
            purpose=ProgramIoPurpose.ACTIVATION,
        ))
    for state in states.values():
        symbol_id, index, definition = hbms[state.hbm_binding_ref]
        target = ProgramHbmTarget(
            ProgramIoTargetKind.HBM, symbol_id, index, definition.name,
            state.id, state.state_ref, state.hbm_binding_ref,
        )
        state_blob = blob(state.size_bytes)
        initializations.append(ProgramSramInitialization.create(
            target=target, offset_bytes=0, length_bytes=state.size_bytes,
            blob_ref=state_blob.id, purpose=ProgramIoPurpose.STATE,
        ))
        if state.access is PersistentStateAccess.READ_WRITE:
            probes.append(ProgramOutputProbe.create(
                target=target, offset_bytes=0, length_bytes=state.size_bytes,
                blob_ref=state_blob.id,
                comparison=ProgramOutputComparison.EXACT_BYTES,
                capture=ProgramOutputCapture.AFTER_PROGRAM,
            ))
    contract = ProgramIoContract.create(
        producer_pass=_PROGRAM_IO_PASS,
        mode=ProgramIoMode.TIMING,
        source_manifest=manifest,
        program_artifact_sha256=program_artifact_sha256,
        blobs=tuple(blobs_by_payload.values()),
        initializations=tuple(initializations),
        output_probes=tuple(probes),
    )
    contract.validate_against(manifest)
    return contract


__all__ = [
    "FlexibleMoeProductionArtifacts",
    "FlexibleMoeRuntimeEvidence",
    "build_flexible_moe_production_program_io",
    "lower_link_flexible_moe_production",
    "observe_flexible_moe_production_runtime",
]
