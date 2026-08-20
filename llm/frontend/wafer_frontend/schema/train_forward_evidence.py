"""Strict timing-only runtime evidence for forward Train execution."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import RecordOpcode, RegionManifest
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .experiment import ExperimentSpec
from .ir0 import OpKind
from .program_io import (
    ProgramIoContract,
    ProgramIoMode,
    ProgramIoTargetKind,
)
from .serde import canonical_digest
from .train_forward_oracle import TrainForwardOracle
from .train_n6 import TrainLinkedProgram


TRAIN_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.train_forward_runtime_report/v1alpha1"
)
TRAIN_FORWARD_MARKER_SCHEMA_VERSION = (
    "wafer_frontend.train_forward_runtime_markers/v1"
)

_DRAIN_NAMES = ("collective", "global", "p2p", "timing")


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(
            "must be a canonical lowercase SHA-256 digest",
            path=path,
        )


def _bool(value: bool, path: str) -> None:
    if type(value) is not bool:
        raise SchemaError("must be a bool", path=path)


def _canonical(values: tuple[object, ...], *, key, path: str) -> None:
    if type(values) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    try:
        keys = tuple(key(value) for value in values)
    except (AttributeError, TypeError) as error:
        raise SchemaError("contains an invalid typed entry", path=path) from error
    if keys != tuple(sorted(set(keys))):
        raise SchemaError("must be unique and canonical", path=path)


@dataclass(frozen=True, slots=True)
class TrainForwardOpcodeCount:
    opcode: RecordOpcode
    count: int

    def validate(self, path: str) -> None:
        if type(self.opcode) is not RecordOpcode:
            raise SchemaError("must be a RecordOpcode", path=f"{path}.opcode")
        validate_uint64(self.count, f"{path}.count")
        if self.count == 0:
            raise SchemaError("must be positive", path=f"{path}.count")


@dataclass(frozen=True, slots=True)
class TrainForwardArtifactEvidence:
    action_count: int
    fragment_count: int
    record_count: int
    runtime_relocation_count: int
    address_relocation_count: int
    relocation_count: int
    address_operand_binding_count: int
    state_operand_binding_count: int
    opcode_counts: tuple[TrainForwardOpcodeCount, ...]

    def validate(self, path: str) -> None:
        for field_name in (
            "action_count",
            "fragment_count",
            "record_count",
            "runtime_relocation_count",
            "address_relocation_count",
            "relocation_count",
            "address_operand_binding_count",
            "state_operand_binding_count",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if min(
            self.action_count,
            self.fragment_count,
            self.record_count,
            self.address_relocation_count,
            self.address_operand_binding_count,
            self.state_operand_binding_count,
        ) == 0:
            raise SchemaError("required artifact counts must be positive", path=path)
        if self.relocation_count != (
            self.runtime_relocation_count + self.address_relocation_count
        ):
            raise SchemaError(
                "relocation count must equal runtime plus address",
                path=f"{path}.relocation_count",
            )
        _canonical(
            self.opcode_counts,
            key=lambda item: int(item.opcode),
            path=f"{path}.opcode_counts",
        )
        for index, item in enumerate(self.opcode_counts):
            if type(item) is not TrainForwardOpcodeCount:
                raise SchemaError(
                    "must be a TrainForwardOpcodeCount",
                    path=f"{path}.opcode_counts[{index}]",
                )
            item.validate(f"{path}.opcode_counts[{index}]")
        if self.record_count != sum(item.count for item in self.opcode_counts):
            raise SchemaError(
                "record count must equal opcode sum",
                path=f"{path}.record_count",
            )


@dataclass(frozen=True, slots=True)
class TrainForwardWorkEvidence:
    parameter_unique_tensor_count: int
    parameter_unique_bytes: int
    parameter_tp_placed_bytes: int
    parameter_dp_replicated_bytes: int
    gemm_flops_per_microbatch: int
    attention_flops_per_microbatch: int
    logical_forward_flops_per_microbatch: int
    rank_forward_flops_per_microbatch: int
    cluster_forward_flops_per_step: int
    collective_node_count: int
    collective_unique_bytes: int
    collective_observed_send_bytes: int

    def validate(self, path: str) -> None:
        for field_name in self.__dataclass_fields__:
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.parameter_unique_tensor_count == 0:
            raise SchemaError(
                "parameter tensor count must be positive",
                path=f"{path}.parameter_unique_tensor_count",
            )
        if self.logical_forward_flops_per_microbatch != (
            self.gemm_flops_per_microbatch
            + self.attention_flops_per_microbatch
        ):
            raise SchemaError(
                "logical FLOPs must equal GEMM plus attention",
                path=f"{path}.logical_forward_flops_per_microbatch",
            )
        if self.collective_node_count == 0 or min(
            self.collective_unique_bytes,
            self.collective_observed_send_bytes,
        ) == 0:
            raise SchemaError(
                "DP2/TP2 Train requires non-zero collective traffic",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class TrainForwardProgramIoEvidence:
    mode: ProgramIoMode
    hbm_initialization_count: int
    sram_initialization_count: int
    hbm_probe_count: int
    sram_probe_count: int
    ce_label_initialization_count: int
    ce_loss_probe_count: int

    def validate(self, path: str) -> None:
        if self.mode is not ProgramIoMode.TIMING:
            raise SchemaError("Train evidence requires timing ProgramIo", path=path)
        for field_name in self.__dataclass_fields__:
            if field_name != "mode":
                validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.hbm_probe_count != 0:
            raise SchemaError(
                "forward-only Train has no HBM output probe",
                path=f"{path}.hbm_probe_count",
            )
        if self.ce_label_initialization_count == 0 or self.ce_loss_probe_count == 0:
            raise SchemaError("CE boundary coverage must be non-zero", path=path)


@dataclass(frozen=True, slots=True)
class TrainForwardMemoryEvidence:
    runtime_core_id: int
    lsu_issued: int
    lsu_completed: int
    lsu_hbm_read_bytes: int
    lsu_hbm_write_bytes: int
    lsu_sram_read_bytes: int
    lsu_sram_write_bytes: int
    lsu_residual: int
    dte_residual: int

    def validate(self, path: str) -> None:
        for field_name in self.__dataclass_fields__:
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.runtime_core_id > 0xFFFF:
            raise SchemaError("must fit uint16", path=f"{path}.runtime_core_id")
        if self.lsu_issued == 0 or self.lsu_issued != self.lsu_completed:
            raise SchemaError("LSU work must issue and complete exactly", path=path)
        if self.lsu_hbm_write_bytes or self.lsu_sram_read_bytes:
            raise SchemaError("forward-only Train LSU path is load-only", path=path)
        if self.lsu_hbm_read_bytes != self.lsu_sram_write_bytes:
            raise SchemaError("HBM loads must close exact SRAM writes", path=path)
        if self.lsu_residual or self.dte_residual:
            raise SchemaError("memory engines must drain", path=path)


@dataclass(frozen=True, slots=True)
class TrainForwardCeMarkerEvidence:
    runtime_core_id: int
    invocation_count: int
    rank_rows: int
    label_read_bytes: int
    loss_write_bytes: int

    def validate(self, path: str) -> None:
        for field_name in self.__dataclass_fields__:
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.runtime_core_id > 0xFFFF:
            raise SchemaError("must fit uint16", path=f"{path}.runtime_core_id")
        if min(
            self.invocation_count,
            self.rank_rows,
            self.label_read_bytes,
            self.loss_write_bytes,
        ) == 0:
            raise SchemaError("CE marker fields must be positive", path=path)
        if (
            self.label_read_bytes != 4 * self.rank_rows
            or self.loss_write_bytes != 4 * self.rank_rows
        ):
            raise SchemaError("CE INT32/FP32 bytes must equal four per row", path=path)


@dataclass(frozen=True, slots=True)
class TrainForwardCoreCount:
    runtime_core_id: int
    count: int

    def validate(self, path: str) -> None:
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        validate_uint64(self.count, f"{path}.count")
        if self.runtime_core_id > 0xFFFF or self.count == 0:
            raise SchemaError("invalid core/count", path=path)


@dataclass(frozen=True, slots=True)
class TrainForwardNamedCount:
    name: str
    count: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.name, f"{path}.name")
        validate_uint64(self.count, f"{path}.count")


@dataclass(frozen=True, slots=True)
class TrainForwardControlEvidence:
    ack_counts: tuple[TrainForwardCoreCount, ...]
    done_counts: tuple[TrainForwardCoreCount, ...]
    drain_residuals: tuple[TrainForwardNamedCount, ...]
    all_done_boundary_reached: bool

    def validate(self, path: str) -> None:
        for field_name in ("ack_counts", "done_counts"):
            values = getattr(self, field_name)
            _canonical(
                values,
                key=lambda item: item.runtime_core_id,
                path=f"{path}.{field_name}",
            )
            if len(values) != 4:
                raise SchemaError("must cover exactly four Train cores", path=path)
            for index, item in enumerate(values):
                if type(item) is not TrainForwardCoreCount:
                    raise SchemaError("invalid core count", path=f"{path}.{field_name}[{index}]")
                item.validate(f"{path}.{field_name}[{index}]")
        if tuple(item.runtime_core_id for item in self.ack_counts) != tuple(
            item.runtime_core_id for item in self.done_counts
        ):
            raise SchemaError("ACK/DONE core coverage differs", path=path)
        _canonical(
            self.drain_residuals,
            key=lambda item: item.name,
            path=f"{path}.drain_residuals",
        )
        if tuple(item.name for item in self.drain_residuals) != _DRAIN_NAMES:
            raise SchemaError("must contain canonical drain classes", path=path)
        for index, item in enumerate(self.drain_residuals):
            if type(item) is not TrainForwardNamedCount:
                raise SchemaError("invalid drain count", path=f"{path}.drain_residuals[{index}]")
            item.validate(f"{path}.drain_residuals[{index}]")
            if item.count:
                raise SchemaError("all engines must drain", path=path)
        _bool(self.all_done_boundary_reached, f"{path}.all_done_boundary_reached")
        if not self.all_done_boundary_reached:
            raise SchemaError("DONE boundary must be reached", path=path)


@dataclass(frozen=True, slots=True)
class TrainForwardRepeatEvidence:
    run_index: int
    makespan_cycles: int
    marker_digest: str
    memory_digest: str
    ce_digest: str
    control_digest: str

    def validate(self, path: str) -> None:
        validate_uint64(self.run_index, f"{path}.run_index")
        validate_uint64(self.makespan_cycles, f"{path}.makespan_cycles")
        if self.makespan_cycles == 0:
            raise SchemaError("must be positive", path=f"{path}.makespan_cycles")
        for field_name in (
            "marker_digest",
            "memory_digest",
            "ce_digest",
            "control_digest",
        ):
            _digest(getattr(self, field_name), f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class TrainForwardRuntimeReport:
    schema_version: str
    producer_pass: str
    id: str
    spec_digest: str
    oracle_id: str
    oracle_digest: str
    train_linked_id: str
    train_linked_digest: str
    linked_manifest_id: str
    linked_manifest_digest: str
    program_io_id: str
    program_io_digest: str
    program_artifact_sha256: str
    artifact_size_bytes: int
    finalizer_sha256: str
    resolver_sha256: str
    npusim_sha256: str
    hardware_digest: str
    simulation_digest: str
    mapping_digest: str
    artifact: TrainForwardArtifactEvidence
    work: TrainForwardWorkEvidence
    program_io: TrainForwardProgramIoEvidence
    memory: tuple[TrainForwardMemoryEvidence, ...]
    ce_markers: tuple[TrainForwardCeMarkerEvidence, ...]
    control: TrainForwardControlEvidence
    marker_schema_version: str
    repeat_count: int
    makespan_cycles: int
    repeats: tuple[TrainForwardRepeatEvidence, ...]
    timing_execution: bool
    train_structure_exact: bool
    analytic_work_exact: bool
    collective_accounting_exact: bool
    program_io_boundary_exact: bool
    compute_functional: bool
    model_functional: bool

    @classmethod
    def create(cls, **semantic_key: object) -> "TrainForwardRuntimeReport":
        result = cls(
            schema_version=TRAIN_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION,
            producer_pass="train_forward_runtime",
            id=stable_artifact_id(
                "train_forward_runtime_report",
                semantic_key,
                schema_version=TRAIN_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "train_forward_runtime_report") -> None:
        if self.schema_version != TRAIN_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "train_forward_runtime":
            raise SchemaError("must be 'train_forward_runtime'", path=f"{path}.producer_pass")
        for field_name in (
            "oracle_id",
            "train_linked_id",
            "linked_manifest_id",
            "program_io_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        for field_name in (
            "spec_digest",
            "oracle_digest",
            "train_linked_digest",
            "linked_manifest_digest",
            "program_io_digest",
            "program_artifact_sha256",
            "finalizer_sha256",
            "resolver_sha256",
            "npusim_sha256",
            "hardware_digest",
            "simulation_digest",
            "mapping_digest",
        ):
            _digest(getattr(self, field_name), f"{path}.{field_name}")
        validate_uint64(self.artifact_size_bytes, f"{path}.artifact_size_bytes")
        if self.artifact_size_bytes == 0:
            raise SchemaError("must be positive", path=f"{path}.artifact_size_bytes")
        for field_name, expected_type in (
            ("artifact", TrainForwardArtifactEvidence),
            ("work", TrainForwardWorkEvidence),
            ("program_io", TrainForwardProgramIoEvidence),
            ("control", TrainForwardControlEvidence),
        ):
            value = getattr(self, field_name)
            if type(value) is not expected_type:
                raise SchemaError(f"must be a {expected_type.__name__}", path=f"{path}.{field_name}")
            value.validate(f"{path}.{field_name}")
        for field_name, expected_type in (
            ("memory", TrainForwardMemoryEvidence),
            ("ce_markers", TrainForwardCeMarkerEvidence),
        ):
            values = getattr(self, field_name)
            _canonical(values, key=lambda item: item.runtime_core_id, path=f"{path}.{field_name}")
            if len(values) != 4:
                raise SchemaError("must cover exactly four runtime cores", path=f"{path}.{field_name}")
            for index, item in enumerate(values):
                if type(item) is not expected_type:
                    raise SchemaError("invalid typed entry", path=f"{path}.{field_name}[{index}]")
                item.validate(f"{path}.{field_name}[{index}]")
        if tuple(item.runtime_core_id for item in self.memory) != tuple(
            item.runtime_core_id for item in self.ce_markers
        ):
            raise SchemaError("memory and CE markers must cover identical cores", path=path)
        if self.marker_schema_version != TRAIN_FORWARD_MARKER_SCHEMA_VERSION:
            raise SchemaError("unsupported marker schema", path=f"{path}.marker_schema_version")
        validate_uint64(self.repeat_count, f"{path}.repeat_count")
        validate_uint64(self.makespan_cycles, f"{path}.makespan_cycles")
        if self.repeat_count < 2 or len(self.repeats) != self.repeat_count or self.makespan_cycles == 0:
            raise SchemaError("requires at least two exact repeats", path=f"{path}.repeats")
        _canonical(self.repeats, key=lambda item: item.run_index, path=f"{path}.repeats")
        expected_digests = (
            canonical_digest(self.memory),
            canonical_digest(self.ce_markers),
            canonical_digest(self.control),
        )
        marker_digest: str | None = None
        for index, repeat in enumerate(self.repeats):
            if type(repeat) is not TrainForwardRepeatEvidence:
                raise SchemaError("invalid repeat", path=f"{path}.repeats[{index}]")
            repeat.validate(f"{path}.repeats[{index}]")
            if (
                repeat.run_index != index
                or repeat.makespan_cycles != self.makespan_cycles
                or (repeat.memory_digest, repeat.ce_digest, repeat.control_digest)
                != expected_digests
            ):
                raise SchemaError("repeat evidence differs", path=f"{path}.repeats[{index}]")
            if marker_digest is None:
                marker_digest = repeat.marker_digest
            elif repeat.marker_digest != marker_digest:
                raise SchemaError("runtime markers are not deterministic", path=f"{path}.repeats[{index}]")
        for field_name in (
            "timing_execution",
            "train_structure_exact",
            "analytic_work_exact",
            "collective_accounting_exact",
            "program_io_boundary_exact",
            "compute_functional",
            "model_functional",
        ):
            _bool(getattr(self, field_name), f"{path}.{field_name}")
        if (
            not self.timing_execution
            or not self.train_structure_exact
            or not self.analytic_work_exact
            or not self.collective_accounting_exact
            or not self.program_io_boundary_exact
            or self.compute_functional
            or self.model_functional
        ):
            raise SchemaError("proof is exact Train timing/accounting only", path=path)
        expected_id = stable_artifact_id(
            "train_forward_runtime_report",
            self._semantic_key(),
            schema_version=TRAIN_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        spec: ExperimentSpec,
        oracle: TrainForwardOracle,
        linked: TrainLinkedProgram,
        program_io: ProgramIoContract,
        path: str = "train_forward_runtime_report",
    ) -> None:
        self.validate(path)
        if type(spec) is not ExperimentSpec:
            raise SchemaError("must be an ExperimentSpec", path="spec")
        if type(oracle) is not TrainForwardOracle:
            raise SchemaError("must be a TrainForwardOracle", path="oracle")
        if type(linked) is not TrainLinkedProgram:
            raise SchemaError("must be a TrainLinkedProgram", path="linked")
        if type(program_io) is not ProgramIoContract:
            raise SchemaError("must be a ProgramIoContract", path="program_io")
        spec.validate("spec")
        oracle.validate_against_spec(spec)
        linked.validate()
        program_io.validate_against(linked.manifest)
        if oracle.tp_degree != 2 or oracle.dp_degree != 2:
            raise SchemaError("N6.5 evidence currently freezes DP2/TP2", path=path)
        if (
            self.spec_digest != canonical_digest(spec)
            or self.oracle_id != oracle.id
            or self.oracle_digest != canonical_digest(oracle)
            or self.train_linked_id != linked.id
            or self.train_linked_digest != canonical_digest(linked)
            or self.linked_manifest_id != linked.manifest.id
            or self.linked_manifest_digest != canonical_digest(linked.manifest)
            or self.program_io_id != program_io.id
            or self.program_io_digest != canonical_digest(program_io)
            or self.program_artifact_sha256 != program_io.program_artifact_sha256
        ):
            raise SchemaError("source ids/digests do not close exact inputs", path=path)

        leaves = tuple(
            item.fragment if isinstance(item, RegionManifest) else item
            for item in linked.manifest.fragments
        )
        records = tuple(
            record
            for leaf in leaves
            for stream in leaf.core_streams
            for record in stream.records
        )
        runtime_relocations = sum(
            len(stream.runtime_relocations)
            for leaf in leaves
            for stream in leaf.core_streams
        )
        address_relocations = sum(
            len(stream.address_relocations)
            for leaf in leaves
            for stream in leaf.core_streams
        )
        opcode_counts: dict[RecordOpcode, int] = {}
        for record in records:
            opcode_counts[record.opcode] = opcode_counts.get(record.opcode, 0) + 1
        expected_artifact = TrainForwardArtifactEvidence(
            len({record.source_global_action_id for record in records}),
            len(linked.manifest.fragments),
            len(records),
            runtime_relocations,
            address_relocations,
            runtime_relocations + address_relocations,
            len(linked.manifest.address_operand_bindings),
            len(linked.manifest.state_operand_bindings),
            tuple(
                TrainForwardOpcodeCount(opcode, opcode_counts[opcode])
                for opcode in sorted(opcode_counts, key=int)
            ),
        )
        if self.artifact != expected_artifact:
            raise SchemaError("artifact evidence differs from manifest", path=f"{path}.artifact")
        observed_send_bytes = sum(
            next(
                operand.literal_value
                for operand in record.operands
                if operand.name == "length_bytes"
            )
            for record in records
            if record.opcode is RecordOpcode.DTE_SEND
        )
        expected_work = TrainForwardWorkEvidence(
            oracle.parameters.unique_tensor_count,
            oracle.parameters.unique_bytes,
            oracle.parameters.tp_placed_bytes,
            oracle.parameters.dp_replicated_bytes,
            oracle.gemm_flops_per_microbatch,
            oracle.attention_flops_per_microbatch,
            oracle.logical_forward_flops_per_microbatch,
            oracle.rank_forward_flops_per_microbatch,
            oracle.cluster_forward_flops_per_step,
            oracle.collectives.node_count,
            oracle.collectives.node_count
            * oracle.collectives.logical_tensor_bytes_per_node
            * oracle.dp_degree,
            observed_send_bytes,
        )
        if self.work != expected_work:
            raise SchemaError("work evidence differs from oracle/records", path=f"{path}.work")

        hbm_initializations = sum(
            item.target.kind is ProgramIoTargetKind.HBM
            for item in program_io.initializations
        )
        sram_initializations = len(program_io.initializations) - hbm_initializations
        hbm_probes = sum(
            item.target.kind is ProgramIoTargetKind.HBM
            for item in program_io.output_probes
        )
        sram_probes = len(program_io.output_probes) - hbm_probes
        ce_nodes = tuple(
            node
            for replica in linked.source.replicas
            for node in replica.lowering_context.ir1.nodes
            if node.kind is OpKind.CE_FORWARD
        )
        if len(ce_nodes) != oracle.dp_degree:
            raise SchemaError(
                "linked Train must contain one CE node per DP replica",
                path=f"{path}.program_io",
            )
        label_value_ids = {node.inputs[1] for node in ce_nodes}
        loss_value_ids = {node.outputs[0] for node in ce_nodes}
        ce_label_initializations = sum(
            item.target.kind is ProgramIoTargetKind.SRAM
            and item.target.value_id in label_value_ids
            and item.target.dtype is DType.INT32
            for item in program_io.initializations
        )
        ce_loss_probes = sum(
            item.target.kind is ProgramIoTargetKind.SRAM
            and item.target.value_id in loss_value_ids
            and item.target.dtype is DType.FP32
            for item in program_io.output_probes
        )
        if (
            ce_label_initializations != oracle.dp_degree * oracle.tp_degree
            or ce_loss_probes != oracle.dp_degree * oracle.tp_degree
        ):
            raise SchemaError(
                "ProgramIo must exactly cover all CE label/loss rank shards",
                path=f"{path}.program_io",
            )
        expected_program_io = TrainForwardProgramIoEvidence(
            program_io.mode,
            hbm_initializations,
            sram_initializations,
            hbm_probes,
            sram_probes,
            ce_label_initializations,
            ce_loss_probes,
        )
        if self.program_io != expected_program_io:
            raise SchemaError("ProgramIo evidence differs from contract", path=f"{path}.program_io")
        runtime_cores = tuple(
            binding.runtime_core_id for binding in linked.manifest.core_bindings
        )
        if tuple(item.runtime_core_id for item in self.memory) != runtime_cores:
            raise SchemaError("memory must cover manifest cores", path=f"{path}.memory")
        if sum(item.lsu_hbm_read_bytes for item in self.memory) != oracle.parameters.dp_replicated_bytes:
            raise SchemaError("HBM marker bytes differ from parameter oracle", path=f"{path}.memory")
        expected_ce = tuple(
            TrainForwardCeMarkerEvidence(
                core,
                1,
                oracle.ce.rank_rows,
                oracle.ce.rank_label_bytes,
                oracle.ce.rank_loss_bytes,
            )
            for core in runtime_cores
        )
        if self.ce_markers != expected_ce:
            raise SchemaError("CE markers differ from oracle/core coverage", path=f"{path}.ce_markers")
        expected_ack = tuple(
            TrainForwardCoreCount(binding.runtime_core_id, 2)
            for binding in linked.manifest.core_bindings
            if binding.logical_core in linked.manifest.envelope.expected_ack_cores
        )
        expected_done = tuple(
            TrainForwardCoreCount(binding.runtime_core_id, 1)
            for binding in linked.manifest.core_bindings
            if binding.logical_core in linked.manifest.envelope.expected_done_cores
        )
        if self.control.ack_counts != expected_ack or self.control.done_counts != expected_done:
            raise SchemaError("control evidence differs from manifest envelope", path=f"{path}.control")


__all__ = [
    "TRAIN_FORWARD_MARKER_SCHEMA_VERSION",
    "TRAIN_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION",
    "TrainForwardArtifactEvidence",
    "TrainForwardCeMarkerEvidence",
    "TrainForwardControlEvidence",
    "TrainForwardCoreCount",
    "TrainForwardMemoryEvidence",
    "TrainForwardNamedCount",
    "TrainForwardOpcodeCount",
    "TrainForwardProgramIoEvidence",
    "TrainForwardRepeatEvidence",
    "TrainForwardRuntimeReport",
    "TrainForwardWorkEvidence",
]
