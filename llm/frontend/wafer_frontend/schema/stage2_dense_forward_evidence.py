"""Strict runtime evidence for the Stage 2 dense-forward timing cases."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import RecordOpcode
from .capability import CapabilityStatus
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .experiment import InferOutput
from .program_io import ProgramIoMode, ProgramIoTargetKind
from .serde import canonical_digest
from .stage2_dense_forward_oracle import Stage2DenseForwardOracle


STAGE2_DENSE_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.stage2_dense_forward_runtime_report/v1alpha1"
)
STAGE2_DENSE_FORWARD_MARKER_SCHEMA_VERSION = (
    "wafer_frontend.stage2_dense_forward_runtime_markers/v1"
)
STAGE2_DENSE_FORWARD_BASELINE_EPOCH = "stage2-dense-forward-v1"

_DRAIN_NAMES = ("collective", "global", "p2p", "timing")
_PACKET_BYTES = 16


def _validate_digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(
            "must be a canonical lowercase SHA-256 digest", path=path
        )


def _validate_bool(value: bool, path: str) -> None:
    if type(value) is not bool:
        raise SchemaError("must be a bool", path=path)


def _validate_canonical(
    values: tuple[object, ...],
    *,
    key,
    path: str,
) -> None:
    if type(values) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    try:
        keys = tuple(key(value) for value in values)
        canonical = tuple(sorted(set(keys)))
    except (AttributeError, TypeError) as error:
        raise SchemaError("contains an invalid typed entry", path=path) from error
    if keys != canonical:
        raise SchemaError("must be unique and in canonical order", path=path)


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardOpcodeCount:
    opcode: RecordOpcode
    count: int

    def validate(self, path: str = "stage2_dense_forward_opcode_count") -> None:
        if type(self.opcode) is not RecordOpcode:
            raise SchemaError("must be a RecordOpcode", path=f"{path}.opcode")
        validate_uint64(self.count, f"{path}.count")
        if self.count == 0:
            raise SchemaError("must be positive", path=f"{path}.count")


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardToolEvidence:
    finalizer_sha256: str
    resolver_sha256: str
    npusim_sha256: str

    def validate(self, path: str = "stage2_dense_forward_tool_evidence") -> None:
        for field_name in self.__dataclass_fields__:
            _validate_digest(getattr(self, field_name), f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardCompileEvidence:
    template_id: str
    template_digest: str
    ir1_id: str
    ir1_digest: str
    global_dag_id: str
    global_dag_digest: str
    lowered_id: str
    lowered_digest: str

    def validate(
        self, path: str = "stage2_dense_forward_compile_evidence"
    ) -> None:
        for field_name in (
            "template_id",
            "ir1_id",
            "global_dag_id",
            "lowered_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        for field_name in (
            "template_digest",
            "ir1_digest",
            "global_dag_digest",
            "lowered_digest",
        ):
            _validate_digest(getattr(self, field_name), f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardArtifactEvidence:
    linked_manifest_id: str
    linked_manifest_digest: str
    program_artifact_sha256: str
    artifact_size_bytes: int
    action_count: int
    leaf_fragment_count: int
    record_count: int
    address_binding_count: int
    relocation_count: int
    opcode_counts: tuple[Stage2DenseForwardOpcodeCount, ...]

    def validate(
        self, path: str = "stage2_dense_forward_artifact_evidence"
    ) -> None:
        validate_nonempty(
            self.linked_manifest_id, f"{path}.linked_manifest_id"
        )
        _validate_digest(
            self.linked_manifest_digest, f"{path}.linked_manifest_digest"
        )
        _validate_digest(
            self.program_artifact_sha256, f"{path}.program_artifact_sha256"
        )
        for field_name in (
            "artifact_size_bytes",
            "action_count",
            "leaf_fragment_count",
            "record_count",
            "address_binding_count",
            "relocation_count",
        ):
            value = getattr(self, field_name)
            validate_uint64(value, f"{path}.{field_name}")
            if value == 0:
                raise SchemaError("must be positive", path=f"{path}.{field_name}")
        _validate_canonical(
            self.opcode_counts,
            key=lambda item: int(item.opcode),
            path=f"{path}.opcode_counts",
        )
        for index, item in enumerate(self.opcode_counts):
            if type(item) is not Stage2DenseForwardOpcodeCount:
                raise SchemaError(
                    "must be a Stage2DenseForwardOpcodeCount",
                    path=f"{path}.opcode_counts[{index}]",
                )
            item.validate(f"{path}.opcode_counts[{index}]")
        if self.record_count != sum(item.count for item in self.opcode_counts):
            raise SchemaError(
                "must equal the opcode count sum", path=f"{path}.record_count"
            )


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardSidecarEvidence:
    contract_id: str
    contract_digest: str
    mode: ProgramIoMode
    hbm_initialization_count: int
    sram_initialization_count: int
    hbm_probe_count: int
    sram_probe_count: int

    def validate(
        self, path: str = "stage2_dense_forward_sidecar_evidence"
    ) -> None:
        validate_nonempty(self.contract_id, f"{path}.contract_id")
        _validate_digest(self.contract_digest, f"{path}.contract_digest")
        if type(self.mode) is not ProgramIoMode:
            raise SchemaError("must be a ProgramIoMode", path=f"{path}.mode")
        for field_name in (
            "hbm_initialization_count",
            "sram_initialization_count",
            "hbm_probe_count",
            "sram_probe_count",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardMemoryEvidence:
    runtime_core_id: int
    lsu_issued: int
    lsu_completed: int
    lsu_hbm_read_bytes: int
    lsu_hbm_write_bytes: int
    lsu_sram_read_bytes: int
    lsu_sram_write_bytes: int
    lsu_residual: int
    dte_residual: int

    def validate(
        self, path: str = "stage2_dense_forward_memory_evidence"
    ) -> None:
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        if self.runtime_core_id > 0xFFFF:
            raise SchemaError(
                "must fit the ProgramArtifact uint16 core id",
                path=f"{path}.runtime_core_id",
            )
        for field_name in self.__dataclass_fields__:
            if field_name != "runtime_core_id":
                validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.lsu_issued != self.lsu_completed:
            raise SchemaError("LSU issued/completed must close exactly", path=path)
        if self.lsu_residual != 0 or self.dte_residual != 0:
            raise SchemaError("memory engines must drain", path=path)


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardProbeEvidence:
    probe_id: str
    target_kind: ProgramIoTargetKind
    length_bytes: int
    expected_sha256: str
    actual_sha256: str
    all_bytes_valid: bool
    exact_match: bool
    passed: bool

    def validate(
        self, path: str = "stage2_dense_forward_probe_evidence"
    ) -> None:
        validate_nonempty(self.probe_id, f"{path}.probe_id")
        if type(self.target_kind) is not ProgramIoTargetKind:
            raise SchemaError(
                "must be a ProgramIoTargetKind", path=f"{path}.target_kind"
            )
        validate_uint64(self.length_bytes, f"{path}.length_bytes")
        if self.length_bytes == 0:
            raise SchemaError("must be positive", path=f"{path}.length_bytes")
        _validate_digest(self.expected_sha256, f"{path}.expected_sha256")
        _validate_digest(self.actual_sha256, f"{path}.actual_sha256")
        for field_name in ("all_bytes_valid", "exact_match", "passed"):
            _validate_bool(getattr(self, field_name), f"{path}.{field_name}")
        expected_pass = (
            self.all_bytes_valid
            and self.exact_match
            and self.expected_sha256 == self.actual_sha256
        )
        if self.passed is not expected_pass or not self.passed:
            raise SchemaError("probe must pass exact valid comparison", path=path)


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardCoreCount:
    runtime_core_id: int
    count: int

    def validate(self, path: str = "stage2_dense_forward_core_count") -> None:
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        if self.runtime_core_id > 0xFFFF:
            raise SchemaError(
                "must fit the ProgramArtifact uint16 core id",
                path=f"{path}.runtime_core_id",
            )
        validate_uint64(self.count, f"{path}.count")
        if self.count == 0:
            raise SchemaError("must be positive", path=f"{path}.count")


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardNamedCount:
    name: str
    count: int

    def validate(self, path: str = "stage2_dense_forward_named_count") -> None:
        validate_nonempty(self.name, f"{path}.name")
        validate_uint64(self.count, f"{path}.count")


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardControlEvidence:
    ack_counts: tuple[Stage2DenseForwardCoreCount, ...]
    done_counts: tuple[Stage2DenseForwardCoreCount, ...]
    drain_residuals: tuple[Stage2DenseForwardNamedCount, ...]
    all_done_boundary_reached: bool

    def validate(
        self, path: str = "stage2_dense_forward_control_evidence"
    ) -> None:
        for field_name in ("ack_counts", "done_counts"):
            values = getattr(self, field_name)
            _validate_canonical(
                values,
                key=lambda item: item.runtime_core_id,
                path=f"{path}.{field_name}",
            )
            if not values:
                raise SchemaError("must be non-empty", path=f"{path}.{field_name}")
            for index, item in enumerate(values):
                if type(item) is not Stage2DenseForwardCoreCount:
                    raise SchemaError(
                        "must be a Stage2DenseForwardCoreCount",
                        path=f"{path}.{field_name}[{index}]",
                    )
                item.validate(f"{path}.{field_name}[{index}]")
        if tuple(item.runtime_core_id for item in self.ack_counts) != tuple(
            item.runtime_core_id for item in self.done_counts
        ):
            raise SchemaError(
                "ACK and DONE must cover the same runtime cores", path=path
            )
        _validate_canonical(
            self.drain_residuals,
            key=lambda item: item.name,
            path=f"{path}.drain_residuals",
        )
        if tuple(item.name for item in self.drain_residuals) != _DRAIN_NAMES:
            raise SchemaError(
                "must contain the four canonical drain classes",
                path=f"{path}.drain_residuals",
            )
        for index, item in enumerate(self.drain_residuals):
            if type(item) is not Stage2DenseForwardNamedCount:
                raise SchemaError(
                    "must be a Stage2DenseForwardNamedCount",
                    path=f"{path}.drain_residuals[{index}]",
                )
            item.validate(f"{path}.drain_residuals[{index}]")
            if item.count != 0:
                raise SchemaError(
                    "all runtime residuals must drain",
                    path=f"{path}.drain_residuals[{index}].count",
                )
        _validate_bool(
            self.all_done_boundary_reached,
            f"{path}.all_done_boundary_reached",
        )
        if not self.all_done_boundary_reached:
            raise SchemaError("DONE boundary must be reached", path=path)


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardD2DLinkEvidence:
    source_die_id: int
    destination_die_id: int
    request_packets: int
    ack_packets: int
    data_packets: int

    def validate(
        self, path: str = "stage2_dense_forward_d2d_link_evidence"
    ) -> None:
        for field_name in self.__dataclass_fields__:
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.source_die_id == self.destination_die_id:
            raise SchemaError("link endpoints must be distinct", path=path)
        if (
            self.request_packets == 0
            or self.ack_packets == 0
            or self.data_packets == 0
        ):
            raise SchemaError("active link packet counts must be positive", path=path)


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardD2DEvidence:
    flow_count: int
    logical_packet_count: int
    physical_packet_count: int
    request_packet_count: int
    ack_packet_count: int
    logical_bytes: int
    byte_hop_bytes: int
    links: tuple[Stage2DenseForwardD2DLinkEvidence, ...]

    def validate(
        self, path: str = "stage2_dense_forward_d2d_evidence"
    ) -> None:
        for field_name in (
            "flow_count",
            "logical_packet_count",
            "physical_packet_count",
            "request_packet_count",
            "ack_packet_count",
            "logical_bytes",
            "byte_hop_bytes",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        _validate_canonical(
            self.links,
            key=lambda item: (item.source_die_id, item.destination_die_id),
            path=f"{path}.links",
        )
        for index, item in enumerate(self.links):
            if type(item) is not Stage2DenseForwardD2DLinkEvidence:
                raise SchemaError(
                    "must be a Stage2DenseForwardD2DLinkEvidence",
                    path=f"{path}.links[{index}]",
                )
            item.validate(f"{path}.links[{index}]")
        if self.logical_bytes != self.logical_packet_count * _PACKET_BYTES:
            raise SchemaError(
                "logical bytes must equal logical packets times 16",
                path=f"{path}.logical_bytes",
            )
        if self.byte_hop_bytes != self.physical_packet_count * _PACKET_BYTES:
            raise SchemaError(
                "byte-hop bytes must equal physical packets times 16",
                path=f"{path}.byte_hop_bytes",
            )
        if (
            sum(item.request_packets for item in self.links)
            != self.request_packet_count
            or sum(item.ack_packets for item in self.links)
            != self.ack_packet_count
            or sum(item.data_packets for item in self.links)
            != self.physical_packet_count
        ):
            raise SchemaError("link packets must close exact D2D totals", path=path)
        empty = not self.links
        if empty is not (
            self.flow_count == 0
            and self.logical_packet_count == 0
            and self.physical_packet_count == 0
            and self.request_packet_count == 0
            and self.ack_packet_count == 0
        ):
            raise SchemaError(
                "empty D2D topology must have exact zero traffic", path=path
            )


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardRepeatEvidence:
    run_index: int
    makespan_cycles: int
    marker_digest: str
    memory_digest: str
    probe_digest: str
    control_digest: str
    d2d_digest: str

    def validate(
        self, path: str = "stage2_dense_forward_repeat_evidence"
    ) -> None:
        validate_uint64(self.run_index, f"{path}.run_index")
        validate_uint64(self.makespan_cycles, f"{path}.makespan_cycles")
        if self.makespan_cycles == 0:
            raise SchemaError("must be positive", path=f"{path}.makespan_cycles")
        for field_name in (
            "marker_digest",
            "memory_digest",
            "probe_digest",
            "control_digest",
            "d2d_digest",
        ):
            _validate_digest(getattr(self, field_name), f"{path}.{field_name}")


def _opcode_counts(**values: int) -> tuple[Stage2DenseForwardOpcodeCount, ...]:
    return tuple(
        Stage2DenseForwardOpcodeCount(RecordOpcode[name], count)
        for name, count in sorted(
            values.items(), key=lambda item: int(RecordOpcode[item[0]])
        )
    )


_CASE_GOLDENS = {
    1: {
        "artifact": (
            22258,
            44,
            44,
            159,
            278,
            297,
            "308152e15697e45329660c160a3428a809015e408079019693e8d7c7d4d3fbad",
        ),
        "opcodes": _opcode_counts(
            ATTENTION_EXACT=2,
            EMBEDDING_LOOKUP=1,
            LSU_LOAD=15,
            LSU_STORE=4,
            MATMUL=9,
            RESIDUAL=4,
            RMSNORM=5,
            ROPE_QK_EXACT=2,
            SRAM_ALLOC_AT=45,
            SRAM_BIND=25,
            SRAM_FREE=45,
            SWIGLU=2,
        ),
        "sidecar": (15, 45, 0, 1),
        "memory": (
            Stage2DenseForwardMemoryEvidence(
                0, 19, 19, 12448, 1024, 1024, 12448, 0, 0
            ),
        ),
        "d2d": Stage2DenseForwardD2DEvidence(0, 0, 0, 0, 0, 0, 0, ()),
        "makespan": 5805,
    },
    2: {
        "artifact": (
            65832,
            160,
            92,
            510,
            804,
            842,
            "1824ceeea8a23150489df9a0e41f4571ac90edbcf86ff82da0feb786888a830c",
        ),
        "opcodes": _opcode_counts(
            ATTENTION_EXACT=4,
            DTE_ISSUE=8,
            DTE_RECV=16,
            DTE_SEND=16,
            DTE_WAIT=16,
            EMBEDDING_LOOKUP=2,
            EVENT_SET=8,
            EVENT_WAIT=8,
            LOCAL_REDUCE=8,
            LSU_LOAD=30,
            LSU_STORE=8,
            MATMUL=26,
            RESIDUAL=8,
            RMSNORM=10,
            ROPE_QK_EXACT=4,
            SRAM_ALLOC_AT=138,
            SRAM_BIND=58,
            SRAM_FREE=138,
            SWIGLU=4,
        ),
        "sidecar": (30, 138, 0, 2),
        "memory": (
            Stage2DenseForwardMemoryEvidence(
                0, 19, 19, 7328, 512, 512, 7328, 0, 0
            ),
            Stage2DenseForwardMemoryEvidence(
                16, 19, 19, 7328, 512, 512, 7328, 0, 0
            ),
        ),
        "d2d": Stage2DenseForwardD2DEvidence(
            16,
            128,
            128,
            16,
            32,
            2048,
            2048,
            (
                Stage2DenseForwardD2DLinkEvidence(0, 1, 8, 16, 64),
                Stage2DenseForwardD2DLinkEvidence(1, 0, 8, 16, 64),
            ),
        ),
        "makespan": 6628,
    },
    4: {
        "artifact": (
            179316,
            544,
            180,
            1452,
            2184,
            2260,
            "fad49b3356701b6e5e65139f7e91104b0693c020710536b97af84bd177fae187",
        ),
        "opcodes": _opcode_counts(
            ATTENTION_EXACT=8,
            DTE_ISSUE=16,
            DTE_RECV=96,
            DTE_SEND=96,
            DTE_WAIT=64,
            EMBEDDING_LOOKUP=4,
            EVENT_SET=24,
            EVENT_WAIT=24,
            LOCAL_REDUCE=16,
            LSU_LOAD=60,
            LSU_STORE=16,
            MATMUL=84,
            RESIDUAL=16,
            RMSNORM=20,
            ROPE_QK_EXACT=8,
            SRAM_ALLOC_AT=372,
            SRAM_BIND=148,
            SRAM_FREE=372,
            SWIGLU=8,
        ),
        "sidecar": (60, 372, 0, 4),
        "memory": (
            Stage2DenseForwardMemoryEvidence(
                0, 19, 19, 4768, 256, 256, 4768, 0, 0
            ),
            Stage2DenseForwardMemoryEvidence(
                4, 19, 19, 4768, 256, 256, 4768, 0, 0
            ),
            Stage2DenseForwardMemoryEvidence(
                8, 19, 19, 4768, 256, 256, 4768, 0, 0
            ),
            Stage2DenseForwardMemoryEvidence(
                12, 19, 19, 4768, 256, 256, 4768, 0, 0
            ),
        ),
        "d2d": Stage2DenseForwardD2DEvidence(
            96,
            384,
            512,
            128,
            256,
            6144,
            8192,
            tuple(
                Stage2DenseForwardD2DLinkEvidence(source, destination, 16, 32, 64)
                for source, destination in (
                    (0, 1),
                    (0, 2),
                    (1, 0),
                    (1, 3),
                    (2, 0),
                    (2, 3),
                    (3, 1),
                    (3, 2),
                )
            ),
        ),
        "makespan": 7503,
    },
}


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardRuntimeReport:
    schema_version: str
    producer_pass: str
    id: str
    baseline_epoch: str
    tp_degree: int
    infer_output: InferOutput
    capability_status: CapabilityStatus
    oracle_id: str
    oracle_digest: str
    compile: Stage2DenseForwardCompileEvidence
    tools: Stage2DenseForwardToolEvidence
    hardware_digest: str
    simulation_digest: str
    mapping_digest: str
    artifact: Stage2DenseForwardArtifactEvidence
    sidecar: Stage2DenseForwardSidecarEvidence
    memory: tuple[Stage2DenseForwardMemoryEvidence, ...]
    probes: tuple[Stage2DenseForwardProbeEvidence, ...]
    control: Stage2DenseForwardControlEvidence
    d2d: Stage2DenseForwardD2DEvidence
    marker_schema_version: str
    repeat_count: int
    makespan_cycles: int
    repeats: tuple[Stage2DenseForwardRepeatEvidence, ...]
    timing_execution: bool
    dense_forward_structure_exact: bool
    analytic_work_exact: bool
    program_io_boundary_exact: bool
    traffic_accounting_exact: bool
    compute_functional: bool
    model_functional: bool

    @classmethod
    def create(cls, **semantic_key: object) -> "Stage2DenseForwardRuntimeReport":
        result = cls(
            schema_version=STAGE2_DENSE_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION,
            producer_pass="stage2_dense_forward_runtime",
            id=stable_artifact_id(
                "stage2_dense_forward_runtime_report",
                semantic_key,
                schema_version=(
                    STAGE2_DENSE_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION
                ),
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

    def validate(
        self, path: str = "stage2_dense_forward_runtime_report"
    ) -> None:
        if (
            self.schema_version
            != STAGE2_DENSE_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION
        ):
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "stage2_dense_forward_runtime":
            raise SchemaError(
                "must be 'stage2_dense_forward_runtime'",
                path=f"{path}.producer_pass",
            )
        if self.baseline_epoch != STAGE2_DENSE_FORWARD_BASELINE_EPOCH:
            raise SchemaError(
                f"must be {STAGE2_DENSE_FORWARD_BASELINE_EPOCH!r}",
                path=f"{path}.baseline_epoch",
            )
        validate_uint64(self.tp_degree, f"{path}.tp_degree")
        golden = _CASE_GOLDENS.get(self.tp_degree)
        if golden is None:
            raise SchemaError(
                "reviewed runtime report requires TP1, TP2 or TP4",
                path=f"{path}.tp_degree",
            )
        if self.infer_output is not InferOutput.LOGITS:
            raise SchemaError(
                "reviewed runtime report covers LOGITS only",
                path=f"{path}.infer_output",
            )
        if self.capability_status is not CapabilityStatus.E2E_TIMING:
            raise SchemaError(
                "dense-forward evidence is timing-only",
                path=f"{path}.capability_status",
            )
        validate_nonempty(self.oracle_id, f"{path}.oracle_id")
        _validate_digest(self.oracle_digest, f"{path}.oracle_digest")
        for field_name in (
            "hardware_digest",
            "simulation_digest",
            "mapping_digest",
        ):
            _validate_digest(getattr(self, field_name), f"{path}.{field_name}")
        for field_name, expected_type in (
            ("compile", Stage2DenseForwardCompileEvidence),
            ("tools", Stage2DenseForwardToolEvidence),
            ("artifact", Stage2DenseForwardArtifactEvidence),
            ("sidecar", Stage2DenseForwardSidecarEvidence),
            ("control", Stage2DenseForwardControlEvidence),
            ("d2d", Stage2DenseForwardD2DEvidence),
        ):
            value = getattr(self, field_name)
            if type(value) is not expected_type:
                raise SchemaError(
                    f"must be a {expected_type.__name__}",
                    path=f"{path}.{field_name}",
                )
            value.validate(f"{path}.{field_name}")

        _validate_canonical(
            self.memory,
            key=lambda item: item.runtime_core_id,
            path=f"{path}.memory",
        )
        for index, item in enumerate(self.memory):
            if type(item) is not Stage2DenseForwardMemoryEvidence:
                raise SchemaError(
                    "must be a Stage2DenseForwardMemoryEvidence",
                    path=f"{path}.memory[{index}]",
                )
            item.validate(f"{path}.memory[{index}]")
        if self.memory != golden["memory"]:
            raise SchemaError(
                "memory evidence disagrees with the frozen TP case",
                path=f"{path}.memory",
            )

        _validate_canonical(
            self.probes,
            key=lambda item: item.probe_id,
            path=f"{path}.probes",
        )
        for index, probe in enumerate(self.probes):
            if type(probe) is not Stage2DenseForwardProbeEvidence:
                raise SchemaError(
                    "must be a Stage2DenseForwardProbeEvidence",
                    path=f"{path}.probes[{index}]",
                )
            probe.validate(f"{path}.probes[{index}]")
            if probe.target_kind is not ProgramIoTargetKind.SRAM:
                raise SchemaError(
                    "reviewed dense-forward probes are SRAM boundary probes",
                    path=f"{path}.probes[{index}].target_kind",
                )
        if len(self.probes) != self.tp_degree:
            raise SchemaError(
                "probe evidence must cover one terminal SRAM output per rank",
                path=f"{path}.probes",
            )

        artifact_values = (
            self.artifact.artifact_size_bytes,
            self.artifact.action_count,
            self.artifact.leaf_fragment_count,
            self.artifact.record_count,
            self.artifact.address_binding_count,
            self.artifact.relocation_count,
            self.artifact.program_artifact_sha256,
        )
        if artifact_values != golden["artifact"]:
            raise SchemaError(
                "artifact evidence disagrees with the frozen TP case",
                path=f"{path}.artifact",
            )
        if self.artifact.opcode_counts != golden["opcodes"]:
            raise SchemaError(
                "opcode evidence disagrees with the frozen TP case",
                path=f"{path}.artifact.opcode_counts",
            )
        sidecar_values = (
            self.sidecar.hbm_initialization_count,
            self.sidecar.sram_initialization_count,
            self.sidecar.hbm_probe_count,
            self.sidecar.sram_probe_count,
        )
        if (
            self.sidecar.mode is not ProgramIoMode.TIMING
            or sidecar_values != golden["sidecar"]
        ):
            raise SchemaError(
                "sidecar evidence disagrees with the frozen TP case",
                path=f"{path}.sidecar",
            )
        if self.d2d != golden["d2d"]:
            raise SchemaError(
                "D2D evidence disagrees with the frozen TP case",
                path=f"{path}.d2d",
            )

        active_cores = tuple(item.runtime_core_id for item in self.memory)
        expected_ack = tuple(
            Stage2DenseForwardCoreCount(core_id, 2)
            for core_id in active_cores
        )
        expected_done = tuple(
            Stage2DenseForwardCoreCount(core_id, 1)
            for core_id in active_cores
        )
        if (
            self.control.ack_counts != expected_ack
            or self.control.done_counts != expected_done
        ):
            raise SchemaError(
                "control evidence disagrees with active core closure",
                path=f"{path}.control",
            )
        if (
            self.marker_schema_version
            != STAGE2_DENSE_FORWARD_MARKER_SCHEMA_VERSION
        ):
            raise SchemaError(
                "unsupported marker schema version",
                path=f"{path}.marker_schema_version",
            )
        validate_uint64(self.repeat_count, f"{path}.repeat_count")
        validate_uint64(self.makespan_cycles, f"{path}.makespan_cycles")
        if (
            self.repeat_count != 2
            or len(self.repeats) != 2
            or self.makespan_cycles != golden["makespan"]
        ):
            raise SchemaError(
                "reviewed runtime evidence requires two exact repeats",
                path=f"{path}.repeats",
            )
        _validate_canonical(
            self.repeats,
            key=lambda item: item.run_index,
            path=f"{path}.repeats",
        )
        memory_digest = canonical_digest(self.memory)
        probe_digest = canonical_digest(self.probes)
        control_digest = canonical_digest(self.control)
        d2d_digest = canonical_digest(self.d2d)
        marker_digest: str | None = None
        for index, repeat in enumerate(self.repeats):
            if type(repeat) is not Stage2DenseForwardRepeatEvidence:
                raise SchemaError(
                    "must be a Stage2DenseForwardRepeatEvidence",
                    path=f"{path}.repeats[{index}]",
                )
            repeat.validate(f"{path}.repeats[{index}]")
            if (
                repeat.run_index != index
                or repeat.makespan_cycles != self.makespan_cycles
                or repeat.memory_digest != memory_digest
                or repeat.probe_digest != probe_digest
                or repeat.control_digest != control_digest
                or repeat.d2d_digest != d2d_digest
            ):
                raise SchemaError(
                    "repeat does not reproduce canonical report evidence",
                    path=f"{path}.repeats[{index}]",
                )
            if marker_digest is None:
                marker_digest = repeat.marker_digest
            elif repeat.marker_digest != marker_digest:
                raise SchemaError(
                    "runtime markers are not deterministic",
                    path=f"{path}.repeats[{index}].marker_digest",
                )

        for field_name in (
            "timing_execution",
            "dense_forward_structure_exact",
            "analytic_work_exact",
            "program_io_boundary_exact",
            "traffic_accounting_exact",
            "compute_functional",
            "model_functional",
        ):
            _validate_bool(getattr(self, field_name), f"{path}.{field_name}")
        if (
            not self.timing_execution
            or not self.dense_forward_structure_exact
            or not self.analytic_work_exact
            or not self.program_io_boundary_exact
            or not self.traffic_accounting_exact
            or self.compute_functional
            or self.model_functional
        ):
            raise SchemaError(
                "proof is exact dense-forward timing/accounting only", path=path
            )
        expected_id = stable_artifact_id(
            "stage2_dense_forward_runtime_report",
            self._semantic_key(),
            schema_version=STAGE2_DENSE_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        oracle: Stage2DenseForwardOracle,
        path: str = "stage2_dense_forward_runtime_report",
    ) -> None:
        self.validate(path)
        if type(oracle) is not Stage2DenseForwardOracle:
            raise SchemaError(
                "must be a Stage2DenseForwardOracle", path=f"{path}.oracle"
            )
        oracle.validate("stage2_dense_forward_oracle")
        if (
            self.oracle_id != oracle.id
            or self.oracle_digest != canonical_digest(oracle)
            or self.tp_degree != oracle.tp_degree
            or self.infer_output is not oracle.infer_output
            or self.compile.template_id != oracle.source_template_id
        ):
            raise SchemaError("does not identify the supplied oracle", path=path)
        hbm_read_bytes = sum(item.lsu_hbm_read_bytes for item in self.memory)
        hbm_write_bytes = sum(item.lsu_hbm_write_bytes for item in self.memory)
        if (
            hbm_read_bytes != oracle.parameters.placed_bytes
            or hbm_write_bytes != oracle.kv.logical_write_bytes
            or hbm_write_bytes != oracle.kv.rank_write_bytes * self.tp_degree
        ):
            raise SchemaError(
                "HBM traffic disagrees with parameter/KV oracle",
                path=f"{path}.memory",
            )
        expected_d2d_bytes = (
            oracle.collectives.all_gather.group_payload_bytes_total
            + oracle.collectives.reduce_scatter.group_payload_bytes_total
        )
        if self.d2d.logical_bytes != expected_d2d_bytes:
            raise SchemaError(
                "logical D2D bytes disagree with collective oracle",
                path=f"{path}.d2d.logical_bytes",
            )
        opcode_counts = {
            item.opcode: item.count for item in self.artifact.opcode_counts
        }
        if (
            self.sidecar.hbm_initialization_count
            != oracle.graph.parameter_declaration_count
            or opcode_counts.get(RecordOpcode.LSU_LOAD, 0)
            != oracle.graph.parameter_declaration_count
            or opcode_counts.get(RecordOpcode.LSU_STORE, 0)
            != oracle.graph.kv_declaration_count
            or len(self.memory) != self.tp_degree
            or len(self.probes) != self.tp_degree
        ):
            raise SchemaError(
                "runtime counts disagree with graph/state oracle", path=path
            )
        if any(
            getattr(oracle.logical_work.greedy, field_name) != 0
            for field_name in oracle.logical_work.greedy.__dataclass_fields__
        ):
            raise SchemaError(
                "reviewed LOGITS evidence must not execute sampling",
                path=f"{path}.oracle",
            )


__all__ = [
    "STAGE2_DENSE_FORWARD_BASELINE_EPOCH",
    "STAGE2_DENSE_FORWARD_MARKER_SCHEMA_VERSION",
    "STAGE2_DENSE_FORWARD_RUNTIME_REPORT_SCHEMA_VERSION",
    "Stage2DenseForwardArtifactEvidence",
    "Stage2DenseForwardCompileEvidence",
    "Stage2DenseForwardControlEvidence",
    "Stage2DenseForwardCoreCount",
    "Stage2DenseForwardD2DEvidence",
    "Stage2DenseForwardD2DLinkEvidence",
    "Stage2DenseForwardMemoryEvidence",
    "Stage2DenseForwardNamedCount",
    "Stage2DenseForwardOpcodeCount",
    "Stage2DenseForwardProbeEvidence",
    "Stage2DenseForwardRepeatEvidence",
    "Stage2DenseForwardRuntimeReport",
    "Stage2DenseForwardSidecarEvidence",
    "Stage2DenseForwardToolEvidence",
]
