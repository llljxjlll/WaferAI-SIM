"""Strict reviewed evidence for the Stage1a persistent-state runtime cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .artifact_manifest import RecordOpcode
from .capability import CapabilityStatus
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .program_io import ProgramIoMode, ProgramIoTargetKind
from .serde import canonical_digest


STAGE1A_ORACLE_SCHEMA_VERSION = "wafer_frontend.stage1a_oracle/v1alpha1"
STAGE1A_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.stage1a_runtime_report/v1alpha1"
)
STAGE1A_BASELINE_EPOCH = "stage1a-persistent-state-v1"


class Stage1aCase(str, Enum):
    P1 = "P1"
    K1 = "K1"
    PD1 = "PD1"


class Stage1aDecodeStartBoundary(str, Enum):
    HOST_AFTER_ALL_DONE = "host_after_all_done/v1"


_COMPLETE_OPCODE_KINDS = {
    Stage1aCase.P1: frozenset(
        {
            RecordOpcode.MATMUL,
            RecordOpcode.SRAM_BIND,
            RecordOpcode.LSU_LOAD,
            RecordOpcode.SRAM_FREE,
            RecordOpcode.SRAM_ALLOC_AT,
        }
    ),
    Stage1aCase.K1: frozenset(
        {
            RecordOpcode.MATMUL,
            RecordOpcode.ROPE_QK_EXACT,
            RecordOpcode.ATTENTION_EXACT,
            RecordOpcode.EMBEDDING_LOOKUP,
            RecordOpcode.SWIGLU,
            RecordOpcode.RESIDUAL,
            RecordOpcode.RMSNORM,
            RecordOpcode.SRAM_BIND,
            RecordOpcode.LSU_LOAD,
            RecordOpcode.LSU_STORE,
            RecordOpcode.SRAM_FREE,
            RecordOpcode.SRAM_ALLOC_AT,
        }
    ),
    Stage1aCase.PD1: frozenset(
        {
            RecordOpcode.ATTENTION_EXACT,
            RecordOpcode.DTE_SEND,
            RecordOpcode.DTE_RECV,
            RecordOpcode.LSU_LOAD,
            RecordOpcode.LSU_STORE,
            RecordOpcode.SRAM_BIND,
            RecordOpcode.SRAM_FREE,
            RecordOpcode.SRAM_ALLOC_AT,
            RecordOpcode.DTE_WAIT,
        }
    ),
}


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
class Stage1aNamedCount:
    name: str
    count: int

    def validate(self, path: str = "stage1a_named_count") -> None:
        validate_nonempty(self.name, f"{path}.name")
        validate_uint64(self.count, f"{path}.count")


@dataclass(frozen=True, slots=True)
class Stage1aCoreCount:
    runtime_core_id: int
    count: int

    def validate(self, path: str = "stage1a_core_count") -> None:
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        if self.runtime_core_id > 0xFFFF:
            raise SchemaError(
                "must fit the ProgramArtifact uint16 core id",
                path=f"{path}.runtime_core_id",
            )
        validate_uint64(self.count, f"{path}.count")


@dataclass(frozen=True, slots=True)
class Stage1aOpcodeCount:
    opcode: RecordOpcode
    count: int

    def validate(self, path: str = "stage1a_opcode_count") -> None:
        if type(self.opcode) is not RecordOpcode:
            raise SchemaError("must be a RecordOpcode", path=f"{path}.opcode")
        validate_uint64(self.count, f"{path}.count")
        if self.count == 0:
            raise SchemaError("must be positive", path=f"{path}.count")


@dataclass(frozen=True, slots=True)
class Stage1aStatePayload:
    state_ref: str
    size_bytes: int
    sha256: str

    def validate(self, path: str = "stage1a_state_payload") -> None:
        validate_nonempty(self.state_ref, f"{path}.state_ref")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if self.size_bytes == 0:
            raise SchemaError("must be positive", path=f"{path}.size_bytes")
        _validate_digest(self.sha256, f"{path}.sha256")


@dataclass(frozen=True, slots=True)
class Stage1aStatePair:
    source_state_ref: str
    destination_state_ref: str
    payload_sha256: str

    def validate(self, path: str = "stage1a_state_pair") -> None:
        validate_nonempty(self.source_state_ref, f"{path}.source_state_ref")
        validate_nonempty(
            self.destination_state_ref, f"{path}.destination_state_ref"
        )
        if self.source_state_ref == self.destination_state_ref:
            raise SchemaError(
                "source and destination states must be distinct", path=path
            )
        _validate_digest(self.payload_sha256, f"{path}.payload_sha256")


_P1_SHA = "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"
_K1_SHAS = (
    "02d449a31fbb267c8f352e9968a79e3e5fc95c1bbeaa502fd6454ebde5a4bedc",
    "9f72ea0cf49536e3c66c787f705186df9a4378083753ae9536d65b3ad7fcddc4",
    "bb391415c05e39d77ca17381d3be3f7d0cd5e5332e5a579311adaa0aa62106e9",
    "deb0e38ced1e41de6f92e70e80c418d2d356afaaa99e26f5939dbc7d3ef4772a",
)
_PD1_SHAS = (
    "60bf07c488aad18fda339df07e4fbc47b4f00be71711936f18d04d352ad01890",
    "60bf07c488aad18fda339df07e4fbc47b4f00be71711936f18d04d352ad01890",
    "fc8b64001c5fdd0f2f40fb67dae4a865a2c5bd17836676d6d5b58b7917e33717",
    "fc8b64001c5fdd0f2f40fb67dae4a865a2c5bd17836676d6d5b58b7917e33717",
)
_REQUIRED_DRAIN_NAMES = ("collective", "global", "p2p", "timing")


@dataclass(frozen=True, slots=True)
class Stage1aOracle:
    schema_version: str
    producer_pass: str
    id: str
    case: Stage1aCase
    capability_status: CapabilityStatus
    state_count: int
    state_payloads: tuple[Stage1aStatePayload, ...]
    expected_probe_payloads: tuple[Stage1aStatePayload, ...]
    state_pairs: tuple[Stage1aStatePair, ...]
    expected_dma_loads: int
    expected_dma_stores: int
    expected_opcode_counts: tuple[Stage1aOpcodeCount, ...]
    expected_hbm_read_bytes: int
    expected_hbm_write_bytes: int
    expected_d2d_bytes: int
    hbm_read_capacity_floor_cycles: int
    hbm_write_capacity_floor_cycles: int
    d2d_capacity_floor_cycles: int
    gemm_flops: int
    expected_sidecar_mode: ProgramIoMode
    expected_hbm_initialization_count: int
    expected_sram_initialization_count: int
    expected_hbm_probe_count: int
    expected_sram_probe_count: int
    expected_ack_counts: tuple[Stage1aCoreCount, ...]
    expected_done_counts: tuple[Stage1aCoreCount, ...]
    expected_drain_names: tuple[str, ...]
    timing_execution: bool
    state_transport_exact: bool
    compute_functional: bool
    model_functional: bool
    synthetic_pd: bool
    decode_start_boundary: Stage1aDecodeStartBoundary | None
    notes: tuple[str, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "Stage1aOracle":
        result = cls(
            schema_version=STAGE1A_ORACLE_SCHEMA_VERSION,
            producer_pass="stage1a_oracle",
            id=stable_artifact_id(
                "stage1a_oracle",
                semantic_key,
                schema_version=STAGE1A_ORACLE_SCHEMA_VERSION,
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

    def validate(self, path: str = "stage1a_oracle") -> None:
        if self.schema_version != STAGE1A_ORACLE_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "stage1a_oracle":
            raise SchemaError(
                "must be 'stage1a_oracle'", path=f"{path}.producer_pass"
            )
        if type(self.case) is not Stage1aCase:
            raise SchemaError("must be a Stage1aCase", path=f"{path}.case")
        if self.capability_status is not CapabilityStatus.E2E_TIMING:
            raise SchemaError(
                "Stage1a foundation cases are timing evidence only",
                path=f"{path}.capability_status",
            )
        for field_name in (
            "state_count",
            "expected_dma_loads",
            "expected_dma_stores",
            "expected_hbm_read_bytes",
            "expected_hbm_write_bytes",
            "expected_d2d_bytes",
            "hbm_read_capacity_floor_cycles",
            "hbm_write_capacity_floor_cycles",
            "d2d_capacity_floor_cycles",
            "gemm_flops",
            "expected_hbm_initialization_count",
            "expected_sram_initialization_count",
            "expected_hbm_probe_count",
            "expected_sram_probe_count",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")

        _validate_canonical(
            self.state_payloads,
            key=lambda item: item.state_ref,
            path=f"{path}.state_payloads",
        )
        for index, payload in enumerate(self.state_payloads):
            if type(payload) is not Stage1aStatePayload:
                raise SchemaError(
                    "must be a Stage1aStatePayload",
                    path=f"{path}.state_payloads[{index}]",
                )
            payload.validate(f"{path}.state_payloads[{index}]")
        if self.state_count != len(self.state_payloads):
            raise SchemaError(
                "must equal the exact state payload count",
                path=f"{path}.state_count",
            )
        _validate_canonical(
            self.expected_probe_payloads,
            key=lambda item: item.state_ref,
            path=f"{path}.expected_probe_payloads",
        )
        for index, payload in enumerate(self.expected_probe_payloads):
            if type(payload) is not Stage1aStatePayload:
                raise SchemaError(
                    "must be a Stage1aStatePayload",
                    path=f"{path}.expected_probe_payloads[{index}]",
                )
            payload.validate(f"{path}.expected_probe_payloads[{index}]")
        expected_probe_count = (
            self.expected_hbm_probe_count + self.expected_sram_probe_count
        )
        if len(self.expected_probe_payloads) != expected_probe_count:
            raise SchemaError(
                "must exactly cover every production sidecar probe",
                path=f"{path}.expected_probe_payloads",
            )
        hbm_probe_payloads = tuple(
            payload
            for payload in self.expected_probe_payloads
            if payload.state_ref.startswith("probe.hbm.")
        )
        sram_probe_payloads = tuple(
            payload
            for payload in self.expected_probe_payloads
            if payload.state_ref.startswith("probe.sram.")
        )
        if (
            len(hbm_probe_payloads) != self.expected_hbm_probe_count
            or len(sram_probe_payloads) != self.expected_sram_probe_count
            or len(hbm_probe_payloads) + len(sram_probe_payloads)
            != len(self.expected_probe_payloads)
        ):
            raise SchemaError(
                "probe witness names must exactly encode HBM/SRAM targets",
                path=f"{path}.expected_probe_payloads",
            )
        if any(
            not any(
                (state.size_bytes, state.sha256)
                == (payload.size_bytes, payload.sha256)
                for state in self.state_payloads
            )
            for payload in hbm_probe_payloads
        ):
            raise SchemaError(
                "every HBM probe witness must match a state payload",
                path=f"{path}.expected_probe_payloads",
            )
        _validate_canonical(
            self.state_pairs,
            key=lambda item: (
                item.source_state_ref,
                item.destination_state_ref,
            ),
            path=f"{path}.state_pairs",
        )
        payload_by_ref = {item.state_ref: item for item in self.state_payloads}
        for index, pair in enumerate(self.state_pairs):
            if type(pair) is not Stage1aStatePair:
                raise SchemaError(
                    "must be a Stage1aStatePair",
                    path=f"{path}.state_pairs[{index}]",
                )
            pair.validate(f"{path}.state_pairs[{index}]")
            source = payload_by_ref.get(pair.source_state_ref)
            destination = payload_by_ref.get(pair.destination_state_ref)
            if (
                source is None
                or destination is None
                or source.sha256 != pair.payload_sha256
                or destination.sha256 != pair.payload_sha256
                or source.size_bytes != destination.size_bytes
            ):
                raise SchemaError(
                    "pair must identify equal source/destination payloads",
                    path=f"{path}.state_pairs[{index}]",
                )

        _validate_canonical(
            self.expected_opcode_counts,
            key=lambda item: int(item.opcode),
            path=f"{path}.expected_opcode_counts",
        )
        opcode_counts: dict[RecordOpcode, int] = {}
        for index, item in enumerate(self.expected_opcode_counts):
            if type(item) is not Stage1aOpcodeCount:
                raise SchemaError(
                    "must be a Stage1aOpcodeCount",
                    path=f"{path}.expected_opcode_counts[{index}]",
                )
            item.validate(f"{path}.expected_opcode_counts[{index}]")
            opcode_counts[item.opcode] = item.count
        if frozenset(opcode_counts) != _COMPLETE_OPCODE_KINDS[self.case]:
            raise SchemaError(
                "must exactly cover every opcode kind in the reviewed full artifact",
                path=f"{path}.expected_opcode_counts",
            )
        if opcode_counts.get(RecordOpcode.LSU_LOAD, 0) != self.expected_dma_loads:
            raise SchemaError(
                "LSU_LOAD count must equal expected_dma_loads",
                path=f"{path}.expected_opcode_counts",
            )
        if opcode_counts.get(RecordOpcode.LSU_STORE, 0) != self.expected_dma_stores:
            raise SchemaError(
                "LSU_STORE count must equal expected_dma_stores",
                path=f"{path}.expected_opcode_counts",
            )
        if self.expected_sidecar_mode is not ProgramIoMode.TIMING:
            raise SchemaError(
                "Stage1a foundation sidecars remain timing mode",
                path=f"{path}.expected_sidecar_mode",
            )
        for field_name in ("expected_ack_counts", "expected_done_counts"):
            values = getattr(self, field_name)
            _validate_canonical(
                values,
                key=lambda item: item.runtime_core_id,
                path=f"{path}.{field_name}",
            )
            if not values:
                raise SchemaError("must be non-empty", path=f"{path}.{field_name}")
            for index, item in enumerate(values):
                if type(item) is not Stage1aCoreCount:
                    raise SchemaError(
                        "must be a Stage1aCoreCount",
                        path=f"{path}.{field_name}[{index}]",
                    )
                item.validate(f"{path}.{field_name}[{index}]")
                if item.count == 0:
                    raise SchemaError(
                        "must be positive", path=f"{path}.{field_name}[{index}].count"
                    )
        if tuple(item.runtime_core_id for item in self.expected_ack_counts) != tuple(
            item.runtime_core_id for item in self.expected_done_counts
        ):
            raise SchemaError(
                "ACK and DONE must cover the same runtime cores", path=path
            )
        if self.expected_drain_names != _REQUIRED_DRAIN_NAMES:
            raise SchemaError(
                "must contain the four reviewed canonical drain classes",
                path=f"{path}.expected_drain_names",
            )
        for index, name in enumerate(self.expected_drain_names):
            validate_nonempty(name, f"{path}.expected_drain_names[{index}]")
        for field_name in (
            "timing_execution",
            "state_transport_exact",
            "compute_functional",
            "model_functional",
            "synthetic_pd",
        ):
            _validate_bool(getattr(self, field_name), f"{path}.{field_name}")
        if (
            not self.timing_execution
            or not self.state_transport_exact
            or self.compute_functional
            or self.model_functional
        ):
            raise SchemaError(
                "foundation proof is timing plus state-transport exact only",
                path=path,
            )
        is_pd = self.case is Stage1aCase.PD1
        if self.synthetic_pd is not is_pd:
            raise SchemaError(
                "synthetic_pd must exactly identify PD1",
                path=f"{path}.synthetic_pd",
            )
        if is_pd:
            if (
                self.decode_start_boundary
                is not Stage1aDecodeStartBoundary.HOST_AFTER_ALL_DONE
                or len(self.state_pairs) != 2
            ):
                raise SchemaError(
                    "PD1 requires two pairs and the reviewed DONE boundary",
                    path=path,
                )
        elif self.decode_start_boundary is not None or self.state_pairs:
            raise SchemaError(
                "only PD1 may carry handoff pairs or a decode boundary", path=path
            )
        if self.notes != tuple(sorted(set(self.notes))) or not self.notes:
            raise SchemaError(
                "must be unique, non-empty and canonical", path=f"{path}.notes"
            )
        for index, note in enumerate(self.notes):
            validate_nonempty(note, f"{path}.notes[{index}]")

        case_values = {
            Stage1aCase.P1: {
                "scalar": (1, 1, 0, 128, 0, 0, 8, 0, 0, 128),
                "sizes": (128,),
                "hashes": (_P1_SHA,),
                "sidecar": (1, 3, 0, 1),
                "cores": ((0, 2), (0, 1)),
            },
            Stage1aCase.K1: {
                "scalar": (4, 4, 4, 128, 128, 0, 8, 8, 0, 0),
                "sizes": (32, 32, 32, 32),
                "hashes": _K1_SHAS,
                "sidecar": (0, 49, 4, 5),
                "cores": ((0, 2), (0, 1)),
            },
            Stage1aCase.PD1: {
                "scalar": (4, 2, 2, 64, 64, 64, 4, 4, 4, 0),
                "sizes": (32, 32, 32, 32),
                "hashes": _PD1_SHAS,
                "sidecar": (2, 8, 2, 2),
                "cores": (((0, 2), (16, 2)), ((0, 1), (16, 1))),
            },
        }[self.case]
        observed_scalar = (
            self.state_count,
            self.expected_dma_loads,
            self.expected_dma_stores,
            self.expected_hbm_read_bytes,
            self.expected_hbm_write_bytes,
            self.expected_d2d_bytes,
            self.hbm_read_capacity_floor_cycles,
            self.hbm_write_capacity_floor_cycles,
            self.d2d_capacity_floor_cycles,
            self.gemm_flops,
        )
        if observed_scalar != case_values["scalar"]:
            raise SchemaError(
                "numeric fields disagree with the frozen case",
                path=path,
            )
        if tuple(sorted(item.size_bytes for item in self.state_payloads)) != case_values[
            "sizes"
        ] or tuple(sorted(item.sha256 for item in self.state_payloads)) != case_values[
            "hashes"
        ]:
            raise SchemaError(
                "state payloads disagree with the frozen case", path=path
            )
        if (
            self.expected_hbm_initialization_count,
            self.expected_sram_initialization_count,
            self.expected_hbm_probe_count,
            self.expected_sram_probe_count,
        ) != case_values["sidecar"]:
            raise SchemaError(
                "sidecar counts disagree with the frozen case", path=path
            )
        ack = tuple((item.runtime_core_id, item.count) for item in self.expected_ack_counts)
        done = tuple(
            (item.runtime_core_id, item.count) for item in self.expected_done_counts
        )
        expected_cores = case_values["cores"]
        if self.case is not Stage1aCase.PD1:
            expected_cores = ((expected_cores[0],), (expected_cores[1],))
        if (ack, done) != expected_cores:
            raise SchemaError(
                "ACK/DONE counts disagree with the frozen case", path=path
            )
        expected_id = stable_artifact_id(
            "stage1a_oracle",
            self._semantic_key(),
            schema_version=STAGE1A_ORACLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id"
            )


@dataclass(frozen=True, slots=True)
class Stage1aArtifactEvidence:
    linked_manifest_id: str
    linked_manifest_digest: str
    program_artifact_sha256: str
    artifact_size_bytes: int
    record_count: int
    relocation_count: int
    opcode_counts: tuple[Stage1aOpcodeCount, ...]

    def validate(self, path: str = "stage1a_artifact_evidence") -> None:
        validate_nonempty(self.linked_manifest_id, f"{path}.linked_manifest_id")
        for field_name in (
            "linked_manifest_digest",
            "program_artifact_sha256",
        ):
            _validate_digest(getattr(self, field_name), f"{path}.{field_name}")
        for field_name in (
            "artifact_size_bytes",
            "record_count",
            "relocation_count",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
            if getattr(self, field_name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{field_name}")
        _validate_canonical(
            self.opcode_counts,
            key=lambda item: int(item.opcode),
            path=f"{path}.opcode_counts",
        )
        for index, item in enumerate(self.opcode_counts):
            if type(item) is not Stage1aOpcodeCount:
                raise SchemaError(
                    "must be a Stage1aOpcodeCount",
                    path=f"{path}.opcode_counts[{index}]",
                )
            item.validate(f"{path}.opcode_counts[{index}]")
        if self.record_count != sum(item.count for item in self.opcode_counts):
            raise SchemaError(
                "must equal the opcode count sum", path=f"{path}.record_count"
            )


@dataclass(frozen=True, slots=True)
class Stage1aSidecarEvidence:
    """Complete counts of every initialization/probe entry in the sidecar."""

    contract_id: str
    contract_digest: str
    mode: ProgramIoMode
    hbm_initialization_count: int
    sram_initialization_count: int
    hbm_probe_count: int
    sram_probe_count: int

    def validate(self, path: str = "stage1a_sidecar_evidence") -> None:
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
class Stage1aMemoryEvidence:
    runtime_core_id: int
    lsu_issued: int
    lsu_completed: int
    lsu_hbm_read_bytes: int
    lsu_hbm_write_bytes: int
    lsu_sram_read_bytes: int
    lsu_sram_write_bytes: int
    lsu_residual: int
    dte_residual: int

    def validate(self, path: str = "stage1a_memory_evidence") -> None:
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
            raise SchemaError(
                "LSU issued/completed must close exactly", path=path
            )
        if self.lsu_residual != 0 or self.dte_residual != 0:
            raise SchemaError("memory engines must drain", path=path)


@dataclass(frozen=True, slots=True)
class Stage1aProbeEvidence:
    probe_id: str
    target_kind: ProgramIoTargetKind
    length_bytes: int
    expected_sha256: str
    actual_sha256: str
    all_bytes_valid: bool
    exact_match: bool
    passed: bool

    def validate(self, path: str = "stage1a_probe_evidence") -> None:
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
class Stage1aControlEvidence:
    ack_counts: tuple[Stage1aCoreCount, ...]
    done_counts: tuple[Stage1aCoreCount, ...]
    drain_residuals: tuple[Stage1aNamedCount, ...]
    all_done_boundary_reached: bool

    def validate(self, path: str = "stage1a_control_evidence") -> None:
        for field_name in ("ack_counts", "done_counts"):
            values = getattr(self, field_name)
            _validate_canonical(
                values,
                key=lambda item: item.runtime_core_id,
                path=f"{path}.{field_name}",
            )
            for index, item in enumerate(values):
                if type(item) is not Stage1aCoreCount:
                    raise SchemaError(
                        "must be a Stage1aCoreCount",
                        path=f"{path}.{field_name}[{index}]",
                    )
                item.validate(f"{path}.{field_name}[{index}]")
        _validate_canonical(
            self.drain_residuals,
            key=lambda item: item.name,
            path=f"{path}.drain_residuals",
        )
        for index, item in enumerate(self.drain_residuals):
            if type(item) is not Stage1aNamedCount:
                raise SchemaError(
                    "must be a Stage1aNamedCount",
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
class Stage1aRepeatEvidence:
    run_index: int
    makespan_cycles: int
    marker_digest: str
    memory_digest: str
    probe_digest: str
    control_digest: str

    def validate(self, path: str = "stage1a_repeat_evidence") -> None:
        validate_uint64(self.run_index, f"{path}.run_index")
        validate_uint64(self.makespan_cycles, f"{path}.makespan_cycles")
        if self.makespan_cycles == 0:
            raise SchemaError("must be positive", path=f"{path}.makespan_cycles")
        for field_name in (
            "marker_digest",
            "memory_digest",
            "probe_digest",
            "control_digest",
        ):
            _validate_digest(getattr(self, field_name), f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class Stage1aPdWitness:
    state_pairs: tuple[Stage1aStatePair, ...]
    completion_action_ids: tuple[str, ...]
    done_runtime_core_ids: tuple[int, ...]
    decode_start_boundary: Stage1aDecodeStartBoundary
    all_completion_actions_before_boundary: bool

    def validate(self, path: str = "stage1a_pd_witness") -> None:
        _validate_canonical(
            self.state_pairs,
            key=lambda item: (item.source_state_ref, item.destination_state_ref),
            path=f"{path}.state_pairs",
        )
        for index, pair in enumerate(self.state_pairs):
            if type(pair) is not Stage1aStatePair:
                raise SchemaError(
                    "must be a Stage1aStatePair",
                    path=f"{path}.state_pairs[{index}]",
                )
            pair.validate(f"{path}.state_pairs[{index}]")
        if len(self.state_pairs) != 2:
            raise SchemaError("must contain two state pairs", path=f"{path}.state_pairs")
        if self.completion_action_ids != tuple(
            sorted(set(self.completion_action_ids))
        ) or len(self.completion_action_ids) != 2:
            raise SchemaError(
                "must contain two canonical completion actions",
                path=f"{path}.completion_action_ids",
            )
        for index, action_id in enumerate(self.completion_action_ids):
            validate_nonempty(action_id, f"{path}.completion_action_ids[{index}]")
        if self.done_runtime_core_ids != tuple(
            sorted(set(self.done_runtime_core_ids))
        ) or not self.done_runtime_core_ids:
            raise SchemaError(
                "must be unique, non-empty and canonical",
                path=f"{path}.done_runtime_core_ids",
            )
        for index, core_id in enumerate(self.done_runtime_core_ids):
            validate_uint64(core_id, f"{path}.done_runtime_core_ids[{index}]")
            if core_id > 0xFFFF:
                raise SchemaError(
                    "must fit the ProgramArtifact uint16 core id",
                    path=f"{path}.done_runtime_core_ids[{index}]",
                )
        if (
            self.decode_start_boundary
            is not Stage1aDecodeStartBoundary.HOST_AFTER_ALL_DONE
        ):
            raise SchemaError(
                "must use the reviewed host-after-DONE boundary",
                path=f"{path}.decode_start_boundary",
            )
        _validate_bool(
            self.all_completion_actions_before_boundary,
            f"{path}.all_completion_actions_before_boundary",
        )
        if not self.all_completion_actions_before_boundary:
            raise SchemaError("handoff must dominate the boundary", path=path)


@dataclass(frozen=True, slots=True)
class Stage1aRuntimeReport:
    schema_version: str
    producer_pass: str
    id: str
    baseline_epoch: str
    case: Stage1aCase
    capability_status: CapabilityStatus
    oracle_id: str
    oracle_digest: str
    hardware_digest: str
    simulation_digest: str
    mapping_digest: str
    artifact: Stage1aArtifactEvidence
    sidecar: Stage1aSidecarEvidence
    memory: tuple[Stage1aMemoryEvidence, ...]
    probes: tuple[Stage1aProbeEvidence, ...]
    control: Stage1aControlEvidence
    observed_hbm_read_bytes: int
    observed_hbm_write_bytes: int
    observed_d2d_bytes: int
    repeat_count: int
    makespan_cycles: int
    repeats: tuple[Stage1aRepeatEvidence, ...]
    timing_execution: bool
    state_transport_exact: bool
    compute_functional: bool
    model_functional: bool
    synthetic_pd: bool
    pd_witness: Stage1aPdWitness | None

    @classmethod
    def create(cls, **semantic_key: object) -> "Stage1aRuntimeReport":
        result = cls(
            schema_version=STAGE1A_RUNTIME_REPORT_SCHEMA_VERSION,
            producer_pass="stage1a_runtime",
            id=stable_artifact_id(
                "stage1a_runtime_report",
                semantic_key,
                schema_version=STAGE1A_RUNTIME_REPORT_SCHEMA_VERSION,
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

    def validate(self, path: str = "stage1a_runtime_report") -> None:
        if self.schema_version != STAGE1A_RUNTIME_REPORT_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "stage1a_runtime":
            raise SchemaError(
                "must be 'stage1a_runtime'", path=f"{path}.producer_pass"
            )
        if self.baseline_epoch != STAGE1A_BASELINE_EPOCH:
            raise SchemaError(
                f"must be {STAGE1A_BASELINE_EPOCH!r}",
                path=f"{path}.baseline_epoch",
            )
        if type(self.case) is not Stage1aCase:
            raise SchemaError("must be a Stage1aCase", path=f"{path}.case")
        if self.capability_status is not CapabilityStatus.E2E_TIMING:
            raise SchemaError(
                "Stage1a foundation reports are timing evidence only",
                path=f"{path}.capability_status",
            )
        validate_nonempty(self.oracle_id, f"{path}.oracle_id")
        for field_name in (
            "oracle_digest",
            "hardware_digest",
            "simulation_digest",
            "mapping_digest",
        ):
            _validate_digest(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.artifact) is not Stage1aArtifactEvidence:
            raise SchemaError(
                "must be Stage1aArtifactEvidence", path=f"{path}.artifact"
            )
        self.artifact.validate(f"{path}.artifact")
        if type(self.sidecar) is not Stage1aSidecarEvidence:
            raise SchemaError(
                "must be Stage1aSidecarEvidence", path=f"{path}.sidecar"
            )
        self.sidecar.validate(f"{path}.sidecar")
        _validate_canonical(
            self.memory,
            key=lambda item: item.runtime_core_id,
            path=f"{path}.memory",
        )
        if not self.memory:
            raise SchemaError("must be non-empty", path=f"{path}.memory")
        for index, item in enumerate(self.memory):
            if type(item) is not Stage1aMemoryEvidence:
                raise SchemaError(
                    "must be a Stage1aMemoryEvidence",
                    path=f"{path}.memory[{index}]",
                )
            item.validate(f"{path}.memory[{index}]")
        for field_name in (
            "observed_hbm_read_bytes",
            "observed_hbm_write_bytes",
            "observed_d2d_bytes",
            "repeat_count",
            "makespan_cycles",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.observed_hbm_read_bytes != sum(
            item.lsu_hbm_read_bytes for item in self.memory
        ) or self.observed_hbm_write_bytes != sum(
            item.lsu_hbm_write_bytes for item in self.memory
        ):
            raise SchemaError(
                "observed HBM bytes must equal the exact per-core sum", path=path
            )
        _validate_canonical(
            self.probes,
            key=lambda item: item.probe_id,
            path=f"{path}.probes",
        )
        for index, probe in enumerate(self.probes):
            if type(probe) is not Stage1aProbeEvidence:
                raise SchemaError(
                    "must be a Stage1aProbeEvidence",
                    path=f"{path}.probes[{index}]",
                )
            probe.validate(f"{path}.probes[{index}]")
        if type(self.control) is not Stage1aControlEvidence:
            raise SchemaError(
                "must be Stage1aControlEvidence", path=f"{path}.control"
            )
        self.control.validate(f"{path}.control")
        if self.repeat_count != 2 or len(self.repeats) != 2:
            raise SchemaError(
                "reviewed runtime evidence requires exactly two repeats",
                path=f"{path}.repeats",
            )
        if self.makespan_cycles == 0:
            raise SchemaError("must be positive", path=f"{path}.makespan_cycles")
        _validate_canonical(
            self.repeats,
            key=lambda item: item.run_index,
            path=f"{path}.repeats",
        )
        expected_memory_digest = canonical_digest(self.memory)
        expected_probe_digest = canonical_digest(self.probes)
        expected_control_digest = canonical_digest(self.control)
        marker_digest: str | None = None
        for index, repeat in enumerate(self.repeats):
            if type(repeat) is not Stage1aRepeatEvidence:
                raise SchemaError(
                    "must be a Stage1aRepeatEvidence",
                    path=f"{path}.repeats[{index}]",
                )
            repeat.validate(f"{path}.repeats[{index}]")
            if (
                repeat.run_index != index
                or repeat.makespan_cycles != self.makespan_cycles
                or repeat.memory_digest != expected_memory_digest
                or repeat.probe_digest != expected_probe_digest
                or repeat.control_digest != expected_control_digest
            ):
                raise SchemaError(
                    "repeat does not reproduce the canonical report evidence",
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
            "state_transport_exact",
            "compute_functional",
            "model_functional",
            "synthetic_pd",
        ):
            _validate_bool(getattr(self, field_name), f"{path}.{field_name}")
        if (
            not self.timing_execution
            or not self.state_transport_exact
            or self.compute_functional
            or self.model_functional
        ):
            raise SchemaError(
                "report may claim timing plus state-transport exact only", path=path
            )
        is_pd = self.case is Stage1aCase.PD1
        if self.synthetic_pd is not is_pd:
            raise SchemaError(
                "synthetic_pd must exactly identify PD1",
                path=f"{path}.synthetic_pd",
            )
        if is_pd:
            if type(self.pd_witness) is not Stage1aPdWitness:
                raise SchemaError(
                    "PD1 requires a Stage1aPdWitness", path=f"{path}.pd_witness"
                )
            self.pd_witness.validate(f"{path}.pd_witness")
        elif self.pd_witness is not None:
            raise SchemaError("only PD1 may carry a witness", path=f"{path}.pd_witness")
        expected_id = stable_artifact_id(
            "stage1a_runtime_report",
            self._semantic_key(),
            schema_version=STAGE1A_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id"
            )

    def validate_against(
        self,
        oracle: Stage1aOracle,
        path: str = "stage1a_runtime_report",
    ) -> None:
        self.validate(path)
        if type(oracle) is not Stage1aOracle:
            raise SchemaError("must be a Stage1aOracle", path=f"{path}.oracle")
        oracle.validate("stage1a_oracle")
        if (
            self.oracle_id != oracle.id
            or self.oracle_digest != canonical_digest(oracle)
            or self.case is not oracle.case
            or self.capability_status is not oracle.capability_status
        ):
            raise SchemaError("does not identify the supplied oracle", path=path)
        if self.artifact.opcode_counts != oracle.expected_opcode_counts:
            raise SchemaError("artifact opcode counts disagree with oracle", path=path)
        if (
            self.sidecar.mode,
            self.sidecar.hbm_initialization_count,
            self.sidecar.sram_initialization_count,
            self.sidecar.hbm_probe_count,
            self.sidecar.sram_probe_count,
        ) != (
            oracle.expected_sidecar_mode,
            oracle.expected_hbm_initialization_count,
            oracle.expected_sram_initialization_count,
            oracle.expected_hbm_probe_count,
            oracle.expected_sram_probe_count,
        ):
            raise SchemaError("sidecar evidence disagrees with oracle", path=path)
        if (
            self.observed_hbm_read_bytes,
            self.observed_hbm_write_bytes,
            self.observed_d2d_bytes,
        ) != (
            oracle.expected_hbm_read_bytes,
            oracle.expected_hbm_write_bytes,
            oracle.expected_d2d_bytes,
        ):
            raise SchemaError("runtime traffic disagrees with oracle", path=path)
        if tuple(item.runtime_core_id for item in self.memory) != tuple(
            item.runtime_core_id for item in oracle.expected_done_counts
        ):
            raise SchemaError("memory evidence has the wrong core closure", path=path)
        if (
            self.control.ack_counts != oracle.expected_ack_counts
            or self.control.done_counts != oracle.expected_done_counts
            or tuple(item.name for item in self.control.drain_residuals)
            != oracle.expected_drain_names
        ):
            raise SchemaError("control evidence disagrees with oracle", path=path)
        if (
            sum(
                probe.target_kind is ProgramIoTargetKind.HBM
                for probe in self.probes
            )
            != oracle.expected_hbm_probe_count
            or sum(
                probe.target_kind is ProgramIoTargetKind.SRAM
                for probe in self.probes
            )
            != oracle.expected_sram_probe_count
        ):
            raise SchemaError("probe target counts disagree with oracle", path=path)
        expected_probe_payloads = {
            payload.state_ref: payload
            for payload in oracle.expected_probe_payloads
        }
        if set(expected_probe_payloads) != {
            probe.probe_id for probe in self.probes
        }:
            raise SchemaError(
                "probes do not exactly cover Oracle payload witnesses", path=path
            )
        state_payloads = {
            (payload.size_bytes, payload.sha256)
            for payload in oracle.state_payloads
        }
        for probe in self.probes:
            witness = expected_probe_payloads[probe.probe_id]
            if (
                probe.length_bytes,
                probe.expected_sha256,
                probe.actual_sha256,
            ) != (
                witness.size_bytes,
                witness.sha256,
                witness.sha256,
            ):
                raise SchemaError(
                    "probe payload disagrees with its Oracle witness", path=path
                )
            if probe.target_kind is ProgramIoTargetKind.HBM:
                if (
                    not probe.probe_id.startswith("probe.hbm.")
                    or (probe.length_bytes, probe.expected_sha256)
                    not in state_payloads
                ):
                    raise SchemaError(
                        "HBM probe requires both state and probe witnesses",
                        path=path,
                    )
            elif not probe.probe_id.startswith("probe.sram."):
                raise SchemaError(
                    "SRAM probe requires its exact SRAM Oracle witness", path=path
                )
        if (
            self.timing_execution,
            self.state_transport_exact,
            self.compute_functional,
            self.model_functional,
            self.synthetic_pd,
        ) != (
            oracle.timing_execution,
            oracle.state_transport_exact,
            oracle.compute_functional,
            oracle.model_functional,
            oracle.synthetic_pd,
        ):
            raise SchemaError("proof claims disagree with oracle", path=path)
        if oracle.case is Stage1aCase.PD1:
            assert self.pd_witness is not None
            if (
                self.pd_witness.state_pairs != oracle.state_pairs
                or self.pd_witness.decode_start_boundary
                is not oracle.decode_start_boundary
                or self.pd_witness.done_runtime_core_ids
                != tuple(item.runtime_core_id for item in oracle.expected_done_counts)
            ):
                raise SchemaError("PD witness disagrees with oracle", path=path)


__all__ = [
    "STAGE1A_BASELINE_EPOCH",
    "STAGE1A_ORACLE_SCHEMA_VERSION",
    "STAGE1A_RUNTIME_REPORT_SCHEMA_VERSION",
    "Stage1aArtifactEvidence",
    "Stage1aCase",
    "Stage1aControlEvidence",
    "Stage1aCoreCount",
    "Stage1aDecodeStartBoundary",
    "Stage1aMemoryEvidence",
    "Stage1aNamedCount",
    "Stage1aOpcodeCount",
    "Stage1aOracle",
    "Stage1aPdWitness",
    "Stage1aProbeEvidence",
    "Stage1aRepeatEvidence",
    "Stage1aRuntimeReport",
    "Stage1aSidecarEvidence",
    "Stage1aStatePair",
    "Stage1aStatePayload",
]
