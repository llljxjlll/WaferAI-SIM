"""Strict standard lower/link and ProgramIO carriers for flexible MoE v2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .artifact_manifest import RecordOpcode
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .flexible_moe import (
    FlexibleMoeExecutablePlan,
    FlexibleMoeMode,
    FlexibleMoeSpec,
    MoeRectActionKind,
)
from .serde import canonical_digest


FLEXIBLE_MOE_STANDARD_LOWERING_PLAN_SCHEMA_VERSION = (
    "wafer_frontend.flexible_moe_standard_lowering_plan/v1alpha1"
)
FLEXIBLE_MOE_STANDARD_PROGRAM_IO_PLAN_SCHEMA_VERSION = (
    "wafer_frontend.flexible_moe_standard_program_io_plan/v1alpha1"
)


class FlexibleMoeStateAccess(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    SCRATCH = "scratch"


class FlexibleMoeIoRole(str, Enum):
    STATE_INITIALIZATION = "state_initialization"
    OUTPUT_PROBE = "output_probe"
    UPDATED_STATE_PROBE = "updated_state_probe"


_OPCODE_BY_ACTION = {
    MoeRectActionKind.STATE_LOAD: RecordOpcode.LSU_LOAD,
    MoeRectActionKind.GATE: RecordOpcode.MATMUL,
    MoeRectActionKind.PACK: RecordOpcode.SRAM_BIND,
    MoeRectActionKind.SEND: RecordOpcode.DTE_SEND,
    MoeRectActionKind.RECV: RecordOpcode.DTE_RECV,
    MoeRectActionKind.WAIT: RecordOpcode.DTE_WAIT,
    MoeRectActionKind.EXPERT_FORWARD: RecordOpcode.MATMUL,
    MoeRectActionKind.WEIGHTED_COMBINE: RecordOpcode.LOCAL_REDUCE,
    MoeRectActionKind.EXPERT_DGRAD: RecordOpcode.MATMUL,
    MoeRectActionKind.EXPERT_WGRAD: RecordOpcode.MATMUL,
    MoeRectActionKind.COMBINE_BACKWARD: RecordOpcode.LOCAL_REDUCE,
    MoeRectActionKind.GATE_WGRAD: RecordOpcode.MATMUL,
    MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE: RecordOpcode.LOCAL_REDUCE,
    MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE: RecordOpcode.DTE_ISSUE,
    MoeRectActionKind.EXPERT_SGD: RecordOpcode.SGD_UPDATE,
    MoeRectActionKind.GATE_SGD: RecordOpcode.SGD_UPDATE,
    MoeRectActionKind.STATE_STORE: RecordOpcode.LSU_STORE,
}


@dataclass(frozen=True, slots=True)
class FlexibleMoeStandardRecord:
    id: str
    source_action_ref: str
    runtime_core_id: int
    opcode: RecordOpcode
    dependency_record_refs: tuple[str, ...]
    flow_ref: str | None
    peer_runtime_core_id: int | None
    transport_tag: int | None
    wave_index: int | None
    logical_bytes: int
    flops: int
    state_refs: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleMoeStandardRecord":
        return cls(
            stable_artifact_id(
                "flexible_moe_standard_record", semantic,
                schema_version=FLEXIBLE_MOE_STANDARD_LOWERING_PLAN_SCHEMA_VERSION,
            ),
            **semantic,
        )

    def validate(self, path: str = "flexible_moe_standard_record") -> None:
        validate_nonempty(self.source_action_ref, f"{path}.source_action_ref")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        if self.runtime_core_id > 0xFFFF:
            raise SchemaError("runtime core must fit U16", path=f"{path}.runtime_core_id")
        if type(self.opcode) is not RecordOpcode:
            raise SchemaError("must use a public RecordOpcode", path=f"{path}.opcode")
        if len(set(self.dependency_record_refs)) != len(self.dependency_record_refs):
            raise SchemaError("duplicate dependency records", path=f"{path}.dependency_record_refs")
        for index, ref in enumerate(self.dependency_record_refs):
            validate_nonempty(ref, f"{path}.dependency_record_refs[{index}]")
        if self.flow_ref is not None:
            validate_nonempty(self.flow_ref, f"{path}.flow_ref")
        for name in ("peer_runtime_core_id", "transport_tag", "wave_index"):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")
        if self.transport_tag is not None and self.transport_tag > 0xFFFF:
            raise SchemaError("transport tag must fit U16", path=f"{path}.transport_tag")
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        validate_uint64(self.flops, f"{path}.flops")
        if len(set(self.state_refs)) != len(self.state_refs):
            raise SchemaError("duplicate state refs", path=f"{path}.state_refs")


@dataclass(frozen=True, slots=True)
class FlexibleMoeCoreStream:
    runtime_core_id: int
    records: tuple[FlexibleMoeStandardRecord, ...]

    def validate(self, path: str = "flexible_moe_core_stream") -> None:
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        if any(item.runtime_core_id != self.runtime_core_id for item in self.records):
            raise SchemaError("record belongs to another core", path=f"{path}.records")
        for index, item in enumerate(self.records):
            item.validate(f"{path}.records[{index}]")


@dataclass(frozen=True, slots=True)
class FlexibleMoeStandardStateAbi:
    state_ref: str
    runtime_core_id: int
    address: int
    size_bytes: int
    dtype: DType
    access: FlexibleMoeStateAccess
    persistent: bool

    def validate(self, path: str = "flexible_moe_state_abi") -> None:
        validate_nonempty(self.state_ref, f"{path}.state_ref")
        for name in ("runtime_core_id", "address", "size_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.address % 64 or self.size_bytes == 0:
            raise SchemaError("state ABI requires aligned nonempty storage", path=path)
        if type(self.dtype) is not DType or type(self.access) is not FlexibleMoeStateAccess:
            raise SchemaError("state ABI types are invalid", path=path)
        if type(self.persistent) is not bool:
            raise SchemaError("persistent must be bool", path=f"{path}.persistent")


@dataclass(frozen=True, slots=True)
class FlexibleMoeEndpointAbi:
    flow_ref: str
    transport_tag: int
    stage: str
    wave_index: int
    source_runtime_core_id: int
    destination_runtime_core_id: int
    logical_bytes: int
    die_path: tuple[int, ...]
    send_record_ref: str
    recv_record_ref: str
    wait_record_ref: str

    def validate(self, path: str = "flexible_moe_endpoint_abi") -> None:
        for name in ("flow_ref", "stage", "send_record_ref", "recv_record_ref", "wait_record_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in (
            "transport_tag", "wave_index", "source_runtime_core_id",
            "destination_runtime_core_id", "logical_bytes",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if not 0 < self.transport_tag <= 0xFFFF or self.logical_bytes == 0:
            raise SchemaError("endpoint tag/bytes are invalid", path=path)


@dataclass(frozen=True, slots=True)
class FlexibleMoeStandardLoweringPlan:
    schema_version: str
    producer_pass: str
    id: str
    source_plan_id: str
    source_plan_digest: str
    source_spec_id: str
    source_spec_digest: str
    core_streams: tuple[FlexibleMoeCoreStream, ...]
    state_abi: tuple[FlexibleMoeStandardStateAbi, ...]
    endpoint_abi: tuple[FlexibleMoeEndpointAbi, ...]
    record_count: int
    max_sessions_per_rank_wave: int
    standard_mapping_verified: bool
    lower_link_verified: bool
    runtime_verified: bool

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleMoeStandardLoweringPlan":
        return cls(
            FLEXIBLE_MOE_STANDARD_LOWERING_PLAN_SCHEMA_VERSION,
            "plan_flexible_moe_standard_mapping",
            stable_artifact_id(
                "flexible_moe_standard_lowering_plan", semantic,
                schema_version=FLEXIBLE_MOE_STANDARD_LOWERING_PLAN_SCHEMA_VERSION,
            ),
            **semantic,
        )

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self)

    def validate_against(
        self,
        plan: FlexibleMoeExecutablePlan,
        spec: FlexibleMoeSpec,
        path: str = "flexible_moe_standard_lowering_plan",
    ) -> None:
        plan.validate_against(spec, f"{path}.plan")
        if self.schema_version != FLEXIBLE_MOE_STANDARD_LOWERING_PLAN_SCHEMA_VERSION or self.producer_pass != "plan_flexible_moe_standard_mapping":
            raise SchemaError("unsupported standard lowering plan", path=path)
        if (
            self.source_plan_id,
            self.source_plan_digest,
            self.source_spec_id,
            self.source_spec_digest,
        ) != (plan.id, canonical_digest(plan), spec.id, spec.digest):
            raise SchemaError("standard manifest provenance drifted", path=path)
        if tuple(item.runtime_core_id for item in self.core_streams) != tuple(range(spec.mesh.rank_count)):
            raise SchemaError("streams must cover every rank exactly", path=f"{path}.core_streams")
        records = tuple(item for stream in self.core_streams for item in stream.records)
        for index, stream in enumerate(self.core_streams):
            stream.validate(f"{path}.core_streams[{index}]")
        if len(records) != self.record_count or self.record_count != len(plan.actions):
            raise SchemaError("one-record-per-action quotient drifted", path=f"{path}.record_count")
        if len({item.id for item in records}) != len(records):
            raise SchemaError("duplicate records", path=f"{path}.core_streams")
        by_action = {item.source_action_ref: item for item in records}
        if set(by_action) != {item.id for item in plan.actions}:
            raise SchemaError("records do not cover plan actions exactly", path=f"{path}.core_streams")
        for action in plan.actions:
            record = by_action[action.id]
            if record.opcode is not _OPCODE_BY_ACTION[action.kind]:
                raise SchemaError("action opcode mapping drifted", path=f"{path}.core_streams")
            expected_deps = tuple(by_action[ref].id for ref in action.deps)
            if record.dependency_record_refs != expected_deps:
                raise SchemaError("record dependencies do not preserve plan DAG", path=f"{path}.core_streams")
            if record.state_refs != action.state_refs:
                raise SchemaError("record state refs drifted", path=f"{path}.core_streams")
        state_refs = {item.state_ref for item in self.state_abi}
        if state_refs != {item.id for item in plan.state_bindings}:
            raise SchemaError("state ABI does not cover plan state", path=f"{path}.state_abi")
        ranges: dict[int, list[tuple[int, int]]] = {}
        for index, item in enumerate(self.state_abi):
            item.validate(f"{path}.state_abi[{index}]")
            ranges.setdefault(item.runtime_core_id, []).append((item.address, item.address + item.size_bytes))
        for core_ranges in ranges.values():
            ordered = sorted(core_ranges)
            if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
                raise SchemaError("state ABI ranges overlap", path=f"{path}.state_abi")
        endpoints = {item.flow_ref: item for item in self.endpoint_abi}
        if set(endpoints) != {item.id for item in plan.flows}:
            raise SchemaError("endpoint ABI does not cover flows", path=f"{path}.endpoint_abi")
        flow_by_id = {item.id: item for item in plan.flows}
        records_by_flow: dict[str, list[FlexibleMoeStandardRecord]] = {}
        for record in records:
            if record.flow_ref is not None:
                records_by_flow.setdefault(record.flow_ref, []).append(record)
        for index, endpoint in enumerate(self.endpoint_abi):
            endpoint.validate(f"{path}.endpoint_abi[{index}]")
            flow = flow_by_id[endpoint.flow_ref]
            opcode_order = {
                RecordOpcode.DTE_SEND: 0,
                RecordOpcode.DTE_RECV: 1,
                RecordOpcode.DTE_WAIT: 2,
            }
            flow_records = tuple(sorted(
                records_by_flow.get(flow.id, ()),
                key=lambda item: opcode_order.get(item.opcode, 99),
            ))
            if (
                tuple(item.opcode for item in flow_records)
                != (RecordOpcode.DTE_SEND, RecordOpcode.DTE_RECV, RecordOpcode.DTE_WAIT)
                or (endpoint.send_record_ref, endpoint.recv_record_ref, endpoint.wait_record_ref)
                != tuple(item.id for item in flow_records)
                or endpoint.die_path != flow.die_path
            ):
                raise SchemaError("SEND/RECV/WAIT endpoint closure drifted", path=f"{path}.endpoint_abi[{index}]")
        if self.max_sessions_per_rank_wave != plan.max_sessions_per_rank_wave or self.max_sessions_per_rank_wave > 3:
            raise SchemaError("session evidence drifted", path=f"{path}.max_sessions_per_rank_wave")
        if self.standard_mapping_verified is not True or self.lower_link_verified is not False or self.runtime_verified is not False:
            raise SchemaError("mapping plan cannot claim lower/link or runtime", path=path)
        expected = stable_artifact_id(
            "flexible_moe_standard_lowering_plan", self._semantic(),
            schema_version=FLEXIBLE_MOE_STANDARD_LOWERING_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleMoeProgramIoEntry:
    role: FlexibleMoeIoRole
    runtime_core_id: int
    size_bytes: int
    state_ref: str | None
    terminal_action_ref: str | None
    address: int | None
    zero_fill: bool

    def validate(self, path: str = "flexible_moe_program_io_entry") -> None:
        if type(self.role) is not FlexibleMoeIoRole:
            raise SchemaError("must use typed IO role", path=f"{path}.role")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if self.size_bytes == 0 or (self.state_ref is None) == (self.terminal_action_ref is None):
            raise SchemaError("IO entry must target exactly one nonempty state/action", path=path)
        if self.address is not None:
            validate_uint64(self.address, f"{path}.address")
        if type(self.zero_fill) is not bool:
            raise SchemaError("zero_fill must be bool", path=f"{path}.zero_fill")


@dataclass(frozen=True, slots=True)
class FlexibleMoeStandardProgramIoPlan:
    schema_version: str
    producer_pass: str
    id: str
    source_manifest_id: str
    source_manifest_digest: str
    program_artifact_sha256: str
    entries: tuple[FlexibleMoeProgramIoEntry, ...]
    timing_execution: bool
    runtime_verified: bool

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleMoeStandardProgramIoPlan":
        return cls(
            FLEXIBLE_MOE_STANDARD_PROGRAM_IO_PLAN_SCHEMA_VERSION,
            "build_flexible_moe_standard_program_io_plan",
            stable_artifact_id(
                "flexible_moe_standard_program_io_plan", semantic,
                schema_version=FLEXIBLE_MOE_STANDARD_PROGRAM_IO_PLAN_SCHEMA_VERSION,
            ),
            **semantic,
        )

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate_against(
        self,
        manifest: FlexibleMoeStandardLoweringPlan,
        plan: FlexibleMoeExecutablePlan,
        spec: FlexibleMoeSpec,
        path: str = "flexible_moe_standard_program_io_plan",
    ) -> None:
        manifest.validate_against(plan, spec, f"{path}.manifest")
        if self.schema_version != FLEXIBLE_MOE_STANDARD_PROGRAM_IO_PLAN_SCHEMA_VERSION or self.producer_pass != "build_flexible_moe_standard_program_io_plan":
            raise SchemaError("unsupported ProgramIO plan", path=path)
        if (self.source_manifest_id, self.source_manifest_digest) != (manifest.id, manifest.digest):
            raise SchemaError("ProgramIO provenance drifted", path=path)
        if len(self.program_artifact_sha256) != 64 or any(item not in "0123456789abcdef" for item in self.program_artifact_sha256):
            raise SchemaError("artifact digest must be canonical SHA256", path=f"{path}.program_artifact_sha256")
        for index, item in enumerate(self.entries):
            item.validate(f"{path}.entries[{index}]")
        initial = {item.state_ref for item in self.entries if item.role is FlexibleMoeIoRole.STATE_INITIALIZATION}
        persistent = {item.state_ref for item in manifest.state_abi if item.persistent}
        if initial != persistent:
            raise SchemaError("ProgramIO must initialize every persistent state", path=f"{path}.entries")
        if spec.mode is FlexibleMoeMode.INFERENCE:
            probes = {item.terminal_action_ref for item in self.entries if item.role is FlexibleMoeIoRole.OUTPUT_PROBE}
            if probes != set(plan.terminal_action_refs):
                raise SchemaError("inference output probes are incomplete", path=f"{path}.entries")
        else:
            updated = {item.state_ref for item in self.entries if item.role is FlexibleMoeIoRole.UPDATED_STATE_PROBE}
            if updated != persistent:
                raise SchemaError("training must probe all updated parameters", path=f"{path}.entries")
        if self.timing_execution is not True or self.runtime_verified is not False:
            raise SchemaError("ProgramIO is timing-only and not runtime evidence", path=path)
        expected = stable_artifact_id(
            "flexible_moe_standard_program_io_plan", self._semantic(),
            schema_version=FLEXIBLE_MOE_STANDARD_PROGRAM_IO_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


__all__ = [
    "FLEXIBLE_MOE_STANDARD_LOWERING_PLAN_SCHEMA_VERSION",
    "FLEXIBLE_MOE_STANDARD_PROGRAM_IO_PLAN_SCHEMA_VERSION",
    "FlexibleMoeCoreStream",
    "FlexibleMoeEndpointAbi",
    "FlexibleMoeIoRole",
    "FlexibleMoeStandardLoweringPlan",
    "FlexibleMoeProgramIoEntry",
    "FlexibleMoeStandardProgramIoPlan",
    "FlexibleMoeStandardRecord",
    "FlexibleMoeStandardStateAbi",
    "FlexibleMoeStateAccess",
]
