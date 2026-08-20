"""Strict runtime evidence for Stage 4 prefill/decode timing cases."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import LinkedProgramManifest, RegionManifest
from .capability import CapabilityStatus
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .program_io import ProgramIoMode
from .s1_naive_evidence import S1_N_BASELINE_EPOCH, S1NaivePolicyEvidence
from .serde import canonical_digest
from .stage4_pd import (
    Stage4KvReshardKind,
    Stage4PdMode,
    Stage4PdOracle,
    Stage4PdPlan,
)


STAGE4_PD_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.stage4_pd_runtime_report/v1alpha2"
)
STAGE4_PD_BASELINE_EPOCH = S1_N_BASELINE_EPOCH

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
class Stage4PdNamedDigest:
    name: str
    digest: str

    def validate(self, path: str = "stage4_pd_named_digest") -> None:
        validate_nonempty(self.name, f"{path}.name")
        _validate_digest(self.digest, f"{path}.digest")


@dataclass(frozen=True, slots=True)
class Stage4PdArtifactEvidence:
    linked_manifest_id: str
    linked_manifest_digest: str
    program_artifact_sha256: str
    artifact_size_bytes: int
    action_count: int
    fragment_count: int
    record_count: int
    runtime_relocation_count: int
    address_relocation_count: int
    relocation_count: int
    address_operand_binding_count: int
    state_operand_binding_count: int

    def validate(self, path: str = "stage4_pd_artifact_evidence") -> None:
        validate_nonempty(
            self.linked_manifest_id, f"{path}.linked_manifest_id"
        )
        _validate_digest(
            self.linked_manifest_digest, f"{path}.linked_manifest_digest"
        )
        _validate_digest(
            self.program_artifact_sha256, f"{path}.program_artifact_sha256"
        )
        for field_name in self.__dataclass_fields__:
            if field_name in (
                "linked_manifest_id",
                "linked_manifest_digest",
                "program_artifact_sha256",
            ):
                continue
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        for field_name in (
            "artifact_size_bytes",
            "action_count",
            "fragment_count",
            "record_count",
        ):
            if getattr(self, field_name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{field_name}")
        if self.relocation_count != (
            self.runtime_relocation_count + self.address_relocation_count
        ):
            raise SchemaError(
                "must equal runtime plus address relocations",
                path=f"{path}.relocation_count",
            )


@dataclass(frozen=True, slots=True)
class Stage4PdProgramIoEvidence:
    contract_id: str
    contract_digest: str
    mode: ProgramIoMode
    hbm_initialization_count: int
    sram_initialization_count: int
    hbm_probe_count: int
    sram_probe_count: int
    all_probes_passed: bool

    def validate(self, path: str = "stage4_pd_program_io_evidence") -> None:
        validate_nonempty(self.contract_id, f"{path}.contract_id")
        _validate_digest(self.contract_digest, f"{path}.contract_digest")
        if self.mode is not ProgramIoMode.TIMING:
            raise SchemaError(
                "Stage 4 evidence requires timing ProgramIo",
                path=f"{path}.mode",
            )
        for field_name in (
            "hbm_initialization_count",
            "sram_initialization_count",
            "hbm_probe_count",
            "sram_probe_count",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        _validate_bool(self.all_probes_passed, f"{path}.all_probes_passed")
        if not self.all_probes_passed:
            raise SchemaError("all ProgramIo probes must pass", path=path)
        if self.hbm_initialization_count + self.sram_initialization_count == 0:
            raise SchemaError("must initialize runtime memory", path=path)
        if self.hbm_probe_count + self.sram_probe_count == 0:
            raise SchemaError("must contain an exact output probe", path=path)


@dataclass(frozen=True, slots=True)
class Stage4PdMemoryEvidence:
    runtime_core_id: int
    lsu_issued: int
    lsu_completed: int
    lsu_hbm_read_bytes: int
    lsu_hbm_write_bytes: int
    lsu_sram_read_bytes: int
    lsu_sram_write_bytes: int
    lsu_residual: int
    dte_residual: int

    def validate(self, path: str = "stage4_pd_memory_evidence") -> None:
        for field_name in self.__dataclass_fields__:
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.runtime_core_id > 0xFFFF:
            raise SchemaError(
                "must fit the ProgramArtifact uint16 core id",
                path=f"{path}.runtime_core_id",
            )
        if self.lsu_issued != self.lsu_completed:
            raise SchemaError("LSU issued/completed must close exactly", path=path)
        if self.lsu_residual != 0 or self.dte_residual != 0:
            raise SchemaError("memory engines must drain", path=path)


@dataclass(frozen=True, slots=True)
class Stage4PdCoreCount:
    runtime_core_id: int
    count: int

    def validate(self, path: str = "stage4_pd_core_count") -> None:
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
class Stage4PdNamedCount:
    name: str
    count: int

    def validate(self, path: str = "stage4_pd_named_count") -> None:
        validate_nonempty(self.name, f"{path}.name")
        validate_uint64(self.count, f"{path}.count")


@dataclass(frozen=True, slots=True)
class Stage4PdControlEvidence:
    ack_counts: tuple[Stage4PdCoreCount, ...]
    done_counts: tuple[Stage4PdCoreCount, ...]
    drain_residuals: tuple[Stage4PdNamedCount, ...]
    all_done_boundary_reached: bool

    def validate(self, path: str = "stage4_pd_control_evidence") -> None:
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
                if type(item) is not Stage4PdCoreCount:
                    raise SchemaError(
                        "must be a Stage4PdCoreCount",
                        path=f"{path}.{field_name}[{index}]",
                    )
                item.validate(f"{path}.{field_name}[{index}]")
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
            if type(item) is not Stage4PdNamedCount:
                raise SchemaError(
                    "must be a Stage4PdNamedCount",
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
class Stage4PdEndpointRouteEvidence:
    source_rank: int
    destination_rank: int
    logical_unique_bytes: int
    delivered_bytes: int
    state_transfer_count: int

    def validate(self, path: str = "stage4_pd_endpoint_route_evidence") -> None:
        validate_uint64(self.source_rank, f"{path}.source_rank")
        validate_uint64(self.destination_rank, f"{path}.destination_rank")
        for field_name in (
            "logical_unique_bytes",
            "delivered_bytes",
            "state_transfer_count",
        ):
            value = getattr(self, field_name)
            validate_uint64(value, f"{path}.{field_name}")
            if value == 0:
                raise SchemaError("must be positive", path=f"{path}.{field_name}")
        if self.delivered_bytes != self.logical_unique_bytes:
            raise SchemaError(
                "v1 has no KV-head replication", path=f"{path}.delivered_bytes"
            )
        if self.state_transfer_count % 2:
            raise SchemaError(
                "must contain complete K/V transfer pairs",
                path=f"{path}.state_transfer_count",
            )


@dataclass(frozen=True, slots=True)
class Stage4PdD2DLinkEvidence:
    source_die_id: int
    destination_die_id: int
    request_in_packets: int
    request_out_packets: int
    ack_in_packets: int
    ack_out_packets: int
    data_in_packets: int
    data_out_packets: int

    def validate(self, path: str = "stage4_pd_d2d_link_evidence") -> None:
        for field_name in self.__dataclass_fields__:
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.source_die_id == self.destination_die_id:
            raise SchemaError("link endpoints must be distinct", path=path)
        if self.request_in_packets != self.request_out_packets:
            raise SchemaError(
                "request ingress and egress must balance", path=path
            )
        if self.ack_in_packets != self.ack_out_packets:
            raise SchemaError("ACK ingress and egress must balance", path=path)
        if self.data_in_packets != self.data_out_packets:
            raise SchemaError("data ingress and egress must balance", path=path)
        if not any(
            (
                self.request_out_packets,
                self.ack_out_packets,
                self.data_out_packets,
            )
        ):
            raise SchemaError("active link cannot have all-zero traffic", path=path)


@dataclass(frozen=True, slots=True)
class Stage4PdD2DEvidence:
    flow_count: int
    logical_packet_count: int
    physical_packet_count: int
    request_packet_count: int
    ack_packet_count: int
    logical_bytes: int
    byte_hop_bytes: int
    state_transport_logical_bytes: int
    state_transfer_count: int
    unique_endpoint_route_count: int
    endpoint_routes: tuple[Stage4PdEndpointRouteEvidence, ...]
    links: tuple[Stage4PdD2DLinkEvidence, ...]

    def validate(self, path: str = "stage4_pd_d2d_evidence") -> None:
        for field_name in (
            "flow_count",
            "logical_packet_count",
            "physical_packet_count",
            "request_packet_count",
            "ack_packet_count",
            "logical_bytes",
            "byte_hop_bytes",
            "state_transport_logical_bytes",
            "state_transfer_count",
            "unique_endpoint_route_count",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        _validate_canonical(
            self.endpoint_routes,
            key=lambda item: (item.source_rank, item.destination_rank),
            path=f"{path}.endpoint_routes",
        )
        for index, item in enumerate(self.endpoint_routes):
            if type(item) is not Stage4PdEndpointRouteEvidence:
                raise SchemaError(
                    "must be a Stage4PdEndpointRouteEvidence",
                    path=f"{path}.endpoint_routes[{index}]",
                )
            item.validate(f"{path}.endpoint_routes[{index}]")
        if self.unique_endpoint_route_count != len(self.endpoint_routes):
            raise SchemaError(
                "must equal the number of unique endpoint routes",
                path=f"{path}.unique_endpoint_route_count",
            )
        if self.state_transport_logical_bytes != sum(
            item.delivered_bytes for item in self.endpoint_routes
        ) or self.state_transfer_count != sum(
            item.state_transfer_count for item in self.endpoint_routes
        ):
            raise SchemaError(
                "endpoint routes must close state transport totals",
                path=f"{path}.endpoint_routes",
            )
        if self.flow_count < self.state_transfer_count:
            raise SchemaError(
                "D2D flow count cannot omit state transfers",
                path=f"{path}.flow_count",
            )
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
        if self.logical_bytes < self.state_transport_logical_bytes:
            raise SchemaError(
                "total logical traffic cannot omit state transport",
                path=f"{path}.logical_bytes",
            )
        _validate_canonical(
            self.links,
            key=lambda item: (item.source_die_id, item.destination_die_id),
            path=f"{path}.links",
        )
        for index, item in enumerate(self.links):
            if type(item) is not Stage4PdD2DLinkEvidence:
                raise SchemaError(
                    "must be a Stage4PdD2DLinkEvidence",
                    path=f"{path}.links[{index}]",
                )
            item.validate(f"{path}.links[{index}]")
        if (
            sum(item.request_out_packets for item in self.links)
            != self.request_packet_count
            or sum(item.ack_out_packets for item in self.links)
            != self.ack_packet_count
            or sum(item.data_out_packets for item in self.links)
            != self.physical_packet_count
            or sum(item.data_out_packets for item in self.links) * _PACKET_BYTES
            != self.byte_hop_bytes
        ):
            raise SchemaError("per-link traffic must close D2D totals", path=path)
        if not self.links and any(
            (
                self.physical_packet_count,
                self.request_packet_count,
                self.ack_packet_count,
                self.byte_hop_bytes,
            )
        ):
            raise SchemaError(
                "empty physical topology must have zero physical traffic",
                path=f"{path}.links",
            )


@dataclass(frozen=True, slots=True)
class Stage4PdRepeatEvidence:
    run_index: int
    makespan_cycles: int
    marker_digest: str
    memory_digest: str
    program_io_digest: str
    control_digest: str
    d2d_digest: str

    def validate(self, path: str = "stage4_pd_repeat_evidence") -> None:
        validate_uint64(self.run_index, f"{path}.run_index")
        validate_uint64(self.makespan_cycles, f"{path}.makespan_cycles")
        if self.makespan_cycles == 0:
            raise SchemaError("must be positive", path=f"{path}.makespan_cycles")
        for field_name in (
            "marker_digest",
            "memory_digest",
            "program_io_digest",
            "control_digest",
            "d2d_digest",
        ):
            _validate_digest(getattr(self, field_name), f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class Stage4PdRuntimeReport:
    schema_version: str
    producer_pass: str
    id: str
    baseline_epoch: str
    case_id: str
    mode: Stage4PdMode
    reshard: Stage4KvReshardKind
    capability_status: CapabilityStatus
    plan_id: str
    plan_digest: str
    oracle_id: str
    oracle_digest: str
    policy: S1NaivePolicyEvidence
    tool_digests: tuple[Stage4PdNamedDigest, ...]
    input_digests: tuple[Stage4PdNamedDigest, ...]
    hardware_digest: str
    simulation_digest: str
    mapping_digest: str
    artifact: Stage4PdArtifactEvidence
    program_io: Stage4PdProgramIoEvidence
    memory: tuple[Stage4PdMemoryEvidence, ...]
    control: Stage4PdControlEvidence
    d2d: Stage4PdD2DEvidence
    repeat_count: int
    makespan_cycles: int
    repeats: tuple[Stage4PdRepeatEvidence, ...]
    timing_execution: bool
    state_transport_exact: bool
    program_io_boundary_exact: bool
    model_functional: bool

    @classmethod
    def create(cls, **semantic_key: object) -> "Stage4PdRuntimeReport":
        result = cls(
            schema_version=STAGE4_PD_RUNTIME_REPORT_SCHEMA_VERSION,
            producer_pass="stage4_pd_runtime",
            id=stable_artifact_id(
                "stage4_pd_runtime_report",
                semantic_key,
                schema_version=STAGE4_PD_RUNTIME_REPORT_SCHEMA_VERSION,
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

    def validate(self, path: str = "stage4_pd_runtime_report") -> None:
        if self.schema_version != STAGE4_PD_RUNTIME_REPORT_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "stage4_pd_runtime":
            raise SchemaError(
                "must be 'stage4_pd_runtime'", path=f"{path}.producer_pass"
            )
        if self.baseline_epoch != S1_N_BASELINE_EPOCH:
            raise SchemaError(
                f"must be {S1_N_BASELINE_EPOCH!r}",
                path=f"{path}.baseline_epoch",
            )
        validate_nonempty(self.case_id, f"{path}.case_id")
        if type(self.mode) is not Stage4PdMode:
            raise SchemaError("must be a Stage4PdMode", path=f"{path}.mode")
        if type(self.reshard) is not Stage4KvReshardKind:
            raise SchemaError(
                "must be a Stage4KvReshardKind", path=f"{path}.reshard"
            )
        if self.capability_status is not CapabilityStatus.E2E_TIMING:
            raise SchemaError(
                "Stage 4 runtime evidence is timing-only",
                path=f"{path}.capability_status",
            )
        validate_nonempty(self.plan_id, f"{path}.plan_id")
        validate_nonempty(self.oracle_id, f"{path}.oracle_id")
        if type(self.policy) is not S1NaivePolicyEvidence:
            raise SchemaError(
                "must be a S1NaivePolicyEvidence", path=f"{path}.policy"
            )
        self.policy.validate(f"{path}.policy")
        for field_name in (
            "plan_digest",
            "oracle_digest",
            "hardware_digest",
            "simulation_digest",
            "mapping_digest",
        ):
            _validate_digest(getattr(self, field_name), f"{path}.{field_name}")
        for field_name, expected_names in (
            (
                "tool_digests",
                ("finalizer", "npusim", "resolver", "runner"),
            ),
            (
                "input_digests",
                (
                    "hardware",
                    "manifest",
                    "mapping",
                    "oracle",
                    "plan",
                    "policy",
                    "program_io",
                    "simulation",
                    "spec",
                ),
            ),
        ):
            values = getattr(self, field_name)
            _validate_canonical(
                values, key=lambda item: item.name, path=f"{path}.{field_name}"
            )
            for index, item in enumerate(values):
                if type(item) is not Stage4PdNamedDigest:
                    raise SchemaError(
                        "must be a Stage4PdNamedDigest",
                        path=f"{path}.{field_name}[{index}]",
                    )
                item.validate(f"{path}.{field_name}[{index}]")
            if tuple(item.name for item in values) != expected_names:
                raise SchemaError(
                    f"must exactly cover {expected_names!r}",
                    path=f"{path}.{field_name}",
                )
        if dict(
            (item.name, item.digest) for item in self.input_digests
        )["policy"] != canonical_digest(self.policy):
            raise SchemaError(
                "must identify the embedded policy evidence",
                path=f"{path}.input_digests",
            )
        for field_name, expected_type in (
            ("artifact", Stage4PdArtifactEvidence),
            ("program_io", Stage4PdProgramIoEvidence),
            ("control", Stage4PdControlEvidence),
            ("d2d", Stage4PdD2DEvidence),
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
        if not self.memory:
            raise SchemaError("must cover every runtime core", path=f"{path}.memory")
        for index, item in enumerate(self.memory):
            if type(item) is not Stage4PdMemoryEvidence:
                raise SchemaError(
                    "must be a Stage4PdMemoryEvidence",
                    path=f"{path}.memory[{index}]",
                )
            item.validate(f"{path}.memory[{index}]")
        if sum(item.lsu_issued for item in self.memory) == 0:
            raise SchemaError(
                "timing proof must issue blocking LSU work",
                path=f"{path}.memory",
            )
        if self.mode is Stage4PdMode.FUSED:
            if (
                self.reshard is not Stage4KvReshardKind.NONE
                or self.d2d.endpoint_routes
                or self.d2d.unique_endpoint_route_count
                or self.d2d.state_transfer_count
                or self.d2d.state_transport_logical_bytes
            ):
                raise SchemaError(
                    "fused PD must report zero state handoff",
                    path=f"{path}.d2d",
                )
        elif (
            self.reshard is Stage4KvReshardKind.NONE
            or not self.d2d.endpoint_routes
        ):
            raise SchemaError(
                "separated PD requires explicit state endpoint routes",
                path=f"{path}.d2d",
            )
        validate_uint64(self.repeat_count, f"{path}.repeat_count")
        validate_uint64(self.makespan_cycles, f"{path}.makespan_cycles")
        if (
            self.repeat_count < 2
            or self.makespan_cycles == 0
            or len(self.repeats) != self.repeat_count
        ):
            raise SchemaError(
                "runtime evidence requires at least two exact repeats",
                path=f"{path}.repeats",
            )
        _validate_canonical(
            self.repeats,
            key=lambda item: item.run_index,
            path=f"{path}.repeats",
        )
        expected_digests = (
            canonical_digest(self.memory),
            canonical_digest(self.program_io),
            canonical_digest(self.control),
            canonical_digest(self.d2d),
        )
        marker_digest: str | None = None
        for index, repeat in enumerate(self.repeats):
            if type(repeat) is not Stage4PdRepeatEvidence:
                raise SchemaError(
                    "must be a Stage4PdRepeatEvidence",
                    path=f"{path}.repeats[{index}]",
                )
            repeat.validate(f"{path}.repeats[{index}]")
            if (
                repeat.run_index != index
                or repeat.makespan_cycles != self.makespan_cycles
                or (
                    repeat.memory_digest,
                    repeat.program_io_digest,
                    repeat.control_digest,
                    repeat.d2d_digest,
                )
                != expected_digests
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
            "state_transport_exact",
            "program_io_boundary_exact",
            "model_functional",
        ):
            _validate_bool(getattr(self, field_name), f"{path}.{field_name}")
        if (
            not self.timing_execution
            or not self.state_transport_exact
            or not self.program_io_boundary_exact
            or self.model_functional
        ):
            raise SchemaError(
                "proof is timing/state-transport/ProgramIo only",
                path=path,
            )
        expected_id = stable_artifact_id(
            "stage4_pd_runtime_report",
            self._semantic_key(),
            schema_version=STAGE4_PD_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        plan: Stage4PdPlan,
        oracle: Stage4PdOracle,
        manifest: LinkedProgramManifest,
        planning_context: object,
        scheduling_context: object,
        path: str = "stage4_pd_runtime_report",
    ) -> None:
        self.validate(path)
        if type(plan) is not Stage4PdPlan:
            raise SchemaError("must be a Stage4PdPlan", path=f"{path}.plan")
        if type(oracle) is not Stage4PdOracle:
            raise SchemaError(
                "must be a Stage4PdOracle", path=f"{path}.oracle"
            )
        if type(manifest) is not LinkedProgramManifest:
            raise SchemaError(
                "must be a LinkedProgramManifest", path=f"{path}.manifest"
            )
        plan.validate("stage4_pd_plan")
        manifest.validate("linked_program_manifest")
        if (
            self.plan_id != plan.id
            or self.plan_digest != canonical_digest(plan)
            or self.mode is not plan.mode
            or self.reshard is not plan.reshard
        ):
            raise SchemaError("does not identify the supplied plan", path=path)
        if (
            self.oracle_id != oracle.id
            or self.oracle_digest != canonical_digest(oracle)
            or self.mode is not oracle.mode
            or self.reshard is not oracle.reshard
        ):
            raise SchemaError("does not identify the supplied oracle", path=path)
        oracle.validate_against(plan, "stage4_pd_oracle")
        self.policy.validate_against(
            planning_context, scheduling_context, f"{path}.policy"
        )
        inputs = {item.name: item.digest for item in self.input_digests}
        expected_inputs = {
            "hardware": self.hardware_digest,
            "manifest": canonical_digest(manifest),
            "mapping": self.mapping_digest,
            "oracle": canonical_digest(oracle),
            "plan": canonical_digest(plan),
            "policy": canonical_digest(self.policy),
            "program_io": self.program_io.contract_digest,
            "simulation": self.simulation_digest,
            "spec": plan.source_spec_digest,
        }
        if inputs != expected_inputs:
            raise SchemaError(
                "input digests do not close plan/oracle/manifest/ProgramIo",
                path=f"{path}.input_digests",
            )
        leaf_fragments = tuple(
            item.fragment if isinstance(item, RegionManifest) else item
            for item in manifest.fragments
        )
        action_ids = {
            record.source_global_action_id
            for stream in manifest.core_streams
            for record in stream.records
        }
        runtime_relocations = sum(
            len(stream.runtime_relocations)
            for fragment in leaf_fragments
            for stream in fragment.core_streams
        )
        address_relocations = sum(
            len(stream.address_relocations)
            for fragment in leaf_fragments
            for stream in fragment.core_streams
        )
        expected_artifact = (
            manifest.id,
            canonical_digest(manifest),
            len(action_ids),
            len(manifest.fragments),
            sum(len(stream.records) for stream in manifest.core_streams),
            runtime_relocations,
            address_relocations,
            runtime_relocations + address_relocations,
            len(manifest.address_operand_bindings),
            len(manifest.state_operand_bindings),
        )
        observed_artifact = (
            self.artifact.linked_manifest_id,
            self.artifact.linked_manifest_digest,
            self.artifact.action_count,
            self.artifact.fragment_count,
            self.artifact.record_count,
            self.artifact.runtime_relocation_count,
            self.artifact.address_relocation_count,
            self.artifact.relocation_count,
            self.artifact.address_operand_binding_count,
            self.artifact.state_operand_binding_count,
        )
        if observed_artifact != expected_artifact:
            raise SchemaError(
                "artifact structure differs from the linked manifest",
                path=f"{path}.artifact",
            )
        expected_routes = tuple(
            Stage4PdEndpointRouteEvidence(
                source_rank=item.source_rank,
                destination_rank=item.destination_rank,
                logical_unique_bytes=item.logical_unique_bytes,
                delivered_bytes=item.delivered_bytes,
                state_transfer_count=item.state_transfer_count,
            )
            for item in oracle.endpoint_pair_metrics
        )
        if (
            self.d2d.endpoint_routes != expected_routes
            or self.d2d.unique_endpoint_route_count
            != oracle.unique_endpoint_route_count
            or self.d2d.state_transport_logical_bytes
            != oracle.delivered_bytes
            or self.d2d.state_transfer_count != oracle.state_transfer_count
        ):
            raise SchemaError(
                "state transport differs from the PD oracle",
                path=f"{path}.d2d",
            )
        expected_memory_cores = tuple(
            sorted(binding.runtime_core_id for binding in manifest.core_bindings)
        )
        if tuple(item.runtime_core_id for item in self.memory) != expected_memory_cores:
            raise SchemaError(
                "memory evidence must cover every manifest runtime core",
                path=f"{path}.memory",
            )
        runtime_by_logical = {
            binding.logical_core: binding.runtime_core_id
            for binding in manifest.core_bindings
        }
        expected_ack = tuple(
            Stage4PdCoreCount(runtime_by_logical[core], 2)
            for core in manifest.envelope.expected_ack_cores
        )
        expected_done = tuple(
            Stage4PdCoreCount(runtime_by_logical[core], 1)
            for core in manifest.envelope.expected_done_cores
        )
        expected_ack = tuple(
            sorted(expected_ack, key=lambda item: item.runtime_core_id)
        )
        expected_done = tuple(
            sorted(expected_done, key=lambda item: item.runtime_core_id)
        )
        if (
            self.control.ack_counts != expected_ack
            or self.control.done_counts != expected_done
        ):
            raise SchemaError(
                "ACK/DONE evidence differs from the manifest envelope",
                path=f"{path}.control",
            )


__all__ = [
    "STAGE4_PD_RUNTIME_REPORT_SCHEMA_VERSION",
    "STAGE4_PD_BASELINE_EPOCH",
    "Stage4PdArtifactEvidence",
    "Stage4PdControlEvidence",
    "Stage4PdCoreCount",
    "Stage4PdD2DEvidence",
    "Stage4PdD2DLinkEvidence",
    "Stage4PdEndpointRouteEvidence",
    "Stage4PdMemoryEvidence",
    "Stage4PdNamedDigest",
    "Stage4PdNamedCount",
    "Stage4PdProgramIoEvidence",
    "Stage4PdRepeatEvidence",
    "Stage4PdRuntimeReport",
]
