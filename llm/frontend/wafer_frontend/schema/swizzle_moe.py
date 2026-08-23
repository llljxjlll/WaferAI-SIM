"""Typed semantic carriers for personalized MoE Swizzle planning.

This module stops at planner witnesses.  It deliberately has no lowering,
ProgramIo, or runtime dependencies.
"""

from __future__ import annotations

from collections import Counter
import math
from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import (
    DType,
    stable_artifact_id,
    validate_dependency_dag,
    validate_nonempty,
    validate_uint64,
    validate_unique_ids,
)
from .global_action import LogicalCoreRef
from .ir0 import FusionPattern
from .ir1 import PhysicalFabric
from .serde import canonical_digest
from .swizzle_moe_scale import MoeSwizzleScaleRole
from .swizzle_moe_calibration import (
    MoeCalibrationStatus,
    MoeSwizzleCalibrationProfile,
)
from .swizzle_moe_placement import (
    MoeCandidateCoreLifecycleFloor,
    MoeWholePairFeasibility,
)

from .swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
    SwizzleDecisionReason,
    SwizzleGroupView,
    SwizzleRouteView,
    SwizzleTensorAxisRole,
)


MOE_SWIZZLE_REGION_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_region/v1alpha1"
MOE_SWIZZLE_PROBLEM_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_problem/v1alpha1"
MOE_SWIZZLE_ACTION_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_action/v1alpha1"
MOE_SWIZZLE_COST_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_cost/v1alpha1"
MOE_SWIZZLE_CANDIDATE_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_candidate/v1alpha1"
MOE_SWIZZLE_DECISION_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_decision/v1alpha1"
MOE_SWIZZLE_WORKLOAD_SELECTION_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_workload_selection/v1alpha1"
)
MOE_HARDWARE_FACTS_SCHEMA_VERSION = "wafer_frontend.moe_hardware_facts/v1alpha1"
MOE_ENDPOINT_SESSION_CONTRACT_SCHEMA_VERSION = "wafer_frontend.moe_endpoint_session_contract/v1alpha1"

_MOE_PATTERNS = (
    FusionPattern.MOE_DISPATCH_GEMM,
    FusionPattern.MOE_GEMM_COMBINE,
)
_MOE_ALGORITHMS = (
    SwizzleAlgorithm.UNFUSED,
    SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A,
    SwizzleAlgorithm.COMET_MESH_PERSONALIZED_A2A,
)


class MoeTrafficScenarioKind(str, Enum):
    ACTUAL = "actual"
    P95 = "p95"
    CAPACITY = "capacity"


class MoeTrafficQuantileMethod(str, Enum):
    DETERMINISTIC_SINGLE_TRACE = "deterministic_single_trace"


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be greater than zero", path=path)


def _refs(value: tuple[str, ...], path: str, *, nonempty: bool = False) -> None:
    if type(value) is not tuple or (nonempty and not value):
        raise SchemaError("must be an immutable non-empty tuple", path=path)
    if len(value) != len(set(value)):
        raise SchemaError("contains duplicate references", path=path)
    for index, ref in enumerate(value):
        validate_nonempty(ref, f"{path}[{index}]")


def _semantic(instance: object) -> dict[str, object]:
    return {
        name: getattr(instance, name)
        for name in instance.__dataclass_fields__
        if name not in ("schema_version", "producer_pass", "id")
    }


def _stable(instance: object, prefix: str, version: str, path: str) -> None:
    expected = stable_artifact_id(prefix, _semantic(instance), schema_version=version)
    if getattr(instance, "id") != expected:
        raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


def _validate_moe_pattern(pattern: FusionPattern, path: str) -> None:
    if type(pattern) is not FusionPattern or pattern not in _MOE_PATTERNS:
        raise SchemaError("requires a MoE fusion pattern", path=path)


@dataclass(frozen=True, slots=True)
class MoeTokenAssignmentView:
    id: str
    token_index: int
    source_rank: int
    expert_index: int
    expert_rank: int
    contributor_ordinal: int
    gate_weight_ref: str | None
    dispatch_flow_ref: str | None
    combine_flow_ref: str | None
    dispatch_route_ref: str | None
    combine_route_ref: str | None
    payload_value_ref: str
    payload_bytes: int
    swiglu_action_ref: str | None

    @classmethod
    def create(cls, **semantic: object) -> "MoeTokenAssignmentView":
        result = cls(
            stable_artifact_id("moe_assignment", semantic, schema_version=MOE_SWIZZLE_REGION_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_assignment") -> None:
        for name in (
            "token_index", "source_rank", "expert_index", "expert_rank",
            "contributor_ordinal", "payload_bytes",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        validate_nonempty(self.payload_value_ref, f"{path}.payload_value_ref")
        if self.payload_bytes == 0:
            raise SchemaError("payload must be non-empty", path=f"{path}.payload_bytes")
        for name in (
            "gate_weight_ref", "dispatch_flow_ref", "combine_flow_ref",
            "dispatch_route_ref", "combine_route_ref", "swiglu_action_ref",
        ):
            ref = getattr(self, name)
            if ref is not None:
                validate_nonempty(ref, f"{path}.{name}")
        local = self.source_rank == self.expert_rank
        p2p_refs = (
            self.dispatch_flow_ref,
            self.combine_flow_ref,
            self.dispatch_route_ref,
            self.combine_route_ref,
        )
        if local != all(ref is None for ref in p2p_refs):
            raise SchemaError("local assignments have no synthetic P2P flows", path=path)
        if not local and any(ref is None for ref in p2p_refs):
            raise SchemaError(
                "remote assignment requires dispatch/combine flows and routes", path=path
            )
        expected = stable_artifact_id(
            "moe_assignment", _semantic(self), schema_version=MOE_SWIZZLE_REGION_SCHEMA_VERSION
        )
        if self.id != expected:
            raise SchemaError(f"unstable assignment id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeExpertGemmView:
    id: str
    expert_index: int
    rank: int
    member_refs: tuple[str, ...]
    m_tokens: int
    n: int
    k: int
    dtype: DType
    accumulation_dtype: DType
    flops: int

    @classmethod
    def create(cls, **semantic: object) -> "MoeExpertGemmView":
        result = cls(
            stable_artifact_id("moe_expert_gemm", semantic, schema_version=MOE_SWIZZLE_REGION_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_expert_gemm") -> None:
        for name in ("expert_index", "rank", "m_tokens", "n", "k", "flops"):
            _positive(getattr(self, name), f"{path}.{name}") if name in ("m_tokens", "n", "k", "flops") else validate_uint64(getattr(self, name), f"{path}.{name}")
        _refs(self.member_refs, f"{path}.member_refs", nonempty=True)
        if type(self.dtype) is not DType or type(self.accumulation_dtype) is not DType:
            raise SchemaError("requires typed input/accumulation dtype", path=path)
        expected = stable_artifact_id(
            "moe_expert_gemm", _semantic(self), schema_version=MOE_SWIZZLE_REGION_SCHEMA_VERSION
        )
        if self.id != expected:
            raise SchemaError(f"unstable expert GEMM id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoePersonalizedTrafficView:
    assignments: tuple[MoeTokenAssignmentView, ...]
    expert_gemms: tuple[MoeExpertGemmView, ...]
    pair_routes: tuple[SwizzleRouteView, ...]
    top_k: int
    capacity_tokens_per_expert: int
    trace_digest: str
    logical_payload_bytes: int
    expert_gemm_flops: int
    region_boundary_output_bytes: int

    def validate(self, path: str = "moe_traffic") -> None:
        if type(self.assignments) is not tuple or not self.assignments:
            raise SchemaError("requires assignments", path=f"{path}.assignments")
        assignment_index = validate_unique_ids(self.assignments, f"{path}.assignments")
        if tuple((item.token_index, item.contributor_ordinal) for item in self.assignments) != tuple(
            sorted((item.token_index, item.contributor_ordinal) for item in self.assignments)
        ):
            raise SchemaError("assignments must use canonical token/contributor order", path=f"{path}.assignments")
        for index, assignment in enumerate(self.assignments):
            assignment.validate(f"{path}.assignments[{index}]")
        del assignment_index
        if type(self.expert_gemms) is not tuple or not self.expert_gemms:
            raise SchemaError("requires expert GEMM groups", path=f"{path}.expert_gemms")
        validate_unique_ids(self.expert_gemms, f"{path}.expert_gemms")
        if tuple(item.expert_index for item in self.expert_gemms) != tuple(range(len(self.expert_gemms))):
            raise SchemaError("expert GEMMs must be dense canonical experts", path=f"{path}.expert_gemms")
        for index, gemm in enumerate(self.expert_gemms):
            gemm.validate(f"{path}.expert_gemms[{index}]")
            count = sum(item.expert_index == gemm.expert_index for item in self.assignments)
            if gemm.rank != gemm.expert_index or gemm.m_tokens != count:
                raise SchemaError("expert GEMM count/home disagrees with assignments", path=f"{path}.expert_gemms[{index}]")
        routes = validate_unique_ids(self.pair_routes, f"{path}.pair_routes")
        pairs = set()
        for index, route in enumerate(self.pair_routes):
            route.validate(f"{path}.pair_routes[{index}]")
            pair = (route.source_rank, route.destination_rank)
            if pair in pairs:
                raise SchemaError("duplicate route endpoint pair", path=f"{path}.pair_routes[{index}]")
            pairs.add(pair)
        route_refs = {
            ref
            for item in self.assignments
            for ref in (item.dispatch_route_ref, item.combine_route_ref)
            if ref is not None
        }
        if not route_refs or not route_refs.issubset(routes):
            raise SchemaError("assignment route is outside typed routes", path=f"{path}.pair_routes")
        _positive(self.top_k, f"{path}.top_k")
        _positive(self.capacity_tokens_per_expert, f"{path}.capacity_tokens_per_expert")
        validate_nonempty(self.trace_digest, f"{path}.trace_digest")
        for name in ("logical_payload_bytes", "expert_gemm_flops", "region_boundary_output_bytes"):
            _positive(getattr(self, name), f"{path}.{name}")
        if self.logical_payload_bytes != sum(
            item.payload_bytes for item in self.assignments if item.source_rank != item.expert_rank
        ):
            raise SchemaError("logical payload bytes do not equal remote assignments", path=f"{path}.logical_payload_bytes")
        if self.expert_gemm_flops != sum(item.flops for item in self.expert_gemms):
            raise SchemaError("expert GEMM FLOPs are not closed", path=f"{path}.expert_gemm_flops")


@dataclass(frozen=True, slots=True)
class MoeSemanticWitness:
    pattern: FusionPattern
    traffic: MoePersonalizedTrafficView
    split_axis: SwizzleTensorAxisRole
    gate_weight_applied: bool
    reduce_dtype: DType | None
    boundary_closed: bool
    route_reversal_closed: bool

    def validate(self, path: str = "moe_semantic_witness") -> None:
        _validate_moe_pattern(self.pattern, f"{path}.pattern")
        self.traffic.validate(f"{path}.traffic")
        if type(self.split_axis) is not SwizzleTensorAxisRole:
            raise SchemaError("requires a typed split axis role", path=f"{path}.split_axis")
        for name in ("gate_weight_applied", "boundary_closed", "route_reversal_closed"):
            if type(getattr(self, name)) is not bool:
                raise SchemaError("semantic predicates must be bool", path=f"{path}.{name}")
        if not self.boundary_closed or not self.route_reversal_closed:
            raise SchemaError("semantic closure predicates must pass", path=path)
        dtype_bytes = {DType.FP16: 2, DType.FP32: 4, DType.INT32: 4}
        expected_boundary_bytes = (
            sum(
                self.traffic.expert_gemms[item.expert_index].n
                * dtype_bytes[self.traffic.expert_gemms[item.expert_index].dtype]
                for item in self.traffic.assignments
            )
            if self.pattern is FusionPattern.MOE_DISPATCH_GEMM
            else sum(item.payload_bytes for item in self.traffic.assignments)
        )
        if self.traffic.region_boundary_output_bytes != expected_boundary_bytes:
            raise SchemaError(
                "region boundary output bytes disagree with typed semantics",
                path=f"{path}.traffic.region_boundary_output_bytes",
            )
        swiglu_refs = tuple(
            item.swiglu_action_ref for item in self.traffic.assignments
        )
        if self.pattern is FusionPattern.MOE_DISPATCH_GEMM:
            if any(ref is None for ref in swiglu_refs) or len(set(swiglu_refs)) != len(swiglu_refs):
                raise SchemaError(
                    "dispatch requires one exact original SWIGLU per assignment",
                    path=f"{path}.traffic.assignments",
                )
            if self.split_axis is not SwizzleTensorAxisRole.FREE_LHS or self.gate_weight_applied or self.reduce_dtype is not None:
                raise SchemaError("dispatch must split token/M without reduction", path=path)
        elif any(ref is not None for ref in swiglu_refs):
            raise SchemaError(
                "combine cannot claim Dispatch SWIGLU originals",
                path=f"{path}.traffic.assignments",
            )
        elif self.split_axis is not SwizzleTensorAxisRole.FREE_RHS:
            raise SchemaError("combine must split output N", path=path)
        elif self.traffic.top_k == 1:
            if self.gate_weight_applied or self.reduce_dtype is not None:
                raise SchemaError("top-1 combine has no numerical reduce", path=path)
        elif not self.gate_weight_applied or self.reduce_dtype is not DType.FP32:
            raise SchemaError("top-k combine requires weighted FP32 SUM", path=path)


@dataclass(frozen=True, slots=True)
class MoeFusionRegion:
    schema_version: str
    producer_pass: str
    id: str
    source_spec_id: str
    source_oracle_id: str
    source_execution_id: str
    pattern: FusionPattern
    member_refs: tuple[str, ...]
    boundary_input_refs: tuple[str, ...]
    boundary_output_refs: tuple[str, ...]
    assignment_refs: tuple[str, ...]
    trace_digest: str
    semantic_witness: MoeSemanticWitness

    @classmethod
    def create(cls, **semantic: object) -> "MoeFusionRegion":
        result = cls(
            MOE_SWIZZLE_REGION_SCHEMA_VERSION,
            "discover_moe_swizzle",
            stable_artifact_id("moe_swizzle_region", semantic, schema_version=MOE_SWIZZLE_REGION_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_swizzle_region") -> None:
        if self.schema_version != MOE_SWIZZLE_REGION_SCHEMA_VERSION or self.producer_pass != "discover_moe_swizzle":
            raise SchemaError("unsupported MoE region schema/producer", path=path)
        validate_nonempty(self.source_spec_id, f"{path}.source_spec_id")
        validate_nonempty(self.source_oracle_id, f"{path}.source_oracle_id")
        validate_nonempty(self.source_execution_id, f"{path}.source_execution_id")
        _validate_moe_pattern(self.pattern, f"{path}.pattern")
        for name in ("member_refs", "boundary_input_refs", "boundary_output_refs", "assignment_refs"):
            _refs(getattr(self, name), f"{path}.{name}", nonempty=True)
        validate_nonempty(self.trace_digest, f"{path}.trace_digest")
        self.semantic_witness.validate(f"{path}.semantic_witness")
        if self.semantic_witness.pattern is not self.pattern:
            raise SchemaError("region/witness pattern mismatch", path=path)
        if self.assignment_refs != tuple(item.id for item in self.semantic_witness.traffic.assignments):
            raise SchemaError("assignment refs are not the traffic assignment order", path=f"{path}.assignment_refs")
        if self.trace_digest != self.semantic_witness.traffic.trace_digest:
            raise SchemaError("region/traffic trace digest mismatch", path=f"{path}.trace_digest")
        _stable(self, "moe_swizzle_region", MOE_SWIZZLE_REGION_SCHEMA_VERSION, path)


@dataclass(frozen=True, slots=True)
class MoeTopologyWitness:
    group: SwizzleGroupView
    row_orders: tuple[tuple[int, ...], ...]
    column_orders: tuple[tuple[int, ...], ...]
    pivot_by_pair: tuple[tuple[int, int, int], ...]
    complete_rectangle: bool

    def validate(self, path: str = "moe_topology") -> None:
        self.group.validate(f"{path}.group")
        if type(self.complete_rectangle) is not bool:
            raise SchemaError("rectangle predicate must be bool", path=f"{path}.complete_rectangle")
        for name in ("row_orders", "column_orders"):
            orders = getattr(self, name)
            if type(orders) is not tuple or any(type(item) is not tuple or not item for item in orders):
                raise SchemaError("row/column orders must be non-empty tuples", path=f"{path}.{name}")
        pairs = tuple((source, destination) for source, destination, _ in self.pivot_by_pair)
        if pairs != tuple(sorted(pairs)) or len(pairs) != len(set(pairs)):
            raise SchemaError("pivot pairs must be unique canonical pairs", path=f"{path}.pivot_by_pair")
        ranks = set(range(len(self.group.placements)))
        if any(source not in ranks or destination not in ranks or pivot not in ranks or source == destination for source, destination, pivot in self.pivot_by_pair):
            raise SchemaError("pivot endpoint is outside topology", path=f"{path}.pivot_by_pair")
        if self.complete_rectangle:
            rows, columns = self.group.logical_shape
            route_pairs = {
                (item.source_rank, item.destination_rank): item
                for item in self.group.routes
            }
            if (
                rows <= 1
                or columns <= 1
                or len(self.pivot_by_pair) != len(ranks) * (len(ranks) - 1)
                or set(pairs) != set(route_pairs)
                or any(
                    pivot not in route_pairs[(source, destination)].die_path
                    for source, destination, pivot in self.pivot_by_pair
                )
            ):
                raise SchemaError("complete rectangle lacks exact routes/pivots", path=path)


@dataclass(frozen=True, slots=True)
class MoeLinkHardwareFact:
    resource_id: str
    physical_resource_ref: str
    source_die: int
    destination_die: int
    bytes_per_cycle: int
    latency_cycles: int

    def validate(self, path: str = "moe_link_hardware_fact") -> None:
        validate_nonempty(self.resource_id, f"{path}.resource_id")
        validate_nonempty(self.physical_resource_ref, f"{path}.physical_resource_ref")
        for name in ("source_die", "destination_die", "bytes_per_cycle", "latency_cycles"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.source_die == self.destination_die or self.bytes_per_cycle == 0:
            raise SchemaError("link incidence/bandwidth is invalid", path=path)
        if self.resource_id != f"moe.scale.link.{self.source_die}.{self.destination_die}":
            raise SchemaError("link resource naming/incidence drifted", path=f"{path}.resource_id")


@dataclass(frozen=True, slots=True)
class MoeCoreHardwareFact:
    logical_core: LogicalCoreRef
    runtime_core_id: int
    sram_profile_ref: str
    region_name: str
    region_base_bytes: int
    region_size_bytes: int
    allocation_alignment_bytes: int
    bank_count: int

    def validate(self, path: str = "moe_core_hardware_fact") -> None:
        self.logical_core.validate(f"{path}.logical_core")
        for name in ("runtime_core_id", "region_base_bytes", "region_size_bytes", "allocation_alignment_bytes", "bank_count"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        validate_nonempty(self.sram_profile_ref, f"{path}.sram_profile_ref")
        validate_nonempty(self.region_name, f"{path}.region_name")
        if self.region_name != "comm" or self.region_size_bytes == 0 or self.allocation_alignment_bytes == 0 or self.bank_count == 0:
            raise SchemaError("core SRAM region facts are invalid", path=path)


@dataclass(frozen=True, slots=True)
class MoeHardwareFacts:
    schema_version: str
    producer_pass: str
    id: str
    source_fabric_digest: str
    ordered_cores_by_die: tuple[tuple[MoeCoreHardwareFact, ...], ...]
    route_resources: tuple[MoeLinkHardwareFact, ...]

    @classmethod
    def from_fabric(cls, fabric: PhysicalFabric) -> "MoeHardwareFacts":
        if type(fabric) is not PhysicalFabric:
            raise SchemaError("requires exact PhysicalFabric", path="moe_hardware_facts.fabric")
        fabric.validate("moe_hardware_facts.fabric")
        profiles = {item.id: item for item in fabric.sram_profiles}
        ordered = tuple(tuple(
            MoeCoreHardwareFact(
                LogicalCoreRef(die.id, core.local_core_id), core.runtime_core_id,
                core.sram_profile_ref, region.name, region.base_bytes, region.size_bytes,
                profiles[core.sram_profile_ref].allocation_alignment_bytes,
                profiles[core.sram_profile_ref].bank_count,
            )
            for core in sorted(die.cores, key=lambda item: item.local_core_id)
            for region in profiles[core.sram_profile_ref].regions if region.name == "comm"
        ) for die in sorted(fabric.dies, key=lambda item: item.id))
        resources = tuple(MoeLinkHardwareFact(f"moe.scale.link.{link.source_die}.{link.destination_die}", link.resource_id, link.source_die, link.destination_die, link.bytes_per_cycle, link.latency_cycles) for link in sorted(fabric.links, key=lambda item: (item.source_die, item.destination_die)))
        semantic = {"source_fabric_digest": canonical_digest(fabric), "ordered_cores_by_die": ordered, "route_resources": resources}
        result = cls(MOE_HARDWARE_FACTS_SCHEMA_VERSION, "build_moe_hardware_facts", stable_artifact_id("moe_hardware_facts", semantic, schema_version=MOE_HARDWARE_FACTS_SCHEMA_VERSION), **semantic)
        result.validate()
        return result

    def validate(self, path: str = "moe_hardware_facts") -> None:
        if self.schema_version != MOE_HARDWARE_FACTS_SCHEMA_VERSION or self.producer_pass != "build_moe_hardware_facts":
            raise SchemaError("unsupported hardware-facts schema/producer", path=path)
        if type(self.source_fabric_digest) is not str or len(self.source_fabric_digest) != 64:
            raise SchemaError("hardware facts require fabric digest", path=f"{path}.source_fabric_digest")
        expected = tuple(LogicalCoreRef(die, local) for die, cores in enumerate(self.ordered_cores_by_die) for local in range(len(cores)))
        actual = tuple(core.logical_core for cores in self.ordered_cores_by_die for core in cores)
        for die, cores in enumerate(self.ordered_cores_by_die):
            for index, core in enumerate(cores):
                core.validate(f"{path}.ordered_cores_by_die[{die}][{index}]")
        if not expected or actual != expected:
            raise SchemaError("ordered logical cores are not contiguous by die", path=f"{path}.ordered_cores_by_die")
        runtime_ids = tuple(core.runtime_core_id for cores in self.ordered_cores_by_die for core in cores)
        if len(runtime_ids) != len(set(runtime_ids)):
            raise SchemaError("runtime core ids are not globally unique", path=f"{path}.ordered_cores_by_die")
        keys = []
        for index, item in enumerate(self.route_resources):
            item.validate(f"{path}.route_resources[{index}]")
            keys.append((item.source_die, item.destination_die))
        if tuple(keys) != tuple(sorted(set(keys))):
            raise SchemaError("route resources are not canonical/unique", path=f"{path}.route_resources")
        _stable(self, "moe_hardware_facts", MOE_HARDWARE_FACTS_SCHEMA_VERSION, path)


@dataclass(frozen=True, slots=True)
class MoeEndpointSessionContract:
    schema_version: str
    producer_pass: str
    id: str
    capacity_per_core: int
    source_contract: str
    config_digest: str

    @classmethod
    def production(cls) -> "MoeEndpointSessionContract":
        config = {"capacity_per_core": 3, "source_contract": "npusim.p2p/MAX_BUFFER_PACKET_SIZE/v1"}
        semantic = {**config, "config_digest": canonical_digest(config)}
        result = cls(MOE_ENDPOINT_SESSION_CONTRACT_SCHEMA_VERSION, "production_moe_endpoint_session_contract", stable_artifact_id("moe_endpoint_session_contract", semantic, schema_version=MOE_ENDPOINT_SESSION_CONTRACT_SCHEMA_VERSION), **semantic)
        result.validate()
        return result

    def validate(self, path: str = "moe_endpoint_session_contract") -> None:
        config = {"capacity_per_core": 3, "source_contract": "npusim.p2p/MAX_BUFFER_PACKET_SIZE/v1"}
        semantic = {**config, "config_digest": canonical_digest(config)}
        if self.schema_version != MOE_ENDPOINT_SESSION_CONTRACT_SCHEMA_VERSION or self.producer_pass != "production_moe_endpoint_session_contract" or self.capacity_per_core != 3 or self.source_contract != config["source_contract"] or self.config_digest != semantic["config_digest"] or self.id != stable_artifact_id("moe_endpoint_session_contract", semantic, schema_version=MOE_ENDPOINT_SESSION_CONTRACT_SCHEMA_VERSION):
            raise SchemaError("unsupported endpoint session contract", path=path)


@dataclass(frozen=True, slots=True)
class MoeTrafficScenario:
    kind: MoeTrafficScenarioKind
    expert_token_counts: tuple[int, ...]
    source_expert_token_counts: tuple[tuple[int, ...], ...]
    logical_payload_bytes: int
    expert_gemm_flops: int
    region_boundary_output_bytes: int
    executable_binding: bool
    quantile_method: MoeTrafficQuantileMethod | None = None

    def validate(self, path: str = "moe_traffic_scenario") -> None:
        if type(self.kind) is not MoeTrafficScenarioKind:
            raise SchemaError("requires a typed scenario kind", path=f"{path}.kind")
        if type(self.expert_token_counts) is not tuple or not self.expert_token_counts:
            raise SchemaError("requires expert token counts", path=f"{path}.expert_token_counts")
        for index, value in enumerate(self.expert_token_counts):
            validate_uint64(value, f"{path}.expert_token_counts[{index}]")
        if not self.source_expert_token_counts or any(len(row) != len(self.expert_token_counts) for row in self.source_expert_token_counts):
            raise SchemaError("source/expert load matrix has invalid shape", path=f"{path}.source_expert_token_counts")
        for source, row in enumerate(self.source_expert_token_counts):
            for expert, value in enumerate(row):
                validate_uint64(value, f"{path}.source_expert_token_counts[{source}][{expert}]")
        if tuple(sum(row[expert] for row in self.source_expert_token_counts) for expert in range(len(self.expert_token_counts))) != self.expert_token_counts:
            raise SchemaError("source/expert load matrix does not close expert counts", path=f"{path}.source_expert_token_counts")
        for name in ("logical_payload_bytes", "expert_gemm_flops", "region_boundary_output_bytes"):
            _positive(getattr(self, name), f"{path}.{name}")
        if type(self.executable_binding) is not bool:
            raise SchemaError("scenario executable binding must be bool", path=f"{path}.executable_binding")
        if (self.kind is MoeTrafficScenarioKind.ACTUAL) != self.executable_binding:
            raise SchemaError("only actual trace has executable assignment binding", path=f"{path}.executable_binding")
        expected_method = (
            MoeTrafficQuantileMethod.DETERMINISTIC_SINGLE_TRACE
            if self.kind is MoeTrafficScenarioKind.P95 else None
        )
        if self.quantile_method is not expected_method:
            raise SchemaError("P95 requires deterministic single-trace method", path=f"{path}.quantile_method")


@dataclass(frozen=True, slots=True)
class MoeSwizzleProblem:
    schema_version: str
    producer_pass: str
    id: str
    source_execution_id: str
    scale_name: str
    scale_role: MoeSwizzleScaleRole
    region: MoeFusionRegion
    topology: MoeTopologyWitness
    traffic_scenarios: tuple[MoeTrafficScenario, ...]
    allowed_algorithms: tuple[SwizzleAlgorithm, ...]
    hardware_facts: MoeHardwareFacts
    endpoint_session_contract: MoeEndpointSessionContract
    max_candidates: int

    @property
    def endpoint_session_capacity(self) -> int:
        return self.endpoint_session_contract.capacity_per_core

    @property
    def sram_capacity_bytes(self) -> int:
        return min(core.region_size_bytes for cores in self.hardware_facts.ordered_cores_by_die for core in cores)

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleProblem":
        result = cls(
            MOE_SWIZZLE_PROBLEM_SCHEMA_VERSION,
            "build_moe_swizzle_problem",
            stable_artifact_id("moe_swizzle_problem", semantic, schema_version=MOE_SWIZZLE_PROBLEM_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_swizzle_problem") -> None:
        if self.schema_version != MOE_SWIZZLE_PROBLEM_SCHEMA_VERSION or self.producer_pass != "build_moe_swizzle_problem":
            raise SchemaError("unsupported MoE problem schema/producer", path=path)
        validate_nonempty(self.source_execution_id, f"{path}.source_execution_id")
        validate_nonempty(self.scale_name, f"{path}.scale_name")
        if type(self.scale_role) is not MoeSwizzleScaleRole:
            raise SchemaError("problem requires typed scale role", path=f"{path}.scale_role")
        self.region.validate(f"{path}.region")
        if self.source_execution_id != self.region.source_execution_id:
            raise SchemaError("problem/region execution provenance mismatch", path=path)
        self.topology.validate(f"{path}.topology")
        self.hardware_facts.validate(f"{path}.hardware_facts")
        self.endpoint_session_contract.validate(f"{path}.endpoint_session_contract")
        route_resources = {ref for route in self.topology.group.routes for ref in route.resource_ids}
        hardware_resources = {item.resource_id for item in self.hardware_facts.route_resources}
        if route_resources != hardware_resources:
            raise SchemaError("topology routes do not close over hardware bandwidth facts", path=f"{path}.hardware_facts.route_resources")
        if len(self.hardware_facts.ordered_cores_by_die) != len(self.topology.group.placements):
            raise SchemaError("hardware dies do not match MoE topology", path=f"{path}.hardware_facts.ordered_cores_by_die")
        if type(self.traffic_scenarios) is not tuple or not self.traffic_scenarios:
            raise SchemaError("requires traffic scenarios", path=f"{path}.traffic_scenarios")
        kinds = []
        for index, scenario in enumerate(self.traffic_scenarios):
            scenario.validate(f"{path}.traffic_scenarios[{index}]")
            kinds.append(scenario.kind)
        if tuple(kinds) != tuple(sorted(set(kinds), key=lambda item: item.value)) or MoeTrafficScenarioKind.ACTUAL not in kinds or MoeTrafficScenarioKind.CAPACITY not in kinds:
            raise SchemaError("requires canonical actual/capacity scenarios", path=f"{path}.traffic_scenarios")
        actual = next(item for item in self.traffic_scenarios if item.kind is MoeTrafficScenarioKind.ACTUAL)
        traffic = self.region.semantic_witness.traffic
        capacity = next(item for item in self.traffic_scenarios if item.kind is MoeTrafficScenarioKind.CAPACITY)
        p95 = next(item for item in self.traffic_scenarios if item.kind is MoeTrafficScenarioKind.P95)
        if (
            p95.expert_token_counts, p95.source_expert_token_counts,
            p95.logical_payload_bytes, p95.expert_gemm_flops,
            p95.region_boundary_output_bytes,
        ) != (
            actual.expert_token_counts, actual.source_expert_token_counts,
            actual.logical_payload_bytes, actual.expert_gemm_flops,
            actual.region_boundary_output_bytes,
        ):
            raise SchemaError("deterministic single-trace P95 must equal ACTUAL", path=f"{path}.traffic_scenarios")
        if any(
            capacity_value < actual_value
            for capacity_value, actual_value in zip(
                (capacity.logical_payload_bytes, capacity.expert_gemm_flops, capacity.region_boundary_output_bytes),
                (actual.logical_payload_bytes, actual.expert_gemm_flops, actual.region_boundary_output_bytes),
                strict=True,
            )
        ):
            raise SchemaError("capacity scenario is not an upper bound", path=f"{path}.traffic_scenarios")
        routes = {(item.source_rank, item.destination_rank): item for item in self.topology.group.routes}
        expected_capacity = [[0] * len(capacity.expert_token_counts) for _ in self.topology.group.placements]
        for expert, count in enumerate(capacity.expert_token_counts):
            home = traffic.expert_gemms[expert].rank
            farthest = max(
                range(len(expected_capacity)),
                key=lambda source: (0 if source == home else len(routes[(source, home)].die_path) - 1, -source),
            )
            expected_capacity[farthest][expert] = count
        if capacity.source_expert_token_counts != tuple(tuple(row) for row in expected_capacity):
            raise SchemaError("capacity load matrix is not the canonical farthest-source bound", path=f"{path}.traffic_scenarios")
        if (actual.logical_payload_bytes, actual.expert_gemm_flops, actual.region_boundary_output_bytes) != (
            traffic.logical_payload_bytes, traffic.expert_gemm_flops, traffic.region_boundary_output_bytes
        ):
            raise SchemaError("actual scenario differs from discovered work", path=f"{path}.traffic_scenarios")
        if (
            type(self.allowed_algorithms) is not tuple
            or len(self.allowed_algorithms) != len(set(self.allowed_algorithms))
            or SwizzleAlgorithm.UNFUSED not in self.allowed_algorithms
            or SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A not in self.allowed_algorithms
            or any(item not in _MOE_ALGORITHMS for item in self.allowed_algorithms)
        ):
            raise SchemaError("requires the scoped MoE algorithm domain", path=f"{path}.allowed_algorithms")
        for algorithm in self.allowed_algorithms:
            if type(algorithm) is not SwizzleAlgorithm:
                raise SchemaError("requires typed algorithms", path=f"{path}.allowed_algorithms")
        _positive(self.max_candidates, f"{path}.max_candidates")
        has_comet = SwizzleAlgorithm.COMET_MESH_PERSONALIZED_A2A in self.allowed_algorithms
        if has_comet != self.topology.complete_rectangle:
            raise SchemaError("Comet mesh admission must equal rectangle witness", path=f"{path}.allowed_algorithms")
        _stable(self, "moe_swizzle_problem", MOE_SWIZZLE_PROBLEM_SCHEMA_VERSION, path)


@dataclass(frozen=True, slots=True)
class MoePacketSlice:
    assignment_ref: str
    source_offset_bytes: int
    destination_offset_bytes: int
    bytes: int
    n_block_index: int | None

    def validate(self, path: str = "moe_packet_slice") -> None:
        validate_nonempty(self.assignment_ref, f"{path}.assignment_ref")
        for name in ("source_offset_bytes", "destination_offset_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        _positive(self.bytes, f"{path}.bytes")
        if self.n_block_index is not None:
            validate_uint64(self.n_block_index, f"{path}.n_block_index")


@dataclass(frozen=True, slots=True)
class MoePacketWitness:
    id: str
    stage: int
    source_rank: int
    destination_rank: int
    pivot_rank: int | None
    route_ref: str
    slices: tuple[MoePacketSlice, ...]
    logical_bytes: int

    @classmethod
    def create(cls, **semantic: object) -> "MoePacketWitness":
        result = cls(
            stable_artifact_id("moe_packet", semantic, schema_version=MOE_SWIZZLE_CANDIDATE_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_packet") -> None:
        for name in ("stage", "source_rank", "destination_rank"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.source_rank == self.destination_rank:
            raise SchemaError("packets cannot encode zero-distance sends", path=path)
        if self.pivot_rank is not None:
            validate_uint64(self.pivot_rank, f"{path}.pivot_rank")
        validate_nonempty(self.route_ref, f"{path}.route_ref")
        if type(self.slices) is not tuple or not self.slices:
            raise SchemaError("packet requires slices", path=f"{path}.slices")
        for index, item in enumerate(self.slices):
            item.validate(f"{path}.slices[{index}]")
        if len({item.assignment_ref for item in self.slices}) != len(self.slices):
            raise SchemaError("packet duplicates an assignment slice", path=f"{path}.slices")
        _positive(self.logical_bytes, f"{path}.logical_bytes")
        if self.logical_bytes != sum(item.bytes for item in self.slices):
            raise SchemaError("packet bytes do not equal slices", path=f"{path}.logical_bytes")
        expected = stable_artifact_id("moe_packet", _semantic(self), schema_version=MOE_SWIZZLE_CANDIDATE_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable packet id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeTileWitness:
    id: str
    expert_index: int
    tile_index: int
    m_block_index: int
    m_block_size: int
    split_axis: SwizzleTensorAxisRole
    assignment_refs: tuple[str, ...]
    arrival_class: int
    required_packet_refs: tuple[str, ...]
    output_value_refs: tuple[str, ...]
    n_block_index: int | None
    output_column_offset: int
    output_column_extent: int

    @classmethod
    def create(cls, **semantic: object) -> "MoeTileWitness":
        result = cls(
            stable_artifact_id("moe_tile", semantic, schema_version=MOE_SWIZZLE_CANDIDATE_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_tile") -> None:
        for name in (
            "expert_index", "tile_index", "m_block_index", "arrival_class",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        _positive(self.m_block_size, f"{path}.m_block_size")
        if self.n_block_index is not None:
            validate_uint64(self.n_block_index, f"{path}.n_block_index")
        validate_uint64(self.output_column_offset, f"{path}.output_column_offset")
        _positive(self.output_column_extent, f"{path}.output_column_extent")
        if type(self.split_axis) is not SwizzleTensorAxisRole:
            raise SchemaError("requires typed split axis", path=f"{path}.split_axis")
        for name in ("assignment_refs", "output_value_refs"):
            _refs(getattr(self, name), f"{path}.{name}", nonempty=True)
        if len(self.assignment_refs) != self.m_block_size:
            raise SchemaError(
                "M-block size must exactly equal assignment coverage",
                path=f"{path}.m_block_size",
            )
        _refs(self.required_packet_refs, f"{path}.required_packet_refs")
        expected = stable_artifact_id("moe_tile", _semantic(self), schema_version=MOE_SWIZZLE_CANDIDATE_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable tile id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeActionWitness:
    schema_version: str
    id: str
    rank: int
    kind: SwizzleActionKind
    deps: tuple[str, ...]
    assignment_refs: tuple[str, ...]
    expert_index: int | None
    tile_index: int | None
    n_block_index: int | None
    packet_ref: str | None
    stage: int | None
    pivot_rank: int | None
    original_action_refs: tuple[str, ...]
    route_ref: str | None
    peer_rank: int | None
    logical_bytes: int
    flops: int
    work_role: str
    pipeline_index: int | None = None
    buffer_slot: int | None = None
    buffer_family: str | None = None
    packed_value_ref: str | None = None

    @classmethod
    def create(cls, **semantic: object) -> "MoeActionWitness":
        semantic = dict(semantic)
        semantic.setdefault("pipeline_index", None)
        semantic.setdefault("buffer_slot", None)
        semantic.setdefault("buffer_family", None)
        semantic.setdefault("packed_value_ref", None)
        if "work_role" not in semantic:
            raise SchemaError("action requires typed semantic work role", path="moe_action.work_role")
        result = cls(
            MOE_SWIZZLE_ACTION_SCHEMA_VERSION,
            stable_artifact_id("moe_swizzle_action", semantic, schema_version=MOE_SWIZZLE_ACTION_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_action") -> None:
        if self.schema_version != MOE_SWIZZLE_ACTION_SCHEMA_VERSION:
            raise SchemaError("unsupported action schema", path=f"{path}.schema_version")
        validate_uint64(self.rank, f"{path}.rank")
        if type(self.kind) is not SwizzleActionKind:
            raise SchemaError("requires typed action kind", path=f"{path}.kind")
        for name in ("deps", "assignment_refs", "original_action_refs"):
            _refs(getattr(self, name), f"{path}.{name}")
        for name in ("expert_index", "tile_index", "n_block_index", "stage", "pivot_rank", "peer_rank", "pipeline_index", "buffer_slot"):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")
        for name in ("packet_ref", "route_ref"):
            value = getattr(self, name)
            if value is not None:
                validate_nonempty(value, f"{path}.{name}")
        if self.buffer_family is not None:
            validate_nonempty(self.buffer_family, f"{path}.buffer_family")
        if self.packed_value_ref is not None:
            validate_nonempty(self.packed_value_ref, f"{path}.packed_value_ref")
        validate_nonempty(self.work_role, f"{path}.work_role")
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        validate_uint64(self.flops, f"{path}.flops")
        if self.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
            if (
                self.packet_ref is None
                or self.stage is None
                or self.route_ref is None
                or self.peer_rank is None
                or not self.assignment_refs
                or self.logical_bytes == 0
                or self.flops
            ):
                raise SchemaError("transport action lacks packet/stage/route/peer/bytes", path=path)
        elif self.kind is SwizzleActionKind.WAIT:
            if (
                self.packet_ref is None
                or self.stage is None
                or not self.assignment_refs
                or self.logical_bytes
                or self.flops
                or self.route_ref is not None
                or self.peer_rank is not None
            ):
                raise SchemaError("WAIT lacks exact packet/stage/assignment closure", path=path)
        elif self.kind is SwizzleActionKind.COMP:
            if (
                self.expert_index is None
                or self.tile_index is None
                or not self.assignment_refs
                or self.flops == 0
                or self.logical_bytes
                or self.packet_ref is not None
                or self.stage is not None
                or self.pivot_rank is not None
                or self.peer_rank is not None
                or self.route_ref is not None
            ):
                raise SchemaError("compute action lacks expert/tile/FLOP closure", path=path)
        elif self.kind is SwizzleActionKind.SWIGLU:
            if (
                self.expert_index is None
                or self.tile_index is None
                or self.pipeline_index is None
                or self.buffer_slot is None
                or self.buffer_family != "dispatch_operand"
                or self.packed_value_ref is None
                or not self.deps
                or not self.assignment_refs
                or not self.original_action_refs
                or self.logical_bytes == 0
                or self.flops
                or self.work_role != "swiglu"
                or any(
                    value is not None
                    for value in (
                        self.n_block_index, self.packet_ref, self.stage,
                        self.pivot_rank, self.route_ref, self.peer_rank,
                    )
                )
            ):
                raise SchemaError("SWIGLU lacks typed grouped-work closure", path=path)
        elif self.kind is SwizzleActionKind.LOCAL_COPY:
            if (
                self.packed_value_ref is None
                or not self.deps
                or not self.assignment_refs
                or self.logical_bytes == 0
                or self.flops
                or any(
                    value is not None
                    for value in (
                        self.packet_ref, self.stage, self.pivot_rank,
                        self.route_ref, self.peer_rank,
                    )
                )
            ):
                raise SchemaError("LOCAL_COPY lacks typed dependency/byte closure", path=path)
        elif self.kind is SwizzleActionKind.BARRIER:
            if (
                not self.deps
                or self.assignment_refs
                or self.logical_bytes
                or self.flops
                or any(
                    value is not None
                    for value in (
                        self.expert_index, self.tile_index, self.n_block_index,
                        self.packet_ref, self.stage, self.pivot_rank,
                        self.route_ref, self.peer_rank,
                    )
                )
            ):
                raise SchemaError("BARRIER lacks typed dependency/control closure", path=path)
        elif self.peer_rank is not None or self.route_ref is not None or self.packet_ref is not None:
            raise SchemaError("non-transport action cannot carry peer/route", path=path)
        if self.kind not in (SwizzleActionKind.LOCAL_COPY, SwizzleActionKind.SWIGLU) and self.packed_value_ref is not None:
            raise SchemaError("only packing actions may carry a packed value ref", path=f"{path}.packed_value_ref")
        _stable(self, "moe_swizzle_action", MOE_SWIZZLE_ACTION_SCHEMA_VERSION, path)


@dataclass(frozen=True, slots=True)
class MoeRankProgram:
    rank: int
    actions: tuple[MoeActionWitness, ...]

    def validate(self, path: str = "moe_rank_program") -> None:
        validate_uint64(self.rank, f"{path}.rank")
        if type(self.actions) is not tuple or not self.actions:
            raise SchemaError("executable rank program cannot be empty", path=f"{path}.actions")
        for index, action in enumerate(self.actions):
            action.validate(f"{path}.actions[{index}]")
            if action.rank != self.rank:
                raise SchemaError("action belongs to another rank", path=f"{path}.actions[{index}]")


@dataclass(frozen=True, slots=True)
class MoeScenarioResourceEstimate:
    scenario: MoeTrafficScenarioKind
    estimated_cycles: float
    lower_cycles: float
    upper_cycles: float
    per_expert_compute_cycles: tuple[float, ...]
    per_link_bytes: tuple[tuple[str, int], ...]
    per_pivot_bytes: tuple[int, ...]
    max_endpoint_sessions: int

    def validate(self, path: str = "moe_scenario_resource_estimate") -> None:
        if type(self.scenario) is not MoeTrafficScenarioKind:
            raise SchemaError("requires typed traffic scenario", path=f"{path}.scenario")
        for name in ("estimated_cycles", "lower_cycles", "upper_cycles"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise SchemaError("scenario cycles must be finite non-negative", path=f"{path}.{name}")
        if not self.lower_cycles <= self.estimated_cycles <= self.upper_cycles:
            raise SchemaError("scenario cost interval is invalid", path=path)
        if not self.per_expert_compute_cycles or any(
            type(item) is not float or not math.isfinite(item) or item < 0.0
            for item in self.per_expert_compute_cycles
        ):
            raise SchemaError("per-expert cycles are invalid", path=f"{path}.per_expert_compute_cycles")
        refs = tuple(ref for ref, _ in self.per_link_bytes)
        if refs != tuple(sorted(set(refs))):
            raise SchemaError("per-link bytes must be canonical unique", path=f"{path}.per_link_bytes")
        for index, (ref, value) in enumerate(self.per_link_bytes):
            validate_nonempty(ref, f"{path}.per_link_bytes[{index}].ref")
            validate_uint64(value, f"{path}.per_link_bytes[{index}].bytes")
        for index, value in enumerate(self.per_pivot_bytes):
            validate_uint64(value, f"{path}.per_pivot_bytes[{index}]")
        validate_uint64(self.max_endpoint_sessions, f"{path}.max_endpoint_sessions")


@dataclass(frozen=True, slots=True)
class MoeSwizzleCost:
    schema_version: str
    id: str
    scenario: MoeTrafficScenarioKind
    estimated_cycles: float
    critical_path_cycles: float
    lower_cycles: float
    upper_cycles: float
    prologue_cycles: float
    steady_state_cycles: float
    epilogue_cycles: float
    logical_payload_bytes: int
    transported_byte_hops: int
    expert_gemm_flops: int
    region_boundary_output_bytes: int
    packet_count: int
    descriptor_count: int
    event_count: int
    max_inflight: int
    max_link_utilization: float
    max_dte_utilization: float
    compute_idle_cycles: float
    communication_idle_cycles: float
    sram_high_water_bytes: int
    scenario_estimates: tuple[MoeScenarioResourceEstimate, ...]
    group_gemm_setup_cycles: float
    swiglu_group_cycles: float
    dte_launch_cycles: float
    dte_sync_cycles: float
    dte_hop_cycles: float
    sram_lifecycle_cycles: float
    event_control_cycles: float
    endpoint_session_cycles: float
    physical_root_count: int
    evidence_scale_name: str
    evidence_scale_role: MoeSwizzleScaleRole
    calibration_status: MoeCalibrationStatus
    calibrated: bool
    calibration_profile: MoeSwizzleCalibrationProfile | None = None
    calibration_profile_ref: str | None = None
    calibration_sample_digest: str | None = None
    calibration_tool_sha256: str | None = None
    calibration_hardware_sha256: str | None = None
    calibration_simulation_sha256: str | None = None
    calibration_mapping_sha256: str | None = None

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleCost":
        semantic = dict(semantic)
        semantic.setdefault("calibration_profile", None)
        semantic.setdefault("calibration_profile_ref", None)
        semantic.setdefault("calibration_sample_digest", None)
        semantic.setdefault("calibration_tool_sha256", None)
        semantic.setdefault("calibration_hardware_sha256", None)
        semantic.setdefault("calibration_simulation_sha256", None)
        semantic.setdefault("calibration_mapping_sha256", None)
        result = cls(
            MOE_SWIZZLE_COST_SCHEMA_VERSION,
            stable_artifact_id("moe_swizzle_cost", semantic, schema_version=MOE_SWIZZLE_COST_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_cost") -> None:
        if self.schema_version != MOE_SWIZZLE_COST_SCHEMA_VERSION or type(self.scenario) is not MoeTrafficScenarioKind:
            raise SchemaError("unsupported cost schema/scenario", path=path)
        for name in (
            "estimated_cycles", "critical_path_cycles", "lower_cycles", "upper_cycles", "prologue_cycles",
            "steady_state_cycles", "epilogue_cycles", "max_link_utilization",
            "max_dte_utilization", "compute_idle_cycles", "communication_idle_cycles",
            "group_gemm_setup_cycles", "swiglu_group_cycles", "dte_launch_cycles", "dte_sync_cycles",
            "dte_hop_cycles", "sram_lifecycle_cycles", "event_control_cycles",
            "endpoint_session_cycles",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise SchemaError("cost values must be finite non-negative floats", path=f"{path}.{name}")
        if not math.isclose(self.critical_path_cycles, self.estimated_cycles, rel_tol=1e-9, abs_tol=1e-9):
            raise SchemaError("critical path must equal total estimate", path=f"{path}.critical_path_cycles")
        if not self.lower_cycles <= self.estimated_cycles <= self.upper_cycles or not math.isclose(
            self.estimated_cycles,
            self.prologue_cycles + self.steady_state_cycles + self.epilogue_cycles,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise SchemaError("cost interval/phase closure failed", path=path)
        if self.max_link_utilization > 1.0 or self.max_dte_utilization > 1.0:
            raise SchemaError("utilization cannot exceed one", path=path)
        for name in (
            "logical_payload_bytes", "transported_byte_hops", "expert_gemm_flops",
            "region_boundary_output_bytes", "packet_count", "descriptor_count", "event_count",
            "max_inflight", "sram_high_water_bytes", "physical_root_count",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if not self.scenario_estimates:
            raise SchemaError("cost requires typed scenario estimates", path=f"{path}.scenario_estimates")
        kinds = []
        for index, estimate in enumerate(self.scenario_estimates):
            estimate.validate(f"{path}.scenario_estimates[{index}]")
            kinds.append(estimate.scenario)
        if tuple(kinds) != tuple(sorted(MoeTrafficScenarioKind, key=lambda item: item.value)):
            raise SchemaError("cost must cover canonical ACTUAL/P95/CAPACITY", path=f"{path}.scenario_estimates")
        validate_nonempty(self.evidence_scale_name, f"{path}.evidence_scale_name")
        if type(self.evidence_scale_role) is not MoeSwizzleScaleRole:
            raise SchemaError("cost requires typed evidence split", path=f"{path}.evidence_scale_role")
        if type(self.calibration_status) is not MoeCalibrationStatus:
            raise SchemaError("cost requires typed calibration status", path=f"{path}.calibration_status")
        if type(self.calibrated) is not bool:
            raise SchemaError("calibrated must be bool", path=f"{path}.calibrated")
        provenance = (
            self.calibration_profile_ref,
            self.calibration_sample_digest,
            self.calibration_tool_sha256,
            self.calibration_hardware_sha256,
            self.calibration_simulation_sha256,
            self.calibration_mapping_sha256,
        )
        if self.calibration_profile is None:
            if (
                self.calibrated
                or self.calibration_status is not MoeCalibrationStatus.PROVISIONAL
                or any(item is not None for item in provenance)
            ):
                raise SchemaError("None profile must remain PROVISIONAL", path=f"{path}.calibration_profile")
        else:
            self.calibration_profile.validate(f"{path}.calibration_profile")
            expected = (
                self.calibration_profile.id,
                canonical_digest(self.calibration_profile.samples),
                self.calibration_profile.tool_sha256,
                self.calibration_profile.hardware_sha256,
                self.calibration_profile.simulation_sha256,
                self.calibration_profile.mapping_sha256,
            )
            if (
                self.calibration_profile.status is not MoeCalibrationStatus.MEASURED
                or not self.calibrated
                or self.calibration_status is not MoeCalibrationStatus.MEASURED
                or provenance != expected
            ):
                raise SchemaError("MEASURED profile/provenance closure failed", path=f"{path}.calibration_profile")
        _stable(self, "moe_swizzle_cost", MOE_SWIZZLE_COST_SCHEMA_VERSION, path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleCandidate:
    schema_version: str
    id: str
    problem_ref: str
    pattern: FusionPattern
    algorithm: SwizzleAlgorithm
    packetization: tuple[MoePacketWitness, ...]
    tile_schedule: tuple[MoeTileWitness, ...]
    rank_programs: tuple[MoeRankProgram, ...]
    expert_wave_count: int
    token_block_size: int
    output_column_block_size: int
    compute_output_block_count: int
    transport_output_block_count: int
    unroll_degree: int
    double_buffer: bool
    compute_core_fraction: float
    communication_core_fraction: float
    original_action_refs: tuple[str, ...]
    cost: MoeSwizzleCost

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleCandidate":
        result = cls(
            MOE_SWIZZLE_CANDIDATE_SCHEMA_VERSION,
            stable_artifact_id("moe_swizzle_candidate", semantic, schema_version=MOE_SWIZZLE_CANDIDATE_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_candidate") -> None:
        if self.schema_version != MOE_SWIZZLE_CANDIDATE_SCHEMA_VERSION:
            raise SchemaError("unsupported candidate schema", path=f"{path}.schema_version")
        validate_nonempty(self.problem_ref, f"{path}.problem_ref")
        _validate_moe_pattern(self.pattern, f"{path}.pattern")
        if type(self.algorithm) is not SwizzleAlgorithm or self.algorithm not in _MOE_ALGORITHMS:
            raise SchemaError("requires a MoE algorithm", path=f"{path}.algorithm")
        for index, packet in enumerate(self.packetization):
            packet.validate(f"{path}.packetization[{index}]")
        packets = validate_unique_ids(self.packetization, f"{path}.packetization")
        for index, tile in enumerate(self.tile_schedule):
            tile.validate(f"{path}.tile_schedule[{index}]")
            if any(ref not in packets for ref in tile.required_packet_refs):
                raise SchemaError("tile requires unknown packet", path=f"{path}.tile_schedule[{index}]")
        validate_unique_ids(self.tile_schedule, f"{path}.tile_schedule")
        if tuple(program.rank for program in self.rank_programs) != tuple(range(len(self.rank_programs))):
            raise SchemaError("rank programs must be dense canonical ranks", path=f"{path}.rank_programs")
        actions = ()
        for index, program in enumerate(self.rank_programs):
            program.validate(f"{path}.rank_programs[{index}]")
            actions += program.actions
        action_index = validate_dependency_dag(actions, f"{path}.rank_programs.actions")
        del action_index
        assignment_refs = {item.assignment_ref for packet in self.packetization for item in packet.slices}
        if (
            self.algorithm is not SwizzleAlgorithm.UNFUSED
            and any(
                ref not in assignment_refs
                for action in actions
                for ref in action.assignment_refs
                if action.kind not in (
                    SwizzleActionKind.COMP, SwizzleActionKind.SWIGLU,
                    SwizzleActionKind.LOCAL_COPY,
                )
            )
        ):
            raise SchemaError("action assignment is outside packetization", path=f"{path}.rank_programs")
        for name in (
            "expert_wave_count", "token_block_size",
            "output_column_block_size", "compute_output_block_count",
            "transport_output_block_count",
            "unroll_degree",
        ):
            _positive(getattr(self, name), f"{path}.{name}")
        if self.token_block_size > 8 or (
            self.token_block_size & (self.token_block_size - 1)
        ):
            raise SchemaError(
                "token M-block must be a supported power of two <= 8",
                path=f"{path}.token_block_size",
            )
        if self.unroll_degree not in (1, 2) or type(self.double_buffer) is not bool:
            raise SchemaError("unroll/double-buffer configuration is invalid", path=path)
        if self.double_buffer != (self.unroll_degree == 2):
            raise SchemaError("double buffer must exactly witness unroll two", path=path)
        invalid_block_contract = (
            (
                self.compute_output_block_count != 1
                or self.transport_output_block_count != 1
            )
            if self.algorithm is SwizzleAlgorithm.UNFUSED
            else (
                self.compute_output_block_count != 1
                or self.transport_output_block_count != 1
            )
            if self.pattern is FusionPattern.MOE_DISPATCH_GEMM
            else (
                self.compute_output_block_count != 1
                or self.transport_output_block_count not in (1, 2)
            )
        )
        if invalid_block_contract:
            raise SchemaError(
                "output-column N-block count is outside the exact pattern/algorithm contract",
                path=f"{path}.compute_output_block_count",
            )
        for name in ("compute_core_fraction", "communication_core_fraction"):
            value = getattr(self, name)
            if type(value) is not float or not 0.0 < value <= 1.0:
                raise SchemaError("core fractions must be in (0,1]", path=f"{path}.{name}")
        if self.compute_core_fraction + self.communication_core_fraction > 1.0:
            raise SchemaError("core fractions exceed one", path=path)
        _refs(self.original_action_refs, f"{path}.original_action_refs", nonempty=True)
        self.cost.validate(f"{path}.cost")
        if self.algorithm is SwizzleAlgorithm.UNFUSED and (self.packetization or self.tile_schedule):
            raise SchemaError("unfused baseline keeps original actions, not fused packets/tiles", path=path)
        if self.algorithm is not SwizzleAlgorithm.UNFUSED and (not self.packetization or not self.tile_schedule):
            raise SchemaError("fused candidate requires packet and tile witnesses", path=path)
        action_original_refs = tuple(
            ref for action in actions for ref in action.original_action_refs
        )
        if (
            len(action_original_refs) != len(set(action_original_refs))
            or set(action_original_refs) != set(self.original_action_refs)
        ):
            raise SchemaError(
                "candidate/action original provenance is not exact",
                path=f"{path}.original_action_refs",
            )
        _stable(self, "moe_swizzle_candidate", MOE_SWIZZLE_CANDIDATE_SCHEMA_VERSION, path)

    def validate_against(
        self, problem: MoeSwizzleProblem, path: str = "moe_candidate"
    ) -> None:
        self.validate(path)
        problem.validate(f"{path}.problem")
        if (
            self.problem_ref != problem.id
            or self.pattern is not problem.region.pattern
            or self.algorithm not in problem.allowed_algorithms
        ):
            raise SchemaError("candidate/problem identity mismatch", path=path)
        actions = tuple(
            action for program in self.rank_programs for action in program.actions
        )
        action_index = {action.id: action for action in actions}
        traffic = problem.region.semantic_witness.traffic
        assignment_index = {item.id: item for item in traffic.assignments}
        if self.algorithm is SwizzleAlgorithm.UNFUSED:
            expected_family = (
                "dispatch_operand"
                if self.pattern is FusionPattern.MOE_DISPATCH_GEMM
                else "combine_output"
            )
            for action in actions:
                remote = any(
                    assignment_index[ref].source_rank
                    != assignment_index[ref].expert_rank
                    for ref in action.assignment_refs
                )
                owns_dynamic_root = remote and (
                    action.kind in (SwizzleActionKind.RECV, SwizzleActionKind.COMP)
                    if self.pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else action.kind in (SwizzleActionKind.COMP, SwizzleActionKind.SEND)
                )
                if owns_dynamic_root:
                    if (
                        action.pipeline_index != action.tile_index
                        or action.buffer_slot != 0
                        or action.buffer_family != expected_family
                    ):
                        raise SchemaError(
                            "UNFUSED dynamic root witness is not exact",
                            path=f"{path}.rank_programs",
                        )
                elif action.kind is SwizzleActionKind.SWIGLU:
                    if (
                        action.pipeline_index != action.tile_index
                        or action.buffer_slot != 0
                        or action.buffer_family != expected_family
                    ):
                        raise SchemaError(
                            "UNFUSED SWIGLU reader root witness is not exact",
                            path=f"{path}.rank_programs",
                        )
                elif (
                    action.pipeline_index is not None
                    or action.buffer_slot is not None
                    or action.buffer_family is not None
                ):
                    raise SchemaError(
                        "UNFUSED non-owner cannot claim a dynamic root",
                        path=f"{path}.rank_programs",
                    )
        else:
            physical_slots = 2 if self.double_buffer else 1
            expected_family = (
                "dispatch_operand"
                if self.pattern is FusionPattern.MOE_DISPATCH_GEMM
                else "combine_output"
            )
            if any(
                action.pipeline_index is None
                or action.buffer_slot is None
                or action.buffer_family != expected_family
                or action.buffer_slot >= physical_slots
                or action.buffer_slot != action.pipeline_index % physical_slots
                for action in actions
            ):
                raise SchemaError("action pipeline/family/physical-slot mapping is not exact", path=f"{path}.rank_programs")
            for program in self.rank_programs:
                used = {action.buffer_slot for action in program.actions}
                if used != set(range(physical_slots)):
                    raise SchemaError("rank program does not witness every physical slot", path=f"{path}.rank_programs")
            by_packet_kind = {
                (action.packet_ref, action.kind): action
                for action in actions
                if action.packet_ref is not None
            }
            comps_by_tile: dict[int, list[MoeActionWitness]] = {}
            for action in actions:
                if action.kind is SwizzleActionKind.COMP:
                    assert action.tile_index is not None
                    comps_by_tile.setdefault(action.tile_index, []).append(action)
            from .swizzle_moe_placement import (
                build_moe_candidate_action_owner_map,
            )
            action_owners = build_moe_candidate_action_owner_map(
                problem, actions
            )
            domains: dict[
                tuple[int, str, int], dict[int, list[MoeTileWitness]]
            ] = {}
            for tile in self.tile_schedule:
                comp = comps_by_tile[tile.tile_index][0]
                assert comp.buffer_slot is not None and comp.buffer_family is not None
                if any(
                    item.pipeline_index != tile.m_block_index
                    for item in comps_by_tile[tile.tile_index]
                ):
                    raise SchemaError(
                        "COMP pipeline must equal typed M-block index",
                        path=f"{path}.rank_programs",
                    )
                domains.setdefault(
                    (
                        action_owners[comp.id].runtime_core_id,
                        comp.buffer_family,
                        comp.buffer_slot,
                    ), {}
                ).setdefault(tile.m_block_index, []).append(tile)
                if self.pattern is FusionPattern.MOE_DISPATCH_GEMM:
                    pair = comps_by_tile[tile.tile_index]
                    if len(pair) != 2 or pair[0].id in pair[1].deps or pair[1].id in pair[0].deps:
                        raise SchemaError("gate/up readers must remain independent", path=f"{path}.rank_programs")
            for occupants in domains.values():
                ordered = sorted(occupants.items())
                for (_, previous_tiles), (_, current_tiles) in zip(
                    ordered, ordered[1:]
                ):
                    previous_comps = tuple(
                        action
                        for tile in previous_tiles
                        for action in comps_by_tile[tile.tile_index]
                    )
                    current_comps = tuple(
                        action
                        for tile in current_tiles
                        for action in comps_by_tile[tile.tile_index]
                    )
                    if self.pattern is FusionPattern.MOE_DISPATCH_GEMM:
                        required = {
                            action.id for action in actions
                            if action.kind is SwizzleActionKind.SWIGLU
                            and action.tile_index in {
                                tile.tile_index for tile in previous_tiles
                            }
                        }
                        targets = tuple(dict.fromkeys(
                            by_packet_kind[(packet_ref, SwizzleActionKind.RECV)]
                            for tile in current_tiles
                            for packet_ref in tile.required_packet_refs
                        )) or current_comps
                    else:
                        previous_assignment_refs = {
                            ref
                            for tile in previous_tiles
                            for ref in tile.assignment_refs
                        }
                        required = {
                            action.id
                            for action in actions
                            if action.packet_ref is not None
                            and set(action.assignment_refs).intersection(
                                previous_assignment_refs
                            )
                            and action.kind is SwizzleActionKind.SEND
                        }
                        targets = current_comps
                    if any(not required.issubset(set(target.deps)) for target in targets):
                        raise SchemaError("physical slot overwrite lacks all prior readers", path=f"{path}.rank_programs")
        if {ref for action in actions for ref in action.assignment_refs} != set(problem.region.assignment_refs):
            raise SchemaError("candidate assignment work is not exact", path=f"{path}.rank_programs")
        if (
            sum(action.flops for action in actions if action.kind is SwizzleActionKind.COMP)
            != traffic.expert_gemm_flops
            or self.cost.expert_gemm_flops != traffic.expert_gemm_flops
            or self.cost.logical_payload_bytes != traffic.logical_payload_bytes
            or self.cost.region_boundary_output_bytes != traffic.region_boundary_output_bytes
            or self.cost.evidence_scale_name != problem.scale_name
            or self.cost.evidence_scale_role is not problem.scale_role
        ):
            raise SchemaError("candidate work totals differ from problem", path=f"{path}.cost")
        if (
            self.algorithm is SwizzleAlgorithm.UNFUSED
            and sum(
                action.logical_bytes
                for action in actions
                if action.kind is SwizzleActionKind.SEND
            ) != traffic.logical_payload_bytes
        ):
            raise SchemaError("UNFUSED payload bytes are not exact", path=f"{path}.rank_programs")

        # Rebuild every transaction instead of trusting individually typed
        # actions. A packet (or original UNFUSED flow) owns one exact triple.
        packet_index = {item.id: item for item in self.packetization}
        transport = tuple(
            action
            for action in actions
            if action.kind in (
                SwizzleActionKind.SEND,
                SwizzleActionKind.RECV,
                SwizzleActionKind.WAIT,
            )
        )
        grouped: dict[str, list[MoeActionWitness]] = {}
        for action in transport:
            assert action.packet_ref is not None
            grouped.setdefault(action.packet_ref, []).append(action)
        expected_packet_refs = (
            set(packet_index)
            if self.algorithm is not SwizzleAlgorithm.UNFUSED
            else {
                ref
                for assignment in traffic.assignments
                for ref in (
                    assignment.dispatch_flow_ref
                    if self.pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else assignment.combine_flow_ref,
                )
                if ref is not None
            }
        )
        if set(grouped) != expected_packet_refs:
            raise SchemaError("transport packet/flow coverage is not exact", path=f"{path}.rank_programs")
        route_index = {item.id: item for item in traffic.pair_routes}
        for packet_ref in sorted(grouped):
            triple = grouped[packet_ref]
            by_kind = {
                kind: tuple(item for item in triple if item.kind is kind)
                for kind in (
                    SwizzleActionKind.SEND,
                    SwizzleActionKind.RECV,
                    SwizzleActionKind.WAIT,
                )
            }
            if any(len(items) != 1 for items in by_kind.values()) or len(triple) != 3:
                raise SchemaError("packet must own one SEND/RECV/WAIT triple", path=f"{path}.rank_programs")
            send = by_kind[SwizzleActionKind.SEND][0]
            recv = by_kind[SwizzleActionKind.RECV][0]
            wait = by_kind[SwizzleActionKind.WAIT][0]
            if self.algorithm is SwizzleAlgorithm.UNFUSED:
                matches = tuple(
                    assignment
                    for assignment in traffic.assignments
                    if (
                        assignment.dispatch_flow_ref
                        if self.pattern is FusionPattern.MOE_DISPATCH_GEMM
                        else assignment.combine_flow_ref
                    ) == packet_ref
                )
                if len(matches) != 1:
                    raise SchemaError("UNFUSED flow does not name one assignment", path=f"{path}.rank_programs")
                assignment = matches[0]
                assignment_refs = (assignment.id,)
                source_rank, destination_rank = (
                    (assignment.source_rank, assignment.expert_rank)
                    if self.pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else (assignment.expert_rank, assignment.source_rank)
                )
                route_ref = (
                    assignment.dispatch_route_ref
                    if self.pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else assignment.combine_route_ref
                )
                stage = 0
                pivot_rank = None
                logical_bytes = assignment.payload_bytes
            else:
                packet = packet_index[packet_ref]
                assignment_refs = tuple(item.assignment_ref for item in packet.slices)
                if any(ref not in assignment_index for ref in assignment_refs):
                    raise SchemaError("packet slice names unknown assignment", path=f"{path}.packetization")
                source_rank, destination_rank = packet.source_rank, packet.destination_rank
                route_ref = packet.route_ref
                stage, pivot_rank = packet.stage, packet.pivot_rank
                logical_bytes = packet.logical_bytes
            route = route_index.get(route_ref or "")
            if (
                route is None
                or route.source_rank != source_rank
                or route.destination_rank != destination_rank
                or send.rank != source_rank
                or send.peer_rank != destination_rank
                or recv.rank != destination_rank
                or recv.peer_rank != source_rank
                or wait.rank != destination_rank
                or send.route_ref != route_ref
                or recv.route_ref != route_ref
                or (send.stage, recv.stage, wait.stage) != (stage, stage, stage)
                or (send.pivot_rank, recv.pivot_rank, wait.pivot_rank)
                != (pivot_rank, pivot_rank, pivot_rank)
                or send.assignment_refs != assignment_refs
                or recv.assignment_refs != assignment_refs
                or wait.assignment_refs != assignment_refs
                or send.logical_bytes != logical_bytes
                or recv.logical_bytes != logical_bytes
                or wait.logical_bytes != 0
                or set(wait.deps) != {send.id, recv.id}
                or any(ref not in action_index for ref in wait.deps)
            ):
                raise SchemaError("transport triple disagrees with packet/route witness", path=f"{path}.rank_programs")

        comp_assignments = tuple(
            ref
            for action in actions
            if action.kind is SwizzleActionKind.COMP
            for ref in action.assignment_refs
        )
        expected_comp_multiplicity = (
            2
            if self.pattern is FusionPattern.MOE_DISPATCH_GEMM
            else self.compute_output_block_count
            if self.algorithm is not SwizzleAlgorithm.UNFUSED
            else 1
        )
        if Counter(comp_assignments) != Counter(
            {ref: expected_comp_multiplicity for ref in problem.region.assignment_refs}
        ):
            raise SchemaError("compute tile/role/block coverage is not exact", path=f"{path}.rank_programs")

        local_copies = tuple(
            action for action in actions
            if action.kind is SwizzleActionKind.LOCAL_COPY
        )
        if local_copies:
            raise SchemaError(
                "grouped SWIGLU forbids packing LOCAL_COPY actions",
                path=f"{path}.rank_programs",
            )
        swiglus = tuple(
            action for action in actions
            if action.kind is SwizzleActionKind.SWIGLU
        )
        if self.pattern is FusionPattern.MOE_GEMM_COMBINE:
            if swiglus:
                raise SchemaError(
                    "Combine cannot claim Dispatch SWIGLU work",
                    path=f"{path}.rank_programs",
                )
        else:
            if Counter(
                ref for action in swiglus for ref in action.assignment_refs
            ) != Counter({ref: 1 for ref in problem.region.assignment_refs}):
                raise SchemaError(
                    "Dispatch SWIGLU assignment coverage is not exact",
                    path=f"{path}.rank_programs",
                )
            for swiglu in swiglus:
                producers = tuple(
                    action for action in actions
                    if action.kind is SwizzleActionKind.COMP
                    and action.tile_index == swiglu.tile_index
                    and action.work_role in ("gate", "up")
                )
                expected_originals = tuple(
                    assignment_index[ref].swiglu_action_ref
                    for ref in swiglu.assignment_refs
                )
                gemm = traffic.expert_gemms[swiglu.expert_index]
                if (
                    len(producers) != 2
                    or {item.work_role for item in producers} != {"gate", "up"}
                    or set(swiglu.deps) != {item.id for item in producers}
                    or any(item.assignment_refs != swiglu.assignment_refs for item in producers)
                    or swiglu.rank != swiglu.expert_index
                    or swiglu.n_block_index is not None
                    or swiglu.pipeline_index is None
                    or swiglu.buffer_slot != swiglu.pipeline_index % (2 if self.double_buffer else 1)
                    or swiglu.buffer_family != "dispatch_operand"
                    or swiglu.logical_bytes != len(swiglu.assignment_refs) * gemm.n * 2
                    or swiglu.original_action_refs != expected_originals
                    or swiglu.packed_value_ref != f"moe.swiglu.{swiglu.tile_index}"
                ):
                    raise SchemaError(
                        "Dispatch grouped SWIGLU provenance/shape/dependency closure is not exact",
                        path=f"{path}.rank_programs",
                    )

        if self.algorithm is not SwizzleAlgorithm.UNFUSED:
            combine = self.pattern is FusionPattern.MOE_GEMM_COMBINE
            compute_blocks = (
                tuple(range(self.compute_output_block_count))
                if combine else (None,)
            )
            transport_blocks = (
                (None,)
                if not combine or self.transport_output_block_count == 1
                else tuple(range(self.transport_output_block_count))
            )
            pivot_index = {
                (source, destination): pivot
                for source, destination, pivot in problem.topology.pivot_by_pair
            }
            route_by_pair = {
                (route.source_rank, route.destination_rank): route.id
                for route in traffic.pair_routes
            }
            slices_by_key: dict[tuple[str, int | None], list[tuple[MoePacketWitness, MoePacketSlice]]] = {
                (ref, block): []
                for ref in problem.region.assignment_refs
                for block in transport_blocks
            }
            for packet in self.packetization:
                experts = {
                    assignment_index[item.assignment_ref].expert_index
                    for item in packet.slices
                }
                blocks = {item.n_block_index for item in packet.slices}
                if len(experts) != 1 or len(blocks) != 1 or next(iter(blocks)) not in transport_blocks:
                    raise SchemaError("packet crosses expert or N-block domain", path=f"{path}.packetization")
                for item in packet.slices:
                    slices_by_key[(item.assignment_ref, item.n_block_index)].append((packet, item))
            final_packet_by_key: dict[tuple[str, int | None], str] = {}
            for assignment in traffic.assignments:
                local = assignment.source_rank == assignment.expert_rank
                source_rank, destination_rank = (
                    (assignment.source_rank, assignment.expert_rank)
                    if self.pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else (assignment.expert_rank, assignment.source_rank)
                )
                whole_source = (
                    assignment.token_index * assignment.payload_bytes
                    if not combine
                    else assignment.contributor_ordinal * assignment.payload_bytes
                )
                whole_destination = (
                    assignment.contributor_ordinal * assignment.payload_bytes
                    if not combine
                    else assignment.token_index * assignment.payload_bytes
                )
                block_bytes = assignment.payload_bytes // len(transport_blocks)
                covered = []
                for block in transport_blocks:
                    entries = slices_by_key[(assignment.id, block)]
                    if local:
                        if entries:
                            raise SchemaError("local assignment cannot create DTE packets", path=f"{path}.packetization")
                        continue
                    if self.algorithm is SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A:
                        expected_segments = ((0, source_rank, destination_rank),)
                        pivot_rank = None
                    else:
                        if not problem.topology.complete_rectangle:
                            raise SchemaError("Comet candidate requires a complete rectangle", path=path)
                        pivot_rank = pivot_index[(source_rank, destination_rank)]
                        expected_segments = tuple(
                            segment
                            for segment in (
                                (0, source_rank, pivot_rank),
                                (1, pivot_rank, destination_rank),
                            )
                            if segment[1] != segment[2]
                        )
                    actual_segments = tuple(sorted(
                        (packet.stage, packet.source_rank, packet.destination_rank)
                        for packet, _ in entries
                    ))
                    if actual_segments != tuple(sorted(expected_segments)):
                        raise SchemaError("packet stages do not reconstruct personalized N-block route", path=f"{path}.packetization")
                    by_stage = {packet.stage: (packet, item) for packet, item in entries}
                    block_offset = 0 if block is None else block * block_bytes
                    source_offset = whole_source + block_offset
                    destination_offset = whole_destination + block_offset
                    for stage, expected_source, expected_destination in expected_segments:
                        packet, item = by_stage[stage]
                        expected_source_offset = (
                            destination_offset
                            if stage == 1 and expected_source != source_rank
                            else source_offset
                        )
                        if (
                            packet.pivot_rank != pivot_rank
                            or packet.route_ref != route_by_pair[(expected_source, expected_destination)]
                            or item.bytes != block_bytes
                            or item.source_offset_bytes != expected_source_offset
                            or item.destination_offset_bytes != destination_offset
                        ):
                            raise SchemaError("packet slice/stage/pivot geometry is not exact", path=f"{path}.packetization")
                    final_packet = by_stage[max(by_stage)][0].id
                    if combine and self.transport_output_block_count == 1:
                        for compute_block in compute_blocks:
                            final_packet_by_key[(assignment.id, compute_block)] = final_packet
                    else:
                        final_packet_by_key[(assignment.id, block)] = final_packet
                    covered.append((block_offset, block_offset + block_bytes))
                if not local and tuple(covered) != tuple(
                    (index * block_bytes, (index + 1) * block_bytes)
                    for index in range(len(transport_blocks))
                ):
                    raise SchemaError("N-block slices do not exactly cover payload", path=f"{path}.packetization")

            tile_keys = Counter(
                (ref, tile.n_block_index)
                for tile in self.tile_schedule
                for ref in tile.assignment_refs
            )
            if tile_keys != Counter(
                {(ref, block): 1 for ref in problem.region.assignment_refs for block in compute_blocks}
            ):
                raise SchemaError("tile assignment/N-block coverage is not exact", path=f"{path}.tile_schedule")
            output_counts = Counter(
                ref for tile in self.tile_schedule for ref in tile.output_value_refs
            )
            expected_output_count = (
                self.compute_output_block_count if combine else 1
            )
            if output_counts != Counter(
                {ref: expected_output_count for ref in problem.region.boundary_output_refs}
            ):
                raise SchemaError("tile output slice coverage is not exact", path=f"{path}.tile_schedule")
            block_columns = None
            comp_by_assignment_block: dict[tuple[str, int], str] = {}
            m_block_occupants: dict[
                tuple[int, int], list[MoeTileWitness]
            ] = {}
            for tile in self.tile_schedule:
                assignments = tuple(assignment_index[ref] for ref in tile.assignment_refs)
                if len({item.expert_index for item in assignments}) != 1:
                    raise SchemaError("tile crosses experts", path=f"{path}.tile_schedule")
                expected_required = tuple(dict.fromkeys(
                    final_packet_by_key[(item.id, tile.n_block_index)]
                    for item in assignments
                    if (item.id, tile.n_block_index) in final_packet_by_key
                ))
                transport_block = (
                    None
                    if combine and self.transport_output_block_count == 1
                    else tile.n_block_index
                )
                packet_counts = [
                    len(slices_by_key[(item.id, transport_block)])
                    for item in assignments
                ]
                if not packet_counts or max(packet_counts) == 0:
                    expected_arrival = 0
                elif self.algorithm is SwizzleAlgorithm.COMET_MESH_PERSONALIZED_A2A:
                    expected_arrival = max(packet_counts)
                else:
                    expected_arrival = max(
                        len(route_index[
                            item.dispatch_route_ref if not combine else item.combine_route_ref
                        ].die_path) - 1
                        for item in assignments
                    )
                if combine:
                    block_columns = self.output_column_block_size
                    expected_offset = tile.n_block_index * block_columns
                    expected_extent = block_columns
                else:
                    expected_offset = 0
                    expected_extent = self.output_column_block_size
                if (
                    tile.expert_index != assignments[0].expert_index
                    or tile.m_block_size > self.token_block_size
                    or tile.split_axis is not problem.region.semantic_witness.split_axis
                    or tile.required_packet_refs != expected_required
                    or tile.arrival_class != expected_arrival
                    or tile.n_block_index not in compute_blocks
                    or tile.output_column_offset != expected_offset
                    or tile.output_column_extent != expected_extent
                ):
                    raise SchemaError("tile arrival/packet/expert/block semantics are not exact", path=f"{path}.tile_schedule")
                tile_comps = comps_by_tile.get(tile.tile_index, [])
                expected_roles = (
                    {"gate", "up"} if not combine else {"down"}
                )
                gemm = traffic.expert_gemms[tile.expert_index]
                expected_n = (
                    gemm.n
                    if not combine
                    else gemm.n // self.compute_output_block_count
                )
                expected_flops = 2 * tile.m_block_size * expected_n * gemm.k
                if (
                    {item.work_role for item in tile_comps} != expected_roles
                    or any(
                        item.assignment_refs != tile.assignment_refs
                        or item.expert_index != tile.expert_index
                        or item.tile_index != tile.tile_index
                        or item.n_block_index != tile.n_block_index
                        or item.pipeline_index != tile.m_block_index
                        or item.flops != expected_flops
                        for item in tile_comps
                    )
                ):
                    raise SchemaError(
                        "M-block COMP role/shape/FLOP closure is not exact",
                        path=f"{path}.rank_programs",
                    )
                if combine:
                    for assignment_ref in tile.assignment_refs:
                        comp_by_assignment_block[(
                            assignment_ref, tile.n_block_index
                        )] = tile_comps[0].id
                m_block_occupants.setdefault(
                    (tile.expert_index, tile.m_block_index), []
                ).append(tile)
            expected_n_blocks = (
                self.compute_output_block_count if combine else 1
            )
            indices_by_expert: dict[int, set[int]] = {}
            canonical_groups: dict[
                tuple[int, int], list[tuple[int, tuple[str, ...]]]
            ] = {}
            for (expert, m_block_index), occupant_tiles in m_block_occupants.items():
                first = occupant_tiles[0]
                if (
                    len(occupant_tiles) != expected_n_blocks
                    or {item.n_block_index for item in occupant_tiles}
                    != set(compute_blocks)
                    or any(
                        item.assignment_refs != first.assignment_refs
                        or item.arrival_class != first.arrival_class
                        or item.m_block_size != first.m_block_size
                        for item in occupant_tiles
                    )
                ):
                    raise SchemaError(
                        "one M-block occupant must close all exact N-blocks",
                        path=f"{path}.tile_schedule",
                    )
                indices_by_expert.setdefault(expert, set()).add(m_block_index)
                canonical_groups.setdefault(
                    (first.arrival_class, expert), []
                ).append((m_block_index, first.assignment_refs))
            for expert, indices in indices_by_expert.items():
                if indices != set(range(len(indices))):
                    raise SchemaError(
                        "expert M-block indices must be dense canonical",
                        path=f"{path}.tile_schedule",
                    )
            for group_tiles in canonical_groups.values():
                group_tiles.sort()
                assignment_refs = tuple(
                    ref for _, refs in group_tiles for ref in refs
                )
                expected_refs = tuple(
                    item.id
                    for item in sorted(
                        (assignment_index[ref] for ref in assignment_refs),
                        key=lambda item: (
                            item.token_index, item.source_rank, item.id,
                        ),
                    )
                )
                expected_chunks = tuple(
                    expected_refs[index:index + self.token_block_size]
                    for index in range(0, len(expected_refs), self.token_block_size)
                )
                if tuple(refs for _, refs in group_tiles) != expected_chunks:
                    raise SchemaError(
                        "M-block assignment chunks are not canonical",
                        path=f"{path}.tile_schedule",
                    )
            if combine and block_columns * len(compute_blocks) * 2 != assignment_index[problem.region.assignment_refs[0]].payload_bytes:
                raise SchemaError("Combine N-block columns do not close payload bytes", path=f"{path}.output_column_block_size")
            if combine:
                for packet in self.packetization:
                    send = next(
                        item
                        for item in grouped[packet.id]
                        if item.kind is SwizzleActionKind.SEND
                    )
                    packet_block = packet.slices[0].n_block_index
                    first_stage = (
                        packet.source_rank
                        == assignment_index[packet.slices[0].assignment_ref].expert_rank
                    )
                    expected_comp_deps = (
                        {
                            comp_by_assignment_block[(slice_.assignment_ref, block)]
                            for slice_ in packet.slices
                            for block in (
                                compute_blocks
                                if packet_block is None or self.compute_output_block_count == 1
                                else (packet_block,)
                            )
                        }
                        if first_stage
                        else set()
                    )
                    actual_comp_deps = {
                        ref
                        for ref in send.deps
                        if action_index[ref].kind is SwizzleActionKind.COMP
                    }
                    if actual_comp_deps != expected_comp_deps:
                        raise SchemaError(
                            "Combine transport does not depend on its exact full-N producer",
                            path=f"{path}.rank_programs",
                        )



@dataclass(frozen=True, slots=True)
class MoeSwizzleWorkloadSelection:
    schema_version: str
    producer_pass: str
    id: str
    source_dispatch_decision_id: str
    source_combine_decision_id: str
    dispatch_candidate_costs: tuple[tuple[str, float], ...]
    combine_candidate_costs: tuple[tuple[str, float], ...]
    dispatch_candidate_lifecycle_costs: tuple[tuple[str, float], ...]
    combine_candidate_lifecycle_costs: tuple[tuple[str, float], ...]
    dispatch_candidate_lifecycle_floors: tuple[MoeCandidateCoreLifecycleFloor, ...]
    combine_candidate_lifecycle_floors: tuple[MoeCandidateCoreLifecycleFloor, ...]
    lifecycle_fixed_cycles: tuple[float, float, float, float] | None
    calibration_profile_ref: str | None
    calibration_sample_digest: str | None
    baseline_pair_ref: tuple[str, str]
    lower_bound_pair_ref: tuple[str, str]
    frontier_pair_refs: tuple[tuple[str, str], ...]
    frontier_stop_lower_bound: float | None
    pair_feasibilities: tuple[MoeWholePairFeasibility, ...]
    ranked_pair_refs: tuple[tuple[str, str], ...]
    selected_dispatch_candidate_ref: str
    selected_combine_candidate_ref: str
    baseline_estimated_cycles: float
    selected_estimated_cycles: float
    decision_reason: SwizzleDecisionReason
    performance_complete: bool

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleWorkloadSelection":
        result = cls(
            MOE_SWIZZLE_WORKLOAD_SELECTION_SCHEMA_VERSION,
            "select_moe_swizzle_workload_deployment",
            stable_artifact_id(
                "moe_swizzle_workload_selection", semantic,
                schema_version=MOE_SWIZZLE_WORKLOAD_SELECTION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_swizzle_workload_selection") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_WORKLOAD_SELECTION_SCHEMA_VERSION
            or self.producer_pass != "select_moe_swizzle_workload_deployment"
        ):
            raise SchemaError("unsupported workload selection schema/producer", path=path)
        for name in (
            "source_dispatch_decision_id", "source_combine_decision_id",
            "selected_dispatch_candidate_ref", "selected_combine_candidate_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        def validate_costs(values, field):
            if type(values) is not tuple or not values:
                raise SchemaError("selection requires full candidate costs", path=f"{path}.{field}")
            refs = tuple(ref for ref, _ in values)
            if refs != tuple(sorted(set(refs))):
                raise SchemaError("candidate costs must be canonical unique", path=f"{path}.{field}")
            for index, (ref, cycles) in enumerate(values):
                validate_nonempty(ref, f"{path}.{field}[{index}].ref")
                if type(cycles) is not float or not math.isfinite(cycles) or cycles <= 0.0:
                    raise SchemaError("candidate cycles must be finite positive", path=f"{path}.{field}[{index}].cycles")
            return dict(values)

        dispatch_costs = validate_costs(
            self.dispatch_candidate_costs, "dispatch_candidate_costs"
        )
        combine_costs = validate_costs(
            self.combine_candidate_costs, "combine_candidate_costs"
        )
        dispatch_lifecycle = validate_costs(
            self.dispatch_candidate_lifecycle_costs,
            "dispatch_candidate_lifecycle_costs",
        )
        combine_lifecycle = validate_costs(
            self.combine_candidate_lifecycle_costs,
            "combine_candidate_lifecycle_costs",
        )
        if set(dispatch_lifecycle) != set(dispatch_costs) or set(combine_lifecycle) != set(combine_costs):
            raise SchemaError("lifecycle cost axes do not close candidate axes", path=path)
        from .swizzle_moe_placement import MoeCandidateCoreLifecycleFloor
        def validate_lifecycle_floors(values, costs, field):
            if (
                type(values) is not tuple or not values
                or values != tuple(sorted(
                    values,
                    key=lambda item: (item.candidate_ref, item.runtime_core_id),
                ))
                or len({(item.candidate_ref, item.runtime_core_id) for item in values})
                != len(values)
            ):
                raise SchemaError(
                    "candidate lifecycle floors must be canonical/unique",
                    path=f"{path}.{field}",
                )
            result = {}
            for index, item in enumerate(values):
                if type(item) is not MoeCandidateCoreLifecycleFloor:
                    raise SchemaError(
                        "requires exact candidate lifecycle floor",
                        path=f"{path}.{field}[{index}]",
                    )
                item.validate(f"{path}.{field}[{index}]")
                if item.candidate_ref not in costs:
                    raise SchemaError(
                        "lifecycle floor candidate is outside cost axis",
                        path=f"{path}.{field}[{index}]",
                    )
                result.setdefault(item.candidate_ref, {})[item.runtime_core_id] = (
                    item.alloc_count, item.bind_count, item.free_count,
                )
            if set(result) != set(costs):
                raise SchemaError(
                    "lifecycle floor axis does not close candidate costs",
                    path=f"{path}.{field}",
                )
            return result
        dispatch_floors = validate_lifecycle_floors(
            self.dispatch_candidate_lifecycle_floors,
            dispatch_costs, "dispatch_candidate_lifecycle_floors",
        )
        combine_floors = validate_lifecycle_floors(
            self.combine_candidate_lifecycle_floors,
            combine_costs, "combine_candidate_lifecycle_floors",
        )
        measured = self.performance_complete
        if measured:
            if (
                type(self.lifecycle_fixed_cycles) is not tuple
                or len(self.lifecycle_fixed_cycles) != 4
                or any(
                    type(value) is not float
                    or not math.isfinite(value) or value < 0.0
                    for value in self.lifecycle_fixed_cycles
                )
                or not isinstance(self.calibration_profile_ref, str)
                or not self.calibration_profile_ref
                or not isinstance(self.calibration_sample_digest, str)
                or len(self.calibration_sample_digest) != 64
            ):
                raise SchemaError("measured joint cost requires exact lifecycle calibration", path=path)
        elif (
            self.lifecycle_fixed_cycles is not None
            or self.calibration_profile_ref is not None
            or self.calibration_sample_digest is not None
        ):
            raise SchemaError("provisional joint cost cannot claim lifecycle calibration", path=path)
        def expected_lifecycle_floor(pair):
            result = dict(dispatch_floors[pair[0]])
            for core, counts in combine_floors[pair[1]].items():
                prior = result.get(core, (0, 0, 0))
                result[core] = tuple(
                    left + right
                    for left, right in zip(prior, counts, strict=True)
                )
            return result
        def lifecycle_floor_cycles(pair):
            if not measured:
                return 0.0
            alloc, bind, free, _ = self.lifecycle_fixed_cycles
            return max(
                counts[0] * alloc + counts[1] * bind + counts[2] * free
                for counts in expected_lifecycle_floor(pair).values()
            )
        def pair_lower(pair):
            total = dispatch_costs[pair[0]] + combine_costs[pair[1]]
            if not measured:
                return total
            _, _, _, terminal = self.lifecycle_fixed_cycles
            return (
                total
                - dispatch_lifecycle[pair[0]]
                - combine_lifecycle[pair[1]]
                - terminal
                + lifecycle_floor_cycles(pair)
            )
        pair_order = tuple(sorted(
            (
                (left, right)
                for left in dispatch_costs
                for right in combine_costs
            ),
            key=lambda pair: (
                pair_lower(pair), pair,
            ),
        ))
        for field in ("baseline_pair_ref", "lower_bound_pair_ref"):
            pair = getattr(self, field)
            if type(pair) is not tuple or len(pair) != 2 or pair not in pair_order:
                raise SchemaError("selection pair proof is outside full grid", path=f"{path}.{field}")
        if self.lower_bound_pair_ref != pair_order[0]:
            raise SchemaError("lower-bound pair is not the full-grid argmin", path=f"{path}.lower_bound_pair_ref")
        if (
            type(self.frontier_pair_refs) is not tuple
            or self.frontier_pair_refs != pair_order[:len(self.frontier_pair_refs)]
        ):
            raise SchemaError("frontier must be an exact cost-ordered prefix", path=f"{path}.frontier_pair_refs")
        if type(self.pair_feasibilities) is not tuple or not self.pair_feasibilities:
            raise SchemaError("selection requires pair feasibility witnesses", path=f"{path}.pair_feasibilities")
        from .swizzle_moe_placement import MoeWholePairFeasibility
        pairs = []
        witness_by_pair = {}
        for index, witness in enumerate(self.pair_feasibilities):
            if type(witness) is not MoeWholePairFeasibility:
                raise SchemaError("requires exact whole-pair feasibility", path=f"{path}.pair_feasibilities[{index}]")
            witness.validate(f"{path}.pair_feasibilities[{index}]")
            pairs.append(witness.candidate_refs)
            witness_by_pair[witness.candidate_refs] = witness
            if witness.feasible:
                actual = {
                    item.runtime_core_id: item
                    for item in witness.core_lifecycle_counts
                }
                for core, counts in expected_lifecycle_floor(witness.candidate_refs).items():
                    if (
                        core not in actual
                        or actual[core].alloc_count < counts[0]
                        or actual[core].bind_count < counts[1]
                        or actual[core].free_count < counts[2]
                    ):
                        raise SchemaError(
                            "whole lifecycle does not enclose candidate lifecycle floor",
                            path=f"{path}.pair_feasibilities[{index}]",
                        )
        if pairs != sorted(pairs) or len(pairs) != len(set(pairs)):
            raise SchemaError("pair witnesses must have canonical unique keys", path=f"{path}.pair_feasibilities")
        dispatch_fused = min(
            (ref for ref in dispatch_costs if ref != self.baseline_pair_ref[0]),
            key=lambda ref: (dispatch_costs[ref], ref),
            default=self.baseline_pair_ref[0],
        )
        combine_fused = min(
            (ref for ref in combine_costs if ref != self.baseline_pair_ref[1]),
            key=lambda ref: (combine_costs[ref], ref),
            default=self.baseline_pair_ref[1],
        )
        mode_pairs = {
            (dispatch_ref, combine_ref)
            for dispatch_ref in (self.baseline_pair_ref[0], dispatch_fused)
            for combine_ref in (self.baseline_pair_ref[1], combine_fused)
        }
        if self.performance_complete:
            if not self.frontier_pair_refs:
                raise SchemaError(
                    "measured selection requires an ordered feasibility frontier",
                    path=f"{path}.frontier_pair_refs",
                )
            def actual_cycles(pair):
                witness = witness_by_pair[pair]
                if not witness.feasible:
                    return None
                alloc, bind, free, _ = self.lifecycle_fixed_cycles
                lifecycle = max(
                    item.alloc_count * alloc
                    + item.bind_count * bind
                    + item.free_count * free
                    for item in witness.core_lifecycle_counts
                )
                return pair_lower(pair) - lifecycle_floor_cycles(pair) + lifecycle
            running_best = math.inf
            for pair in self.frontier_pair_refs:
                if pair not in witness_by_pair:
                    raise SchemaError("frontier lacks exact witness", path=f"{path}.frontier_pair_refs")
                if pair_lower(pair) >= running_best:
                    raise SchemaError("frontier evaluates beyond its lower-bound stop", path=f"{path}.frontier_pair_refs")
                actual = actual_cycles(pair)
                if actual is not None:
                    running_best = min(running_best, actual)
            if not math.isfinite(running_best):
                raise SchemaError("frontier has no feasible corrected-cost pair", path=f"{path}.frontier_pair_refs")
            next_lower = (
                None
                if len(self.frontier_pair_refs) == len(pair_order)
                else pair_lower(pair_order[len(self.frontier_pair_refs)])
            )
            if next_lower is not None and next_lower < running_best:
                raise SchemaError("frontier stops before corrected-cost proof closes", path=f"{path}.frontier_pair_refs")
            if self.frontier_stop_lower_bound != next_lower:
                raise SchemaError("frontier stop lower bound is not exact", path=f"{path}.frontier_stop_lower_bound")
            required_pairs = mode_pairs | set(self.frontier_pair_refs)
        else:
            if self.frontier_pair_refs:
                raise SchemaError(
                    "provisional selection cannot claim a measured frontier",
                    path=f"{path}.frontier_pair_refs",
                )
            required_pairs = mode_pairs
            if self.frontier_stop_lower_bound is not None:
                raise SchemaError("provisional selection cannot claim a frontier stop", path=f"{path}.frontier_stop_lower_bound")
        if set(pairs) != required_pairs:
            raise SchemaError(
                "pair witnesses must exactly cover mode fallbacks and ordered frontier",
                path=f"{path}.pair_feasibilities",
            )
        if (
            type(self.ranked_pair_refs) is not tuple
            or set(self.ranked_pair_refs) != set(pairs)
            or len(self.ranked_pair_refs) != len(set(self.ranked_pair_refs))
            or not self.ranked_pair_refs
            or self.ranked_pair_refs[0] != (self.selected_dispatch_candidate_ref, self.selected_combine_candidate_ref)
        ):
            raise SchemaError("ranked pair/selection closure failed", path=f"{path}.ranked_pair_refs")
        for name in ("baseline_estimated_cycles", "selected_estimated_cycles"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise SchemaError("selection cycles must be finite positive", path=f"{path}.{name}")
        if type(self.decision_reason) is not SwizzleDecisionReason or type(self.performance_complete) is not bool:
            raise SchemaError("selection reason/completion must be typed", path=path)
        def corrected(pair):
            if not measured:
                return pair_lower(pair)
            witness = witness_by_pair[pair]
            if not witness.feasible:
                return math.inf
            alloc, bind, free, _ = self.lifecycle_fixed_cycles
            lifecycle = max(
                item.alloc_count * alloc
                + item.bind_count * bind
                + item.free_count * free
                for item in witness.core_lifecycle_counts
            )
            return pair_lower(pair) - lifecycle_floor_cycles(pair) + lifecycle
        baseline_cycles = corrected(self.baseline_pair_ref)
        selected_pair = (
            self.selected_dispatch_candidate_ref,
            self.selected_combine_candidate_ref,
        )
        selected_cycles = corrected(selected_pair)
        if (
            self.baseline_estimated_cycles != baseline_cycles
            or self.selected_estimated_cycles != selected_cycles
        ):
            raise SchemaError("selection cycle sums disagree with full candidate costs", path=path)
        if self.performance_complete:
            feasible_frontier = tuple(
                pair for pair in self.frontier_pair_refs
                if witness_by_pair[pair].feasible
            )
            first_feasible = min(feasible_frontier, key=lambda pair: (corrected(pair), pair))
            if selected_pair != (
                first_feasible
                if selected_cycles < baseline_cycles
                else self.baseline_pair_ref
            ):
                raise SchemaError("selection does not follow exact frontier economics", path=path)
            expected_reason = (
                SwizzleDecisionReason.LOWEST_ESTIMATED_CYCLES
                if selected_cycles < baseline_cycles
                else SwizzleDecisionReason.NO_PROFITABLE_FUSION
            )
        else:
            if selected_pair != self.baseline_pair_ref:
                raise SchemaError("provisional selection must retain baseline pair", path=path)
            expected_reason = SwizzleDecisionReason.NO_PROFITABLE_FUSION
        if self.decision_reason is not expected_reason:
            raise SchemaError("selection reason disagrees with exact economics", path=path)
        selected = next(
            item for item in self.pair_feasibilities
            if item.candidate_refs == self.ranked_pair_refs[0]
        )
        if not selected.feasible:
            raise SchemaError("selected pair lacks exact feasible cost witness", path=path)
        _stable(
            self, "moe_swizzle_workload_selection",
            MOE_SWIZZLE_WORKLOAD_SELECTION_SCHEMA_VERSION, path,
        )


@dataclass(frozen=True, slots=True)
class MoeSwizzleDecision:
    schema_version: str
    id: str
    problem: MoeSwizzleProblem
    baseline: MoeSwizzleCandidate
    ranked_candidates: tuple[MoeSwizzleCandidate, ...]
    selected_candidate_ref: str
    decision_reason: SwizzleDecisionReason
    performance_complete: bool

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleDecision":
        result = cls(
            MOE_SWIZZLE_DECISION_SCHEMA_VERSION,
            stable_artifact_id(
                "moe_swizzle_decision",
                semantic,
                schema_version=MOE_SWIZZLE_DECISION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_decision") -> None:
        if self.schema_version != MOE_SWIZZLE_DECISION_SCHEMA_VERSION:
            raise SchemaError("unsupported decision schema", path=f"{path}.schema_version")
        self.problem.validate(f"{path}.problem")
        self.baseline.validate(f"{path}.baseline")
        if self.baseline.algorithm is not SwizzleAlgorithm.UNFUSED:
            raise SchemaError("baseline must be executable UNFUSED", path=f"{path}.baseline")
        candidates = validate_unique_ids(
            self.ranked_candidates, f"{path}.ranked_candidates"
        )
        if (
            not self.ranked_candidates
            or self.ranked_candidates[0].id != self.selected_candidate_ref
            or self.baseline.id not in candidates
        ):
            raise SchemaError("decision ranking/selection/baseline closure failed", path=path)
        for index, candidate in enumerate(self.ranked_candidates):
            candidate.validate_against(
                self.problem, f"{path}.ranked_candidates[{index}]"
            )
            if (
                candidate.problem_ref != self.problem.id
                or candidate.pattern is not self.problem.region.pattern
            ):
                raise SchemaError(
                    "candidate belongs to another MoE problem",
                    path=f"{path}.ranked_candidates[{index}]",
                )
        if (
            type(self.decision_reason) is not SwizzleDecisionReason
            or type(self.performance_complete) is not bool
        ):
            raise SchemaError("decision reason/completion must be typed", path=path)
        if (
            not all(item.cost.calibrated for item in self.ranked_candidates)
            and self.performance_complete
        ):
            raise SchemaError(
                "provisional costs cannot complete performance",
                path=f"{path}.performance_complete",
            )
        if self.performance_complete and len({
            item.cost.calibration_profile_ref for item in self.ranked_candidates
        }) != 1:
            raise SchemaError(
                "performance decision requires one exact calibration profile",
                path=f"{path}.ranked_candidates",
            )
        _stable(self, "moe_swizzle_decision", MOE_SWIZZLE_DECISION_SCHEMA_VERSION, path)


__all__ = [name for name in globals() if name.startswith("Moe") or name.startswith("MOE_")]
