"""Typed source, topology, IR0, and N4 carriers for fixed EP4 S3-Lite MoE."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .action import FusionPlan, StandaloneCollectivePlan
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import IR0, OpKind
from .ir1 import IR1
from .lite_moe import (
    LiteMoeRoutingKind,
    LiteMoeStaticTrace,
    LiteMoeTransferRole,
)
from .persistent_state import PersistentStateAccess, StateKind
from .serde import canonical_digest


S3_LITE_MOE_DP4_INFER_CASE_ID = "case.s3_lite.dp4_ep4.static_moe_infer"
S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID = (
    "case.s3_lite.dp4_ep4.static_moe_train_forward"
)
S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID = (
    "case.s3_lite.dp4_ep4.static_moe_down_wgrad"
)
LITE_MOE_DP4_SPEC_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_spec/v1alpha1"
)
LITE_MOE_DP4_TOPOLOGY_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_topology/v1alpha1"
)
LITE_MOE_DP4_ORACLE_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_oracle/v1alpha1"
)
LITE_MOE_DP4_P2P_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_p2p_binding/v1alpha1"
)
LITE_MOE_DP4_IR0_ADAPTER_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_ir0_adapter/v1alpha1"
)
LITE_MOE_DP4_PLACED_IR1_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_placed_ir1/v1alpha1"
)
LITE_MOE_DP4_N4_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_n4/v1alpha1"
)

_DIES = (0, 1, 2, 3)
_EXPERT_HOMES = (0, 1, 2, 3)
_TOKEN_SOURCES = (0, 1, 2, 3, 0, 1, 2, 3)
_LOCAL_TOKENS = (0, 7)
_REMOTE_TOKENS = (1, 2, 3, 4, 5, 6)
_ASSIGNMENTS = (0, 0, 1, 1, 2, 2, 3, 3)
_SLOTS = (0, 1, 0, 1, 0, 1, 0, 1)


class LiteMoeDp4WorkloadKind(str, Enum):
    INFER_FORWARD = "infer_forward"
    TRAIN_FORWARD = "train_forward"
    DOWN_WGRAD = "down_wgrad"


def _semantic(instance: object) -> dict[str, object]:
    return {
        name: getattr(instance, name)
        for name in instance.__dataclass_fields__
        if name not in ("schema_version", "producer_pass", "id")
    }


def _check_id(
    instance: object,
    prefix: str,
    version: str,
    path: str,
) -> None:
    expected = stable_artifact_id(prefix, _semantic(instance), schema_version=version)
    if getattr(instance, "id") != expected:
        raise SchemaError(
            f"unstable artifact id; expected {expected!r}", path=f"{path}.id"
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4Spec:
    schema_version: str
    producer_pass: str
    id: str
    case_id: str
    die_count: int
    ep_degree: int
    expert_count: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    dtype: DType
    capacity_per_expert: int
    routing_kind: LiteMoeRoutingKind
    trace: LiteMoeStaticTrace

    @classmethod
    def create(cls, *, trace: LiteMoeStaticTrace) -> "LiteMoeDp4Spec":
        semantic = {
            "case_id": S3_LITE_MOE_DP4_INFER_CASE_ID,
            "die_count": 4,
            "ep_degree": 4,
            "expert_count": 4,
            "top_k": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "dtype": DType.FP16,
            "capacity_per_expert": 2,
            "routing_kind": LiteMoeRoutingKind.STATIC_TRACE,
            "trace": trace,
        }
        result = cls(
            LITE_MOE_DP4_SPEC_SCHEMA_VERSION,
            "lite_moe_dp4_spec",
            stable_artifact_id(
                "s3_lite_moe_dp4_spec",
                semantic,
                schema_version=LITE_MOE_DP4_SPEC_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_spec") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_SPEC_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_spec"
        ):
            raise SchemaError("unsupported DP4 spec schema/producer", path=path)
        if (
            self.case_id,
            self.die_count,
            self.ep_degree,
            self.expert_count,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.dtype,
            self.capacity_per_expert,
            self.routing_kind,
        ) != (
            S3_LITE_MOE_DP4_INFER_CASE_ID,
            4,
            4,
            4,
            1,
            16,
            32,
            DType.FP16,
            2,
            LiteMoeRoutingKind.STATIC_TRACE,
        ):
            raise SchemaError("requires exact DP4/EP4/T8/H16/I32 static MoE", path=path)
        if type(self.trace) is not LiteMoeStaticTrace:
            raise SchemaError("must reuse LiteMoeStaticTrace", path=f"{path}.trace")
        self.trace.validate(f"{path}.trace")
        observed = tuple(
            (item.token_index, item.expert_index, item.slot_index)
            for item in self.trace.assignments
        )
        expected = tuple(zip(range(8), _ASSIGNMENTS, _SLOTS, strict=True))
        if (
            self.trace.token_count != 8
            or self.trace.expert_histogram != (2, 2, 2, 2)
            or observed != expected
        ):
            raise SchemaError("requires exact balanced T8 assignment/slot trace", path=f"{path}.trace")
        _check_id(self, "s3_lite_moe_dp4_spec", LITE_MOE_DP4_SPEC_SCHEMA_VERSION, path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4Topology:
    schema_version: str
    producer_pass: str
    id: str
    source_spec_id: str
    source_spec_digest: str
    die_grid: tuple[int, int]
    die_ids: tuple[int, ...]
    expert_home_die_ids: tuple[int, ...]
    token_source_die_ids: tuple[int, ...]
    local_token_indices: tuple[int, ...]
    remote_token_indices: tuple[int, ...]

    @classmethod
    def create(cls, *, spec: LiteMoeDp4Spec) -> "LiteMoeDp4Topology":
        semantic = {
            "source_spec_id": spec.id,
            "source_spec_digest": canonical_digest(spec),
            "die_grid": (2, 2),
            "die_ids": _DIES,
            "expert_home_die_ids": _EXPERT_HOMES,
            "token_source_die_ids": _TOKEN_SOURCES,
            "local_token_indices": _LOCAL_TOKENS,
            "remote_token_indices": _REMOTE_TOKENS,
        }
        result = cls(
            LITE_MOE_DP4_TOPOLOGY_SCHEMA_VERSION,
            "lite_moe_dp4_topology",
            stable_artifact_id(
                "s3_lite_moe_dp4_topology",
                semantic,
                schema_version=LITE_MOE_DP4_TOPOLOGY_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate_against(spec)
        return result

    def validate(self, path: str = "lite_moe_dp4_topology") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_TOPOLOGY_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_topology"
        ):
            raise SchemaError("unsupported DP4 topology schema/producer", path=path)
        validate_nonempty(self.source_spec_id, f"{path}.source_spec_id")
        if (
            self.die_grid,
            self.die_ids,
            self.expert_home_die_ids,
            self.token_source_die_ids,
            self.local_token_indices,
            self.remote_token_indices,
        ) != (
            (2, 2),
            _DIES,
            _EXPERT_HOMES,
            _TOKEN_SOURCES,
            _LOCAL_TOKENS,
            _REMOTE_TOKENS,
        ):
            raise SchemaError("requires exact 2x2 expert/token topology", path=path)
        _check_id(
            self,
            "s3_lite_moe_dp4_topology",
            LITE_MOE_DP4_TOPOLOGY_SCHEMA_VERSION,
            path,
        )

    def validate_against(
        self, spec: LiteMoeDp4Spec, path: str = "lite_moe_dp4_topology"
    ) -> None:
        self.validate(path)
        spec.validate(f"{path}.source_spec")
        if (
            self.source_spec_id != spec.id
            or self.source_spec_digest != canonical_digest(spec)
        ):
            raise SchemaError("topology/spec provenance mismatch", path=path)
        remote = tuple(
            assignment.token_index
            for assignment in spec.trace.assignments
            if self.token_source_die_ids[assignment.token_index]
            != self.expert_home_die_ids[assignment.expert_index]
        )
        if remote != self.remote_token_indices:
            raise SchemaError("remote-token set is not trace-derived", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4Oracle:
    schema_version: str
    producer_pass: str
    id: str
    source_spec_id: str
    source_spec_digest: str
    source_topology_id: str
    source_topology_digest: str
    expert_token_counts: tuple[int, ...]
    expert_gemm_flops: tuple[int, ...]
    remote_token_indices: tuple[int, ...]
    dispatch_logical_bytes: int
    combine_logical_bytes: int
    total_expert_gemm_flops: int
    logical_p2p_bytes: int
    data_packets: int

    @classmethod
    def create(
        cls, *, spec: LiteMoeDp4Spec, topology: LiteMoeDp4Topology
    ) -> "LiteMoeDp4Oracle":
        counts = spec.trace.expert_histogram
        flops = tuple(6 * count * 16 * 32 for count in counts)
        semantic = {
            "source_spec_id": spec.id,
            "source_spec_digest": canonical_digest(spec),
            "source_topology_id": topology.id,
            "source_topology_digest": canonical_digest(topology),
            "expert_token_counts": counts,
            "expert_gemm_flops": flops,
            "remote_token_indices": topology.remote_token_indices,
            "dispatch_logical_bytes": 192,
            "combine_logical_bytes": 192,
            "total_expert_gemm_flops": sum(flops),
            "logical_p2p_bytes": 384,
            "data_packets": 24,
        }
        result = cls(
            LITE_MOE_DP4_ORACLE_SCHEMA_VERSION,
            "lite_moe_dp4_oracle",
            stable_artifact_id(
                "s3_lite_moe_dp4_oracle",
                semantic,
                schema_version=LITE_MOE_DP4_ORACLE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate_against(spec, topology)
        return result

    def validate(self, path: str = "lite_moe_dp4_oracle") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_ORACLE_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_oracle"
        ):
            raise SchemaError("unsupported DP4 oracle schema/producer", path=path)
        if (
            self.expert_token_counts,
            self.expert_gemm_flops,
            self.remote_token_indices,
            self.dispatch_logical_bytes,
            self.combine_logical_bytes,
            self.total_expert_gemm_flops,
            self.logical_p2p_bytes,
            self.data_packets,
        ) != (
            (2, 2, 2, 2),
            (6144, 6144, 6144, 6144),
            _REMOTE_TOKENS,
            192,
            192,
            24576,
            384,
            24,
        ):
            raise SchemaError("oracle does not equal fixed DP4 work", path=path)
        _check_id(
            self,
            "s3_lite_moe_dp4_oracle",
            LITE_MOE_DP4_ORACLE_SCHEMA_VERSION,
            path,
        )

    def validate_against(
        self,
        spec: LiteMoeDp4Spec,
        topology: LiteMoeDp4Topology,
        path: str = "lite_moe_dp4_oracle",
    ) -> None:
        self.validate(path)
        spec.validate(f"{path}.spec")
        topology.validate_against(spec, f"{path}.topology")
        if (
            self.source_spec_id,
            self.source_spec_digest,
            self.source_topology_id,
            self.source_topology_digest,
        ) != (
            spec.id,
            canonical_digest(spec),
            topology.id,
            canonical_digest(topology),
        ):
            raise SchemaError("oracle source provenance mismatch", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4P2PBinding:
    schema_version: str
    id: str
    node_ref: str
    role: LiteMoeTransferRole
    token_index: int
    expert_index: int
    slot_index: int
    source_die_id: int
    destination_die_id: int
    bytes: int

    @classmethod
    def create(
        cls,
        *,
        node_ref: str,
        role: LiteMoeTransferRole,
        token_index: int,
        expert_index: int,
        slot_index: int,
    ) -> "LiteMoeDp4P2PBinding":
        token_home = token_index % 4
        expert_home = expert_index
        source, destination = (
            (token_home, expert_home)
            if role is LiteMoeTransferRole.MOE_DISPATCH
            else (expert_home, token_home)
        )
        semantic = {
            "node_ref": node_ref,
            "role": role,
            "token_index": token_index,
            "expert_index": expert_index,
            "slot_index": slot_index,
            "source_die_id": source,
            "destination_die_id": destination,
            "bytes": 32,
        }
        result = cls(
            LITE_MOE_DP4_P2P_BINDING_SCHEMA_VERSION,
            stable_artifact_id(
                "s3_lite_moe_dp4_p2p_binding",
                semantic,
                schema_version=LITE_MOE_DP4_P2P_BINDING_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_p2p_binding") -> None:
        if self.schema_version != LITE_MOE_DP4_P2P_BINDING_SCHEMA_VERSION:
            raise SchemaError("unsupported P2P binding schema", path=path)
        if (
            self.token_index not in _REMOTE_TOKENS
            or self.expert_index != _ASSIGNMENTS[self.token_index]
            or self.slot_index != _SLOTS[self.token_index]
            or self.bytes != 32
            or type(self.role) is not LiteMoeTransferRole
        ):
            raise SchemaError("binding is not an exact remote T8 assignment", path=path)
        token_home = self.token_index % 4
        expert_home = self.expert_index
        endpoints = (
            (token_home, expert_home)
            if self.role is LiteMoeTransferRole.MOE_DISPATCH
            else (expert_home, token_home)
        )
        semantic = {
            "node_ref": self.node_ref,
            "role": self.role,
            "token_index": self.token_index,
            "expert_index": self.expert_index,
            "slot_index": self.slot_index,
            "source_die_id": endpoints[0],
            "destination_die_id": endpoints[1],
            "bytes": 32,
        }
        expected_id = stable_artifact_id(
            "s3_lite_moe_dp4_p2p_binding",
            semantic,
            schema_version=LITE_MOE_DP4_P2P_BINDING_SCHEMA_VERSION,
        )
        if _semantic(self) != semantic or self.id != expected_id:
            raise SchemaError("binding endpoint/id is not canonical", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4IR0Adapter:
    schema_version: str
    producer_pass: str
    id: str
    experiment_digest: str
    spec: LiteMoeDp4Spec
    topology: LiteMoeDp4Topology
    oracle: LiteMoeDp4Oracle
    graph: IR0
    p2p_bindings: tuple[LiteMoeDp4P2PBinding, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4IR0Adapter":
        result = cls(
            LITE_MOE_DP4_IR0_ADAPTER_SCHEMA_VERSION,
            "lite_moe_dp4_graph",
            stable_artifact_id(
                "s3_lite_moe_dp4_ir0_adapter",
                semantic,
                schema_version=LITE_MOE_DP4_IR0_ADAPTER_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_ir0_adapter") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_IR0_ADAPTER_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_graph"
        ):
            raise SchemaError("unsupported DP4 adapter schema/producer", path=path)
        self.spec.validate(f"{path}.spec")
        self.topology.validate_against(self.spec, f"{path}.topology")
        self.oracle.validate_against(self.spec, self.topology, f"{path}.oracle")
        self.graph.validate(f"{path}.graph")
        if (
            len(self.graph.nodes),
            len(self.graph.values),
            len(self.graph.edges),
            len(self.graph.persistent_states),
            len(self.graph.state_accesses),
            len(self.p2p_bindings),
        ) != (44, 64, 42, 12, 24, 12):
            raise SchemaError("graph does not have exact DP4 cardinality", path=f"{path}.graph")
        keys = []
        for index, binding in enumerate(self.p2p_bindings):
            if type(binding) is not LiteMoeDp4P2PBinding:
                raise SchemaError("must be a DP4 P2P binding", path=f"{path}.p2p_bindings[{index}]")
            binding.validate(f"{path}.p2p_bindings[{index}]")
            keys.append((binding.token_index, 0 if binding.role is LiteMoeTransferRole.MOE_DISPATCH else 1))
        if tuple(keys) != tuple(sorted(keys)) or len(set(keys)) != 12:
            raise SchemaError("bindings must use canonical token/role order", path=f"{path}.p2p_bindings")
        _check_id(
            self,
            "s3_lite_moe_dp4_ir0_adapter",
            LITE_MOE_DP4_IR0_ADAPTER_SCHEMA_VERSION,
            path,
        )


def _expert_from_tensor_ref(value: str) -> int:
    prefix = "S3M4.expert"
    if not value.startswith(prefix):
        raise SchemaError("invalid DP4 expert tensor ref", path="persistent_states")
    text = value[len(prefix):].split(".", 1)[0]
    if text not in ("0", "1", "2", "3"):
        raise SchemaError("invalid DP4 expert index", path="persistent_states")
    return int(text)


def _validate_physical(
    graph: IR1,
    bindings: tuple[LiteMoeDp4P2PBinding, ...],
    path: str,
) -> None:
    graph.validate(path)
    if len(graph.groups) != 1:
        raise SchemaError("requires one EP4 group", path=f"{path}.groups")
    group = graph.groups[0]
    if (
        group.logical_shape != (4,)
        or tuple((item.rank, item.die_id) for item in group.placements)
        != ((0, 0), (1, 1), (2, 2), (3, 3))
        or len(group.embedding.routes) != 12
    ):
        raise SchemaError("requires canonical EP4 placement/routes", path=f"{path}.groups")
    node_refs = tuple(item.node_ref for item in bindings)
    if node_refs != tuple(node.id for node in graph.nodes if node.kind is OpKind.P2P):
        raise SchemaError("P2P bindings must cover physical P2P nodes", path=f"{path}.bindings")
    manifest = graph.persistent_state_manifest
    if manifest is None or len(manifest.declarations) != 12 or len(manifest.bindings) != 12:
        raise SchemaError("requires all 12 expert states", path=f"{path}.persistent_states")
    hbm = {item.state_ref: item for item in manifest.bindings}
    for declaration in manifest.declarations:
        tensor_ref = declaration.identity.tensor_ref or ""
        expert = _expert_from_tensor_ref(tensor_ref)
        if (
            declaration.identity.kind is not StateKind.PARAMETER
            or declaration.identity.shard_index != expert
            or declaration.access is not PersistentStateAccess.READ_ONLY
            or hbm[declaration.id].die_id != expert
        ):
            raise SchemaError("expert state is not READ_ONLY at home die", path=f"{path}.persistent_states")
    if len(graph.state_accesses) != 24:
        raise SchemaError("requires 24 parameter reads", path=f"{path}.state_accesses")


@dataclass(frozen=True, slots=True)
class LiteMoeDp4PlacedIR1:
    schema_version: str
    producer_pass: str
    id: str
    source: LiteMoeDp4IR0Adapter
    placement_context_id: str
    graph: IR1
    p2p_bindings: tuple[LiteMoeDp4P2PBinding, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4PlacedIR1":
        result = cls(
            LITE_MOE_DP4_PLACED_IR1_SCHEMA_VERSION,
            "lite_moe_dp4_placement",
            stable_artifact_id(
                "s3_lite_moe_dp4_placed_ir1",
                semantic,
                schema_version=LITE_MOE_DP4_PLACED_IR1_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_placed_ir1") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_PLACED_IR1_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_placement"
        ):
            raise SchemaError("unsupported DP4 placed schema/producer", path=path)
        self.source.validate(f"{path}.source")
        validate_nonempty(self.placement_context_id, f"{path}.placement_context_id")
        if self.p2p_bindings != self.source.p2p_bindings:
            raise SchemaError("placed carrier changed P2P bindings", path=f"{path}.p2p_bindings")
        _validate_physical(self.graph, self.p2p_bindings, f"{path}.graph")
        if self.graph.source_ir0_id != self.source.graph.id:
            raise SchemaError("placed source IR0 mismatch", path=f"{path}.graph")
        _check_id(
            self,
            "s3_lite_moe_dp4_placed_ir1",
            LITE_MOE_DP4_PLACED_IR1_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4N4IR1:
    schema_version: str
    producer_pass: str
    id: str
    source: LiteMoeDp4PlacedIR1
    partition_context_id: str
    planning_context_id: str
    graph: IR1
    fusion_plans: tuple[FusionPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]
    p2p_bindings: tuple[LiteMoeDp4P2PBinding, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4N4IR1":
        result = cls(
            LITE_MOE_DP4_N4_SCHEMA_VERSION,
            "lite_moe_dp4_n4",
            stable_artifact_id(
                "s3_lite_moe_dp4_n4",
                semantic,
                schema_version=LITE_MOE_DP4_N4_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_n4") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_N4_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_n4"
        ):
            raise SchemaError("unsupported DP4 N4 schema/producer", path=path)
        self.source.validate(f"{path}.source")
        validate_nonempty(self.partition_context_id, f"{path}.partition_context_id")
        validate_nonempty(self.planning_context_id, f"{path}.planning_context_id")
        if self.fusion_plans or self.standalone_plans:
            raise SchemaError("unrolled DP4 MoE has no Dense plans", path=path)
        if self.p2p_bindings != self.source.p2p_bindings:
            raise SchemaError("N4 changed P2P bindings", path=f"{path}.p2p_bindings")
        _validate_physical(self.graph, self.p2p_bindings, f"{path}.graph")
        if (
            self.graph.producer_pass != "fusion_partition"
            or self.graph.source_ir0_id != self.source.source.graph.id
        ):
            raise SchemaError("N4 graph provenance mismatch", path=f"{path}.graph")
        _check_id(
            self,
            "s3_lite_moe_dp4_n4",
            LITE_MOE_DP4_N4_SCHEMA_VERSION,
            path,
        )


__all__ = [
    "LITE_MOE_DP4_IR0_ADAPTER_SCHEMA_VERSION",
    "LITE_MOE_DP4_N4_SCHEMA_VERSION",
    "LITE_MOE_DP4_ORACLE_SCHEMA_VERSION",
    "LITE_MOE_DP4_P2P_BINDING_SCHEMA_VERSION",
    "LITE_MOE_DP4_PLACED_IR1_SCHEMA_VERSION",
    "LITE_MOE_DP4_SPEC_SCHEMA_VERSION",
    "LITE_MOE_DP4_TOPOLOGY_SCHEMA_VERSION",
    "S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID",
    "S3_LITE_MOE_DP4_INFER_CASE_ID",
    "S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID",
    "LiteMoeDp4IR0Adapter",
    "LiteMoeDp4N4IR1",
    "LiteMoeDp4Oracle",
    "LiteMoeDp4P2PBinding",
    "LiteMoeDp4PlacedIR1",
    "LiteMoeDp4Spec",
    "LiteMoeDp4Topology",
    "LiteMoeDp4WorkloadKind",
]
