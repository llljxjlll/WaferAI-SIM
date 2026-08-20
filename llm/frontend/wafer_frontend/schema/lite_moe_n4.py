"""Dedicated placement and N4 carriers for the isolated S3-Lite graph."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .action import FusionPlan, StandaloneCollectivePlan
from .common import stable_artifact_id, validate_nonempty
from .ir0 import OpKind
from .ir1 import IR1
from .lite_moe_graph import LiteMoeP2PBinding


LITE_MOE_PLACED_IR1_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_static_moe_placed_ir1/v1alpha1"
)
LITE_MOE_N4_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_static_moe_n4/v1alpha1"
)


def _bindings(
    bindings: tuple[LiteMoeP2PBinding, ...],
    graph: IR1,
    path: str,
) -> None:
    if type(bindings) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    node_index = {node.id: node for node in graph.nodes}
    for index, binding in enumerate(bindings):
        binding_path = f"{path}[{index}]"
        if type(binding) is not LiteMoeP2PBinding:
            raise SchemaError("must be a LiteMoeP2PBinding", path=binding_path)
        binding.validate(binding_path)
        node = node_index.get(binding.node_ref)
        if node is None or node.kind is not OpKind.P2P:
            raise SchemaError(
                "must reference one physical P2P node", path=binding_path
            )
    if tuple(binding.node_ref for binding in bindings) != tuple(
        node.id for node in graph.nodes if node.kind is OpKind.P2P
    ):
        raise SchemaError(
            "bindings must exactly cover physical P2P nodes in source order",
            path=path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoePlacedIR1:
    schema_version: str
    producer_pass: str
    id: str
    source_adapter_id: str
    source_ir0_id: str
    placement_context_id: str
    graph: IR1
    p2p_bindings: tuple[LiteMoeP2PBinding, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "LiteMoePlacedIR1":
        result = cls(
            schema_version=LITE_MOE_PLACED_IR1_SCHEMA_VERSION,
            producer_pass="lite_moe_placement",
            id=stable_artifact_id(
                "s3_lite_static_moe_placed_ir1",
                semantic_key,
                schema_version=LITE_MOE_PLACED_IR1_SCHEMA_VERSION,
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

    def validate(self, path: str = "lite_moe_placed_ir1") -> None:
        if self.schema_version != LITE_MOE_PLACED_IR1_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "lite_moe_placement":
            raise SchemaError(
                "must be 'lite_moe_placement'", path=f"{path}.producer_pass"
            )
        for field_name in (
            "source_adapter_id",
            "source_ir0_id",
            "placement_context_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if (
            self.graph.producer_pass != "placement"
            or self.graph.source_ir0_id != self.source_ir0_id
        ):
            raise SchemaError(
                "must carry the exact placed source graph", path=f"{path}.graph"
            )
        _bindings(self.p2p_bindings, self.graph, f"{path}.p2p_bindings")
        expected_id = stable_artifact_id(
            "s3_lite_static_moe_placed_ir1",
            self._semantic_key(),
            schema_version=LITE_MOE_PLACED_IR1_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class LiteMoeN4IR1:
    schema_version: str
    producer_pass: str
    id: str
    source_placed_id: str
    source_adapter_id: str
    source_ir0_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    graph: IR1
    fusion_plans: tuple[FusionPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]
    p2p_bindings: tuple[LiteMoeP2PBinding, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "LiteMoeN4IR1":
        result = cls(
            schema_version=LITE_MOE_N4_SCHEMA_VERSION,
            producer_pass="lite_moe_n4",
            id=stable_artifact_id(
                "s3_lite_static_moe_n4",
                semantic_key,
                schema_version=LITE_MOE_N4_SCHEMA_VERSION,
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

    def validate(self, path: str = "lite_moe_n4") -> None:
        if self.schema_version != LITE_MOE_N4_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "lite_moe_n4":
            raise SchemaError(
                "must be 'lite_moe_n4'", path=f"{path}.producer_pass"
            )
        for field_name in (
            "source_placed_id",
            "source_adapter_id",
            "source_ir0_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if (
            self.graph.producer_pass != "fusion_partition"
            or self.graph.source_ir0_id != self.source_ir0_id
        ):
            raise SchemaError(
                "must carry the exact partitioned source graph",
                path=f"{path}.graph",
            )
        if self.fusion_plans or self.standalone_plans:
            raise SchemaError(
                "unrolled S3-Lite has no Dense fusion or collective plan",
                path=path,
            )
        _bindings(self.p2p_bindings, self.graph, f"{path}.p2p_bindings")
        expected_id = stable_artifact_id(
            "s3_lite_static_moe_n4",
            self._semantic_key(),
            schema_version=LITE_MOE_N4_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


__all__ = [
    "LITE_MOE_N4_SCHEMA_VERSION",
    "LITE_MOE_PLACED_IR1_SCHEMA_VERSION",
    "LiteMoeN4IR1",
    "LiteMoePlacedIR1",
]
