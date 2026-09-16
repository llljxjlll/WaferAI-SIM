"""Exact unmet RoPE and residual reverse leaf requirements from Dense IR0.

This derives rank shapes and forward tensor identities. It is not a backward
producer, public opcode, physical BufferABI closure or numeric gradient proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from ..errors import SchemaError
from .common import DType
from .flexible_dense_train import FlexibleDenseTrainPlan
from .full_dense_gradient_requirements import build_dense_full_train_requirements
from .ir0 import (
    OpKind,
    ResidualWorkload,
    RopeQkWorkload,
    SwiGluWorkload,
)
from .serde import canonical_digest


@dataclass(frozen=True, slots=True)
class RopeQkReverseRequirement:
    forward_ref: str
    forward_input_ref: str
    forward_output_ref: str
    required_position_trace_ref: str
    logical_tokens: int
    rank_tokens: int
    logical_query_heads: int
    logical_kv_heads: int
    rank_query_heads: int
    rank_kv_heads: int
    tp_degree: int
    head_dim: int
    rotary_dim: int
    max_position_embeddings: int
    position_bytes: int
    fp16_upstream_bytes: int
    fp16_output_bytes: int
    inverse_rotary_pairs: int
    pass_through_v_elements: int


@dataclass(frozen=True, slots=True)
class ResidualDualDxReverseRequirement:
    forward_ref: str
    left_forward_value_ref: str
    right_forward_value_ref: str
    forward_output_ref: str
    required_left_gradient_ref: str
    required_right_gradient_ref: str
    logical_rows: int
    rank_rows: int
    tp_degree: int
    hidden_size: int
    fp16_upstream_bytes: int
    fp16_left_output_bytes: int
    fp16_right_output_bytes: int


@dataclass(frozen=True, slots=True)
class DenseRopeResidualReverseRequirements:
    source_forward_graph_digest: str
    rope: tuple[RopeQkReverseRequirement, ...]
    residual: tuple[ResidualDualDxReverseRequirement, ...]

    def validate_against(self, plan: FlexibleDenseTrainPlan) -> None:
        if self != build_dense_rope_residual_reverse_requirements(plan):
            raise SchemaError("Dense RoPE/residual source rank geometry drifted",
                              path="dense_rope_residual_reverse_requirements")


class DenseReverseSourceFamily(str, Enum):
    GEMM_DX = "gemm_dx"
    NORM_DX = "norm_dx"
    COLLECTIVE_DX = "collective_dx"
    ROPE_QK_DX = "rope_qk_dx"
    ATTENTION_DX = "attention_dx"
    RESIDUAL_DUAL_DX = "residual_dual_dx"
    SWIGLU_DX = "swiglu_dx"


@dataclass(frozen=True, slots=True)
class DenseReverseSourceLeaf:
    forward_ref: str
    backward_ref: str
    family: DenseReverseSourceFamily


@dataclass(frozen=True, slots=True)
class DenseFullReverseSourceGapGate:
    """Exact source-contract inventory; never physical/runtime admission."""

    source_forward_graph_digest: str
    required: tuple[DenseReverseSourceLeaf, ...]
    contracted: tuple[DenseReverseSourceLeaf, ...]
    missing: tuple[DenseReverseSourceLeaf, ...]

    @property
    def contracted_by_ref(self) -> Mapping[str, DenseReverseSourceLeaf]:
        return MappingProxyType({item.backward_ref: item for item in self.contracted})

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def require_complete_source_contracts(self) -> None:
        if self.missing:
            raise SchemaError(
                f"Dense full reverse source contracts missing: {len(self.missing)}",
                path="dense_full_reverse_source_gap.missing",
            )

    def validate_against(
        self,
        plan: FlexibleDenseTrainPlan,
        contracted_families: Mapping[str, DenseReverseSourceFamily],
    ) -> None:
        if self != build_dense_full_reverse_source_gap_gate(
            plan, contracted_families=contracted_families
        ):
            raise SchemaError(
                "Dense full reverse source gap inventory drifted",
                path="dense_full_reverse_source_gap",
            )


def _reverse_source_family(node: object) -> DenseReverseSourceFamily:
    kind = node.kind
    if kind is OpKind.GEMM:
        return DenseReverseSourceFamily.GEMM_DX
    if kind is OpKind.NORM:
        return DenseReverseSourceFamily.NORM_DX
    if kind is OpKind.COLLECTIVE:
        return DenseReverseSourceFamily.COLLECTIVE_DX
    if kind is OpKind.ROPE:
        return DenseReverseSourceFamily.ROPE_QK_DX
    if kind is OpKind.ATTENTION:
        return DenseReverseSourceFamily.ATTENTION_DX
    if kind is OpKind.ELEMENTWISE and type(node.workload) is ResidualWorkload:
        return DenseReverseSourceFamily.RESIDUAL_DUAL_DX
    if kind is OpKind.ELEMENTWISE and type(node.workload) is SwiGluWorkload:
        return DenseReverseSourceFamily.SWIGLU_DX
    raise SchemaError(
        "Dense reverse source family has no explicit contract class",
        path=f"forward[{node.id}]",
    )


def build_dense_full_reverse_source_gap_gate(
    plan: FlexibleDenseTrainPlan,
    *,
    contracted_families: Mapping[str, DenseReverseSourceFamily],
) -> DenseFullReverseSourceGapGate:
    """Bind explicit source contracts to every required two-layer reverse leaf.

    Callers must provide an exact backward-ref -> family witness. A name,
    public opcode, or legacy MATMUL is not inferred as a source contract.
    """

    if not isinstance(contracted_families, Mapping):
        raise SchemaError(
            "contracted families must be a mapping", path="contracted_families"
        )
    requirements = build_dense_full_train_requirements(plan, steps=2)
    nodes = {node.id: node for node in plan.forward_graph.nodes}
    required: list[DenseReverseSourceLeaf] = []
    for backward_ref in requirements.required_backbone_backward_refs:
        if not backward_ref.startswith("backward::"):
            raise SchemaError("backward source ref is not canonical", path="requirements")
        forward_ref = backward_ref.removeprefix("backward::")
        node = nodes.get(forward_ref)
        if node is None:
            raise SchemaError(
                "backward source ref is not in forward graph", path="requirements"
            )
        required.append(DenseReverseSourceLeaf(
            forward_ref=forward_ref,
            backward_ref=backward_ref,
            family=_reverse_source_family(node),
        ))
    required_by_ref = {item.backward_ref: item for item in required}
    if len(required_by_ref) != len(required):
        raise SchemaError("duplicate required reverse source", path="requirements")
    if not set(contracted_families) <= set(required_by_ref):
        raise SchemaError(
            "source contract is not a required reverse leaf",
            path="contracted_families",
        )
    for ref, family in contracted_families.items():
        if type(family) is not DenseReverseSourceFamily:
            raise SchemaError(
                "source contract family must be typed",
                path=f"contracted_families[{ref}]",
            )
        if family is not required_by_ref[ref].family:
            raise SchemaError(
                "source contract family differs from forward source",
                path=f"contracted_families[{ref}]",
            )
    contracted = tuple(
        item for item in required if item.backward_ref in contracted_families
    )
    missing = tuple(
        item for item in required if item.backward_ref not in contracted_families
    )
    return DenseFullReverseSourceGapGate(
        source_forward_graph_digest=canonical_digest(plan.forward_graph),
        required=tuple(required),
        contracted=contracted,
        missing=missing,
    )


def build_dense_rope_residual_source_gap_gate(
    plan: FlexibleDenseTrainPlan,
) -> DenseFullReverseSourceGapGate:
    """Admit only the source contracts implemented by this NEW-only slice."""

    witnesses = build_dense_rope_residual_reverse_requirements(plan)
    contracted = {
        **{
            f"backward::{item.forward_ref}": DenseReverseSourceFamily.ROPE_QK_DX
            for item in witnesses.rope
        },
        **{
            f"backward::{item.forward_ref}": DenseReverseSourceFamily.RESIDUAL_DUAL_DX
            for item in witnesses.residual
        },
    }
    return build_dense_full_reverse_source_gap_gate(
        plan, contracted_families=contracted
    )


def build_dense_rope_residual_reverse_requirements(
    plan: FlexibleDenseTrainPlan,
) -> DenseRopeResidualReverseRequirements:
    if type(plan) is not FlexibleDenseTrainPlan:
        raise SchemaError("requires production FlexibleDenseTrainPlan", path="plan")
    plan.validate()
    graph = plan.forward_graph
    graph.validate("dense_rope_residual_forward_source")
    if plan.full_model_backward_materialized:
        raise SchemaError("source reverse leaves are already materialized",
                          path="plan.full_model_backward_materialized")
    values = {value.id: value for value in graph.values}
    nodes = {node.id: node for node in graph.nodes}
    tp = graph.instances[0].parallel.tp
    rope: list[RopeQkReverseRequirement] = []
    residual: list[ResidualDualDxReverseRequirement] = []
    for node in graph.nodes:
        if node.kind is OpKind.ROPE:
            workload = node.workload
            if type(workload) is not RopeQkWorkload or len(node.inputs) != 1 or len(node.outputs) != 1:
                raise SchemaError("RoPE source node is not exact packed QKV",
                                  path=f"forward[{node.id}]")
            src, dst = values[node.inputs[0]], values[node.outputs[0]]
            tokens, packed_width = workload.rank_input_shape
            if (src.dtype is not DType.FP16 or dst.dtype is not DType.FP16
                    or src.producer is None or nodes[src.producer].kind is not OpKind.GEMM
                    or dst.producer != node.id
                    or src.shape != workload.logical_input_shape
                    or dst.shape != workload.logical_output_shape
                    or workload.rank_output_shape != workload.rank_input_shape
                    or tokens != workload.profile.prefill_tokens
                    or workload.profile.decode_tokens != 0
                    or workload.num_heads != workload.rank_num_heads * tp
                    or workload.num_kv_heads != workload.rank_num_kv_heads * tp
                    or packed_width != (workload.rank_num_heads +
                                        2 * workload.rank_num_kv_heads) * workload.head_dim):
                raise SchemaError("RoPE inverse lacks source GEMM/packed GQA rank geometry",
                                  path=f"forward[{node.id}]")
            bytes_fp16 = 2 * tokens * packed_width
            rope.append(RopeQkReverseRequirement(
                forward_ref=node.id, forward_input_ref=src.id,
                forward_output_ref=dst.id,
                required_position_trace_ref=f"{node.id}.position_ids",
                logical_tokens=workload.profile.prefill_tokens,
                rank_tokens=tokens,
                logical_query_heads=workload.num_heads,
                logical_kv_heads=workload.num_kv_heads,
                rank_query_heads=workload.rank_num_heads,
                rank_kv_heads=workload.rank_num_kv_heads,
                tp_degree=tp, head_dim=workload.head_dim,
                rotary_dim=workload.rotary_dim,
                max_position_embeddings=workload.max_position_embeddings,
                position_bytes=4 * tokens,
                fp16_upstream_bytes=bytes_fp16,
                fp16_output_bytes=bytes_fp16,
                inverse_rotary_pairs=(tokens *
                    (workload.rank_num_heads + workload.rank_num_kv_heads) *
                    workload.rotary_dim // 2),
                pass_through_v_elements=(tokens * workload.rank_num_kv_heads *
                                         workload.head_dim),
            ))
        elif node.kind is OpKind.ELEMENTWISE and type(node.workload) is ResidualWorkload:
            workload = node.workload
            if len(node.inputs) != 2 or len(node.outputs) != 1:
                raise SchemaError("residual source requires two independent inputs",
                                  path=f"forward[{node.id}]")
            left, right = (values[ref] for ref in node.inputs)
            dst = values[node.outputs[0]]
            logical_rows, hidden = workload.logical_shape
            rank_rows, rank_hidden = workload.rank_shape
            if (left.id == right.id or left.dtype is not DType.FP16
                    or right.dtype is not DType.FP16 or dst.dtype is not DType.FP16
                    or left.shape != workload.logical_shape
                    or right.shape != workload.logical_shape
                    or dst.shape != workload.logical_shape
                    or left.producer is None or right.producer is None
                    or dst.producer != node.id
                    or logical_rows != rank_rows * tp
                    or hidden != rank_hidden):
                raise SchemaError("residual reverse lacks two real FP16 forward branches",
                                  path=f"forward[{node.id}]")
            rank_bytes = 2 * rank_rows * hidden
            residual.append(ResidualDualDxReverseRequirement(
                forward_ref=node.id,
                left_forward_value_ref=left.id,
                right_forward_value_ref=right.id,
                forward_output_ref=dst.id,
                required_left_gradient_ref=f"backward::{node.id}.left_gradient",
                required_right_gradient_ref=f"backward::{node.id}.right_gradient",
                logical_rows=logical_rows, rank_rows=rank_rows,
                tp_degree=tp, hidden_size=hidden,
                fp16_upstream_bytes=rank_bytes,
                fp16_left_output_bytes=rank_bytes,
                fp16_right_output_bytes=rank_bytes,
            ))
    if not rope or not residual:
        raise SchemaError("two-layer Dense source requires RoPE and residual leaves",
                          path="forward")
    return DenseRopeResidualReverseRequirements(
        source_forward_graph_digest=canonical_digest(graph),
        rope=tuple(reversed(rope)), residual=tuple(reversed(residual)),
    )


__all__ = [
    "RopeQkReverseRequirement",
    "ResidualDualDxReverseRequirement",
    "DenseRopeResidualReverseRequirements",
    "DenseReverseSourceFamily",
    "DenseReverseSourceLeaf",
    "DenseFullReverseSourceGapGate",
    "build_dense_rope_residual_reverse_requirements",
    "build_dense_full_reverse_source_gap_gate",
    "build_dense_rope_residual_source_gap_gate",
]
