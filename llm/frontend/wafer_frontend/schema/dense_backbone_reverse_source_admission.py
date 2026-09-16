"""Exact two-step Dense backbone reverse source admission.

This module closes only source identities and rank-local derivative geometry.
It does not create IR0 backward nodes, physical buffers, public records, or
runtime evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType
from .dense_rope_residual_reverse_requirements import (
    DenseFullReverseSourceGapGate,
    DenseReverseSourceFamily,
    build_dense_full_reverse_source_gap_gate,
    build_dense_rope_residual_reverse_requirements,
)
from .flexible_dense_train import FlexibleDenseTrainPlan
from .full_dense_gradient_requirements import build_dense_full_train_requirements
from .ir0 import (
    AttentionMode,
    AttentionWorkload,
    CollectiveKind,
    CollectiveWorkload,
    GemmWorkload,
    OpKind,
    ReduceOp,
    ResidualWorkload,
    RmsNormWorkload,
    RopeQkWorkload,
    SwiGluWorkload,
)
from .serde import canonical_digest


@dataclass(frozen=True, slots=True)
class DenseBackboneReverseSourceLeaf:
    forward_ref: str
    backward_ref: str
    family: DenseReverseSourceFamily
    forward_input_refs: tuple[str, ...]
    forward_output_ref: str
    parameter_state_refs: tuple[str, ...]
    saved_forward_bytes: int
    rank_upstream_bytes: int
    rank_output_bytes: int
    output_count: int
    profile: tuple[tuple[str, int], ...]
    reverse_collective: CollectiveKind | None


@dataclass(frozen=True, slots=True)
class DenseStepReverseSourceLeaf:
    step: int
    backward_ref: str
    family: DenseReverseSourceFamily


@dataclass(frozen=True, slots=True)
class DenseTwoStepBackboneSourceAdmission:
    source_plan_id: str
    source_plan_digest: str
    source_forward_graph_digest: str
    steps: int
    leaves: tuple[DenseBackboneReverseSourceLeaf, ...]
    step_leaves: tuple[DenseStepReverseSourceLeaf, ...]
    source_gap_gate: DenseFullReverseSourceGapGate
    physical_program_admitted: bool

    def validate_against(self, plan: FlexibleDenseTrainPlan) -> None:
        if self != build_dense_two_step_backbone_source_admission(plan):
            raise SchemaError(
                "Dense two-step backbone source admission drifted",
                path="dense_two_step_backbone_source_admission",
            )


def _states_for(
    plan: FlexibleDenseTrainPlan, forward_ref: str
) -> tuple[str, ...]:
    states = tuple(
        item.state_ref
        for item in plan.parameter_templates
        if forward_ref in item.forward_consumer_refs
    )
    expected = plan.spec.tp_degree
    if len(states) != expected or len(set(states)) != expected:
        raise SchemaError(
            "parameterized reverse source lacks every TP StateDecl shard",
            path=f"forward[{forward_ref}]",
        )
    return states


def _leaf(
    *,
    node: object,
    family: DenseReverseSourceFamily,
    states: tuple[str, ...] = (),
    saved: int,
    upstream: int,
    output: int,
    output_count: int = 1,
    profile: tuple[tuple[str, int], ...],
    reverse_collective: CollectiveKind | None = None,
) -> DenseBackboneReverseSourceLeaf:
    return DenseBackboneReverseSourceLeaf(
        forward_ref=node.id,
        backward_ref=f"backward::{node.id}",
        family=family,
        forward_input_refs=node.inputs,
        forward_output_ref=node.outputs[0],
        parameter_state_refs=states,
        saved_forward_bytes=saved,
        rank_upstream_bytes=upstream,
        rank_output_bytes=output,
        output_count=output_count,
        profile=profile,
        reverse_collective=reverse_collective,
    )


def build_dense_two_step_backbone_source_admission(
    plan: FlexibleDenseTrainPlan,
) -> DenseTwoStepBackboneSourceAdmission:
    if type(plan) is not FlexibleDenseTrainPlan:
        raise SchemaError("requires production FlexibleDenseTrainPlan", path="plan")
    plan.validate()
    graph = plan.forward_graph
    graph.validate("dense_two_step_backbone_source")
    requirements = build_dense_full_train_requirements(plan, steps=2)
    rope_residual = build_dense_rope_residual_reverse_requirements(plan)
    rope_by_ref = {item.forward_ref: item for item in rope_residual.rope}
    residual_by_ref = {item.forward_ref: item for item in rope_residual.residual}
    values = {item.id: item for item in graph.values}
    nodes = {item.id: item for item in graph.nodes}
    leaves: list[DenseBackboneReverseSourceLeaf] = []

    for backward_ref in requirements.required_backbone_backward_refs:
        forward_ref = backward_ref.removeprefix("backward::")
        node = nodes.get(forward_ref)
        if node is None or len(node.outputs) != 1:
            raise SchemaError(
                "required reverse leaf lacks one exact forward source",
                path=f"reverse[{backward_ref}]",
            )
        tensors = tuple(values[ref] for ref in (*node.inputs, *node.outputs))
        if any(item.dtype is not DType.FP16 for item in tensors):
            raise SchemaError(
                "Dense backbone source tensors must be FP16",
                path=f"forward[{forward_ref}]",
            )

        if node.kind is OpKind.GEMM:
            workload = node.workload
            if type(workload) is not GemmWorkload or len(node.inputs) != 2:
                raise SchemaError("GEMM dX source arity differs", path=forward_ref)
            rows, out_width, in_width = workload.rank_shape
            logical_rows, logical_out, logical_in = workload.logical_shape
            activation, weight = (values[ref] for ref in node.inputs)
            output = values[node.outputs[0]]
            if (
                activation.shape != (logical_rows, logical_in)
                or weight.shape != (logical_in, logical_out)
                or weight.producer is not None
                or output.shape != (logical_rows, logical_out)
                or output.producer != node.id
            ):
                raise SchemaError("GEMM dX lacks exact X/W/Y source geometry",
                                  path=forward_ref)
            leaf = _leaf(
                node=node, family=DenseReverseSourceFamily.GEMM_DX,
                states=_states_for(plan, node.id),
                saved=2 * rows * in_width,
                upstream=2 * rows * out_width,
                output=4 * rows * in_width,
                profile=(("rows", rows), ("input_width", in_width),
                         ("output_width", out_width)),
            )
        elif node.kind is OpKind.NORM:
            workload = node.workload
            if type(workload) is not RmsNormWorkload or len(node.inputs) != 2:
                raise SchemaError("Norm dX source arity differs", path=forward_ref)
            rows, hidden = workload.rank_activation_shape
            activation, weight = (values[ref] for ref in node.inputs)
            output = values[node.outputs[0]]
            if (
                activation.shape != workload.logical_activation_shape
                or weight.shape != workload.logical_weight_shape
                or weight.producer is not None
                or output.shape != workload.logical_output_shape
                or output.producer != node.id
            ):
                raise SchemaError("Norm dX lacks exact activation/gamma source",
                                  path=forward_ref)
            leaf = _leaf(
                node=node, family=DenseReverseSourceFamily.NORM_DX,
                states=_states_for(plan, node.id),
                saved=2 * rows * hidden, upstream=2 * rows * hidden,
                output=2 * rows * hidden,
                profile=(("rows", rows), ("hidden", hidden)),
            )
        elif node.kind is OpKind.ATTENTION:
            workload = node.workload
            if (
                type(workload) is not AttentionWorkload
                or workload.mode is not AttentionMode.TRAIN_FORWARD
                or len(node.inputs) != 1
            ):
                raise SchemaError("Attention dX source profile differs", path=forward_ref)
            tokens = workload.query_tokens
            packed_heads = workload.rank_num_heads + 2 * workload.rank_num_kv_heads
            packed_bytes = 2 * tokens * packed_heads * workload.head_dim
            upstream_bytes = (
                2 * tokens * workload.rank_num_heads * workload.head_dim
            )
            if (
                workload.query_key_pairs !=
                workload.profile.num_seqs * workload.context_max *
                (workload.context_max + 1) // 2
                or values[node.outputs[0]].producer != node.id
            ):
                raise SchemaError("Attention dX causal source geometry differs",
                                  path=forward_ref)
            leaf = _leaf(
                node=node, family=DenseReverseSourceFamily.ATTENTION_DX,
                saved=packed_bytes, upstream=upstream_bytes, output=packed_bytes,
                profile=(("tokens", tokens),
                         ("rank_query_heads", workload.rank_num_heads),
                         ("rank_kv_heads", workload.rank_num_kv_heads),
                         ("head_dim", workload.head_dim),
                         ("sequences", workload.profile.num_seqs),
                         ("pairs", workload.query_key_pairs)),
            )
        elif node.kind is OpKind.COLLECTIVE:
            workload = node.workload
            if type(workload) is not CollectiveWorkload or len(node.inputs) != 1:
                raise SchemaError("Collective dX source differs", path=forward_ref)
            if workload.collective is CollectiveKind.ALL_GATHER:
                reverse = CollectiveKind.REDUCE_SCATTER
                if workload.reduce_op is not None:
                    raise SchemaError("AllGather source cannot reduce", path=forward_ref)
            elif workload.collective is CollectiveKind.REDUCE_SCATTER:
                reverse = CollectiveKind.ALL_GATHER
                if workload.reduce_op is not ReduceOp.SUM:
                    raise SchemaError("ReduceScatter source must SUM", path=forward_ref)
            else:
                raise SchemaError("Dense reverse collective family unsupported",
                                  path=forward_ref)
            leaf = _leaf(
                node=node, family=DenseReverseSourceFamily.COLLECTIVE_DX,
                saved=0, upstream=workload.rank_output_bytes,
                output=workload.rank_input_bytes,
                profile=(("participants", workload.participant_count),
                         ("logical_bytes", workload.logical_tensor_bytes),
                         ("rank_input_bytes", workload.rank_input_bytes),
                         ("rank_output_bytes", workload.rank_output_bytes)),
                reverse_collective=reverse,
            )
        elif node.kind is OpKind.ROPE:
            workload = node.workload
            witness = rope_by_ref.get(node.id)
            if type(workload) is not RopeQkWorkload or witness is None:
                raise SchemaError("RoPE dX source witness missing", path=forward_ref)
            leaf = _leaf(
                node=node, family=DenseReverseSourceFamily.ROPE_QK_DX,
                saved=witness.position_bytes,
                upstream=witness.fp16_upstream_bytes,
                output=witness.fp16_output_bytes,
                profile=(("tokens", witness.rank_tokens),
                         ("rank_query_heads", witness.rank_query_heads),
                         ("rank_kv_heads", witness.rank_kv_heads),
                         ("head_dim", witness.head_dim),
                         ("rotary_dim", witness.rotary_dim)),
            )
        elif (
            node.kind is OpKind.ELEMENTWISE
            and type(node.workload) is ResidualWorkload
        ):
            witness = residual_by_ref.get(node.id)
            if witness is None:
                raise SchemaError("Residual dX source witness missing", path=forward_ref)
            leaf = _leaf(
                node=node, family=DenseReverseSourceFamily.RESIDUAL_DUAL_DX,
                saved=0, upstream=witness.fp16_upstream_bytes,
                output=(witness.fp16_left_output_bytes +
                        witness.fp16_right_output_bytes),
                output_count=2,
                profile=(("rows", witness.rank_rows),
                         ("hidden", witness.hidden_size)),
            )
        elif (
            node.kind is OpKind.ELEMENTWISE
            and type(node.workload) is SwiGluWorkload
        ):
            workload = node.workload
            if len(node.inputs) != 1:
                raise SchemaError("SwiGLU dX source arity differs", path=forward_ref)
            rows, two_intermediate = workload.rank_input_shape
            out_rows, intermediate = workload.rank_output_shape
            if rows != out_rows or two_intermediate != 2 * intermediate:
                raise SchemaError("SwiGLU dX rank geometry differs", path=forward_ref)
            leaf = _leaf(
                node=node, family=DenseReverseSourceFamily.SWIGLU_DX,
                saved=2 * rows * two_intermediate,
                upstream=2 * rows * intermediate,
                output=2 * rows * two_intermediate,
                profile=(("rows", rows), ("intermediate", intermediate)),
            )
        else:
            raise SchemaError(
                "required reverse leaf has no source admission contract",
                path=f"reverse[{backward_ref}]",
            )
        if leaf.backward_ref != backward_ref:
            raise SchemaError("reverse source ordering drifted", path=backward_ref)
        leaves.append(leaf)

    families = {item.backward_ref: item.family for item in leaves}
    gate = build_dense_full_reverse_source_gap_gate(
        plan, contracted_families=families
    )
    gate.require_complete_source_contracts()
    step_leaves = tuple(
        DenseStepReverseSourceLeaf(step, leaf.backward_ref, leaf.family)
        for step in range(2)
        for leaf in leaves
    )
    result = DenseTwoStepBackboneSourceAdmission(
        source_plan_id=plan.id,
        source_plan_digest=canonical_digest(plan),
        source_forward_graph_digest=canonical_digest(graph),
        steps=2,
        leaves=tuple(leaves),
        step_leaves=step_leaves,
        source_gap_gate=gate,
        physical_program_admitted=False,
    )
    if len(step_leaves) != 2 * len(leaves):
        raise SchemaError("two-step reverse source expansion differs", path="step_leaves")
    return result


__all__ = [
    "DenseBackboneReverseSourceLeaf",
    "DenseStepReverseSourceLeaf",
    "DenseTwoStepBackboneSourceAdmission",
    "build_dense_two_step_backbone_source_admission",
]
