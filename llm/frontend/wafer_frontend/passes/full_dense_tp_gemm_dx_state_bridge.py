"""Bind one logical Dense GEMM dX to its real per-TP forward weight reads."""

from __future__ import annotations

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import DType
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.ir0 import OpKind, StateAccessMode


def dense_tp_gemm_dx_state_refs(
    plan: FlexibleDenseTrainPlan, forward_ref: str,
) -> tuple[str, ...]:
    """Return StateDecl IDs in exact physical TP rank order, never one surrogate.

    A logical GEMM dX will have a single output TensorValue shared by all
    ranks, while each rank reads its own weight shard from the HBM state home.
    The returned tuple is a source proof, not a generated native action.
    """
    plan.validate("dense_tp_gemm_dx_plan")
    if plan.spec.dp_degree != 1:
        raise UnsupportedFeatureError(
            "TP GEMM dX source binding first requires one DP group",
            path="plan.spec.dp_degree",
        )
    graph = plan.forward_graph
    forward = next((node for node in graph.nodes if node.id == forward_ref), None)
    if forward is None or forward.kind is not OpKind.GEMM:
        raise SchemaError("TP dX must name a real forward GEMM", path=forward_ref)
    if len(forward.inputs) != 2:
        raise SchemaError("source GEMM must name one parameter value", path=forward_ref)
    weight_ref = forward.inputs[1]
    weight = next((item for item in graph.values if item.id == weight_ref), None)
    if weight is None or weight.dtype is not DType.FP16 or weight.producer is not None:
        raise SchemaError("TP dX weight must be an external FP16 source", path=forward_ref)
    _, out_width, in_width = forward.workload.rank_shape
    states = {item.id: item for item in graph.persistent_states}
    selected = tuple(template for template in plan.parameter_templates
                     if forward_ref in template.forward_consumer_refs)
    ordered = []
    for rank in range(plan.spec.tp_degree):
        matches = tuple(template for template in selected
                        if template.tp_shard_index == rank)
        if len(matches) != 1:
            raise SchemaError("TP GEMM dX lacks one parameter shard per rank",
                              path=f"{forward_ref}.tp{rank}")
        template = matches[0]
        declaration = states[template.state_ref]
        if (declaration.identity.tensor_ref != weight_ref
                or declaration.identity.shard_index != rank
                or declaration.shape != (in_width, out_width)
                or declaration.dtype is not DType.FP16
                or template.owner_ranks != (rank,)
                or template.weight_bytes != 2 * in_width * out_width
                or len(tuple(access for access in graph.state_accesses
                             if access.node_ref == forward_ref
                             and access.state_ref == declaration.id
                             and access.rank == rank
                             and access.mode is StateAccessMode.READ)) != 1):
            raise SchemaError("TP dX StateDecl does not match actual forward rank READ",
                              path=f"{forward_ref}.tp{rank}")
        ordered.append(declaration.id)
    if len(selected) != plan.spec.tp_degree or len(set(ordered)) != len(ordered):
        raise SchemaError("TP dX parameter shard inventory is duplicated",
                          path=forward_ref)
    return tuple(ordered)



__all__ = ["dense_tp_gemm_dx_state_refs"]
