"""Build the source-backed complete Dense backward graph for one SGD step."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import DType, MeshAxisName, Sharding, TensorValue
from ..schema.dense_backward_workloads import (
    AttentionBackwardWorkload, ResidualBackwardWorkload,
    RmsNormBackwardWorkload, RopeBackwardWorkload, SwiGluBackwardWorkload,
)
from ..schema.dense_backbone_reverse_source_admission import (
    DenseReverseSourceFamily, build_dense_two_step_backbone_source_admission,
)
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.gemm_input_dx_workload import GemmInputDxWorkload
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase, ResidualWorkload, StateAccess, StateAccessMode,
    CollectiveKind, CollectiveWorkload, ReduceOp,
)
from ..schema.moe_training_ir0_workloads import (
    EmbeddingTableWgradWorkload, NormGammaWgradWorkload,
)
from .full_dense_training_ce_ir0 import append_dense_training_ce_backward_source
from .full_dense_tp_gemm_dx_state_bridge import dense_tp_gemm_dx_state_refs


def _pure() -> NodeEffects:
    return NodeEffects(EffectKind.PURE, None, None)


def build_full_dense_training_backward_ir0(
    plan: FlexibleDenseTrainPlan,
) -> IR0:
    """Append every native backbone dX and parameter WGRAD to real forward+CE.

    DP1 uses typed, rank-local TP shards: each reverse node consumes its actual
    downstream gradient and saved forward value; residual branches are added
    explicitly and each parameter shard has a public FP32 WGRAD node.  The
    following two-step pass expands optimizer and parameter-state versions.
    """
    if type(plan) is not FlexibleDenseTrainPlan:
        raise SchemaError("requires production FlexibleDenseTrainPlan", path="plan")
    plan.validate()
    if plan.spec.dp_degree != 1:
        raise UnsupportedFeatureError(
            "complete backward source requires one DP replica before real FP32 SUM",
            path="plan.spec",
        )
    admission = build_dense_two_step_backbone_source_admission(plan)
    admission.validate_against(plan)
    graph = append_dense_training_ce_backward_source(plan.forward_graph)
    nodes = list(graph.nodes)
    values = {value.id: value for value in graph.values}
    original_nodes = {node.id: node for node in plan.forward_graph.nodes}
    states = {state.id: state for state in plan.forward_graph.persistent_states}
    templates_by_forward = {
        ref: tuple(template for template in plan.parameter_templates
                   if ref in template.forward_consumer_refs)
        for ref in (node.id for node in plan.forward_graph.nodes)
    }
    gradients: dict[str, list[str]] = {
        f"{graph.instances[0].id}.logits": [
            f"{graph.instances[0].id}.logits_gradient"
        ]
    }
    new_accesses = list(graph.state_accesses)
    position_values: dict[str, str] = {}

    def add_value(value: TensorValue) -> str:
        if value.id in values:
            raise SchemaError("duplicate backward value", path=value.id)
        values[value.id] = value
        return value.id

    def add_node(node: LogicalNode) -> None:
        if any(existing.id == node.id for existing in nodes):
            raise SchemaError("duplicate backward node", path=node.id)
        nodes.append(node)

    def accumulated(forward_value_ref: str) -> str:
        refs = gradients.get(forward_value_ref, [])
        if not refs:
            raise SchemaError("reverse source lacks downstream gradient",
                              path=forward_value_ref)
        while len(refs) > 1:
            left, right, *rest = refs
            source = values[forward_value_ref]
            node_id = f"gradient_sum::{forward_value_ref}::{len(rest)}"
            output_id = f"{node_id}.output"
            add_value(TensorValue(
                output_id, source.shape, DType.FP16,
                f"{source.logical_layout}.gradient_sum", source.sharding,
                node_id, (), None,
            ))
            add_node(LogicalNode(
                node_id, graph.instances[0].id, OpKind.ELEMENTWISE,
                OpPhase.DGRAD, 0, source.sharding.mesh_ref,
                (left, right), (output_id,),
                ResidualWorkload(
                    source.shape,
                    tuple(extent // plan.spec.tp_degree
                          if axis is MeshAxisName.TP else extent
                          for extent, axis in zip(
                              source.shape, source.sharding.dim_map, strict=True
                          )),
                    DType.FP16,
                ),
                original_nodes[next(iter(original_nodes))].math, _pure(),
                "residual",
            ))
            refs = [output_id, *rest]
        gradients[forward_value_ref] = refs
        return refs[0]

    def append_gradient(forward_value_ref: str, gradient_ref: str) -> None:
        gradients.setdefault(forward_value_ref, []).append(gradient_ref)

    def add_parameter_wgrad(forward: LogicalNode, upstream_ref: str) -> None:
        templates = templates_by_forward.get(forward.id, ())
        if not templates:
            return
        if len(templates) != plan.spec.tp_degree:
            raise SchemaError("every source parameter needs all TP WGRAD shards",
                              path=forward.id)
        for template in sorted(templates, key=lambda item: item.tp_shard_index):
            state = states[template.state_ref]
            output_id = f"{template.wgrad_ref}.output"
            parameter_value = values[forward.inputs[1]]
            if forward.kind is OpKind.GEMM:
                activation = values[forward.inputs[0]]
                rows, out_width, in_width = forward.workload.rank_shape
                workload = GemmWeightWgradWorkload(
                    m=in_width, n=out_width, k=rows,
                    source_forward_op_ref=forward.id,
                    source_parameter_state_ref=state.id,
                )
                kind = OpKind.GEMM_WEIGHT_WGRAD
                inputs = (activation.id, upstream_ref)
                impl = "gemm_weight_wgrad_timing"
            elif forward.kind is OpKind.NORM:
                rows, hidden = forward.workload.rank_activation_shape
                workload = NormGammaWgradWorkload(
                    rows * plan.spec.tp_degree, rows,
                    plan.spec.tp_degree, hidden, 0,
                )
                kind = OpKind.NORM_GAMMA_WGRAD
                inputs = (forward.inputs[0], upstream_ref)
                impl = "norm_gamma_wgrad_timing"
            else:
                return
            add_value(TensorValue(
                output_id, parameter_value.shape, DType.FP32,
                f"{parameter_value.logical_layout}.gradient",
                parameter_value.sharding, template.wgrad_ref, (), None,
            ))
            add_node(LogicalNode(
                template.wgrad_ref, forward.instance_id, kind, OpPhase.WGRAD,
                forward.stage, forward.mesh_ref, inputs, (output_id,), workload,
                forward.math, _pure(), impl,
            ))

    leaves = {leaf.backward_ref: leaf for leaf in admission.leaves}
    for reverse_ref in admission.source_gap_gate.required:
        # required is already in reverse forward-node order.
        forward = original_nodes[reverse_ref.forward_ref]
        leaf = leaves[reverse_ref.backward_ref]
        upstream = accumulated(forward.outputs[0])
        add_parameter_wgrad(forward, upstream)
        output_ids: tuple[str, ...]
        if leaf.family is DenseReverseSourceFamily.GEMM_DX:
            state_refs = dense_tp_gemm_dx_state_refs(plan, forward.id)
            rows, out_width, in_width = forward.workload.rank_shape
            output_id = f"{reverse_ref.backward_ref}.input_gradient"
            add_value(TensorValue(
                output_id, values[forward.inputs[0]].shape, DType.FP16,
                f"{values[forward.inputs[0]].logical_layout}.gradient",
                Sharding(
                    forward.mesh_ref,
                    values[forward.inputs[0]].sharding.dim_map,
                    (MeshAxisName.TP,),
                ) if (plan.spec.tp_degree > 1
                      and values[forward.inputs[0]].sharding.partial == ()
                      and values[forward.inputs[0]].sharding.dim_map
                         == (None, None)
                      and values[forward.outputs[0]].sharding.dim_map[-1]
                         is MeshAxisName.TP)
                else values[forward.inputs[0]].sharding,
                reverse_ref.backward_ref, (), None,
            ))
            workload = GemmInputDxWorkload(
                k=rows, m=in_width, n=out_width,
                source_forward_op_ref=forward.id,
                source_parameter_state_ref=state_refs[0],
                output_dtype=DType.FP16,
            )
            kind, inputs, output_ids, impl = (
                OpKind.GEMM_INPUT_DX, (forward.inputs[1], upstream),
                (output_id,), "gemm_input_dx_timing",
            )
            for rank, state_ref in enumerate(state_refs):
                new_accesses.append(StateAccess.create(
                    node_ref=reverse_ref.backward_ref, state_ref=state_ref,
                    mode=StateAccessMode.READ, rank=rank,
                ))
        elif leaf.family is DenseReverseSourceFamily.NORM_DX:
            rows, hidden = dict(leaf.profile)["rows"], dict(leaf.profile)["hidden"]
            output_id = f"{reverse_ref.backward_ref}.input_gradient"
            source = values[forward.inputs[0]]
            add_value(TensorValue(output_id, source.shape, DType.FP16,
                f"{source.logical_layout}.gradient", source.sharding,
                reverse_ref.backward_ref, (), None))
            workload = RmsNormBackwardWorkload(
                rows, hidden, plan.spec.tp_degree,
            )
            kind, inputs, output_ids, impl = (
                OpKind.RMSNORM_BACKWARD, (forward.inputs[0], upstream),
                (output_id,), "rmsnorm_backward_timing",
            )
        elif leaf.family is DenseReverseSourceFamily.ATTENTION_DX:
            profile = dict(leaf.profile)
            output_id = f"{reverse_ref.backward_ref}.input_gradient"
            source = values[forward.inputs[0]]
            add_value(TensorValue(output_id, source.shape, DType.FP16,
                f"{source.logical_layout}.gradient", source.sharding,
                reverse_ref.backward_ref, (), None))
            workload = AttentionBackwardWorkload(
                profile["tokens"], profile["rank_query_heads"],
                profile["rank_kv_heads"], profile["head_dim"],
                plan.spec.tp_degree,
                profile["sequences"], profile["pairs"],
            )
            kind, inputs, output_ids, impl = (
                OpKind.ATTENTION_BACKWARD, (forward.inputs[0], upstream),
                (output_id,), "attention_backward_timing",
            )
        elif leaf.family is DenseReverseSourceFamily.ROPE_QK_DX:
            profile = dict(leaf.profile)
            forward_workload = forward.workload
            position_id = position_values.get(forward.id)
            if position_id is None:
                position_id = f"{forward.id}.position_ids"
                position_values[forward.id] = position_id
                add_value(TensorValue(
                    position_id, (profile["tokens"],), DType.INT32,
                    "M_position_ids",
                    Sharding(values[forward.inputs[0]].sharding.mesh_ref,
                             (None,), ()),
                    None, (), None,
                ))
            output_id = f"{reverse_ref.backward_ref}.input_gradient"
            source = values[forward.inputs[0]]
            add_value(TensorValue(output_id, source.shape, DType.FP16,
                f"{source.logical_layout}.gradient", source.sharding,
                reverse_ref.backward_ref, (), None))
            workload = RopeBackwardWorkload(
                forward_workload.profile.prefill_tokens, profile["tokens"],
                forward_workload.num_heads, forward_workload.num_kv_heads,
                profile["rank_query_heads"], profile["rank_kv_heads"],
                plan.spec.tp_degree,
                profile["head_dim"], profile["head_dim"],
                forward_workload.max_position_embeddings,
            )
            kind, inputs, output_ids, impl = (
                OpKind.ROPE_BACKWARD, (position_id, upstream),
                (output_id,), "rope_backward_timing",
            )
        elif leaf.family is DenseReverseSourceFamily.RESIDUAL_DUAL_DX:
            profile = dict(leaf.profile)
            output_ids = tuple(
                f"{reverse_ref.backward_ref}.{side}_gradient"
                for side in ("left", "right")
            )
            for output_id, input_ref in zip(output_ids, forward.inputs):
                source = values[input_ref]
                add_value(TensorValue(output_id, source.shape, DType.FP16,
                    f"{source.logical_layout}.gradient", source.sharding,
                    reverse_ref.backward_ref, (), None))
            workload = ResidualBackwardWorkload(
                forward.workload.logical_shape[0], profile["rows"],
                plan.spec.tp_degree,
                profile["hidden"],
            )
            kind, inputs, impl = (
                OpKind.RESIDUAL_BACKWARD,
                (forward.outputs[0], upstream), "residual_backward_timing",
            )
        elif leaf.family is DenseReverseSourceFamily.SWIGLU_DX:
            profile = dict(leaf.profile)
            output_id = f"{reverse_ref.backward_ref}.input_gradient"
            source = values[forward.inputs[0]]
            add_value(TensorValue(output_id, source.shape, DType.FP16,
                f"{source.logical_layout}.gradient", source.sharding,
                reverse_ref.backward_ref, (), None))
            workload = SwiGluBackwardWorkload(
                profile["rows"], profile["intermediate"])
            kind, inputs, output_ids, impl = (
                OpKind.SWIGLU_BACKWARD, (forward.inputs[0], upstream),
                (output_id,), "swiglu_backward_timing",
            )
        elif leaf.family is DenseReverseSourceFamily.COLLECTIVE_DX:
            source = values[forward.inputs[0]]
            output_id = f"{reverse_ref.backward_ref}.input_gradient"
            output_layout = f"{source.logical_layout}.gradient"
            output_sharding = (
                Sharding(source.sharding.mesh_ref, source.sharding.dim_map, ())
                if source.sharding.partial else source.sharding
            )
            add_value(TensorValue(
                output_id, source.shape, DType.FP16, output_layout,
                output_sharding, reverse_ref.backward_ref, (), None,
            ))
            original = forward.workload
            is_rs = leaf.reverse_collective is CollectiveKind.REDUCE_SCATTER
            workload = replace(
                original, collective=leaf.reverse_collective,
                reduce_op=ReduceOp.SUM if is_rs else None,
                reduction_mesh_axes=(MeshAxisName.TP,) if is_rs else (),
                scatter_tensor_axis=original.gather_tensor_axis if is_rs else None,
                gather_tensor_axis=original.scatter_tensor_axis if not is_rs else None,
                rank_input_bytes=original.rank_output_bytes,
                rank_output_bytes=original.rank_input_bytes,
                input_layout=values[upstream].logical_layout,
                output_layout=output_layout,
            )
            kind, inputs, output_ids, impl = (
                OpKind.COLLECTIVE, (upstream,), (output_id,),
                "collective_derived",
            )
        else:
            raise SchemaError("Dense reverse source has unknown derivative family",
                              path=forward.id)
        add_node(LogicalNode(
            reverse_ref.backward_ref, forward.instance_id, kind, OpPhase.DGRAD,
            forward.stage, forward.mesh_ref, inputs, output_ids, workload,
            forward.math, _pure(), impl,
        ))
        parameter_input = (
            forward.inputs[1]
            if forward.kind in (OpKind.GEMM, OpKind.NORM)
            else None
        )
        for input_ref, gradient_ref in zip(
            (ref for ref in forward.inputs if ref != parameter_input), output_ids
        ):
            append_gradient(input_ref, gradient_ref)

    embedding = original_nodes[f"{graph.instances[0].id}.embedding"]
    embedding_upstream = accumulated(embedding.outputs[0])
    table = values[embedding.inputs[1]]
    rows = embedding.workload.rank_output_shape[0]
    trace = tuple(range(rows)) + (0,) * (16 - rows)
    for template in templates_by_forward[embedding.id]:
        state = states[template.state_ref]
        output_id = f"{template.wgrad_ref}.output"
        add_value(TensorValue(
            output_id, table.shape, DType.FP32,
            f"{table.logical_layout}.gradient", table.sharding,
            template.wgrad_ref, (), None,
        ))
        add_node(LogicalNode(
            template.wgrad_ref, embedding.instance_id,
            OpKind.EMBEDDING_TABLE_WGRAD, OpPhase.WGRAD, embedding.stage,
            embedding.mesh_ref,
            (embedding.inputs[0], embedding.inputs[1], embedding_upstream),
            (output_id,), EmbeddingTableWgradWorkload(
                rows * plan.spec.tp_degree, rows,
                plan.spec.tp_degree,
                state.shape[0], 0, state.shape[0], state.shape[1], trace,
            ), embedding.math, _pure(), "embedding_table_wgrad_timing",
        ))
        new_accesses.append(StateAccess.create(
            node_ref=template.wgrad_ref, state_ref=state.id,
            mode=StateAccessMode.READ, rank=template.tp_shard_index,
        ))

    # Rebuild value consumers and all data edges from the actual typed nodes.
    consumers: dict[str, list[str]] = {value_id: [] for value_id in values}
    for node in nodes:
        for value_id in node.inputs:
            consumers[value_id].append(node.id)
    final_values = tuple(replace(value, consumers=tuple(consumers[value.id]))
                         for value in values.values())
    data_edges = tuple(
        GraphEdge(f"{value.id}.edge_to.{consumer}", EdgeKind.DATA,
                  value.producer, consumer, value.id)
        for value in final_values if value.producer is not None
        for consumer in value.consumers
    )
    ce_forward = f"{graph.instances[0].id}.cross_entropy"
    ce_backward = f"{ce_forward}_backward"
    result = IR0.create(
        producer_pass="full_dense_training_backward_source",
        job=graph.job, instances=graph.instances, nodes=tuple(nodes),
        values=final_values,
        edges=(*data_edges, GraphEdge(
            f"{ce_forward}.control_to.{ce_backward}", EdgeKind.CONTROL,
            ce_forward, ce_backward, None,
        )),
        fusion_candidates=(), profile=graph.profile, train=graph.train,
        persistent_states=graph.persistent_states,
        state_accesses=tuple(sorted(set(new_accesses),
                                    key=lambda item: (item.node_ref,
                                                      item.state_ref,
                                                      item.rank, item.id))),
    )
    node_ids = {node.id for node in result.nodes}
    required_reverse = {
        leaf.backward_ref for leaf in admission.source_gap_gate.required
    }
    required_wgrad = {
        template.wgrad_ref for template in plan.parameter_templates
    }
    if not required_reverse or not required_wgrad:
        raise SchemaError("complete reverse source is empty", path="result.nodes")
    if not required_reverse <= node_ids:
        raise SchemaError("complete reverse source omits a required dX",
                          path="result.nodes")
    if not required_wgrad <= node_ids:
        raise SchemaError("complete reverse source omits a required WGRAD",
                          path="result.nodes")
    return result


__all__ = ["build_full_dense_training_backward_ir0"]
