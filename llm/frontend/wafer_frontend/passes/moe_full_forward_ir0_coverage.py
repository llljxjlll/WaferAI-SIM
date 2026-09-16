"""Independent per-layer forward MoE replacement/dense-spine DATA oracle."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.ir0 import IR0
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..schema.moe_full_training_block_workload import MoeFullTrainingBlockWorkload
from ..schema.serde import canonical_digest


def require_moe_full_forward_ir0_coverage(
    graph: IR0, sequence: MoeCompileSequence, *, step: int,
) -> None:
    """Every original Dense residual now physically depends on true MoE output.

    This is a source graph oracle, not a claim about native record execution;
    a complete TRAIN verifier also needs full backbone/expert reverse and SGD.
    """
    graph.validate("moe_full_forward_coverage.graph")
    sequence.validate("moe_full_forward_coverage.sequence")
    request = sequence.materialization.request
    nodes = {node.id: node for node in graph.nodes}
    values = {value.id: value for value in graph.values}
    if (graph.job.value != "train" or len(graph.instances) != 1
            or graph.instances[0].parallel.ep != request.parallel.ep
            or graph.instances[0].parallel.tp != request.parallel.tp):
        raise SchemaError("TRAIN source geometry does not match MoE model",
                          path="moe_full_forward_coverage")
    for layer in range(request.model.num_layers):
        prefix = f"{graph.instances[0].id}.layer{layer}."
        old_mlp = tuple(prefix+name for name in ("gate_up","swiglu","down"))
        if any(ref in nodes for ref in old_mlp):
            raise SchemaError("old Dense MLP still executes alongside its replacement",
                              path=f"moe_full_forward_coverage.layer{layer}")
        norm = values[prefix+"norm2_out"]
        router = nodes.get(prefix+"moe.router")
        freeze = nodes.get(prefix+"moe.route_freeze")
        dispatch = nodes.get(prefix+"moe.dispatch")
        combine = nodes.get(prefix+"moe.combine")
        residual = nodes.get(prefix+"residual2")
        expert_nodes = tuple(nodes.get(prefix+f"moe.expert{expert}")
                             for expert in range(request.model.num_experts))
        route = values.get(prefix+"moe.route_ids")
        route_source = values.get(prefix+"moe.route_table_source")
        score = values.get(prefix+"moe.router_scores")
        combined = values.get(prefix+"moe.combine_out")
        trace = next((trace for trace in
                      sequence.materialization.logical_graph.route_traces
                      if (trace.step,trace.layer)==(step,layer)),None)
        if (None in (router,freeze,dispatch,combine,residual,route,route_source,score,
                     combined,trace) or any(node is None for node in
                                             expert_nodes)):
            raise SchemaError("shared router/dispatch/expert/combine/residual layer incomplete",
                              path=f"moe_full_forward_coverage.layer{layer}")
        expected_router_weights = tuple(prefix+f"moe.router.weight.ep{rank}"
                                        for rank in range(request.parallel.ep))
        expected_dispatches = tuple(prefix+f"moe.dispatch{expert}"
                                    for expert in range(request.model.num_experts))
        expected_expert_out = tuple(prefix+f"moe.expert{expert}.output"
                                    for expert in range(request.model.num_experts))
        if (norm.dtype is not DType.FP16 or norm.producer != prefix+"norm2"
                or norm.consumers != (router.id,dispatch.id)
                or router.inputs != (norm.id,*expected_router_weights)
                or router.outputs != (score.id,)
                or freeze.inputs != (score.id,route_source.id)
                or route_source.shape != route.shape
                or route_source.dtype is not DType.INT32
                or route_source.producer is not None
                or route_source.consumers != (freeze.id,)
                or freeze.outputs != (route.id,)
                or route.dtype is not DType.INT32
                or route.shape != (trace.token_count, 5)
                or dispatch.inputs != (norm.id,route.id)
                or dispatch.outputs != expected_dispatches
                or combine.inputs != (*expected_expert_out,route.id)
                or combine.outputs != (combined.id,)
                or residual.inputs[1] != combined.id
                or combined.dtype is not DType.FP16
                or combined.producer != combine.id
                or combined.consumers != (residual.id,)
                or score.dtype is not DType.FP16):
            raise SchemaError("real hidden→router/dispatch→combine→residual DATA chain broke",
                              path=f"moe_full_forward_coverage.layer{layer}")
        for expert, expert_node in enumerate(expert_nodes):
            required_weights = tuple(prefix+f"moe.expert{expert}.{name}.weight"
                                     for name in ("gate","up","down"))
            output = values[expected_expert_out[expert]]
            payload = values[expected_dispatches[expert]]
            m = trace.expert_token_counts[expert]
            if (m == 0 or expert_node.inputs != (payload.id,*required_weights)
                    or expert_node.outputs != (output.id,)
                    or payload.shape != (m,request.model.hidden_size)
                    or output.shape != payload.shape
                    or payload.consumers != (expert_node.id,)
                    or output.consumers != (combine.id,)
                    or any(values[weight].producer is not None for weight
                           in required_weights)
                    or any(values[weight].consumers != (expert_node.id,)
                           for weight in required_weights)):
                raise SchemaError("one expert omits its routed input, projection or combine output",
                                  path=f"moe_full_forward_coverage.layer{layer}.expert{expert}")
        expected_ids = (
            trace.id, canonical_digest(trace),
            request.case_id, step, layer, tuple(trace.expert_by_token),
        )
        for source_node in (router,freeze,dispatch,combine,*expert_nodes):
            workload = source_node.workload
            if (type(workload) is not MoeFullTrainingBlockWorkload
                    or (workload.source_route_trace_ref,
                        workload.source_route_trace_digest,
                        workload.source_case_ref,workload.step,
                        workload.layer,workload.frozen_expert_by_token)
                        != expected_ids):
                raise SchemaError("MoE physical source action lost original route/step identity",
                                  path=f"moe_full_forward_coverage.layer{layer}")


__all__ = ["require_moe_full_forward_ir0_coverage"]
