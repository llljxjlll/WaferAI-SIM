"""Bind all P3 MoE step/layer blocks to the production Flexible-MoE lowerer."""

from __future__ import annotations

from ..errors import SchemaError, UnsupportedFeatureError
from ..lowering.flexible_moe_production import (
    lower_link_flexible_moe_production,
)
from ..schema.common import DType
from ..schema.e2e_workload_graph import E2EOperationKind, E2ERouteTrace
from ..schema.flexible_moe import (
    FlexibleMoeLimits,
    FlexibleMoeMode,
    FlexibleMoeSpec,
    MoeRectActionKind,
    MoeRectStateRole,
    MoeRectStaticTrace,
    MoeRectTraceAssignment,
)
from ..schema.memory_plan import MemoryPlanExecution
from ..schema.moe_compile_sequence import (
    MoeCompileSequence,
    MoeCompileUnit,
    MoeOperationBinding,
    MoeParameterBinding,
    MoeParameterLowering,
    MoeParameterReadBinding,
)
from ..schema.rect_mesh import RectMeshSpec
from ..schema.serde import canonical_digest
from ..schema.workload_materialization import (
    WorkloadMaterializationManifest,
    WorkloadMaterializationStatus,
)
from ..schema.workload_run import (
    WorkloadExecutionStrategy,
    WorkloadFamily,
    WorkloadMemoryMode,
    WorkloadModelArchitecture,
    WorkloadOptimizerKind,
)
from .flexible_moe import compile_flexible_moe_baseline


def _reject(reasons: list[str]) -> None:
    if reasons:
        raise UnsupportedFeatureError(
            "request is outside MoE compile-sequence subset: "
            + ", ".join(sorted(set(reasons))),
            path="moe_compile_sequence",
            code="moe_compile_sequence_unsupported",
        )


def _validate_inputs(manifest: WorkloadMaterializationManifest) -> None:
    manifest.validate("manifest")
    request = manifest.request
    reasons: list[str] = []
    if manifest.status is not WorkloadMaterializationStatus.PARTIAL:
        reasons.append("manifest.partial_required")
    if request.family not in (
        WorkloadFamily.MOE_INFERENCE,
        WorkloadFamily.MOE_TRAINING,
    ):
        reasons.append("request.moe_family_required")
    if request.model.architecture is not WorkloadModelArchitecture.LLAMA_MOE:
        reasons.append("request.moe_model_required")
    if request.model.dtype is not DType.FP16:
        reasons.append("request.fp16_required")
    if (
        request.parallel.tp,
        request.parallel.dp,
        request.parallel.pp,
    ) != (1, 1, 1):
        reasons.append("request.tp_dp_pp_must_equal_one")
    if (
        request.parallel.ep != request.mesh.rank_count
        or request.model.num_experts != request.mesh.rank_count
    ):
        reasons.append("request.ep_experts_must_cover_mesh")
    expected_dies = tuple(range(request.mesh.rank_count))
    if request.parallel.active_die_ids and request.parallel.active_die_ids != expected_dies:
        reasons.append("request.full_row_major_mesh_required")
    if manifest.placement.active_die_ids != expected_dies:
        reasons.append("manifest.full_row_major_mesh_required")
    if request.memory.mode is not WorkloadMemoryMode.RESIDENT_HBM:
        reasons.append("request.resident_hbm_required")
    if (
        manifest.memory_plan.execution
        is not MemoryPlanExecution.NO_EXTERNAL_TRANSPORT_REQUIRED
    ):
        reasons.append("manifest.external_transport_not_supported")
    if (
        not request.execution.timing
        or request.execution.functional
        or request.execution.strategy is not WorkloadExecutionStrategy.BASELINE
    ):
        reasons.append("request.baseline_timing_only_required")
    if request.model.num_layers < 2:
        reasons.append("request.at_least_two_layers_required")
    if request.family is WorkloadFamily.MOE_INFERENCE:
        inference = request.steps.inference
        if (
            inference is None
            or inference.prefill_tokens == 0
            or inference.decode_steps != 2
        ):
            reasons.append("request.prefill_plus_two_decode_required")
    else:
        training = request.steps.training
        if training is None or training.step_count != 2:
            reasons.append("request.exactly_two_training_steps_required")
        if (
            request.optimizer is None
            or request.optimizer.kind is not WorkloadOptimizerKind.SGD
        ):
            reasons.append("request.sgd_required")
    _reject(reasons)


def _one_operation(graph, kind, step, layer, expert=None, parameter=None):
    matches = tuple(
        item
        for item in graph.operations
        if (
            item.kind,
            item.step,
            item.layer,
            item.expert,
        ) == (kind, step, layer, expert)
        and (parameter is None or item.parameter_ref == parameter)
    )
    if len(matches) != 1:
        raise SchemaError(
            "requires one exact P3 MoE operation",
            path=f"logical_graph.{kind.value}",
        )
    return matches[0]


def _flexible_spec(
    manifest: WorkloadMaterializationManifest,
    trace: E2ERouteTrace,
    source_rank_policy: str,
) -> FlexibleMoeSpec:
    request = manifest.request
    next_slot = [0] * request.model.num_experts
    assignments = []
    for token, expert in enumerate(trace.expert_by_token):
        assignments.append(MoeRectTraceAssignment(
            token_index=token,
            source_rank=(
                0
                if source_rank_policy == "rank0_shared_spine"
                else token % request.parallel.ep
            ),
            expert_index=expert,
            expert_home_rank=expert,
            slot_index=next_slot[expert],
        ))
        next_slot[expert] += 1
    flexible_trace = MoeRectStaticTrace.create(
        token_count=trace.token_count,
        expert_count=request.model.num_experts,
        capacity_per_expert=max(1, max(next_slot)),
        assignments=tuple(assignments),
    )
    return FlexibleMoeSpec.create(
        mesh=RectMeshSpec(request.mesh.rows, request.mesh.columns),
        mode=(
            FlexibleMoeMode.TRAIN
            if request.family is WorkloadFamily.MOE_TRAINING
            else FlexibleMoeMode.INFERENCE
        ),
        hidden_size=request.model.hidden_size,
        intermediate_size=request.model.intermediate_size,
        expert_count=request.model.num_experts,
        expert_parallel_degree=request.parallel.ep,
        top_k=request.model.experts_per_token,
        trace_mode="static",
        trace=flexible_trace,
        limits=FlexibleMoeLimits(),
        expert_dtype=DType.FP16,
        combine_dtype=DType.FP32,
        token_drop=False,
    )


def _operation_binding(graph, trace, training: bool) -> MoeOperationBinding:
    common = {
        "step": trace.step,
        "layer": trace.layer,
    }
    return MoeOperationBinding.create(
        router_operation_ref=_one_operation(
            graph, E2EOperationKind.ROUTER, **common
        ).id,
        route_freeze_operation_ref=_one_operation(
            graph, E2EOperationKind.ROUTE_FREEZE, **common
        ).id,
        dispatch_operation_ref=_one_operation(
            graph, E2EOperationKind.DISPATCH, **common
        ).id,
        expert_forward_operation_refs=tuple(
            _one_operation(
                graph,
                E2EOperationKind.EXPERT_FORWARD,
                expert=expert,
                **common,
            ).id
            for expert in range(len(trace.expert_token_counts))
        ),
        combine_operation_ref=_one_operation(
            graph, E2EOperationKind.COMBINE, **common
        ).id,
        grad_dispatch_operation_ref=(
            _one_operation(
                graph, E2EOperationKind.GRAD_DISPATCH, **common
            ).id
            if training
            else None
        ),
        expert_backward_operation_refs=(
            tuple(
                _one_operation(
                    graph,
                    E2EOperationKind.EXPERT_BACKWARD,
                    expert=expert,
                    **common,
                ).id
                for expert in range(len(trace.expert_token_counts))
            )
            if training
            else ()
        ),
        dx_combine_operation_ref=(
            _one_operation(
                graph, E2EOperationKind.DX_COMBINE, **common
            ).id
            if training
            else None
        ),
    )


def _parameter_binding(
    graph,
    plan,
    *,
    step: int,
    layer: int,
    expert: int | None,
) -> MoeParameterBinding:
    if expert is None:
        parameters = (f"layer.{layer}.router.weight",)
        gradient_kind = E2EOperationKind.ROUTER_GRADIENT
        lowering = MoeParameterLowering.ROUTER_REPLICATED_GATE
        state_role = MoeRectStateRole.GATE_PARAMETER
        ranks = tuple(range(plan.state_bindings[-1].owner_rank + 1))
        wgrad_kind = MoeRectActionKind.GATE_WGRAD
        sync_kind = MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE
        sgd_kind = MoeRectActionKind.GATE_SGD
    else:
        parameters = tuple(
            f"layer.{layer}.expert.{expert}.{projection}.weight"
            for projection in ("gate", "up", "down")
        )
        gradient_kind = E2EOperationKind.EXPERT_GRADIENT
        lowering = MoeParameterLowering.EXPERT_FUSED_GATE_UP_DOWN
        state_role = MoeRectStateRole.EXPERT_PARAMETER
        ranks = (expert,)
        wgrad_kind = MoeRectActionKind.EXPERT_WGRAD
        sync_kind = None
        sgd_kind = MoeRectActionKind.EXPERT_SGD
    gradients = tuple(
        _one_operation(
            graph, gradient_kind, step, layer, expert, parameter
        )
        for parameter in parameters
    )
    syncs = tuple(
        _one_operation(
            graph, E2EOperationKind.GRADIENT_SYNC, step, layer, expert, parameter
        )
        for parameter in parameters
    )
    updates = tuple(
        _one_operation(
            graph, E2EOperationKind.SGD_UPDATE, step, layer, expert, parameter
        )
        for parameter in parameters
    )
    stores = tuple(
        _one_operation(
            graph, E2EOperationKind.PARAMETER_STORE, step, layer, expert, parameter
        )
        for parameter in parameters
    )
    production_states = tuple(
        sorted(
            (
                item for item in plan.state_bindings
                if item.role is state_role and item.owner_rank in ranks
                and (expert is None or item.expert_index == expert)
            ),
            key=lambda item: item.owner_rank,
        )
    )
    actions = tuple(plan.actions)
    action_refs = lambda kind: tuple(sorted(
        item.id
        for item in actions
        if kind is not None and item.kind is kind and item.rank in ranks
    ))
    return MoeParameterBinding.create(
        step=step,
        layer=layer,
        expert=expert,
        parameter_refs=parameters,
        input_parameter_state_refs=tuple(item.reads[0] for item in updates),
        raw_gradient_state_refs=tuple(item.writes[0] for item in gradients),
        synced_gradient_state_refs=tuple(item.writes[0] for item in syncs),
        output_parameter_state_refs=tuple(item.writes[0] for item in updates),
        gradient_operation_refs=tuple(item.id for item in gradients),
        sync_operation_refs=tuple(item.id for item in syncs),
        sgd_operation_refs=tuple(item.id for item in updates),
        store_operation_refs=tuple(item.id for item in stores),
        production_parameter_state_refs=tuple(item.id for item in production_states),
        production_wgrad_action_refs=action_refs(wgrad_kind),
        production_sync_action_refs=action_refs(sync_kind),
        production_sgd_action_refs=action_refs(sgd_kind),
        production_store_action_refs=tuple(sorted(
            item.id
            for item in actions
            if item.kind is MoeRectActionKind.STATE_STORE
            and item.rank in ranks
            and any(state.id in item.state_refs for state in production_states)
        )),
        lowering=lowering,
    )


def _parameter_read_binding(
    graph,
    plan,
    operation_binding: MoeOperationBinding,
    *,
    step: int,
    layer: int,
    expert: int | None,
) -> MoeParameterReadBinding:
    if expert is None:
        parameters = (f"layer.{layer}.router.weight",)
        operation = next(
            item for item in graph.operations
            if item.id == operation_binding.router_operation_ref
        )
        state_role = MoeRectStateRole.GATE_PARAMETER
        lowering = MoeParameterLowering.ROUTER_REPLICATED_GATE
    else:
        parameters = tuple(
            f"layer.{layer}.expert.{expert}.{projection}.weight"
            for projection in ("gate", "up", "down")
        )
        operation = next(
            item for item in graph.operations
            if item.id == operation_binding.expert_forward_operation_refs[expert]
        )
        state_role = MoeRectStateRole.EXPERT_PARAMETER
        lowering = MoeParameterLowering.EXPERT_FUSED_GATE_UP_DOWN
    state_index = {item.id: item for item in graph.state_versions}
    parameter_states = tuple(
        state_ref
        for state_ref in operation.reads
        if state_index[state_ref].logical_name in parameters
    )
    production_states = tuple(sorted(
        (
            item for item in plan.state_bindings
            if item.role is state_role
            and (expert is None or item.expert_index == expert)
        ),
        key=lambda item: item.owner_rank,
    ))
    owner_ranks = {item.owner_rank for item in production_states}
    loads = tuple(sorted(
        item.id
        for item in plan.actions
        if item.kind is MoeRectActionKind.STATE_LOAD
        and item.rank in owner_ranks
        and any(state.id in item.state_refs for state in production_states)
    ))
    return MoeParameterReadBinding.create(
        step=step,
        layer=layer,
        expert=expert,
        parameter_refs=parameters,
        parameter_state_refs=parameter_states,
        production_parameter_state_refs=tuple(
            item.id for item in production_states
        ),
        production_load_action_refs=loads,
        lowering=lowering,
    )


def compile_moe_sequence(
    manifest: WorkloadMaterializationManifest,
    *,
    source_rank_policy: str = "token_index_mod_ep",
    runtime_core_ids: tuple[int, ...] | None = None,
) -> MoeCompileSequence:
    """Compile every P3 MoE block without claiming full-model runtime."""

    if type(manifest) is not WorkloadMaterializationManifest:
        raise SchemaError("must be a WorkloadMaterializationManifest", path="manifest")
    if source_rank_policy not in ("token_index_mod_ep", "rank0_shared_spine"):
        raise SchemaError("unsupported source-rank policy", path="source_rank_policy")
    _validate_inputs(manifest)
    graph = manifest.logical_graph
    training = manifest.request.family is WorkloadFamily.MOE_TRAINING
    units = []
    for trace in sorted(graph.route_traces, key=lambda item: (item.step, item.layer)):
        spec = _flexible_spec(manifest, trace, source_rank_policy)
        plan = compile_flexible_moe_baseline(spec)
        artifacts = lower_link_flexible_moe_production(
            plan, spec,
            full_model_dataflow=source_rank_policy == "rank0_shared_spine",
            runtime_core_ids=runtime_core_ids,
        )
        operation_binding = _operation_binding(graph, trace, training)
        units.append(MoeCompileUnit.create(
            phase=trace.phase,
            step=trace.step,
            layer=trace.layer,
            route_trace_ref=trace.id,
            route_trace_digest=canonical_digest(trace),
            operation_binding=operation_binding,
            parameter_reads=(
                _parameter_read_binding(
                    graph,
                    plan,
                    operation_binding,
                    step=trace.step,
                    layer=trace.layer,
                    expert=None,
                ),
                *(
                    _parameter_read_binding(
                        graph,
                        plan,
                        operation_binding,
                        step=trace.step,
                        layer=trace.layer,
                        expert=expert,
                    )
                    for expert in range(manifest.request.model.num_experts)
                ),
            ),
            parameter_bindings=(
                (
                    _parameter_binding(
                        graph, plan, step=trace.step, layer=trace.layer, expert=None
                    ),
                    *(
                        _parameter_binding(
                            graph,
                            plan,
                            step=trace.step,
                            layer=trace.layer,
                            expert=expert,
                        )
                        for expert in range(manifest.request.model.num_experts)
                    ),
                )
                if training
                else ()
            ),
            source_rank_policy=source_rank_policy,
            spec=spec,
            plan=plan,
            linked_manifest=artifacts.manifest,
            linked_manifest_digest=canonical_digest(artifacts.manifest),
            lower_link_verified=artifacts.lower_link_verified,
            runtime_verified=artifacts.runtime_verified,
        ))
    return MoeCompileSequence.create(materialization=manifest, units=tuple(units))


__all__ = ["compile_moe_sequence"]
