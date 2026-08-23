"""Deterministic bounded evaluator for the first intra-die v2 templates."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.intra_die_optimization import IntraDieOptimizationMode
from ..schema.intra_die_timing_model import IntraDieTimingModel
from ..schema.intra_die_refine import (
    IntraDieOptimizationOptions,
    SplitKRefineOptions,
)
from ..schema.intra_die_v2_search import (
    IntraDieV2AnalyticCost,
    IntraDieV2Candidate,
    IntraDieV2CandidateRejection,
    IntraDieV2CandidateKind,
    IntraDieV2SearchBudget,
    IntraDieV2SearchDecision,
)
from ..schema.ir0 import GemmWorkload
from .intra_die_timing_model import (
    estimate_gemm_cycles,
    estimate_reduce_cycles,
    estimate_transport_cycles,
    resolve_intra_die_timing_model,
)
from ..schema.ir1 import IR1, MemoryInitiator
from ..schema.ir2 import (
    IR2ProjectionResult,
    OrdinaryNodeOrigin,
    SemanticTask,
    SemanticTaskKind,
)


_DTYPE_BYTES = {DType.FP16: 2, DType.FP32: 4}

def _available_compute_groups(ir1: IR1, die_id: int, *, double_buffer: bool) -> int:
    die = next(item for item in ir1.fabric.dies if item.id == die_id)
    profiles = {item.id: item for item in ir1.fabric.sram_profiles}
    common = {MemoryInitiator.COMPUTE}
    if double_buffer:
        common.add(MemoryInitiator.LSU)
    source_required = common | {MemoryInitiator.DTE}
    destination_required = common | {MemoryInitiator.NOC_RX}

    def supports(core: object, required: set[MemoryInitiator]) -> bool:
        return any(
            required.issubset(region.access)
            for region in profiles[core.sram_profile_ref].regions
        )

    return max(
        (
            1 + sum(
                destination.runtime_core_id != source.runtime_core_id
                and supports(destination, destination_required)
                for destination in die.cores
            )
            for source in die.cores
            if supports(source, source_required)
        ),
        default=1,
    )


def _eligible(task: SemanticTask, output_consumers: dict[str, tuple[str, ...]]) -> bool:
    return (
        task.kind is SemanticTaskKind.COMP
        and isinstance(task.origin_ref, OrdinaryNodeOrigin)
        and task.compute is not None
        and type(task.compute.workload) is GemmWorkload
        and len(task.read_values) == 2
        and len(task.write_values) == 1
        and len(task.compute.inputs) == 2
        and len(task.compute.outputs) == 1
        and output_consumers.get(task.write_values[0]) == ()
    )


def evaluate_intra_die_v2_candidates(
    projection: IR2ProjectionResult,
    ir1: IR1,
    options: SplitKRefineOptions | IntraDieOptimizationOptions,
    *,
    budget: IntraDieV2SearchBudget = IntraDieV2SearchBudget(),
    timing_model: IntraDieTimingModel | None = None,
    hardware_digest: str | None = None,
    simulation_digest: str | None = None,
) -> IntraDieV2SearchDecision:
    """Bounded OFF/AUTO/FORCE evaluator without simulator feedback.

    ``SplitKRefineOptions`` remains the legacy FORCE interface. AUTO always
    retains identity, removes candidates that fail the analytic break-even
    test, and chooses the minimum predicted makespan (identity wins ties).
    """

    if type(projection) is not IR2ProjectionResult:
        raise SchemaError("must be an IR2ProjectionResult", path="projection")
    if type(ir1) is not IR1:
        raise SchemaError("must be an IR1", path="ir1")
    if type(options) not in (SplitKRefineOptions, IntraDieOptimizationOptions):
        raise SchemaError(
            "must be SplitKRefineOptions or IntraDieOptimizationOptions",
            path="options",
        )
    if type(budget) is not IntraDieV2SearchBudget:
        raise SchemaError("must be IntraDieV2SearchBudget", path="budget")
    projection.validate("projection")
    ir1.validate("ir1")
    options.validate("options")
    budget.validate("budget")
    if projection.source_ir1_id != ir1.id:
        raise SchemaError("projection belongs to another IR-1", path="projection.source_ir1_id")
    if type(options) is IntraDieOptimizationOptions:
        option_hardware_digest = options.timing_hardware_digest
        option_simulation_digest = options.timing_simulation_digest
        if (hardware_digest is not None and option_hardware_digest is not None
                and hardware_digest != option_hardware_digest):
            raise SchemaError("conflicts with options timing binding", path="hardware_digest")
        if (simulation_digest is not None and option_simulation_digest is not None
                and simulation_digest != option_simulation_digest):
            raise SchemaError("conflicts with options timing binding", path="simulation_digest")
        hardware_digest = hardware_digest or option_hardware_digest
        simulation_digest = simulation_digest or option_simulation_digest
    resolved_timing = resolve_intra_die_timing_model(
        ir1, timing_model,
        hardware_digest=hardware_digest,
        simulation_digest=simulation_digest,
    )

    legacy_force = type(options) is SplitKRefineOptions
    if legacy_force:
        assert type(options) is SplitKRefineOptions
        if options.split_k_parts < 2:
            raise SchemaError("v2 search requires a requested split-K fallback", path="options.split_k_parts")
        mode = IntraDieOptimizationMode.FORCE
        configurations = ((
            (
                "split_k_tree_direct_dma"
                if options.enable_tree_reduce and options.enable_direct_dma
                else "split_k_double_buffer"
                if options.enable_double_buffer
                else (
                    "split_k"
                    if options.enable_streaming_reduce
                    else "split_k_barrier"
                )
            ),
            options.split_k_parts,
            options.enable_reduce,
            options.enable_double_buffer,
            options.enable_streaming_reduce,
            options.enable_tree_reduce,
            options.enable_direct_dma,
        ),)
        force_candidate = configurations[0][0]
        max_candidates = budget.max_candidates
        compute_groups_per_die = options.compute_groups_per_die
        require_full_compute_groups = False
    else:
        assert type(options) is IntraDieOptimizationOptions
        mode = options.mode
        if options.max_candidates > budget.max_candidates:
            raise SchemaError("options exceed search budget", path="options.max_candidates")
        max_candidates = options.max_candidates
        compute_groups_per_die = options.compute_groups_per_die
        require_full_compute_groups = options.require_full_compute_groups
        force_candidate = options.force_candidate
        configurations = tuple(
            (
                candidate_name,
                parts,
                True,
                candidate_name == "split_k_double_buffer",
                candidate_name != "split_k_barrier",
                candidate_name == "split_k_tree_direct_dma",
                candidate_name == "split_k_tree_direct_dma",
            )
            for candidate_name in options.allowed_candidates
            if candidate_name != "identity"
            and (
                mode is not IntraDieOptimizationMode.FORCE
                or candidate_name == force_candidate
            )
            for parts in options.split_k_parts
        )

    eligible: list[SemanticTask] = []
    all_gemms: list[SemanticTask] = []
    gemms_by_dag: list[tuple[SemanticTask, ...]] = []
    eligible_by_dag: list[tuple[SemanticTask, ...]] = []
    task_dag_index: dict[str, int] = {}
    for dag_index, dag in enumerate(projection.dags):
        output_consumers = {value.id: value.consumer_tasks for value in dag.values}
        dag_gemms: list[SemanticTask] = []
        dag_eligible: list[SemanticTask] = []
        for task in dag.tasks:
            task_dag_index[task.id] = dag_index
            if task.compute is not None and type(task.compute.workload) is GemmWorkload:
                all_gemms.append(task)
                dag_gemms.append(task)
            if _eligible(task, output_consumers):
                eligible.append(task)
                dag_eligible.append(task)
        gemms_by_dag.append(tuple(dag_gemms))
        eligible_by_dag.append(tuple(dag_eligible))

    def gemm_cycles(task: SemanticTask, divisor: int = 1) -> int:
        return estimate_gemm_cycles(
            task, resolved_timing, parallel_divisor=divisor
        )

    identity_compute = max(
        (sum(gemm_cycles(task) for task in dag_gemms) for dag_gemms in gemms_by_dag),
        default=0,
    )
    common_issue = max(
        (len(dag_gemms) * resolved_timing.sync_issue_cycles for dag_gemms in gemms_by_dag),
        default=0,
    )
    fixed_pipeline = resolved_timing.fixed_pipeline_cycles
    memory_cycles = max(
        (
            sum(
                task.kind is SemanticTaskKind.DMA_IN
                for task in dag.tasks
            ) * resolved_timing.hbm_load_setup_cycles
            + (
                sum(
                    task.bytes
                    for task in dag.tasks
                    if task.kind is SemanticTaskKind.DMA_IN
                )
                + resolved_timing.hbm_load_bytes_per_cycle - 1
            ) // resolved_timing.hbm_load_bytes_per_cycle
            for dag in projection.dags
        ),
        default=0,
    )
    identity_overlap = min(identity_compute, memory_cycles)
    identity_cost = IntraDieV2AnalyticCost(
        compute_cycles=identity_compute,
        memory_cycles=memory_cycles,
        compute_memory_overlap_cycles=identity_overlap,
        transport_cycles=0,
        reduction_cycles=0,
        sync_cycles=common_issue,
        fixed_pipeline_cycles=fixed_pipeline,
        predicted_makespan_cycles=(
            identity_compute + memory_cycles - identity_overlap
            + common_issue + fixed_pipeline
        ),
    )
    identity = IntraDieV2Candidate.create(
        kind=IntraDieV2CandidateKind.IDENTITY,
        split_k_parts=1,
        enable_reduce=False,
        enable_double_buffer=False,
        enable_streaming_reduce=False,
        enable_tree_reduce=False,
        enable_direct_dma=False,
        eligible_gemm_count=len(eligible),
        analytic_cost=identity_cost,
    )

    if mode is IntraDieOptimizationMode.OFF:
        candidates = (identity,)
        rejections: tuple[IntraDieV2CandidateRejection, ...] = ()
        selected = identity
        reason = "off_identity_only"
    else:
        retained: list[IntraDieV2Candidate] = [identity]
        rejected: list[IntraDieV2CandidateRejection] = []
        eligible_ids = {task.id for task in eligible}
        for (
            candidate_name,
            parts,
            enable_reduce,
            enable_double_buffer,
            enable_streaming_reduce,
            enable_tree_reduce,
            enable_direct_dma,
        ) in configurations:
            rejection_reason: str | None = None
            if not eligible:
                rejection_reason = "no_eligible_gemm"
            elif any(
                (
                    task.compute.workload.logical_shape[2] % parts
                    or task.compute.workload.rank_shape[2] % parts
                )
                for task in eligible
                if task.compute is not None
                and type(task.compute.workload) is GemmWorkload
            ):
                rejection_reason = "k_not_divisible"

            if rejection_reason is not None:
                if mode is IntraDieOptimizationMode.FORCE and candidate_name == force_candidate:
                    raise SchemaError(
                        "forced split-K candidate is infeasible",
                        path=f"options.split_k_parts[{parts}]",
                    )
                rejected.append(IntraDieV2CandidateRejection.create(
                    candidate_name=candidate_name,
                    split_k_parts=parts,
                    enable_double_buffer=enable_double_buffer,
                    reason=rejection_reason,
                    identity_predicted_makespan_cycles=identity_cost.predicted_makespan_cycles,
                    candidate_predicted_makespan_cycles=None,
                    compute_savings_cycles=0,
                    overhead_cycles=0,
                ))
                continue

            split_compute = max(
                (
                    sum(
                        gemm_cycles(
                            task,
                            min(
                                parts,
                                _available_compute_groups(
                                    ir1,
                                    projection.dags[task_dag_index[task.id]].die_id,
                                    double_buffer=enable_double_buffer,
                                ),
                                compute_groups_per_die,
                            )
                            if task.id in eligible_ids else 1,
                        )
                        for task in dag_gemms
                    )
                    for dag_gemms in gemms_by_dag
                ),
                default=0,
            )
            transport_bytes_by_dag = [0] * len(projection.dags)
            transport_operations_by_dag = [0] * len(projection.dags)
            reduction_bytes_by_dag = [0] * len(projection.dags)
            for task in eligible:
                assert task.compute is not None
                workload = task.compute.workload
                assert type(workload) is GemmWorkload
                dtype_bytes = _DTYPE_BYTES.get(workload.dtype)
                if dtype_bytes is None:
                    raise SchemaError(
                        "split-K analytic model supports FP16/FP32",
                        path=f"projection.tasks[{task.id}]",
                    )
                m, n, k = workload.rank_shape
                dag_index = task_dag_index[task.id]
                # Exact first implementation: each M row is packed per part;
                # operand and partial handoffs each close through send/recv/wait.
                input_elements = 0 if enable_direct_dma else m * k + k * n
                transport_bytes_by_dag[dag_index] += (
                    input_elements + (parts - 1) * m * n
                ) * dtype_bytes
                transport_operations_by_dag[dag_index] += (
                    (0 if enable_direct_dma else parts * m + 4 * parts)
                    + 3 * (parts - 1)
                    + (2 * parts if enable_double_buffer else 0)
                )
                if enable_reduce:
                    reduction_bytes_by_dag[dag_index] += parts * m * n * dtype_bytes
            def transport_parallelism(dag_index: int) -> int:
                groups = min(
                    parts,
                    _available_compute_groups(
                        ir1,
                        projection.dags[dag_index].die_id,
                        double_buffer=enable_double_buffer,
                    ),
                    compute_groups_per_die,
                )
                if enable_tree_reduce:
                    return max(1, groups // 2)
                return min(3, groups) if groups > 2 else 1

            transport_cycles = max(
                (
                    (
                        estimate_transport_cycles(
                            payload, resolved_timing,
                            operation_count=transport_operations_by_dag[index],
                        )
                        + transport_parallelism(index)
                        - 1
                    ) // transport_parallelism(index)
                    if payload else 0
                    for index, payload in enumerate(transport_bytes_by_dag)
                ),
                default=0,
            )
            reduction_cycles = max(
                (
                    (estimate_reduce_cycles(payload, resolved_timing)
                     + max(1, compute_groups_per_die // 2) - 1)
                    // max(1, compute_groups_per_die // 2)
                    if payload and enable_tree_reduce
                    else estimate_reduce_cycles(payload, resolved_timing)
                    if payload else 0
                    for payload in reduction_bytes_by_dag
                ),
                default=0,
            )
            sync_cycles = max(
                (
                    len(dag_eligible) * (
                        parts.bit_length() if enable_tree_reduce else parts
                    ) * resolved_timing.sync_issue_cycles
                    * (3 if enable_double_buffer else 1)
                    for dag_eligible in eligible_by_dag
                ),
                default=0,
            )
            if not enable_streaming_reduce:
                sync_cycles += reduction_cycles
            predicted = (
                split_compute + memory_cycles - min(split_compute, memory_cycles)
                + transport_cycles + reduction_cycles
                + sync_cycles + fixed_pipeline
            )
            candidate = IntraDieV2Candidate.create(
                kind=IntraDieV2CandidateKind.SPLIT_K_FALLBACK,
                split_k_parts=parts,
                enable_reduce=enable_reduce,
                enable_double_buffer=enable_double_buffer,
                enable_streaming_reduce=enable_streaming_reduce,
                enable_tree_reduce=enable_tree_reduce,
                enable_direct_dma=enable_direct_dma,
                eligible_gemm_count=len(eligible),
                analytic_cost=IntraDieV2AnalyticCost(
                    compute_cycles=split_compute,
                    memory_cycles=memory_cycles,
                    compute_memory_overlap_cycles=min(
                        split_compute, memory_cycles
                    ),
                    transport_cycles=transport_cycles,
                    reduction_cycles=reduction_cycles,
                    sync_cycles=sync_cycles,
                    fixed_pipeline_cycles=fixed_pipeline,
                    predicted_makespan_cycles=predicted,
                ),
            )
            compute_savings = max(
                0,
                identity_compute + memory_cycles - identity_overlap
                - (
                    split_compute + memory_cycles
                    - min(split_compute, memory_cycles)
                ),
            )
            overhead = max(
                0,
                transport_cycles + reduction_cycles + sync_cycles - common_issue,
            )
            profitable = (
                compute_savings > overhead
                and predicted < identity_cost.predicted_makespan_cycles
            )
            if (
                mode is IntraDieOptimizationMode.AUTO
                and not require_full_compute_groups
                and not profitable
            ):
                rejected.append(IntraDieV2CandidateRejection.create(
                    candidate_name=candidate_name,
                    split_k_parts=parts,
                    enable_double_buffer=enable_double_buffer,
                    reason="break_even_not_met",
                    identity_predicted_makespan_cycles=identity_cost.predicted_makespan_cycles,
                    candidate_predicted_makespan_cycles=predicted,
                    compute_savings_cycles=compute_savings,
                    overhead_cycles=overhead,
                ))
            else:
                retained.append(candidate)

        candidates = tuple(sorted(
            retained,
            key=lambda item: (
                item.analytic_cost.predicted_makespan_cycles,
                0 if item.kind is IntraDieV2CandidateKind.IDENTITY else 1,
                item.split_k_parts, item.enable_double_buffer,
                item.enable_streaming_reduce, item.enable_tree_reduce,
                item.enable_direct_dma, item.id,
            ),
        ))
        rejections = tuple(sorted(
            rejected,
            key=lambda item: (
                item.candidate_name, item.split_k_parts,
                item.enable_double_buffer, item.id,
            ),
        ))
        if mode is IntraDieOptimizationMode.FORCE:
            if force_candidate == "identity":
                selected = identity
                reason = "explicit_identity_request"
            else:
                selected = next(
                    item for item in candidates
                    if item.kind is IntraDieV2CandidateKind.SPLIT_K_FALLBACK
                    and item.enable_double_buffer == (force_candidate == "split_k_double_buffer")
                    and item.enable_streaming_reduce
                    == (force_candidate != "split_k_barrier")
                    and item.enable_tree_reduce
                    == (force_candidate == "split_k_tree_direct_dma")
                    and item.enable_direct_dma
                    == (force_candidate == "split_k_tree_direct_dma")
                )
                reason = "explicit_split_k_request"
        else:
            selectable = (
                tuple(
                    item for item in candidates
                    if item.kind is not IntraDieV2CandidateKind.IDENTITY
                )
                if require_full_compute_groups
                else candidates
            )
            if not selectable:
                raise SchemaError(
                    "AUTO has no feasible full-compute-group candidate",
                    path="options.require_full_compute_groups",
                )
            selected = selectable[0]
            reason = (
                "auto_minimum_required_compute_groups"
                if require_full_compute_groups
                else (
                    "auto_identity_no_profitable_candidate"
                    if selected.kind is IntraDieV2CandidateKind.IDENTITY
                    else "auto_minimum_predicted_makespan"
                )
            )

    generated_count = len(candidates) + len(rejections)
    if generated_count > max_candidates:
        raise SchemaError("candidate count exceeds max_candidates", path="options.max_candidates")
    decision_budget = (
        budget
        if max_candidates == budget.max_candidates
        else IntraDieV2SearchBudget(
            max_candidates=max_candidates,
            simulator_call_budget=budget.simulator_call_budget,
            analytic_model_version=budget.analytic_model_version,
        )
    )
    result = IntraDieV2SearchDecision.create(
        source_projection_id=projection.id,
        source_ir1_id=ir1.id,
        budget=decision_budget,
        timing_model_ref=resolved_timing.id,
        hardware_digest=resolved_timing.hardware_digest,
        simulation_digest=resolved_timing.simulation_digest,
        mode=mode,
        candidates=candidates,
        rejected_candidates=rejections,
        selected_candidate_ref=selected.id,
        selection_reason=reason,
        generated_candidate_count=generated_count,
        full_analytic_evaluation_count=generated_count,
        simulator_calls_during_search=0,
        reserved_simulator_calls_for_final_evidence=(
            2 if legacy_force else 3
        ),
        require_full_compute_groups=require_full_compute_groups,
    )
    result.validate()
    return result


__all__ = ["evaluate_intra_die_v2_candidates"]
