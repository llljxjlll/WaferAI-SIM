"""Build joint MoE deployment feasibility from tentative whole lowerings."""

from __future__ import annotations

from ..errors import SchemaError
from ..lowering.moe_swizzle_workload_abi import build_moe_swizzle_workload_abi
from ..schema.ir0 import FusionPattern
from ..schema.ir1 import IR1
from ..schema.swizzle_moe import MoeSwizzleDecision
from ..schema.swizzle_moe_execution import MoeScaleExecution
from ..schema.swizzle_moe_placement import (
    MoeWholePairFeasibility,
    MoeWholePairPlacementFeasibility,
    MoeWholePairPlacementReason,
    build_moe_candidate_core_lifecycle_floor,
    build_moe_candidate_dynamic_root_keys,
    build_moe_swizzle_whole_pair_feasibility,
    build_moe_swizzle_workload_placement,
)
from ..schema.serde import canonical_digest
from ..schema.swizzle_moe_scale import MoeSwizzleScaleSpec
from ..schema.swizzle_moe_state import MoeSwizzleWorkloadStateABI
from .build_moe_scale_swizzle_overlay import build_moe_scale_swizzle_overlay
from .build_moe_swizzle_workload_value_bridge import (
    build_moe_swizzle_workload_value_bridge,
)
from .project_moe_scale_swizzle_ir2 import project_moe_scale_swizzle_ir2
from .project_moe_swizzle_whole_workload import (
    project_moe_swizzle_whole_workload,
)
from .schedule_moe_swizzle_workload_endpoints import (
    schedule_moe_swizzle_workload_endpoints,
)
from .schedule_moe_swizzle_workload_storage_reuse import (
    schedule_moe_swizzle_workload_storage_reuse,
)


_PATTERNS = (
    FusionPattern.MOE_DISPATCH_GEMM,
    FusionPattern.MOE_GEMM_COMBINE,
)


_PLACEMENT_FAILURES = {
    (
        "projection.values",
        "value crosses cores without an explicit LOCAL_COPY",
    ): MoeWholePairPlacementReason.ORDINARY_VALUE_CROSS_CORE,
    (
        "build_moe_swizzle_workload_placement.workload_projection",
        "preserved action spans multiple M-block owners without LOCAL_COPY",
    ): MoeWholePairPlacementReason.PRESERVED_OWNER_MISMATCH,
}


def _placement_infeasible_witness(
    candidate_refs: tuple[str, str],
    workload: object,
    replacement: object,
    error: SchemaError,
    *,
    placement: tuple[object, ...] | None,
    dynamic_sram_capacity_bytes: int,
) -> MoeWholePairFeasibility:
    reason = _PLACEMENT_FAILURES.get((error.path, error.message))
    if (
        reason is None
        and error.path.startswith("value_bridge.bindings[")
        and error.message.startswith("value bridge lacks physical root ")
        and "linked_owners=" in error.message
    ):
        reason = MoeWholePairPlacementReason.ORDINARY_VALUE_CROSS_CORE
    if (
        reason is None
        and error.path.startswith(
            "schedule_moe_swizzle_workload_storage_reuse.bindings["
        )
        and error.message == "SWIGLU occupant crosses cores without LOCAL_COPY"
    ):
        reason = MoeWholePairPlacementReason.ORDINARY_VALUE_CROSS_CORE
    if reason is None:
        raise error
    placement_witness = MoeWholePairPlacementFeasibility.create(
        workload_projection_id=workload.id,
        replacement_projection_id=replacement.id,
        placement_digest=(
            None if placement is None else canonical_digest(placement)
        ),
        reason=reason,
        failure_path=error.path,
        failure_message=error.message,
        feasible=False,
    )
    return MoeWholePairFeasibility.create(
        candidate_refs=candidate_refs,
        placement=placement_witness,
        endpoint=None,
        workload_abi_id=None,
        dynamic_root_keys=(),
        dynamic_sram_high_water_bytes=0,
        dynamic_sram_capacity_bytes=dynamic_sram_capacity_bytes,
        storage_color_depths=(),
        core_lifecycle_counts=(),
        whole_physical_root_count=0,
        whole_alloc_count=0,
        whole_free_count=0,
        feasible=False,
    )


def _ordered_decisions(
    execution: MoeScaleExecution,
    decisions: tuple[MoeSwizzleDecision, ...],
) -> tuple[MoeSwizzleDecision, MoeSwizzleDecision]:
    if type(execution) is not MoeScaleExecution:
        raise SchemaError(
            "requires exact MoeScaleExecution",
            path="moe_pair_feasibility.execution",
        )
    execution.validate("moe_pair_feasibility.execution")
    if type(decisions) is not tuple or len(decisions) != 2:
        raise SchemaError(
            "requires exactly two decisions",
            path="moe_pair_feasibility.decisions",
        )
    by_pattern = {}
    for index, decision in enumerate(decisions):
        if type(decision) is not MoeSwizzleDecision:
            raise SchemaError(
                "requires exact MoeSwizzleDecision",
                path=f"moe_pair_feasibility.decisions[{index}]",
            )
        decision.validate(f"moe_pair_feasibility.decisions[{index}]")
        if decision.problem.source_execution_id != execution.id:
            raise SchemaError(
                "decision belongs to another execution",
                path=f"moe_pair_feasibility.decisions[{index}]",
            )
        pattern = decision.problem.region.pattern
        if pattern in by_pattern:
            raise SchemaError(
                "decision pattern is duplicated",
                path="moe_pair_feasibility.decisions",
            )
        by_pattern[pattern] = decision
    if set(by_pattern) != set(_PATTERNS):
        raise SchemaError(
            "decisions do not cover Dispatch and Combine",
            path="moe_pair_feasibility.decisions",
        )
    return by_pattern[_PATTERNS[0]], by_pattern[_PATTERNS[1]]


def _tentative_decision(
    decision: MoeSwizzleDecision,
    candidate_ref: str,
) -> MoeSwizzleDecision:
    candidates = {item.id: item for item in decision.ranked_candidates}
    selected = candidates.get(candidate_ref)
    if selected is None:
        raise SchemaError(
            "pair candidate is outside its source decision",
            path="moe_pair_feasibility.candidate_refs",
        )
    ranked = (selected,) + tuple(
        item for item in decision.ranked_candidates if item.id != candidate_ref
    )
    return MoeSwizzleDecision.create(
        problem=decision.problem,
        baseline=decision.baseline,
        ranked_candidates=ranked,
        selected_candidate_ref=candidate_ref,
        decision_reason=decision.decision_reason,
        performance_complete=decision.performance_complete,
    )


def build_moe_swizzle_pair_feasibility_witness(
    ir1: IR1,
    execution: MoeScaleExecution,
    spec: MoeSwizzleScaleSpec,
    decisions: tuple[MoeSwizzleDecision, ...],
    state_abi: MoeSwizzleWorkloadStateABI,
    candidate_refs: tuple[str, str],
) -> MoeWholePairFeasibility:
    """Lower one ordered candidate pair and freeze endpoint/SRAM truth.

    This builder intentionally does not perform admission: infeasible endpoint
    width or SRAM high-water is returned as a typed ``feasible=False`` witness
    so the joint selector can rank the complete candidate cross-product.
    """

    ir1.validate("moe_pair_feasibility.ir1")
    spec.validate("moe_pair_feasibility.spec")
    state_abi.validate("moe_pair_feasibility.state_abi")
    ordered = _ordered_decisions(execution, decisions)
    if type(candidate_refs) is not tuple or len(candidate_refs) != 2:
        raise SchemaError(
            "candidate refs must be ordered Dispatch/Combine pair",
            path="moe_pair_feasibility.candidate_refs",
        )
    tentative = tuple(
        _tentative_decision(decision, candidate_ref)
        for decision, candidate_ref in zip(ordered, candidate_refs, strict=True)
    )
    if state_abi.source_ir1_id != ir1.id:
        raise SchemaError(
            "state ABI belongs to another IR1",
            path="moe_pair_feasibility.state_abi",
        )
    hardware_facts = ordered[0].problem.hardware_facts
    capacity = ordered[0].problem.endpoint_session_capacity
    sram_capacity = ordered[0].problem.sram_capacity_bytes
    if any(
        decision.problem.hardware_facts != hardware_facts
        or decision.problem.endpoint_session_capacity != capacity
        or decision.problem.sram_capacity_bytes != sram_capacity
        for decision in ordered
    ):
        raise SchemaError(
            "joint decisions disagree on hardware resource truth",
            path="moe_pair_feasibility.decisions",
        )

    overlay = build_moe_scale_swizzle_overlay(execution, tentative)
    replacement = project_moe_scale_swizzle_ir2(
        overlay, execution, spec, tentative,
        endpoint_session_capacity=capacity,
    )
    workload = project_moe_swizzle_whole_workload(
        overlay, execution, replacement, state_abi,
    )
    try:
        initial_placement = build_moe_swizzle_workload_placement(
            ir1, workload, replacement, hardware_facts,
        )
    except SchemaError as error:
        return _placement_infeasible_witness(
            candidate_refs, workload, replacement, error,
            placement=None,
            dynamic_sram_capacity_bytes=sram_capacity,
        )
    workload = schedule_moe_swizzle_workload_endpoints(
        workload, replacement, initial_placement,
        capacity_per_core=capacity,
    )
    try:
        placement = build_moe_swizzle_workload_placement(
            ir1, workload, replacement, hardware_facts,
        )
        bridge = build_moe_swizzle_workload_value_bridge(
            execution, workload, replacement,
        )
        workload = schedule_moe_swizzle_workload_storage_reuse(
            workload, replacement, bridge, placement,
        )
        placement = build_moe_swizzle_workload_placement(
            ir1, workload, replacement, hardware_facts,
        )
        bridge = build_moe_swizzle_workload_value_bridge(
            execution, workload, replacement,
        )
        workload_abi = build_moe_swizzle_workload_abi(
            ir1, workload, replacement, state_abi, bridge, hardware_facts,
        )
    except SchemaError as error:
        return _placement_infeasible_witness(
            candidate_refs, workload, replacement, error,
            placement=locals().get("placement", initial_placement),
            dynamic_sram_capacity_bytes=sram_capacity,
        )
    result = build_moe_swizzle_whole_pair_feasibility(
        candidate_refs, workload, replacement, placement, workload_abi,
        capacity_per_core=capacity,
        dynamic_sram_capacity_bytes=sram_capacity,
    )
    actual = {
        item.runtime_core_id: item for item in result.core_lifecycle_counts
    }
    expected_dynamic_roots = tuple(sorted({
        key
        for decision in tentative
        for key in build_moe_candidate_dynamic_root_keys(
            decision.problem, decision.ranked_candidates[0],
        )
    }))
    if result.dynamic_root_keys != expected_dynamic_roots:
        raise SchemaError(
            "whole dynamic roots differ from pure candidate roots: "
            f"pair={candidate_refs!r}, "
            f"missing={tuple(sorted(set(expected_dynamic_roots) - set(result.dynamic_root_keys)))!r}, "
            f"unexpected={tuple(sorted(set(result.dynamic_root_keys) - set(expected_dynamic_roots)))!r}",
            path="moe_pair_feasibility.dynamic_root_keys",
        )
    floor = {}
    for decision in tentative:
        candidate = decision.ranked_candidates[0]
        for item in build_moe_candidate_core_lifecycle_floor(
            decision.problem, candidate,
        ):
            counts = floor.setdefault(item.runtime_core_id, [0, 0, 0])
            counts[0] += item.alloc_count
            counts[1] += item.bind_count
            counts[2] += item.free_count
    for runtime_core_id, counts in sorted(floor.items()):
        observed = actual.get(runtime_core_id)
        if observed is None or any(
            actual_value < floor_value
            for actual_value, floor_value in zip(
                (observed.alloc_count, observed.bind_count, observed.free_count),
                counts,
                strict=True,
            )
        ):
            raise SchemaError(
                "whole lifecycle is below its pure candidate floor: "
                f"pair={candidate_refs!r}, floor={tuple(counts)!r}, "
                f"actual={None if observed is None else (observed.alloc_count, observed.bind_count, observed.free_count)!r}, "
                f"roots={tuple((item.family, item.slot, item.allocate, item.free) for item in workload_abi.roots if item.runtime_core_id == runtime_core_id)!r}",
                path=(
                    "moe_pair_feasibility.core_lifecycle_counts"
                    f"[{runtime_core_id}]"
                ),
            )
    return result


def build_moe_swizzle_pair_feasibility_witnesses(
    ir1: IR1,
    execution: MoeScaleExecution,
    spec: MoeSwizzleScaleSpec,
    decisions: tuple[MoeSwizzleDecision, ...],
    state_abi: MoeSwizzleWorkloadStateABI,
) -> tuple[MoeWholePairFeasibility, ...]:
    """Build the canonical complete Dispatch x Combine witness set."""

    dispatch, combine = _ordered_decisions(execution, decisions)
    pairs = tuple(sorted(
        (left.id, right.id)
        for left in dispatch.ranked_candidates
        for right in combine.ranked_candidates
    ))
    return tuple(
        build_moe_swizzle_pair_feasibility_witness(
            ir1, execution, spec, (dispatch, combine), state_abi, pair,
        )
        for pair in pairs
    )


__all__ = [
    "build_moe_swizzle_pair_feasibility_witness",
    "build_moe_swizzle_pair_feasibility_witnesses",
]
