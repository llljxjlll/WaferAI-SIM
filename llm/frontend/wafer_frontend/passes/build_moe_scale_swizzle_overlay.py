"""Generalized MoeScaleExecution overlay construction for C1+ scale cases."""

from __future__ import annotations

from collections import Counter

from ..errors import SchemaError
from ..schema.ir0 import FusionPattern
from ..schema.swizzle import SwizzleActionKind, SwizzleAlgorithm
from ..schema.swizzle_moe import (
    MoeActionWitness, MoeRankProgram, MoeSwizzleDecision,
    MoeSwizzleWorkloadSelection,
)
from ..schema.swizzle_moe_execution import (
    MoeScaleExecution,
    MoeScaleExecutionAction,
    MoeScaleExecutionActionKind,
    MoeScaleExecutionFlowRole,
    MoeScaleExecutionMode,
)
from ..schema.swizzle_moe_plan import (
    MoeBoundaryValueBinding,
    MoeSwizzleDeploymentMode,
    MoeSwizzleDeploymentSelection,
    MoeSwizzleLinkedAction,
    MoeSwizzleOverlay,
)
from .build_moe_swizzle_overlay import _PATTERNS, _topological, _unique
from ..policies.swizzle.moe_cost import build_moe_action_owner_map
from ..policies.swizzle.moe_unfused import _endpoint_session_dependencies


def build_moe_scale_swizzle_overlay(
    execution: MoeScaleExecution,
    decisions: tuple[MoeSwizzleDecision, ...],
    workload_selection: MoeSwizzleWorkloadSelection | None = None,
) -> MoeSwizzleOverlay:
    """Replace both regions directly in generalized execution truth.

    The LiteMoE 92-action bridge is intentionally absent from this path.
    """
    if type(execution) is not MoeScaleExecution:
        raise SchemaError("requires exact generalized execution", path="moe_scale_overlay.execution")
    execution.validate("moe_scale_overlay.execution")
    if not execution.execution_ready:
        raise SchemaError("capacity-probe execution is not deployable", path="moe_scale_overlay.execution")
    if type(decisions) is not tuple or len(decisions) != 2:
        raise SchemaError("requires exactly two decisions", path="moe_scale_overlay.decisions")
    for index, decision in enumerate(decisions):
        if type(decision) is not MoeSwizzleDecision:
            raise SchemaError("requires typed decisions", path=f"moe_scale_overlay.decisions[{index}]")
        decision.validate(f"moe_scale_overlay.decisions[{index}]")
        if decision.problem.source_execution_id != execution.id:
            raise SchemaError("decision belongs to another execution", path=f"moe_scale_overlay.decisions[{index}]")
    by_pattern = {item.problem.region.pattern: item for item in decisions}
    if set(by_pattern) != set(_PATTERNS):
        raise SchemaError("decisions must cover both MoE regions", path="moe_scale_overlay.decisions")
    ordered_decisions = tuple(by_pattern[item] for item in _PATTERNS)
    if workload_selection is None:
        candidate_refs = tuple(item.selected_candidate_ref for item in ordered_decisions)
    else:
        if type(workload_selection) is not MoeSwizzleWorkloadSelection:
            raise SchemaError(
                "workload deployment requires an exact joint selection",
                path="moe_scale_overlay.workload_selection",
            )
        workload_selection.validate("moe_scale_overlay.workload_selection")
        if (
            workload_selection.source_dispatch_decision_id,
            workload_selection.source_combine_decision_id,
        ) != tuple(item.id for item in ordered_decisions):
            raise SchemaError(
                "joint selection belongs to another decision pair",
                path="moe_scale_overlay.workload_selection",
            )
        candidate_refs = (
            workload_selection.selected_dispatch_candidate_ref,
            workload_selection.selected_combine_candidate_ref,
        )
    candidates = tuple(
        next(
            (candidate for candidate in decision.ranked_candidates if candidate.id == candidate_ref),
            None,
        )
        for decision, candidate_ref in zip(ordered_decisions, candidate_refs, strict=True)
    )
    if any(candidate is None for candidate in candidates):
        raise SchemaError(
            "selected workload candidate is unavailable from its decision",
            path="moe_scale_overlay.workload_selection",
        )
    for candidate, decision in zip(candidates, ordered_decisions, strict=True):
        assert candidate is not None
        candidate.validate_against(decision.problem, "moe_scale_overlay.candidate")

    source = {item.id: item for item in execution.actions}
    region_refs = tuple(
        ref for decision in ordered_decisions for ref in decision.problem.region.member_refs
    )
    originals = tuple(
        ref for candidate in candidates for program in candidate.rank_programs
        for action in program.actions for ref in action.original_action_refs
    )
    if (
        len(region_refs) != len(set(region_refs))
        or any(ref not in source for ref in region_refs)
        or len(originals) != len(set(originals))
        or set(originals) != set(region_refs)
    ):
        raise SchemaError("candidate/source region partition is not exact", path="moe_scale_overlay.regions")

    token_count = sum(item.kind is MoeScaleExecutionActionKind.DMA_IN for item in execution.actions) // 3
    remote_count = len(execution.flows) // 2
    if tuple(len(item.problem.region.member_refs) for item in ordered_decisions) != (
        3 * token_count + 3 * remote_count,
        token_count + 3 * remote_count,
    ):
        raise SchemaError("region cardinality formula drifted", path="moe_scale_overlay.regions")
    preserved = tuple(item for item in execution.actions if item.id not in set(region_refs))
    expected = Counter({
        MoeScaleExecutionActionKind.DMA_IN: 3 * token_count,
    })
    if execution.mode is MoeScaleExecutionMode.TRAIN_FORWARD:
        expected[MoeScaleExecutionActionKind.TAPE_COPY] = token_count
    if Counter(item.kind for item in preserved) != expected:
        raise SchemaError("preserved action formula drifted", path="moe_scale_overlay.preserved")
    if len(execution.actions) != 7 * token_count + 6 * remote_count + (
        token_count if execution.mode is MoeScaleExecutionMode.TRAIN_FORWARD else 0
    ):
        raise SchemaError("execution action formula drifted", path="moe_scale_overlay.execution")

    programs = tuple(MoeRankProgram(rank, tuple(
        action for candidate in candidates for program in candidate.rank_programs
        if program.rank == rank for action in program.actions
    )) for rank in range(4))
    replacement = tuple(action for program in programs for action in program.actions)
    old_pattern = {
        action.id: decision.problem.region.pattern
        for decision, candidate in zip(
            ordered_decisions, candidates, strict=True,
        )
        for program in candidate.rank_programs
        for action in program.actions
    }
    owners = {}
    for decision, candidate in zip(
        ordered_decisions, candidates, strict=True,
    ):
        candidate_actions = tuple(
            action for program in candidate.rank_programs
            for action in program.actions
        )
        candidate_owners = build_moe_action_owner_map(
            decision.problem, candidate_actions,
        )
        if set(owners).intersection(candidate_owners):
            raise SchemaError(
                "replacement action ids collide across regions",
                path="moe_scale_overlay.sessions",
            )
        owners.update(candidate_owners)
    old_replacement_by_id = {item.id: item for item in replacement}
    old_source_owner = {
        ref: action.id
        for action in replacement
        for ref in action.original_action_refs
    }
    assignments = {
        pattern: {
            item.id: item
            for item in decision.problem.region.semantic_witness.traffic.assignments
        }
        for pattern, decision in by_pattern.items()
    }
    flow_by_token_role = {
        (item.token_index, item.role): item for item in execution.flows
    }

    def scheduling_sources(
        witness: MoeActionWitness,
    ) -> tuple[MoeScaleExecutionAction, ...]:
        if witness.original_action_refs:
            return tuple(source[ref] for ref in witness.original_action_refs)
        pattern = old_pattern[witness.id]
        tokens = {
            assignments[pattern][ref].token_index
            for ref in witness.assignment_refs
        }
        if witness.kind is SwizzleActionKind.LOCAL_COPY:
            return ()
        if witness.kind is SwizzleActionKind.COMP:
            role = (
                "down"
                if pattern is FusionPattern.MOE_GEMM_COMBINE else None
            )
            return tuple(item for item in execution.actions if (
                item.kind is MoeScaleExecutionActionKind.GEMM
                and item.token_index in tokens
                and item.expert_index == witness.expert_index
                and (role is None or item.role == role)
            ))
        if witness.kind in (
            SwizzleActionKind.SEND,
            SwizzleActionKind.RECV,
            SwizzleActionKind.WAIT,
        ):
            flow_role = (
                MoeScaleExecutionFlowRole.DISPATCH
                if pattern is FusionPattern.MOE_DISPATCH_GEMM
                else MoeScaleExecutionFlowRole.COMBINE
            )
            attr = {
                SwizzleActionKind.SEND: "send_action_ref",
                SwizzleActionKind.RECV: "recv_action_ref",
                SwizzleActionKind.WAIT: "wait_action_ref",
            }[witness.kind]
            return tuple(
                source[getattr(flow_by_token_role[(token, flow_role)], attr)]
                for token in sorted(tokens)
                if (token, flow_role) in flow_by_token_role
            )
        return ()

    base_dependencies = {}
    replacement_ids = set(old_replacement_by_id)
    for witness in replacement:
        semantic = scheduling_sources(witness)
        covered = set(witness.original_action_refs)
        external = {
            old_source_owner.get(dependency, dependency)
            for item in semantic
            for dependency in item.deps
            if covered and dependency not in covered
        }
        base_dependencies[witness.id] = set(witness.deps).union(
            external.intersection(replacement_ids)
        )
    strengthened = _endpoint_session_dependencies(
        ordered_decisions[0].problem,
        replacement,
        base_dependencies,
        owners=owners,
    )
    lane_dependencies = {
        ref: strengthened[ref] - base_dependencies[ref]
        for ref in strengthened
    }
    old_order = {action.id: index for index, action in enumerate(replacement)}
    by_old_id = {action.id: action for action in replacement}
    pending = set(old_order)
    rebuilt = {}
    ordered_replacement = []
    while pending:
        ready = sorted(
            (
                ref for ref in pending
                if strengthened[ref].issubset(rebuilt)
            ),
            key=old_order.__getitem__,
        )
        if not ready:
            raise SchemaError(
                "whole replacement endpoint scheduling creates a cycle",
                path="moe_scale_overlay.sessions",
            )
        ref = ready[0]
        action = by_old_id[ref]
        semantic = {
            name: getattr(action, name)
            for name in action.__dataclass_fields__
            if name not in ("schema_version", "id")
        }
        semantic["deps"] = tuple(
            rebuilt[dependency].id
            for dependency in sorted(
                set(action.deps).union(lane_dependencies[ref]),
                key=old_order.__getitem__,
            )
        )
        rebuilt[ref] = MoeActionWitness.create(**semantic)
        ordered_replacement.append(rebuilt[ref])
        pending.remove(ref)
    replacement = tuple(ordered_replacement)
    programs = tuple(
        MoeRankProgram(
            rank,
            tuple(action for action in replacement if action.rank == rank),
        )
        for rank in range(4)
    )
    source_owner = {
        ref: action.id for action in replacement for ref in action.original_action_refs
    }
    replacement_by_id = {item.id: item for item in replacement}
    action_pattern = {
        rebuilt[ref].id: pattern for ref, pattern in old_pattern.items()
    }
    assignments = {
        pattern: {item.id: item for item in decision.problem.region.semantic_witness.traffic.assignments}
        for pattern, decision in by_pattern.items()
    }
    flow_by_token_role = {(item.token_index, item.role): item for item in execution.flows}

    def semantic_sources(witness: MoeActionWitness) -> tuple[MoeScaleExecutionAction, ...]:
        if witness.original_action_refs:
            return tuple(source[ref] for ref in witness.original_action_refs)
        pattern = action_pattern[witness.id]
        tokens = {assignments[pattern][ref].token_index for ref in witness.assignment_refs}
        if witness.kind is SwizzleActionKind.LOCAL_COPY:
            dependency = (
                replacement_by_id.get(witness.deps[0])
                if len(witness.deps) == 1 else None
            )
            role = witness.work_role.removesuffix(".pack")
            assignment_ref = witness.assignment_refs[0] if len(witness.assignment_refs) == 1 else None
            if (
                pattern is not FusionPattern.MOE_DISPATCH_GEMM
                or witness.work_role not in ("gate.pack", "up.pack")
                or dependency is None
                or dependency.kind is not SwizzleActionKind.COMP
                or dependency.work_role != role
                or assignment_ref not in dependency.assignment_refs
                or witness.packed_value_ref != f"moe.pack.{assignment_ref}.{role}"
                or witness.original_action_refs
            ):
                raise SchemaError(
                    "empty-provenance packing copy is not exactly reconstructed",
                    path=f"moe_scale_overlay.replacement[{witness.id}]",
                )
            # This is a physical lowering action, not a second semantic owner.
            return ()
        if witness.kind is SwizzleActionKind.COMP:
            role = "down" if pattern is FusionPattern.MOE_GEMM_COMBINE else None
            found = tuple(item for item in execution.actions if (
                item.kind is MoeScaleExecutionActionKind.GEMM
                and item.token_index in tokens
                and item.expert_index == witness.expert_index
                and (role is None or item.role == role)
            ))
        elif witness.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV, SwizzleActionKind.WAIT):
            flow_role = (
                MoeScaleExecutionFlowRole.DISPATCH
                if pattern is FusionPattern.MOE_DISPATCH_GEMM
                else MoeScaleExecutionFlowRole.COMBINE
            )
            attr = {
                SwizzleActionKind.SEND: "send_action_ref",
                SwizzleActionKind.RECV: "recv_action_ref",
                SwizzleActionKind.WAIT: "wait_action_ref",
            }[witness.kind]
            found = tuple(
                source[getattr(flow_by_token_role[(token, flow_role)], attr)]
                for token in sorted(tokens) if (token, flow_role) in flow_by_token_role
            )
        else:
            found = ()
        if not found and witness.kind is not SwizzleActionKind.LOCAL_COPY:
            raise SchemaError("empty-provenance action is an orphan", path=f"moe_scale_overlay.replacement[{witness.id}]")
        return found

    def mapped(ref: str) -> str:
        return source_owner.get(ref, ref)

    producer = {ref: item.id for item in execution.actions for ref in item.write_values}
    consumers: dict[str, list[str]] = {}
    for item in execution.actions:
        for ref in item.read_values:
            consumers.setdefault(ref, []).append(item.id)
    terminal_values = {item.value_ref for item in execution.terminals}
    linked = []
    for witness in replacement:
        semantic = semantic_sources(witness)
        covered = set(witness.original_action_refs)
        external_deps = _unique(
            mapped(dep) for item in semantic for dep in item.deps
            if covered and dep not in covered and mapped(dep) != witness.id
        )
        reads = _unique(
            ref for item in semantic for ref in item.read_values
            if producer.get(ref) not in covered
        )
        writes = _unique(
            ref for item in semantic for ref in item.write_values
            if ref in terminal_values or not consumers.get(ref) or not set(consumers[ref]).issubset(covered)
        )
        linked.append(MoeSwizzleLinkedAction(
            witness.id, witness.rank, witness.rank, f"replacement.{witness.kind.value}", False,
            witness.original_action_refs, _unique(witness.deps + external_deps), (), reads, writes,
            witness.assignment_refs, witness.expert_index, witness.tile_index,
            witness.n_block_index, witness.packet_ref, witness.stage, witness.pivot_rank,
        ))
    for item in preserved:
        linked.append(MoeSwizzleLinkedAction(
            item.id, item.die_id, item.die_id, f"preserved.{item.kind.value}", True,
            (item.id,), _unique(mapped(ref) for ref in item.deps), (),
            item.read_values, item.write_values, (), item.expert_index, item.token_index,
            None, None, None, None,
        ))
    linked = list(_topological(tuple(linked)))
    produced: dict[str, list[str]] = {}
    consumed: dict[str, list[str]] = {}
    for item in linked:
        for ref in item.write_value_refs:
            produced.setdefault(ref, []).append(item.id)
        for ref in item.read_value_refs:
            consumed.setdefault(ref, []).append(item.id)
    linked_by_id = {item.id: item for item in linked}
    boundaries = []
    for ref in _unique(tuple(produced) + tuple(consumed) + tuple(sorted(terminal_values))):
        producer_refs = tuple(produced.get(ref, ()))
        consumer_refs = tuple(consumed.get(ref, ()))
        producer_modes = {linked_by_id[item].preserved for item in producer_refs}
        consumer_modes = {linked_by_id[item].preserved for item in consumer_refs}
        if ref in terminal_values or not producer_refs or (
            producer_modes and consumer_modes and producer_modes != consumer_modes
        ):
            boundaries.append(MoeBoundaryValueBinding(
                ref, ref, producer_refs, consumer_refs, ref in terminal_values
            ))
    selections = tuple(MoeSwizzleDeploymentSelection(
        decision.problem.region.id, decision.id, candidate.id,
        MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE
        if candidate.algorithm is SwizzleAlgorithm.UNFUSED
        else (
            MoeSwizzleDeploymentMode.FUSED_AUTO
            if workload_selection is not None
            else MoeSwizzleDeploymentMode.FUSED_FORCED
        ),
        candidate.double_buffer,
    ) for decision, candidate in zip(ordered_decisions, candidates, strict=True))
    return MoeSwizzleOverlay.create(
        source_execution_id=execution.id,
        source_forward_execution_id=execution.id,
        source_workload_selection_id=(
            None if workload_selection is None else workload_selection.id
        ),
        decision_refs=tuple(item.id for item in ordered_decisions),
        deployment_selections=selections,
        replaced_action_refs=region_refs,
        replacement_rank_programs=programs,
        preserved_action_refs=tuple(item.id for item in preserved),
        boundary_value_bindings=tuple(boundaries),
        linked_actions=tuple(linked),
        terminal_value_refs=tuple(sorted(terminal_values)),
        replacement_packetization=tuple(
            packet for candidate in candidates for packet in candidate.packetization
        ),
        replacement_tile_schedule=tuple(
            tile for candidate in candidates for tile in candidate.tile_schedule
        ),
        train_forward=execution.mode is MoeScaleExecutionMode.TRAIN_FORWARD,
    )


__all__ = ["build_moe_scale_swizzle_overlay"]
