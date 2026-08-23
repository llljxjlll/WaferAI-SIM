"""Replace both selected MoE regions exactly once in a LiteMoE global DAG."""

from __future__ import annotations

from collections import Counter
import re

from ..errors import SchemaError
from ..schema.ir0 import FusionPattern
from ..schema.lite_moe_dp4_execution import (
    LiteMoeDp4ExecutionCase,
    LiteMoeDp4TaskKind,
)
from ..schema.lite_moe_dp4_train_forward import LiteMoeDp4TrainForward
from ..schema.swizzle import SwizzleAlgorithm
from ..schema.swizzle_moe import (
    MoeActionWitness,
    MoeRankProgram,
    MoeSwizzleCandidate,
    MoeSwizzleDecision,
)
from ..schema.swizzle_moe_execution import (
    MoeScaleExecution,
    MoeScaleExecutionAction,
    MoeScaleExecutionActionKind,
)
from ..schema.swizzle_moe_plan import (
    MoeBoundaryValueBinding,
    MoeSwizzleCandidateAdapter,
    MoeSwizzleDeploymentMode,
    MoeSwizzleDeploymentSelection,
    MoeSwizzleLinkedAction,
    MoeSwizzleOverlay,
)


_PATTERNS = (
    FusionPattern.MOE_DISPATCH_GEMM,
    FusionPattern.MOE_GEMM_COMBINE,
)


def _selected(decision: MoeSwizzleDecision) -> MoeSwizzleCandidate:
    candidates = {
        item.id: item for item in decision.ranked_candidates
    }
    try:
        return candidates[decision.selected_candidate_ref]
    except KeyError as error:
        raise SchemaError(
            "decision selected candidate is absent",
            path="build_moe_swizzle_overlay.decisions",
        ) from error


def _unique(refs: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(refs))


def _topological(
    actions: tuple[MoeSwizzleLinkedAction, ...],
) -> tuple[MoeSwizzleLinkedAction, ...]:
    by_id = {item.id: item for item in actions}
    if len(by_id) != len(actions):
        raise SchemaError(
            "linked action ids collide",
            path="build_moe_swizzle_overlay.linked_actions",
        )
    dependencies = {
        item.id: set(item.deps + item.metadata_action_refs) for item in actions
    }
    if any(ref not in by_id for refs in dependencies.values() for ref in refs):
        raise SchemaError(
            "linked action references missing dependency",
            path="build_moe_swizzle_overlay.linked_actions",
        )
    ready = sorted(ref for ref, refs in dependencies.items() if not refs)
    result = []
    while ready:
        ref = ready.pop(0)
        result.append(by_id[ref])
        for other in sorted(dependencies):
            if ref in dependencies[other]:
                dependencies[other].remove(ref)
                if (
                    not dependencies[other]
                    and other not in ready
                    and all(item.id != other for item in result)
                ):
                    ready.append(other)
                    ready.sort()
    if len(result) != len(actions):
        raise SchemaError(
            "linked action dependency cycle",
            path="build_moe_swizzle_overlay.linked_actions",
        )
    return tuple(result)
_LEGACY_NODE = re.compile(
    r"S3M4\.token(?P<token>\d+)\.expert(?P<expert>\d+)\."
    r"(?P<role>gate|up|down|dispatch|combine|swiglu)"
)


def _legacy_key(action: object, task: object) -> tuple[str, int, int, str]:
    match = _LEGACY_NODE.fullmatch(task.node_ref)
    if match is None:
        raise SchemaError("legacy action lacks typed token/expert/role", path="moe_swizzle_c0_bridge")
    role = match.group("role")
    if action.kind is LiteMoeDp4TaskKind.DMA_IN:
        role = f"{role}.weight"
    elif action.kind in (
        LiteMoeDp4TaskKind.SEND,
        LiteMoeDp4TaskKind.RECV,
        LiteMoeDp4TaskKind.WAIT,
    ):
        role = f"{role}.{action.kind.value}"
    return (
        action.kind.value,
        int(match.group("token")),
        int(match.group("expert")),
        role,
    )


def _scale_key(action: MoeScaleExecutionAction) -> tuple[str, int, int, str]:
    return (action.kind.value, action.token_index, action.expert_index, action.role)


def _build_c0_legacy_action_bridge(
    forward: LiteMoeDp4ExecutionCase,
    execution: MoeScaleExecution,
) -> dict[str, str]:
    execution.validate("moe_swizzle_c0_bridge.execution")
    if len(execution.actions) != 92:
        raise SchemaError("C0 bridge requires exact 92-action execution", path="moe_swizzle_c0_bridge")
    tasks = {
        task.id: task for die in forward.projection.dies for task in die.tasks
    }
    legacy = {}
    for action in forward.global_dag.actions:
        task = tasks[action.task_ref]
        key = _legacy_key(action, task)
        if key in legacy:
            raise SchemaError("legacy semantic action key is not unique", path="moe_swizzle_c0_bridge")
        legacy[key] = (action, task)
    generalized = {}
    for action in execution.actions:
        key = _scale_key(action)
        if key in generalized:
            raise SchemaError("generalized semantic action key is not unique", path="moe_swizzle_c0_bridge")
        generalized[key] = action
    if len(legacy) != 92 or set(legacy) != set(generalized):
        raise SchemaError("C0 semantic action keys are not a 92-item bijection", path="moe_swizzle_c0_bridge")

    bridge = {
        generalized[key].id: legacy[key][0].id for key in generalized
    }
    legacy_by_id = {item.id: item for item in forward.global_dag.actions}
    generalized_by_id = {item.id: item for item in execution.actions}
    value_bridge: dict[str, str] = {}

    def bind(scale_ref: str, legacy_ref: str) -> None:
        prior = value_bridge.setdefault(scale_ref, legacy_ref)
        if prior != legacy_ref:
            raise SchemaError("C0 operand value mapping is inconsistent", path="moe_swizzle_c0_bridge.values")

    for key in sorted(generalized):
        scale = generalized[key]
        source, task = legacy[key]
        if scale.die_id != source.die_id:
            raise SchemaError("C0 semantic action die placement drifted", path="moe_swizzle_c0_bridge.actions")
        if scale.kind is MoeScaleExecutionActionKind.WAIT:
            if (
                len(scale.read_values) != 1
                or scale.read_values != scale.write_values
                or task.read_values
                or task.write_values
                or source.flow_ref is None
            ):
                raise SchemaError("C0 WAIT operand normalization drifted", path="moe_swizzle_c0_bridge.values")
            flow = next(
                (item for item in forward.projection.flows if item.id == source.flow_ref),
                None,
            )
            if flow is None:
                raise SchemaError("C0 WAIT lacks legacy flow", path="moe_swizzle_c0_bridge.values")
            bind(scale.read_values[0], flow.destination_value_ref)
        elif scale.kind is MoeScaleExecutionActionKind.SWIGLU:
            if len(scale.read_values) != 2 or len(task.read_values) != 1 or len(scale.write_values) != len(task.write_values):
                raise SchemaError("C0 packed SwiGLU operand normalization drifted", path="moe_swizzle_c0_bridge.values")
            bind(scale.read_values[0], task.read_values[0])
            bind(scale.read_values[1], task.read_values[0])
            for scale_ref, legacy_ref in zip(scale.write_values, task.write_values, strict=True):
                bind(scale_ref, legacy_ref)
        else:
            if len(scale.read_values) != len(task.read_values) or len(scale.write_values) != len(task.write_values):
                raise SchemaError("C0 operand cardinality drifted", path="moe_swizzle_c0_bridge.values")
            for scale_ref, legacy_ref in zip(scale.read_values, task.read_values, strict=True):
                bind(scale_ref, legacy_ref)
            for scale_ref, legacy_ref in zip(scale.write_values, task.write_values, strict=True):
                bind(scale_ref, legacy_ref)

    for key in sorted(generalized):
        scale = generalized[key]
        source, _ = legacy[key]
        scale_deps = {bridge[ref] for ref in scale.deps}
        legacy_deps = set(source.deps)
        if scale.kind is MoeScaleExecutionActionKind.RECV:
            paired_send = (
                scale.kind.value.replace("recv", "send"),
                scale.token_index,
                scale.expert_index,
                scale.role.replace(".recv", ".send"),
            )
            expected = {legacy[paired_send][0].id}
            if scale_deps != expected or legacy_deps:
                raise SchemaError("C0 RECV dependency normalization drifted", path="moe_swizzle_c0_bridge.deps")
        elif scale_deps != legacy_deps:
            raise SchemaError("C0 action dependencies drifted", path="moe_swizzle_c0_bridge.deps")
        for ref in scale.read_values + scale.write_values:
            if ref not in value_bridge:
                raise SchemaError("C0 operand value is outside typed bridge", path="moe_swizzle_c0_bridge.values")
    del legacy_by_id, generalized_by_id
    return bridge


def adapt_moe_swizzle_candidate_to_c0_legacy(
    adapter: MoeSwizzleCandidateAdapter,
    forward: LiteMoeDp4ExecutionCase,
    execution: MoeScaleExecution,
) -> MoeSwizzleCandidateAdapter:
    """Rebuild witness ids/deps after mapping generalized originals to legacy."""

    adapter.validate("adapt_moe_swizzle_candidate_to_c0_legacy.adapter")
    bridge = _build_c0_legacy_action_bridge(forward, execution)
    actions = {
        item.id: item for program in adapter.rank_programs for item in program.actions
    }
    dependencies = {ref: set(item.deps) for ref, item in actions.items()}
    ready = sorted(ref for ref, deps in dependencies.items() if not deps)
    rebuilt: dict[str, MoeActionWitness] = {}
    while ready:
        ref = ready.pop(0)
        action = actions[ref]
        if any(item not in bridge for item in action.original_action_refs):
            raise SchemaError("candidate original is outside C0 typed bridge", path="adapt_moe_swizzle_candidate_to_c0_legacy")
        rebuilt[ref] = MoeActionWitness.create(
            rank=action.rank,
            kind=action.kind,
            deps=tuple(rebuilt[item].id for item in action.deps),
            assignment_refs=action.assignment_refs,
            expert_index=action.expert_index,
            tile_index=action.tile_index,
            n_block_index=action.n_block_index,
            packet_ref=action.packet_ref,
            stage=action.stage,
            pivot_rank=action.pivot_rank,
            original_action_refs=tuple(bridge[item] for item in action.original_action_refs),
            route_ref=action.route_ref,
            pipeline_index=action.pipeline_index,
            buffer_slot=action.buffer_slot,
            buffer_family=action.buffer_family,
            packed_value_ref=action.packed_value_ref,
            peer_rank=action.peer_rank,
            logical_bytes=action.logical_bytes,
            flops=action.flops,
            work_role=action.work_role,
        )
        for other in sorted(dependencies):
            dependencies[other].discard(ref)
            if not dependencies[other] and other not in rebuilt and other not in ready:
                ready.append(other)
                ready.sort()
    if len(rebuilt) != len(actions):
        raise SchemaError("candidate witness dependency graph is not closed", path="adapt_moe_swizzle_candidate_to_c0_legacy")
    programs = tuple(
        MoeRankProgram(
            rank=program.rank,
            actions=tuple(rebuilt[item.id] for item in program.actions),
        )
        for program in adapter.rank_programs
    )
    result = MoeSwizzleCandidateAdapter(
        pattern=adapter.pattern,
        region_ref=adapter.region_ref,
        decision_ref=adapter.decision_ref,
        candidate_ref=adapter.candidate_ref,
        algorithm=adapter.algorithm,
        rank_programs=programs,
        packetization=adapter.packetization,
        double_buffer=adapter.double_buffer,
    )
    result.validate("adapt_moe_swizzle_candidate_to_c0_legacy.result")
    return result




def build_moe_swizzle_overlay_from_adapters(
    forward: LiteMoeDp4ExecutionCase,
    adapters: tuple[MoeSwizzleCandidateAdapter, ...],
    *,
    train_forward: LiteMoeDp4TrainForward | None = None,
    source_execution_ref: str | None = None,
) -> MoeSwizzleOverlay:
    """Build one exact overlay from the narrow candidate deployment protocol."""

    if type(forward) is not LiteMoeDp4ExecutionCase:
        raise SchemaError(
            "requires exact LiteMoE forward execution",
            path="build_moe_swizzle_overlay.forward",
        )
    forward.validate("build_moe_swizzle_overlay.forward")
    if type(adapters) is not tuple or len(adapters) != 2:
        raise SchemaError(
            "requires dispatch and combine candidate adapters",
            path="build_moe_swizzle_overlay.adapters",
        )
    for index, adapter in enumerate(adapters):
        if type(adapter) is not MoeSwizzleCandidateAdapter:
            raise SchemaError(
                "requires typed candidate adapter",
                path=f"build_moe_swizzle_overlay.adapters[{index}]",
            )
        adapter.validate(f"build_moe_swizzle_overlay.adapters[{index}]")
    by_pattern = {item.pattern: item for item in adapters}
    if set(by_pattern) != set(_PATTERNS):
        raise SchemaError(
            "adapters must cover Dispatch+GEMM and GEMM+Combine exactly",
            path="build_moe_swizzle_overlay.adapters",
        )
    selected = tuple(by_pattern[item] for item in _PATTERNS)
    expected_counts = (42, 26)
    if tuple(sum(len(action.original_action_refs) for program in item.rank_programs for action in program.actions) for item in selected) != expected_counts:
        raise SchemaError(
            "selected candidates changed the exact 42/26 source slices",
            path="build_moe_swizzle_overlay.adapters",
        )

    source_actions = {item.id: item for item in forward.global_dag.actions}
    source_tasks = {
        item.id: item
        for die in forward.projection.dies
        for item in die.tasks
    }
    if len(source_actions) != 92 or len(source_tasks) != 92:
        raise SchemaError(
            "LiteMoE source quotient changed",
            path="build_moe_swizzle_overlay.forward",
        )
    replaced_refs = tuple(
        ref for candidate in selected for program in candidate.rank_programs for action in program.actions for ref in action.original_action_refs
    )
    if (
        len(replaced_refs) != 68
        or len(set(replaced_refs)) != 68
        or any(ref not in source_actions for ref in replaced_refs)
    ):
        raise SchemaError(
            "selected source slices overlap or reference another execution",
            path="build_moe_swizzle_overlay.replaced_action_refs",
        )
    replaced = set(replaced_refs)
    preserved_global = tuple(
        item for item in forward.global_dag.actions if item.id not in replaced
    )
    preserved_counts = Counter(item.kind for item in preserved_global)
    if preserved_counts != Counter({LiteMoeDp4TaskKind.DMA_IN: 24}):
        raise SchemaError(
            "overlay must preserve exactly 24 DMA_IN actions",
            path="build_moe_swizzle_overlay.preserved_action_refs",
        )

    replacement_programs = tuple(
        MoeRankProgram(
            rank=rank,
            actions=tuple(
                action
                for candidate in selected
                for program in candidate.rank_programs
                if program.rank == rank
                for action in program.actions
            ),
        )
        for rank in range(4)
    )
    replacement_actions = tuple(
        action for program in replacement_programs for action in program.actions
    )
    source_to_replacement = {}
    for action in replacement_actions:
        for ref in action.original_action_refs:
            if ref in source_to_replacement:
                raise SchemaError(
                    "one source action is replaced twice",
                    path="build_moe_swizzle_overlay.replacement_rank_programs",
                )
            source_to_replacement[ref] = action.id
    if set(source_to_replacement) != replaced:
        raise SchemaError(
            "replacement programs do not exactly cover selected slices",
            path="build_moe_swizzle_overlay.replacement_rank_programs",
        )

    value_producers: dict[str, list[str]] = {}
    value_consumers: dict[str, list[str]] = {}
    for source in forward.global_dag.actions:
        task = source_tasks[source.task_ref]
        for ref in task.write_values:
            value_producers.setdefault(ref, []).append(source.id)
        for ref in task.read_values:
            value_consumers.setdefault(ref, []).append(source.id)
    terminal_values = set(forward.global_dag.combined_output_refs)

    def mapped(ref: str) -> str:
        return source_to_replacement.get(ref, ref)

    linked = []
    for witness in replacement_actions:
        covered = set(witness.original_action_refs)
        originals = tuple(source_actions[ref] for ref in witness.original_action_refs)
        tasks = tuple(source_tasks[item.task_ref] for item in originals)
        source_deps = _unique(
            mapped(dep)
            for item in originals
            for dep in item.deps
            if dep not in covered and mapped(dep) != witness.id
        )
        reads = _unique(
            ref
            for task in tasks
            for ref in task.read_values
            if not value_producers.get(ref)
            or not set(value_producers[ref]).issubset(covered)
        )
        writes = _unique(
            ref
            for task in tasks
            for ref in task.write_values
            if (
                ref in terminal_values
                or not value_consumers.get(ref)
                or not set(value_consumers[ref]).issubset(covered)
            )
        )
        linked.append(
            MoeSwizzleLinkedAction(
                id=witness.id,
                rank=witness.rank,
                die_id=witness.rank,
                kind=f"replacement.{witness.kind.value}",
                preserved=False,
                source_action_refs=witness.original_action_refs,
                deps=_unique(witness.deps + source_deps),
                metadata_action_refs=(),
                read_value_refs=reads,
                write_value_refs=writes,
                assignment_refs=witness.assignment_refs,
                expert_index=witness.expert_index,
                tile_index=witness.tile_index,
                n_block_index=witness.n_block_index,
                packet_ref=witness.packet_ref,
                stage=witness.stage,
                pivot_rank=witness.pivot_rank,
            )
        )

    for source in preserved_global:
        task = source_tasks[source.task_ref]
        linked.append(
            MoeSwizzleLinkedAction(
                id=source.id,
                rank=source.die_id,
                die_id=source.die_id,
                kind=f"preserved.{source.kind.value}",
                preserved=True,
                source_action_refs=(source.id,),
                deps=_unique(mapped(ref) for ref in source.deps),
                metadata_action_refs=(),
                read_value_refs=task.read_values,
                write_value_refs=task.write_values,
                assignment_refs=(),
                expert_index=None,
                tile_index=None,
                n_block_index=None,
                packet_ref=None,
                stage=None,
                pivot_rank=None,
            )
        )

    preserved_refs = [item.id for item in preserved_global]
    source_execution_id = source_execution_ref or forward.id
    is_train = train_forward is not None
    if train_forward is not None:
        if (
            type(train_forward) is not LiteMoeDp4TrainForward
            or train_forward.forward != forward
        ):
            raise SchemaError(
                "training wrapper belongs to another forward execution",
                path="build_moe_swizzle_overlay.train_forward",
            )
        train_forward.validate("build_moe_swizzle_overlay.train_forward")
        tape_by_id = {item.id: item for item in train_forward.tape_buffers}
        for copy in train_forward.tape_copies:
            buffer = tape_by_id[copy.destination_buffer_ref]
            linked.append(
                MoeSwizzleLinkedAction(
                    id=copy.id,
                    rank=copy.die_id,
                    die_id=copy.die_id,
                    kind="preserved.local_copy",
                    preserved=True,
                    source_action_refs=(copy.id,),
                    deps=(mapped(copy.source_action_ref),),
                    metadata_action_refs=(mapped(copy.down_action_ref),),
                    read_value_refs=(copy.source_value_ref,),
                    write_value_refs=(buffer.value_ref,),
                    assignment_refs=(),
                    expert_index=copy.expert_index,
                    tile_index=copy.token_index,
                    n_block_index=None,
                    packet_ref=None,
                    stage=None,
                    pivot_rank=None,
                )
            )
            preserved_refs.append(copy.id)
            terminal_values.add(buffer.value_ref)
        source_execution_id = train_forward.id

    ordered_linked = _topological(tuple(linked))
    producers: dict[str, list[str]] = {}
    consumers: dict[str, list[str]] = {}
    for action in ordered_linked:
        for ref in action.write_value_refs:
            producers.setdefault(ref, []).append(action.id)
        for ref in action.read_value_refs:
            consumers.setdefault(ref, []).append(action.id)
    by_linked_id = {item.id: item for item in ordered_linked}
    boundary_refs = []
    for ref in _unique(
        tuple(producers) + tuple(consumers) + tuple(sorted(terminal_values))
    ):
        producer_refs = tuple(producers.get(ref, ()))
        consumer_refs = tuple(consumers.get(ref, ()))
        producer_modes = {by_linked_id[item].preserved for item in producer_refs}
        consumer_modes = {by_linked_id[item].preserved for item in consumer_refs}
        if (
            ref in terminal_values
            or not producer_refs
            or (producer_modes and consumer_modes and producer_modes != consumer_modes)
        ):
            boundary_refs.append(
                MoeBoundaryValueBinding(
                    source_value_ref=ref,
                    replacement_value_ref=ref,
                    producer_action_refs=producer_refs,
                    consumer_action_refs=consumer_refs,
                    terminal=ref in terminal_values,
                )
            )

    selections = tuple(
        MoeSwizzleDeploymentSelection(
            region_ref=candidate.region_ref,
            decision_ref=candidate.decision_ref,
            candidate_ref=candidate.candidate_ref,
            mode=(
                MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE
                if candidate.algorithm is SwizzleAlgorithm.UNFUSED
                else MoeSwizzleDeploymentMode.FUSED_FORCED
            ),
            double_buffer=candidate.double_buffer,
        )
        for candidate in selected
    )
    return MoeSwizzleOverlay.create(
        source_execution_id=source_execution_id,
        source_forward_execution_id=forward.id,
        source_workload_selection_id=None,
        decision_refs=tuple(item.decision_ref for item in selected),
        deployment_selections=selections,
        replaced_action_refs=replaced_refs,
        replacement_rank_programs=replacement_programs,
        preserved_action_refs=tuple(preserved_refs),
        boundary_value_bindings=tuple(boundary_refs),
        linked_actions=ordered_linked,
        terminal_value_refs=tuple(sorted(terminal_values)),
        replacement_packetization=tuple(packet for item in selected for packet in item.packetization),
        replacement_tile_schedule=(),
        train_forward=is_train,
    )


def build_moe_swizzle_overlay(
    forward: LiteMoeDp4ExecutionCase,
    decisions: tuple[MoeSwizzleDecision, ...],
    *,
    train_forward: LiteMoeDp4TrainForward | None = None,
    source_execution: MoeScaleExecution | None = None,
) -> MoeSwizzleOverlay:
    """Extract the narrow deployment protocol from planner decisions."""

    if type(decisions) is not tuple or len(decisions) != 2:
        raise SchemaError("requires exactly two decisions", path="build_moe_swizzle_overlay.decisions")
    for index, decision in enumerate(decisions):
        if type(decision) is not MoeSwizzleDecision:
            raise SchemaError("requires typed decision", path=f"build_moe_swizzle_overlay.decisions[{index}]")
        decision.validate(f"build_moe_swizzle_overlay.decisions[{index}]")
    by_pattern = {item.problem.region.pattern: item for item in decisions}
    if set(by_pattern) != set(_PATTERNS):
        raise SchemaError("decisions do not cover both MoE regions", path="build_moe_swizzle_overlay.decisions")
    adapters = []
    for pattern in _PATTERNS:
        decision = by_pattern[pattern]
        candidate = _selected(decision)
        adapters.append(
            MoeSwizzleCandidateAdapter(
                pattern=pattern,
                region_ref=decision.problem.region.id,
                decision_ref=decision.id,
                candidate_ref=candidate.id,
                algorithm=candidate.algorithm,
                rank_programs=candidate.rank_programs,
                packetization=candidate.packetization,
                double_buffer=candidate.double_buffer,
            )
        )
    legacy_ids = {item.id for item in forward.global_dag.actions}
    original_refs = {
        ref
        for adapter in adapters
        for program in adapter.rank_programs
        for action in program.actions
        for ref in action.original_action_refs
    }
    if not original_refs.issubset(legacy_ids):
        if type(source_execution) is not MoeScaleExecution:
            raise SchemaError(
                "generalized candidate requires typed C0 source execution",
                path="build_moe_swizzle_overlay.source_execution",
            )
        if any(item.problem.source_execution_id != source_execution.id for item in decisions):
            raise SchemaError(
                "decision/source execution provenance mismatch",
                path="build_moe_swizzle_overlay.source_execution",
            )
        adapters = [
            adapt_moe_swizzle_candidate_to_c0_legacy(item, forward, source_execution)
            for item in adapters
        ]
    return build_moe_swizzle_overlay_from_adapters(
        forward,
        tuple(adapters),
        train_forward=train_forward,
        source_execution_ref=None if source_execution is None else source_execution.id,
    )


def validate_moe_swizzle_overlay_against_source(
    overlay: MoeSwizzleOverlay,
    forward: LiteMoeDp4ExecutionCase,
    *,
    adapters: tuple[MoeSwizzleCandidateAdapter, ...],
    train_forward: LiteMoeDp4TrainForward | None = None,
) -> None:
    """Cross-validate the complete overlay by deterministic reconstruction."""

    overlay.validate("validate_moe_swizzle_overlay_against_source.overlay")
    rebuilt = build_moe_swizzle_overlay_from_adapters(
        forward,
        adapters,
        train_forward=train_forward,
        source_execution_ref=overlay.source_execution_id,
    )
    if overlay != rebuilt:
        raise SchemaError(
            "overlay is not the exact deterministic source replacement",
            path="validate_moe_swizzle_overlay_against_source",
        )


__all__ = [
    "build_moe_swizzle_overlay",
    "validate_moe_swizzle_overlay_against_source",
    "build_moe_swizzle_overlay_from_adapters",
]
