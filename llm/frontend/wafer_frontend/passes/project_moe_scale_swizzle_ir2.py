"""Typed generalized scale projection for selected fused MoE replacements."""

from __future__ import annotations

from collections import defaultdict
from math import prod
from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.common import DType, stable_artifact_id
from ..schema.ir0 import FusionPattern
from ..schema.swizzle import SwizzleActionKind, SwizzleAlgorithm
from ..schema.swizzle_moe import (
    MoePacketSlice, MoeSwizzleDecision, MoeSwizzleWorkloadSelection,
)
from ..schema.swizzle_moe_execution import (
    MoeScaleExecution,
    MoeScaleExecutionActionKind,
    MoeScaleExecutionTerminalKind,
)
from ..schema.swizzle_moe_ir2 import (
    MOE_SWIZZLE_IR2_SCHEMA_VERSION,
    MoeSwizzleIr2Buffer,
    MoeSwizzleIr2BufferUse,
    MoeSwizzleIr2Flow,
    MoeSwizzleIr2PacketSlice,
    MoeSwizzleIr2Projection,
    MoeSwizzleIr2Task,
    MoeSwizzleIr2TerminalSlice,
    MoeSwizzleIr2Value,
)
from ..schema.swizzle_moe_plan import MoeSwizzleOverlay
from ..schema.swizzle_moe_scale import MoeSwizzleScaleSpec


def _id(kind: str, semantic: object) -> str:
    return stable_artifact_id(
        f"moe_scale_ir2_{kind}", semantic,
        schema_version=MOE_SWIZZLE_IR2_SCHEMA_VERSION,
    )


def _action_pattern_key(action: object) -> tuple[tuple[str, object], ...]:
    """Identify one replacement action independent of strengthened deps/id."""

    return tuple(
        (name, getattr(action, name))
        for name in action.__dataclass_fields__
        if name not in ("schema_version", "id", "deps")
    )


@dataclass(frozen=True, slots=True)
class _UnfusedPacketView:
    id: str
    stage: int
    source_rank: int
    destination_rank: int
    pivot_rank: int | None
    route_ref: str
    slices: tuple[MoePacketSlice, ...]
    logical_bytes: int


@dataclass(frozen=True, slots=True)
class _UnfusedTileView:
    id: str
    expert_index: int
    tile_index: int
    m_block_index: int
    m_block_size: int
    assignment_refs: tuple[str, ...]
    required_packet_refs: tuple[str, ...]
    output_value_refs: tuple[str, ...]
    n_block_index: int | None
    output_column_extent: int


def _unfused_views(
    problem: object,
    candidate: object,
    execution: MoeScaleExecution,
    spec: MoeSwizzleScaleSpec,
) -> tuple[tuple[_UnfusedPacketView, ...], tuple[_UnfusedTileView, ...]]:
    actions = tuple(item for program in candidate.rank_programs for item in program.actions)
    action_index = {item.id: item for item in actions}
    assignment_index = {
        item.id: item for item in problem.region.semantic_witness.traffic.assignments
    }
    execution_index = {item.id: item for item in execution.actions}
    grouped = defaultdict(list)
    for action in actions:
        if action.packet_ref is not None:
            grouped[action.packet_ref].append(action)
    packets = []
    for packet_ref, group in sorted(grouped.items()):
        by_kind = {item.kind: item for item in group}
        if set(by_kind) != {
            SwizzleActionKind.SEND, SwizzleActionKind.RECV, SwizzleActionKind.WAIT,
        } or len(group) != 3:
            raise SchemaError("UNFUSED transport lacks one typed triple", path="moe_scale_ir2.unfused")
        send = by_kind[SwizzleActionKind.SEND]
        recv = by_kind[SwizzleActionKind.RECV]
        if len(send.assignment_refs) != 1 or send.assignment_refs != recv.assignment_refs:
            raise SchemaError("UNFUSED transport assignment closure drifted", path="moe_scale_ir2.unfused")
        assignment = assignment_index[send.assignment_refs[0]]
        source, destination, route_ref = (
            (assignment.source_rank, assignment.expert_rank, assignment.dispatch_route_ref)
            if candidate.pattern is FusionPattern.MOE_DISPATCH_GEMM
            else (assignment.expert_rank, assignment.source_rank, assignment.combine_route_ref)
        )
        if (
            (send.rank, recv.rank, send.route_ref, recv.route_ref)
            != (source, destination, route_ref, route_ref)
            or route_ref is None
        ):
            raise SchemaError("UNFUSED route endpoints drifted", path="moe_scale_ir2.unfused")
        packets.append(_UnfusedPacketView(
            packet_ref, 0, source, destination, None, route_ref,
            (MoePacketSlice(assignment.id, 0, 0, assignment.payload_bytes, send.n_block_index),),
            assignment.payload_bytes,
        ))
    tiles = {}
    for action in actions:
        if action.kind is not SwizzleActionKind.COMP:
            continue
        key = (action.expert_index, action.assignment_refs, action.n_block_index)
        if key in tiles:
            continue
        m = len(action.assignment_refs)
        k = (
            spec.hidden_size if candidate.pattern is FusionPattern.MOE_DISPATCH_GEMM
            else spec.intermediate_size
        )
        if m == 0 or action.flops % (2 * m * k):
            raise SchemaError("UNFUSED COMP shape is not exact", path="moe_scale_ir2.unfused")
        originals = tuple(execution_index[ref] for ref in action.original_action_refs)
        output_refs = tuple(ref for original in originals for ref in original.write_values)
        required = tuple(
            action_index[ref].packet_ref for ref in action.deps
            if ref in action_index and action_index[ref].kind is SwizzleActionKind.WAIT
        )
        tile_index = 0 if action.tile_index is None else action.tile_index
        tiles[key] = _UnfusedTileView(
            _id("unfused_tile", {"candidate": candidate.id, "key": key}),
            action.expert_index, tile_index,
            0 if action.pipeline_index is None else action.pipeline_index,
            m, action.assignment_refs, required, output_refs,
            action.n_block_index, action.flops // (2 * m * k),
        )
    return tuple(packets), tuple(tiles[key] for key in sorted(tiles))




def project_moe_scale_swizzle_ir2(
    overlay: MoeSwizzleOverlay,
    execution: MoeScaleExecution,
    spec: MoeSwizzleScaleSpec,
    decisions: tuple[MoeSwizzleDecision, MoeSwizzleDecision],
    workload_selection: MoeSwizzleWorkloadSelection | None = None,
    *,
    endpoint_session_capacity: int,
) -> MoeSwizzleIr2Projection:
    spec.validate("moe_scale_ir2.spec")
    execution.validate("moe_scale_ir2.execution")
    overlay.validate("moe_scale_ir2.overlay")
    if type(decisions) is not tuple or len(decisions) != 2:
        raise SchemaError("scale projection requires exactly two decisions", path="moe_scale_ir2.decisions")
    for index, decision in enumerate(decisions):
        if type(decision) is not MoeSwizzleDecision:
            raise SchemaError("requires exact typed decisions", path=f"moe_scale_ir2.decisions[{index}]")
        decision.validate(f"moe_scale_ir2.decisions[{index}]")
        if decision.problem.source_execution_id != execution.id:
            raise SchemaError("decision belongs to another execution", path=f"moe_scale_ir2.decisions[{index}]")
    by_pattern = {decision.problem.region.pattern: decision for decision in decisions}
    expected_patterns = (
        FusionPattern.MOE_DISPATCH_GEMM,
        FusionPattern.MOE_GEMM_COMBINE,
    )
    if set(by_pattern) != set(expected_patterns):
        raise SchemaError("decisions do not cover both MoE regions", path="moe_scale_ir2.decisions")
    ordered_decisions = tuple(by_pattern[pattern] for pattern in expected_patterns)
    if (
        workload_selection is not None
        and type(workload_selection) is not MoeSwizzleWorkloadSelection
    ):
        raise SchemaError(
            "workload projection requires an exact joint selection",
            path="moe_scale_ir2.workload_selection",
        )
    expected_selection_id = (
        None if workload_selection is None else workload_selection.id
    )
    if (
        overlay.source_execution_id != execution.id
        or tuple(item.id for item in ordered_decisions) != overlay.decision_refs
        or overlay.source_workload_selection_id != expected_selection_id
    ):
        raise SchemaError("scale projection lineage is not exact", path="moe_scale_ir2")
    if workload_selection is None:
        candidate_refs = tuple(item.selected_candidate_ref for item in ordered_decisions)
    else:
        if type(workload_selection) is not MoeSwizzleWorkloadSelection:
            raise SchemaError(
                "workload projection requires an exact joint selection",
                path="moe_scale_ir2.workload_selection",
            )
        workload_selection.validate("moe_scale_ir2.workload_selection")
        if (
            workload_selection.source_dispatch_decision_id,
            workload_selection.source_combine_decision_id,
        ) != tuple(item.id for item in ordered_decisions):
            raise SchemaError(
                "joint selection belongs to another decision pair",
                path="moe_scale_ir2.workload_selection",
            )
        candidate_refs = (
            workload_selection.selected_dispatch_candidate_ref,
            workload_selection.selected_combine_candidate_ref,
        )
    selected = []
    pattern_by_action_semantic = {}
    assignments = {}
    unfused_views = []
    routes = {}
    for decision, candidate_ref in zip(ordered_decisions, candidate_refs, strict=True):
        candidate = next(
            (item for item in decision.ranked_candidates if item.id == candidate_ref),
            None,
        )
        if candidate is None:
            raise SchemaError("selected workload candidate is unavailable", path="moe_scale_ir2.decisions")
        candidate.validate_against(decision.problem, "moe_scale_ir2.candidate")
        selected.append(candidate)
        for program in candidate.rank_programs:
            for action in program.actions:
                key = _action_pattern_key(action)
                previous = pattern_by_action_semantic.get(key)
                if previous is not None and previous is not candidate.pattern:
                    raise SchemaError(
                        "replacement action semantic key crosses regions",
                        path="moe_scale_ir2.decisions",
                    )
                pattern_by_action_semantic[key] = candidate.pattern
        for item in decision.problem.region.semantic_witness.traffic.assignments:
            assignments[item.id] = item
        if candidate.algorithm is SwizzleAlgorithm.UNFUSED:
            unfused_views.append(_unfused_views(decision.problem, candidate, execution, spec))
        for item in decision.problem.topology.group.routes:
            routes[item.id] = item
    if tuple(item.id for item in selected) != tuple(item.candidate_ref for item in overlay.deployment_selections):
        raise SchemaError("overlay selected candidate lineage drifted", path="moe_scale_ir2.decisions")

    witnesses = {
        item.id: item
        for program in overlay.replacement_rank_programs for item in program.actions
    }
    action_pattern = {}
    for ref, action in witnesses.items():
        pattern = pattern_by_action_semantic.get(_action_pattern_key(action))
        if pattern is None:
            raise SchemaError(
                "overlay action lacks exact selected-candidate semantics",
                path="moe_scale_ir2.overlay.replacement_rank_programs",
            )
        action_pattern[ref] = pattern
    linked = tuple(item for item in overlay.linked_actions if not item.preserved)
    if set(witnesses) != {item.id for item in linked}:
        raise SchemaError("replacement witness/action closure drifted", path="moe_scale_ir2.overlay")
    adapted_packets = tuple(
        item for packet_views, _ in unfused_views for item in packet_views
    )
    adapted_tiles = tuple(item for _, tile_views in unfused_views for item in tile_views)
    packet_values = overlay.replacement_packetization + adapted_packets
    tile_values = overlay.replacement_tile_schedule + adapted_tiles
    packets = {item.id: item for item in packet_values}
    packet_actions: dict[str, dict[SwizzleActionKind, object]] = defaultdict(dict)
    for item in witnesses.values():
        if item.packet_ref is not None:
            if item.kind in packet_actions[item.packet_ref]:
                raise SchemaError("packet duplicates an action kind", path="moe_scale_ir2.packets")
            packet_actions[item.packet_ref][item.kind] = item
    if set(packet_actions) != set(packets) or any(
        set(group) != {SwizzleActionKind.SEND, SwizzleActionKind.RECV, SwizzleActionKind.WAIT}
        for group in packet_actions.values()
    ):
        raise SchemaError("packet SEND/RECV/WAIT closure is not exact", path="moe_scale_ir2.packets")

    tile_by_key = {}
    for tile in tile_values:
        pattern = FusionPattern.MOE_GEMM_COMBINE if tile.n_block_index is not None else FusionPattern.MOE_DISPATCH_GEMM
        key = (pattern, tile.expert_index, tile.assignment_refs, tile.n_block_index)
        if key in tile_by_key:
            raise SchemaError("tile semantic key is not unique", path="moe_scale_ir2.tiles")
        tile_by_key[key] = tile
    dispatch_tile_by_assignment = {}
    for tile in tile_values:
        if tile.n_block_index is not None:
            continue
        for row, assignment_ref in enumerate(tile.assignment_refs):
            if assignment_ref in dispatch_tile_by_assignment:
                raise SchemaError("dispatch assignment belongs to multiple M-block tiles", path="moe_scale_ir2.tiles")
            dispatch_tile_by_assignment[assignment_ref] = (tile, row)

    task_operands: dict[str, tuple[list[str], list[str]]] = {
        item.id: ([], []) for item in linked
    }
    value_meta = {}
    producer = {}
    value_terminal_slices = {}
    value_alias_sources = {}
    terminal_by_token = {
        item.token_index: item for item in execution.terminals
        if item.kind is MoeScaleExecutionTerminalKind.COMBINED
    }

    def add_value(
        ref: str, rank: int, origin: str, shape: tuple[int, ...],
        producer_ref: str | None, *, terminal_ref: str | None = None,
        buffer_family: str | None = None,
        byte_offset: int = 0,
        alias_source_refs: tuple[str, ...] = (),
    ) -> str:
        if ref in value_meta:
            raise SchemaError("scale projection value id collision", path="moe_scale_ir2.values")
        value_meta[ref] = (rank, origin, shape, terminal_ref, buffer_family, byte_offset)
        if alias_source_refs:
            value_alias_sources[ref] = alias_source_refs
        if producer_ref is not None:
            producer[ref] = producer_ref
            task_operands[producer_ref][1].append(ref)
        return ref

    recv_value = {}
    for packet_ref, group in packet_actions.items():
        packet = packets[packet_ref]
        recv = group[SwizzleActionKind.RECV]
        pattern = action_pattern[recv.id]
        family = "dispatch_operand" if pattern is FusionPattern.MOE_DISPATCH_GEMM else "combine_output"
        columns = spec.hidden_size if pattern is FusionPattern.MOE_DISPATCH_GEMM else next(
            item.bytes // 2 for item in packet.slices
        )
        if packet.logical_bytes % (2 * columns):
            raise SchemaError("packet bytes do not close a dense value shape", path="moe_scale_ir2.packets")
        rows = packet.logical_bytes // (2 * columns)
        byte_offset = 0
        if pattern is FusionPattern.MOE_DISPATCH_GEMM:
            tile_rows = tuple(dispatch_tile_by_assignment.get(item.assignment_ref) for item in packet.slices)
            if any(item is None for item in tile_rows):
                raise SchemaError("dispatch packet lacks an exact M-block row", path="moe_scale_ir2.packets")
            tiles = {item[0].id for item in tile_rows}
            positions = tuple(item[1] for item in tile_rows)
            if (
                len(tiles) != 1
                or positions != tuple(range(positions[0], positions[0] + len(positions)))
            ):
                raise SchemaError("dispatch packet is not a contiguous subview of one M-block", path="moe_scale_ir2.packets")
            tile = tile_rows[0][0]
            # One dispatch slot is laid out as
            # [gate MxI][up MxI][activation MxH].  Keeping the activation
            # after both outputs prevents the gate MATMUL from clobbering
            # the input still needed by the up MATMUL.
            byte_offset = (
                2 * tile.m_block_size * spec.intermediate_size * 2
                + positions[0] * columns * 2
            )
        ref = _id("recv_value", {"overlay": overlay.id, "packet": packet_ref})
        terminal_ref = None
        terminal_slices = ()
        if pattern is FusionPattern.MOE_GEMM_COMBINE:
            source_ranks = {
                assignments[item.assignment_ref].source_rank for item in packet.slices
            }
            if source_ranks == {recv.rank}:
                built = []
                cursor = 0
                for part in packet.slices:
                    assignment = assignments[part.assignment_ref]
                    terminal = terminal_by_token[assignment.token_index]
                    built.append(MoeSwizzleIr2TerminalSlice(
                        f"{terminal.value_ref}.nblock{part.n_block_index}",
                        part.assignment_ref, cursor, part.bytes,
                        (1, part.bytes // 2),
                    ))
                    cursor += part.bytes
                terminal_slices = tuple(built)
                if len(terminal_slices) == 1 and rows == 1:
                    terminal_ref = terminal_slices[0].terminal_ref
                    terminal_slices = ()
        recv_value[packet_ref] = add_value(
            ref, recv.rank, packet_ref, (rows, columns), recv.id,
            terminal_ref=terminal_ref, buffer_family=family,
            byte_offset=byte_offset,
        )
        if terminal_slices:
            value_terminal_slices[ref] = terminal_slices

    comp_output = {}
    for item in witnesses.values():
        if item.kind is not SwizzleActionKind.COMP:
            continue
        pattern = action_pattern[item.id]
        tile = tile_by_key.get((pattern, item.expert_index, item.assignment_refs, item.n_block_index))
        if tile is None:
            raise SchemaError("COMP lacks exact tile slice witness", path="moe_scale_ir2.comp")
        m = len(item.assignment_refs)
        k = spec.hidden_size if pattern is FusionPattern.MOE_DISPATCH_GEMM else spec.intermediate_size
        denominator = 2 * m * k
        if not m or item.flops % denominator:
            raise SchemaError("COMP FLOPs do not close m/n/k", path="moe_scale_ir2.comp")
        n = item.flops // denominator
        if n != tile.output_column_extent or m != tile.m_block_size:
            raise SchemaError("COMP N differs from typed tile extent", path="moe_scale_ir2.comp")
        terminal_slices = ()
        if pattern is FusionPattern.MOE_DISPATCH_GEMM:
            sources = [
                action for ref in item.original_action_refs
                for action in execution.actions if action.id == ref
            ]
            assignment_tokens = {
                assignments[ref].token_index for ref in item.assignment_refs
            }
            if (
                len(sources) != m
                or any(
                    source.kind is not MoeScaleExecutionActionKind.GEMM
                    or source.role != item.work_role
                    or source.expert_index != item.expert_index
                    for source in sources
                )
                or {source.token_index for source in sources} != assignment_tokens
            ):
                raise SchemaError("Dispatch COMP role/original closure drifted", path="moe_scale_ir2.comp")
            origin = f"{tile.id}.{item.work_role}"
            terminal_ref = None
            family = "dispatch_operand"
            output_byte_offset = 0 if item.work_role == "gate" else m * n * 2
        else:
            if item.work_role != "down" or tile.output_value_refs == ():
                raise SchemaError("Combine COMP lacks typed down slice", path="moe_scale_ir2.comp")
            origin = f"{tile.id}.down"
            built = []
            for row, assignment_ref in enumerate(item.assignment_refs):
                assignment = assignments[assignment_ref]
                if assignment.source_rank == item.rank:
                    terminal = terminal_by_token[assignment.token_index]
                    built.append(MoeSwizzleIr2TerminalSlice(
                        f"{terminal.value_ref}.nblock{item.n_block_index}",
                        assignment_ref, row * n * 2, n * 2, (1, n),
                    ))
            terminal_slices = tuple(built)
            terminal_ref = None
            if len(terminal_slices) == 1 and m == 1:
                terminal_ref = terminal_slices[0].terminal_ref
                terminal_slices = ()
            family = "combine_output"
            output_byte_offset = 0
        ref = _id("comp_value", {"overlay": overlay.id, "task": item.id, "origin": origin})
        comp_output[item.id] = add_value(
            ref, item.rank, origin, (m, n), item.id,
            terminal_ref=terminal_ref, buffer_family=family,
            byte_offset=output_byte_offset,
        )
        if terminal_slices:
            value_terminal_slices[ref] = terminal_slices

    # Grouped SWIGLU consumes one flat [gate MxI][up MxI] contiguous view and
    # produces one contiguous MxI output.
    swiglu_output_by_group = {}
    for item in witnesses.values():
        if item.kind is not SwizzleActionKind.SWIGLU:
            continue
        if (
            action_pattern[item.id] is not FusionPattern.MOE_DISPATCH_GEMM
            or item.work_role != "swiglu"
            or item.packed_value_ref is None
            or item.logical_bytes
            != len(item.assignment_refs) * spec.intermediate_size * 2
        ):
            raise SchemaError("grouped SWIGLU typed closure drifted", path="moe_scale_ir2.swiglu")
        dependencies = tuple(witnesses.get(ref) for ref in item.deps)
        by_role = {
            dependency.work_role: dependency
            for dependency in dependencies
            if dependency is not None
        }
        if (
            set(by_role) != {"gate", "up"}
            or len(dependencies) != 2
            or any(
                dependency.kind is not SwizzleActionKind.COMP
                or dependency.assignment_refs != item.assignment_refs
                or dependency.id not in comp_output
                for dependency in dependencies
            )
            or (item.expert_index, item.assignment_refs) in swiglu_output_by_group
        ):
            raise SchemaError("grouped SWIGLU producer closure drifted", path="moe_scale_ir2.swiglu")
        gate_ref = comp_output[by_role["gate"].id]
        up_ref = comp_output[by_role["up"].id]
        input_ref = _id("swiglu_input", {
            "overlay": overlay.id, "task": item.id,
            "gate": gate_ref, "up": up_ref,
        })
        add_value(
            input_ref, item.rank, item.packed_value_ref,
            (2 * len(item.assignment_refs), spec.intermediate_size), None,
            buffer_family="dispatch_operand", byte_offset=0,
            alias_source_refs=(gate_ref, up_ref),
        )
        add_value(
            item.packed_value_ref, item.rank, item.packed_value_ref,
            (len(item.assignment_refs), spec.intermediate_size), item.id,
        )
        task_operands[item.id][0].append(input_ref)
        swiglu_output_by_group[
            (item.expert_index, item.assignment_refs)
        ] = item.packed_value_ref

    # Transport sources and COMP operands are connected from typed assignment,
    # packet, and tile witnesses only.
    for packet_ref, group in packet_actions.items():
        packet = packets[packet_ref]
        send = group[SwizzleActionKind.SEND]
        pattern = action_pattern[send.id]
        dep_comp = next((ref for ref in send.deps if ref in comp_output), None)
        if dep_comp is not None:
            task_operands[send.id][0].append(comp_output[dep_comp])
        else:
            current_slices = tuple(
                (item.assignment_ref, item.n_block_index, item.bytes)
                for item in packet.slices
            )
            relay_packets = tuple(
                witnesses[ref].packet_ref for ref in send.deps
                if ref in witnesses
                and witnesses[ref].kind is SwizzleActionKind.WAIT
                and witnesses[ref].packet_ref in packets
                and tuple(
                    (item.assignment_ref, item.n_block_index, item.bytes)
                    for item in packets[witnesses[ref].packet_ref].slices
                ) == current_slices
            )
            if len(relay_packets) > 1:
                raise SchemaError("relay SEND has ambiguous prior packet", path="moe_scale_ir2.transport")
            if relay_packets:
                task_operands[send.id][0].append(recv_value[relay_packets[0]])
                continue
            columns = spec.hidden_size if pattern is FusionPattern.MOE_DISPATCH_GEMM else packet.slices[0].bytes // 2
            rows = packet.logical_bytes // (2 * columns)
            ref = _id("borrowed_send", {"overlay": overlay.id, "packet": packet_ref})
            add_value(ref, send.rank, packet_ref, (rows, columns), None)
            task_operands[send.id][0].append(ref)

    dispatch_recv_by_assignment = {}
    for packet_ref, packet in packets.items():
        recv = packet_actions[packet_ref][SwizzleActionKind.RECV]
        if action_pattern[recv.id] is FusionPattern.MOE_DISPATCH_GEMM:
            for part in packet.slices:
                assignment = assignments[part.assignment_ref]
                if recv.rank != assignment.expert_rank:
                    continue
                if part.assignment_ref in dispatch_recv_by_assignment:
                    raise SchemaError("dispatch assignment has multiple final receive values", path="moe_scale_ir2.comp")
                dispatch_recv_by_assignment[part.assignment_ref] = recv_value[packet_ref]
    activation_by_tile = {}
    for item in witnesses.values():
        if item.kind is not SwizzleActionKind.COMP:
            continue
        pattern = action_pattern[item.id]
        m = len(item.assignment_refs)
        if pattern is FusionPattern.MOE_DISPATCH_GEMM:
            activation_key = (item.expert_index, item.assignment_refs)
            activation = activation_by_tile.get(activation_key)
            if activation is None:
                row_sources = []
                for row, assignment_ref in enumerate(item.assignment_refs):
                    source = dispatch_recv_by_assignment.get(assignment_ref)
                    if source is None:
                        source = _id(
                            "local_activation_row",
                            {"overlay": overlay.id, "tile": tile_by_key[(pattern, item.expert_index, item.assignment_refs, item.n_block_index)].id, "assignment": assignment_ref},
                        )
                        add_value(
                            source, item.rank, assignment_ref,
                            (1, spec.hidden_size), None,
                            buffer_family="dispatch_operand",
                            byte_offset=(
                                2 * m * spec.intermediate_size * 2
                                + row * spec.hidden_size * 2
                            ),
                        )
                    row_sources.append(source)
                sources = tuple(dict.fromkeys(row_sources))
                if len(sources) == 1 and value_meta[sources[0]][2] == (m, spec.hidden_size):
                    activation = sources[0]
                else:
                    tile = tile_by_key[(pattern, item.expert_index, item.assignment_refs, item.n_block_index)]
                    activation = _id("dispatch_activation_view", {"overlay": overlay.id, "tile": tile.id})
                    add_value(
                        activation, item.rank, tile.id,
                        (m, spec.hidden_size), None,
                        buffer_family="dispatch_operand",
                        byte_offset=2 * m * spec.intermediate_size * 2,
                        alias_source_refs=sources,
                    )
                activation_by_tile[activation_key] = activation
            weight_shape = (spec.hidden_size, spec.intermediate_size)
        else:
            activation = swiglu_output_by_group.get(
                (item.expert_index, item.assignment_refs)
            )
            if activation is None:
                activation = _id("swiglu_boundary", {"overlay": overlay.id, "task": item.id})
                add_value(activation, item.rank, item.assignment_refs[0], (m, spec.intermediate_size), None)
            weight_shape = (spec.intermediate_size, value_meta[comp_output[item.id]][2][1])
        weight = _id("weight", {"overlay": overlay.id, "task": item.id, "role": item.work_role})
        add_value(weight, item.rank, item.work_role, weight_shape, None)
        task_operands[item.id][0].extend((activation, weight))

    consumers: dict[str, list[str]] = defaultdict(list)
    for task_ref, (reads, _) in task_operands.items():
        for ref in reads:
            consumers[ref].append(task_ref)
    replacement_ids = set(witnesses)
    tasks = []
    flow_ref_by_task = {}
    flows = []
    for packet_ref, group in packet_actions.items():
        packet = packets[packet_ref]
        route = routes.get(packet.route_ref)
        if route is None:
            raise SchemaError("packet route lacks topology witness", path="moe_scale_ir2.routes")
        flow_id = _id("flow", {"overlay": overlay.id, "packet": packet_ref})
        for action in group.values():
            flow_ref_by_task[action.id] = flow_id
        flows.append(MoeSwizzleIr2Flow(
            flow_id, packet_ref, packet.stage, packet.pivot_rank,
            packet.source_rank, packet.destination_rank,
            packet.source_rank, packet.destination_rank,
            packet.route_ref, tuple(route.die_path), packet.logical_bytes,
            tuple(MoeSwizzleIr2PacketSlice(
                item.assignment_ref, item.source_offset_bytes,
                item.destination_offset_bytes, item.bytes,
            ) for item in packet.slices),
            group[SwizzleActionKind.SEND].id,
            group[SwizzleActionKind.RECV].id,
            group[SwizzleActionKind.WAIT].id,
        ))
    buffer_values: dict[tuple[int, str], list[str]] = defaultdict(list)
    for ref, (rank, _, _, terminal_ref, family, _) in value_meta.items():
        if family is not None and terminal_ref is None and ref not in value_terminal_slices:
            buffer_values[(rank, family)].append(ref)
    buffer_ref = {
        ref: f"moe.scale.ir2.buffer.r{rank}.{family}"
        for (rank, family), refs in buffer_values.items() for ref in refs
    }
    slots_by_family = defaultdict(set)
    for item in witnesses.values():
        if item.buffer_family is not None:
            slots_by_family[item.buffer_family].add(item.buffer_slot)
    buffers = tuple(MoeSwizzleIr2Buffer(
        rank, f"moe.scale.ir2.buffer.r{rank}.{family}", tuple(sorted(refs)),
        max(value_meta[ref][5] + 2 * prod(value_meta[ref][2]) for ref in refs),
        max(slots_by_family[family]) + 1,
    ) for (rank, family), refs in sorted(buffer_values.items()))
    for action in linked:
        item = witnesses[action.id]
        reads, writes = task_operands[action.id]
        if item.kind is SwizzleActionKind.WAIT:
            reads = []
            writes = []
        referenced = tuple(sorted({buffer_ref[ref] for ref in reads + writes if ref in buffer_ref}))
        slot = 0 if item.buffer_slot is None else item.buffer_slot
        flow_ref = flow_ref_by_task.get(item.id)
        m = n = k = None
        if item.kind is SwizzleActionKind.COMP:
            shape = value_meta[comp_output[item.id]][2]
            m, n = shape
            k = spec.hidden_size if action_pattern[item.id] is FusionPattern.MOE_DISPATCH_GEMM else spec.intermediate_size
        deps = tuple(ref for ref in item.deps if ref in replacement_ids)
        if item.kind is SwizzleActionKind.WAIT:
            deps = (packet_actions[item.packet_ref][SwizzleActionKind.RECV].id,)
        tasks.append(MoeSwizzleIr2Task(
            id=item.id, rank=item.rank, die_id=item.rank, kind=item.kind,
            work_role=item.work_role,
            deps=deps,
            read_value_refs=tuple(reads), write_value_refs=tuple(writes),
            buffer_uses=tuple(MoeSwizzleIr2BufferUse(ref, slot) for ref in referenced),
            assignment_refs=item.assignment_refs, expert_index=item.expert_index,
            tile_index=item.tile_index, n_block=item.n_block_index,
            packet_ref=item.packet_ref, stage=item.stage, pivot_rank=item.pivot_rank,
            original_action_refs=item.original_action_refs,
            pipeline_index=0 if item.pipeline_index is None else item.pipeline_index,
            buffer_slot=item.buffer_slot, buffer_family=item.buffer_family,
            peer_rank=item.peer_rank, flow_ref=flow_ref,
            route_ref=item.route_ref if item.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV) else None,
            logical_bytes=item.logical_bytes, flops=item.flops,
            matmul_m=m, matmul_n=n, matmul_k=k,
            dtype=spec.dtype if m is not None else None,
            accumulation_dtype=DType.FP32 if m is not None else None,
        ))
    values = tuple(MoeSwizzleIr2Value(
        id=ref, rank=meta[0], origin_ref=meta[1], shape=meta[2],
        layout="row_major", dtype=spec.dtype, byte_offset=meta[5],
        size_bytes=2 * prod(meta[2]), producer_task_ref=producer.get(ref),
        consumer_task_refs=tuple(consumers.get(ref, ())),
        buffer_ref=buffer_ref.get(ref), terminal_ref=meta[3],
        terminal_slices=value_terminal_slices.get(ref, ()),
        alias_source_refs=value_alias_sources.get(ref, ()),
        borrowed=ref not in producer and ref not in value_alias_sources, replicated=False,
    ) for ref, meta in sorted(value_meta.items()))
    terminal_refs = tuple(sorted(
        ref
        for value in values
        for ref in (
            (value.terminal_ref,) if value.terminal_ref is not None
            else tuple(item.terminal_ref for item in value.terminal_slices)
        )
    ))
    if sum(
        value.size_bytes if value.terminal_ref is not None
        else sum(item.size_bytes for item in value.terminal_slices)
        for value in values
    ) != sum(
        item.bytes for item in execution.terminals if item.kind is MoeScaleExecutionTerminalKind.COMBINED
    ):
        raise SchemaError("projected combined terminal bytes drifted", path="moe_scale_ir2.terminals")
    result = MoeSwizzleIr2Projection.create(
        source_execution_id=execution.id, source_overlay_id=overlay.id,
        tasks=tuple(tasks), values=values, buffers=buffers,
        flows=tuple(sorted(flows, key=lambda item: item.id)),
        terminal_refs=terminal_refs,
        endpoint_session_capacity=endpoint_session_capacity,
    )
    return result


__all__ = ["project_moe_scale_swizzle_ir2"]
