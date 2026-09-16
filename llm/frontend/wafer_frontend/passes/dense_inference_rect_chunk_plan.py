"""Source-bound parameter chunk preflight for a bounded rectangular Die mesh.

This is deliberately an admission check, not a pager.  Splitting a transfer
cannot make a linked LSU_LOAD that still reads the entire state executable.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.artifact_manifest import (
    LinkedProgramManifest, RecordOpcode, RegionManifest, StateABI, StateKind,
)


@dataclass(frozen=True, slots=True)
class ParameterChunk:
    state_abi_id: str
    die_id: int
    source_offset_bytes: int
    size_bytes: int
    hbm_address: int


@dataclass(frozen=True, slots=True)
class RectChunkPreflight:
    chunks: tuple[ParameterChunk, ...]
    hbm_capacity_bytes_per_die: int
    source_state_count: int
    native_execution_admitted: bool = False


def _window(
    capacity: int, reserved: tuple[tuple[int, int], ...], alignment: int,
) -> tuple[int, int]:
    if type(capacity) is not int or capacity <= 0 or type(alignment) is not int or alignment <= 0:
        raise SchemaError("bounded HBM capacity and alignment must be positive", path="chunk.window")
    previous = 0
    holes = []
    for start, end in sorted(reserved):
        if (type(start) is not int or type(end) is not int
                or start < previous or start >= end or end > capacity):
            raise SchemaError("KV/activation HBM reservations overlap or exceed capacity",
                              path="chunk.reserved")
        aligned = (previous + alignment - 1) // alignment * alignment
        if aligned < start:
            holes.append((aligned, start - aligned))
        previous = end
    aligned = (previous + alignment - 1) // alignment * alignment
    if aligned < capacity:
        holes.append((aligned, capacity - aligned))
    if not holes:
        raise SchemaError("no HBM parameter window remains", path="chunk.window")
    return max(holes, key=lambda hole: (hole[1], -hole[0]))


def plan_bounded_parameter_chunks(
    states: tuple[StateABI, ...],
    consumer_sizes: dict[str, tuple[int, ...]],
    *,
    hbm_capacity_bytes_per_die: int,
    reserved_by_die: dict[int, tuple[tuple[int, int], ...]],
    alignment_bytes: int = 64,
) -> RectChunkPreflight:
    """Use real parameter StateABIs and bound LSU read sizes; fail on untiled reads.

    The returned candidate partitions source byte ranges exactly, but does not
    claim native execution: record offset rewrites, live activation lifetimes,
    DTE scheduling and external backing must be proven separately.
    """
    if not states or len({state.id for state in states}) != len(states):
        raise SchemaError("parameter StateABI inventory is empty or duplicated",
                          path="chunk.states")
    if (set(reserved_by_die) != {state.die_id for state in states}
            or any(not spans for spans in reserved_by_die.values())):
        raise SchemaError("every physical Die needs explicit KV and activation reservations",
                          path="chunk.reserved")
    if set(consumer_sizes) != {state.id for state in states}:
        raise SchemaError("every parameter needs bound LSU consumers", path="chunk.consumers")
    chunks: list[ParameterChunk] = []
    for state in sorted(states, key=lambda item: (item.die_id, -item.size_bytes, item.id)):
        if state.kind is not StateKind.PARAMETER or state.size_bytes <= 0:
            raise SchemaError("only true positive parameter StateABIs are pageable",
                              path=f"chunk.states.{state.id}")
        if state.alignment_bytes > alignment_bytes:
            raise SchemaError("parameter home is stricter than the chunk alignment",
                              path=f"chunk.states.{state.id}")
        base, available = _window(
            hbm_capacity_bytes_per_die, reserved_by_die[state.die_id],
            alignment_bytes,
        )
        available = available // alignment_bytes * alignment_bytes
        if available == 0:
            raise SchemaError("aligned HBM parameter window is empty",
                              path=f"chunk.states.{state.id}")
        if state.size_bytes > available and state.size_bytes % alignment_bytes:
            raise SchemaError("split parameter lacks a full aligned source range",
                              path=f"chunk.states.{state.id}")
        sizes = consumer_sizes[state.id]
        if not sizes or any(type(size) is not int or size <= 0 or size > state.size_bytes
                            for size in sizes):
            raise SchemaError("bound LSU read exceeds parameter StateABI",
                              path=f"chunk.consumers.{state.id}")
        if state.size_bytes > available and any(size > available for size in sizes):
            raise UnsupportedFeatureError(
                "requires_tiled_consumer: linked LSU reads the full parameter "
                "without a source-bound tiled compute/record rewrite",
                path=f"chunk.consumers.{state.id}",
            )
        for offset in range(0, state.size_bytes, available):
            chunks.append(ParameterChunk(
                state.id, state.die_id, offset,
                min(available, state.size_bytes - offset), base,
            ))
    return RectChunkPreflight(
        tuple(chunks), hbm_capacity_bytes_per_die, len(states),
    )


def preflight_dense_rect_linked_segment(
    manifest: LinkedProgramManifest,
    *,
    hbm_capacity_bytes_per_die: int,
    reserved_by_die: dict[int, tuple[tuple[int, int], ...]],
    alignment_bytes: int = 64,
) -> RectChunkPreflight:
    """Extract parameter ABIs and actual LSU sizes from a validated linked program."""
    if type(manifest) is not LinkedProgramManifest:
        raise SchemaError("requires production linked manifest", path="chunk.manifest")
    manifest.validate("chunk.manifest")
    fragments = {}
    states = {}
    for item in manifest.fragments:
        fragment = item.fragment if isinstance(item, RegionManifest) else item
        fragments[fragment.id] = fragment
        fragments[item.id] = fragment
        for state in fragment.state_abi:
            if state.kind is StateKind.PARAMETER:
                prior = states.setdefault(state.id, state)
                if prior != state:
                    raise SchemaError("conflicting physical parameter StateABI",
                                      path="chunk.manifest.fragments")
    consumers: dict[str, list[int]] = {state_id: [] for state_id in states}
    for binding in manifest.state_operand_bindings:
        if binding.state_abi_id not in states:
            continue
        fragment = fragments[binding.fragment_id]
        stream = next((stream for stream in fragment.core_streams
                       if stream.logical_core == binding.logical_core), None)
        if stream is None or binding.fragment_record_index >= len(stream.records):
            raise SchemaError("parameter binding lacks physical record", path="chunk.bindings")
        record = stream.records[binding.fragment_record_index]
        if record.opcode is not RecordOpcode.LSU_LOAD:
            raise SchemaError("parameter binding must be an LSU load", path="chunk.bindings")
        if binding.logical_core.die_id != states[binding.state_abi_id].die_id:
            raise SchemaError("LSU consumer is on wrong physical Die", path="chunk.bindings")
        consumers[binding.state_abi_id].append(record.operands[1].literal_value)
    return plan_bounded_parameter_chunks(
        tuple(states.values()),
        {state: tuple(sizes) for state, sizes in consumers.items()},
        hbm_capacity_bytes_per_die=hbm_capacity_bytes_per_die,
        reserved_by_die=reserved_by_die,
        alignment_bytes=alignment_bytes,
    )


__all__ = [
    "ParameterChunk", "RectChunkPreflight", "plan_bounded_parameter_chunks",
    "preflight_dense_rect_linked_segment",
]
