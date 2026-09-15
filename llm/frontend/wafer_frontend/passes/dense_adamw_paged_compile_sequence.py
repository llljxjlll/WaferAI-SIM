"""Relink true Dense AdamW production StateABI into signed bounded timed slots."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    CommandFragment, LinkedProgramManifest, ManifestInputDigest,
    ManifestInputKind, StateABI,
)
from ..schema.dense_adamw_linked import DenseAdamwLinkedProgram
from ..schema.flexible_dense_backward import FlexibleDenseBackwardLinkedProgram
from ..schema.serde import canonical_digest
from .dense_adamw_compile_sequence import compile_dense_adamw_step
from .dense_adamw_mid_program_residency import DenseAdamwBoundedSlots
from .dense_adamw_offload_preflight import DenseAdamwOffloadWindow


def compile_dense_adamw_paged_step(
    window: DenseAdamwOffloadWindow,
    backward: FlexibleDenseBackwardLinkedProgram,
    step_index: int,
    slots: DenseAdamwBoundedSlots,
) -> DenseAdamwLinkedProgram:
    """Relocate each actual LSU StateABI without touching records or SRAM work."""

    if step_index not in (0, 1):
        raise SchemaError("requires genuine step0/step1 source", path="step_index")
    offload = window.materialization
    initial = compile_dense_adamw_step(offload, backward, step_index)
    source = initial.manifest
    fragment = source.fragments[0]
    if type(fragment) is not CommandFragment:
        raise SchemaError("requires production Dense AdamW command fragment", path="source")
    addresses = dict(slots.state_addresses)
    if (
        len(addresses) != 83
        or set(addresses) != {abi.state_ref for abi in fragment.state_abi}
        or slots.workspace_end_bytes != window.hbm_workspace_peak_bytes
        or slots.highest_state_end_bytes > window.resident_hbm_capacity_bytes
    ):
        raise SchemaError("paged slots do not close 83 signed StateABI", path="slots")
    remapped = {
        abi.id: StateABI.create(
            state_ref=abi.state_ref, hbm_binding_ref=abi.hbm_binding_ref,
            kind=abi.kind, lifetime=abi.lifetime, access=abi.access,
            shape=abi.shape, dtype=abi.dtype,
            layout=abi.layout, die_id=abi.die_id,
            address=addresses[abi.state_ref], size_bytes=abi.size_bytes,
            alignment_bytes=abi.alignment_bytes,
        )
        for abi in fragment.state_abi
    }
    if any(
        new.address < slots.workspace_end_bytes
        or new.address + new.size_bytes > window.resident_hbm_capacity_bytes
        for new in remapped.values()
    ):
        raise SchemaError("paged physical ABI collides with P3 workspace", path="slots")
    fragment_key = fragment._semantic_key()
    fragment_key["state_abi"] = tuple(sorted(remapped.values(), key=lambda item: item.id))
    relocated = CommandFragment.create(
        producer_pass=fragment.producer_pass, **fragment_key,
    )
    relocated.validate("paged.fragment")
    state_by_binding = {
        item.hbm_binding_ref: item for item in remapped.values()
    }
    definitions = tuple(
        replace(item, value=state_by_binding[item.symbol.source_ref].address)
        if item.symbol.source_ref in state_by_binding
        else item
        for item in source.program_symbol_definitions
    )
    inputs = tuple(sorted((
        ManifestInputDigest(
            ManifestInputKind.COMMAND_FRAGMENT,
            relocated.id, relocated.schema_version, canonical_digest(relocated),
        ) if item.kind is ManifestInputKind.COMMAND_FRAGMENT else item
        for item in source.input_digests
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    key = source._semantic_key()
    key.update(
        input_digests=inputs,
        fragments=(relocated,),
        fragment_interfaces=tuple(
            replace(interface, fragment_id=relocated.id)
            for interface in source.fragment_interfaces
        ),
        core_streams=tuple(
            replace(stream, records=tuple(
                replace(record, fragment_id=relocated.id)
                for record in stream.records
            )) for stream in source.core_streams
        ),
        program_symbol_definitions=definitions,
        address_operand_bindings=tuple(
            replace(binding, fragment_id=relocated.id)
            for binding in source.address_operand_bindings
        ),
        state_operand_bindings=tuple(
            replace(binding, fragment_id=relocated.id,
                    state_abi_id=remapped[binding.state_abi_id].id)
            for binding in source.state_operand_bindings
        ),
    )
    linked = LinkedProgramManifest.create(
        producer_pass=source.producer_pass, **key,
    )
    linked.validate("paged.linked")
    return DenseAdamwLinkedProgram.create(
        materialization=offload, backward_source=backward,
        manifest=linked, step_index=step_index,
    )


__all__ = ["compile_dense_adamw_paged_step"]
