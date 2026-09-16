"""Replace full TRAIN frozen route HBM seeds with signed P2 source bytes.

The caller must supply an actual *fully linked* manifest and a valid base
ProgramIO contract.  An isolated LSU_LOAD leaf cannot be passed off as a
whole-model executable or as a ProgramIO success.
"""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import LinkedProgramManifest, ProgramSymbolKind, RegionManifest
from ..schema.persistent_state import StateKind
from ..schema.program_io import (
    ProgramBlob, ProgramHbmTarget, ProgramIoContract, ProgramIoPurpose,
    ProgramIoTargetKind, ProgramSramInitialization, _entry_order,
)
from .moe_full_train_route_table_source import MoeFullTrainRouteTableSource
from .moe_full_train_forward_ir0 import FullMoeForwardIr0Phase
from .moe_full_train_ep_placement import MoeFullTrainEpPlacement


def bind_full_moe_route_state_program_io(
    manifest: LinkedProgramManifest,
    base: ProgramIoContract,
    route_source: MoeFullTrainRouteTableSource,
    phase: FullMoeForwardIr0Phase,
    sequence,
    placement: MoeFullTrainEpPlacement,
    *,
    original_dense,
    dense_manifest: LinkedProgramManifest,
    context,
) -> ProgramIoContract:
    """Seed only two real route StateABI homes from the frozen source trace."""
    phase.validate_against(original_dense, sequence)
    placement.validate(phase, original_dense, dense_manifest, sequence, context)
    route_source.validate_against(phase, sequence)
    manifest.validate("moe_route_program_io.manifest")
    base.validate_against(manifest, "moe_route_program_io.base")
    if len(route_source.seeds) != 2:
        raise SchemaError("both full-model route states require signed bytes",
                          path="moe_route_program_io.source")
    state_abis = [state for fragment in manifest.fragments
                  for state in (fragment.fragment if isinstance(fragment, RegionManifest)
                                else fragment).state_abi]
    definitions = tuple(enumerate(manifest.program_symbol_definitions))
    blobs = {blob.id: blob for blob in base.blobs}
    entries = list(base.initializations)
    for seed, home in zip(route_source.seeds, placement.hbm_layout.routes,
                          strict=True):
        matches = [state for state in state_abis
                   if state.state_ref == home.declaration_ref]
        if (len(matches) != 1 or seed.layer != home.layer
                or home.declaration_ref != phase.route_state_refs[seed.layer]
                or seed.route_bytes != home.tensor_size
                or seed.payload_sha256 != home.source_seed_sha256):
            raise SchemaError("linked route StateABI does not cover signed two-layer source",
                              path=f"moe_route_program_io.layer{seed.layer}")
        abi = matches[0]
        if (abi.kind is not StateKind.MOE_STATIC_ROUTE
                or abi.die_id != 0 or abi.address != home.physical_address
                or abi.size_bytes != seed.route_bytes):
            raise SchemaError("route StateABI home differs from physical Die0 source",
                              path=f"moe_route_program_io.layer{seed.layer}")
        symbols = [(index, definition) for index, definition in definitions
                   if definition.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
                   and definition.symbol.source_ref == abi.hbm_binding_ref]
        if (len(symbols) != 1 or symbols[0][1].value != abi.address):
            raise SchemaError("route HBM home needs one resolved physical symbol",
                              path=f"moe_route_program_io.layer{seed.layer}")
        index, definition = symbols[0]
        target = ProgramHbmTarget(
            ProgramIoTargetKind.HBM, definition.symbol.id, index,
            definition.name, abi.id, abi.state_ref, abi.hbm_binding_ref,
        )
        blob = ProgramBlob.create(seed.payload)
        if blob.sha256 != seed.payload_sha256:
            raise SchemaError("route ProgramIO bytes differ from signed trace",
                              path=f"moe_route_program_io.layer{seed.layer}")
        blobs[blob.id] = blob
        entry = ProgramSramInitialization.create(
            target=target, offset_bytes=0, length_bytes=seed.route_bytes,
            blob_ref=blob.id, purpose=ProgramIoPurpose.STATE,
        )
        existing = [position for position, item in enumerate(entries)
                    if isinstance(item.target, ProgramHbmTarget)
                    and item.target.state_ref == abi.state_ref]
        if len(existing) > 1:
            raise SchemaError("route HBM state has duplicate base initializers",
                              path=f"moe_route_program_io.layer{seed.layer}")
        if existing:
            entries[existing[0]] = entry
        else:
            entries.append(entry)
    used = {item.blob_ref for item in (*entries, *base.output_probes)}
    result = ProgramIoContract.create(
        producer_pass="source_bound_full_moe_route_program_io",
        mode=base.mode, source_manifest=manifest,
        program_artifact_sha256=base.program_artifact_sha256,
        blobs=tuple(blobs[ref] for ref in sorted(used)),
        initializations=tuple(sorted(entries, key=_entry_order)),
        output_probes=base.output_probes,
    )
    result.validate_against(manifest, "moe_route_program_io.result")
    return result


__all__ = ["bind_full_moe_route_state_program_io"]
