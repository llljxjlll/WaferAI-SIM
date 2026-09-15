"""Defer typed HBM StateABI initialization to a signed external DMA bring-in.

This is a compile-time binding only. The NpuSim startup gate must independently
prove the HBM ranges absent before DMA and restored before a worker can run.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..errors import SchemaError
from ..schema.artifact_manifest import RegionManifest
from ..schema.external_dma_program import ExternalDmaProgram
from ..schema.external_memory import ExternalTransferDirection
from ..schema.program_io import (
    ProgramHbmTarget,
    ProgramIoContract,
)


def defer_external_state_initializations(
    contract: ProgramIoContract,
    manifest,
    program: ExternalDmaProgram,
    state_seed_overrides: Mapping[str, bytes],
) -> ProgramIoContract:
    """Bind every deferred HBM StateABI byte to an external seed and E2H DMA.

    Resident StateABIs are deliberately not affected. An H2E final phase must
    also cover every deferred span, so the final owner can be external.
    """

    contract.validate_against(manifest)
    program.validate()
    if not state_seed_overrides or any(
        type(value) is not bytes or not value for value in state_seed_overrides.values()
    ):
        raise SchemaError("external states need exact non-empty seed payloads",
                          path="state_seed_overrides")
    state_abis = {
        abi.id: abi
        for fragment in manifest.fragments
        for abi in (
            fragment.fragment.state_abi
            if isinstance(fragment, RegionManifest) else fragment.state_abi
        )
        if abi.state_ref in state_seed_overrides
    }
    if not state_abis or {abi.state_ref for abi in state_abis.values()} != set(
        state_seed_overrides
    ):
        raise SchemaError("deferred state refs differ from linked StateABI",
                          path="state_seed_overrides")
    connections = {item.id: item for item in program.fabric.connections}
    links = {item.id: item for item in program.fabric.links}
    seeds = program.external_seeds
    deferred = []
    for item in contract.initializations:
        if not isinstance(item.target, ProgramHbmTarget):
            continue
        if item.target.state_ref not in state_seed_overrides:
            continue
        abi = state_abis.get(item.target.state_abi_id)
        if abi is None or item.offset_bytes != 0 or item.length_bytes != abi.size_bytes:
            raise SchemaError("deferred StateABI requires an exact whole-state HBM seed",
                              path="program_io.initializations")
        deferred.append(item.target.state_abi_id)
    if set(deferred) != set(state_abis) or len(deferred) != len(state_abis):
        raise SchemaError("all deferred StateABIs must have one host seed before deferral",
                          path="program_io.initializations")

    for abi in state_abis.values():
        seed = state_seed_overrides[abi.state_ref]
        if len(seed) != abi.size_bytes:
            raise SchemaError("external seed differs from whole StateABI bytes",
                              path=f"state_seed_overrides[{abi.state_ref!r}]")
        restore_location = None
        for direction in (
            ExternalTransferDirection.EXTERNAL_TO_HBM,
            ExternalTransferDirection.HBM_TO_EXTERNAL,
        ):
            matches = []
            for descriptor in program.descriptors:
                if descriptor.direction is not direction:
                    continue
                connection = connections[descriptor.connection_ref]
                if connection.target_die_id != abi.die_id or not (
                    descriptor.hbm_address <= abi.address and
                    abi.address + abi.size_bytes <=
                    descriptor.hbm_address + descriptor.size_bytes
                ):
                    continue
                external_ref = links[connection.link_ref].external_capacity_ref
                address = descriptor.external_address + (
                    abi.address - descriptor.hbm_address
                )
                location = (external_ref, address)
                if direction is ExternalTransferDirection.EXTERNAL_TO_HBM:
                    if not any(
                        item.external_capacity_ref == external_ref and
                        item.address <= address and
                        address + abi.size_bytes <=
                        item.address + len(bytes.fromhex(item.payload_hex)) and
                        bytes.fromhex(item.payload_hex)[
                            address - item.address:
                            address - item.address + abi.size_bytes
                        ] == seed
                        for item in seeds
                    ):
                        continue
                elif location != restore_location:
                    continue
                matches.append(location)
            if len(matches) != 1:
                detail = ("with authentic payload" if direction is
                          ExternalTransferDirection.EXTERNAL_TO_HBM
                          else "at the same external authority")
                raise SchemaError("whole StateABI needs exactly one signed external "
                                  f"{direction.value} DMA span {detail}",
                                  path=f"external_dma_program.descriptors[{abi.id}]")
            if direction is ExternalTransferDirection.EXTERNAL_TO_HBM:
                restore_location = matches[0]
        assert restore_location is not None
        external_ref, external_address = restore_location
        if not any(
            item.external_capacity_ref == external_ref and
            item.address <= external_address and
            external_address + abi.size_bytes <=
            item.address + len(bytes.fromhex(item.expected_payload_hex))
            for item in program.external_probes
        ):
            raise SchemaError("final authoritative StateABI lacks an external probe",
                              path=f"external_dma_program.external_probes[{abi.id}]")
    if not any(seed != bytes(len(seed)) for seed in state_seed_overrides.values()):
        raise SchemaError("external seed has no non-zero restoration witness",
                          path="state_seed_overrides")
    deferred_refs = set(state_seed_overrides)
    result = ProgramIoContract.create(
        producer_pass="external_state_authority",
        mode=contract.mode,
        source_manifest=manifest,
        program_artifact_sha256=contract.program_artifact_sha256,
        blobs=contract.blobs,
        initializations=tuple(
            item for item in contract.initializations
            if not (
                isinstance(item.target, ProgramHbmTarget) and
                item.target.state_ref in deferred_refs
            )
        ),
        output_probes=contract.output_probes,
    )
    result.validate_against(manifest)
    return result
