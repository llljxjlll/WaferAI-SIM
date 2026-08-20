from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest,
    ProgramSymbolKind,
    RegionManifest,
    StateABI,
)
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess,
)
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramBlob,
    ProgramHbmTarget,
    ProgramIoContract,
    ProgramIoMode,
    ProgramIoPurpose,
    ProgramIoTargetKind,
    ProgramOutputCapture,
    ProgramOutputComparison,
    ProgramOutputProbe,
    ProgramSramInitialization,
)

from test_n6_pipeline import _compile_through_n6
from test_program_io_schema import _abis, _initialization, _probe


def _state_abis(manifest: LinkedProgramManifest) -> tuple[StateABI, ...]:
    result: dict[str, StateABI] = {}
    for linked in manifest.fragments:
        fragment = linked.fragment if type(linked) is RegionManifest else linked
        for abi in fragment.state_abi:
            previous = result.setdefault(abi.id, abi)
            assert previous == abi
    return tuple(sorted(result.values(), key=lambda abi: abi.id))


def _hbm_target(
    manifest: LinkedProgramManifest,
    abi: StateABI,
) -> ProgramHbmTarget:
    matches = tuple(
        (index, definition)
        for index, definition in enumerate(manifest.program_symbol_definitions)
        if definition.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
        and definition.symbol.source_ref == abi.hbm_binding_ref
    )
    assert len(matches) == 1
    symbol_index, definition = matches[0]
    return ProgramHbmTarget(
        kind=ProgramIoTargetKind.HBM,
        program_symbol_ref=definition.symbol.id,
        finalized_symbol_index=symbol_index,
        expected_symbol_name=definition.name,
        state_abi_id=abi.id,
        state_ref=abi.state_ref,
        hbm_binding_ref=abi.hbm_binding_ref,
    )


def _state_initialization(
    manifest: LinkedProgramManifest,
    abi: StateABI,
    blob: ProgramBlob,
) -> ProgramSramInitialization:
    return ProgramSramInitialization.create(
        target=_hbm_target(manifest, abi),
        offset_bytes=0,
        length_bytes=abi.size_bytes,
        blob_ref=blob.id,
        purpose=ProgramIoPurpose.STATE,
    )


def _state_probe(
    manifest: LinkedProgramManifest,
    abi: StateABI,
    blob: ProgramBlob,
) -> ProgramOutputProbe:
    return ProgramOutputProbe.create(
        target=_hbm_target(manifest, abi),
        offset_bytes=0,
        length_bytes=abi.size_bytes,
        blob_ref=blob.id,
        comparison=ProgramOutputComparison.EXACT_BYTES,
        capture=ProgramOutputCapture.AFTER_PROGRAM,
    )


def _stateful_contract(
    manifest: LinkedProgramManifest,
) -> tuple[ProgramIoContract, StateABI, StateABI]:
    borrowed = tuple(
        abi for abi in _abis(manifest) if abi.ownership is BufferOwnership.BORROWED
    )
    owned = tuple(
        abi for abi in _abis(manifest) if abi.ownership is BufferOwnership.OWNED
    )
    parameter = next(
        abi
        for abi in _state_abis(manifest)
        if abi.access is PersistentStateAccess.READ_ONLY
    )
    kv = next(
        abi
        for abi in _state_abis(manifest)
        if abi.access is PersistentStateAccess.READ_WRITE
    )

    blobs: list[ProgramBlob] = []
    initializations: list[ProgramSramInitialization] = []
    for index, abi in enumerate(borrowed):
        blob = ProgramBlob.create(bytes([0x10 + index]) * abi.size_bytes)
        blobs.append(blob)
        initializations.append(_initialization(manifest, abi, blob))
    for fill, abi in ((0x51, parameter), (0x52, kv)):
        blob = ProgramBlob.create(bytes([fill]) * abi.size_bytes)
        blobs.append(blob)
        initializations.append(_state_initialization(manifest, abi, blob))

    owned_blob = ProgramBlob.create(bytes([0xA5]) * owned[0].size_bytes)
    kv_blob = ProgramBlob.create(bytes([0xB6]) * kv.size_bytes)
    blobs.extend((owned_blob, kv_blob))
    probes = (
        _probe(manifest, owned[0], owned_blob),
        _state_probe(manifest, kv, kv_blob),
    )
    contract = ProgramIoContract.create(
        producer_pass="program_io_hbm_fixture",
        mode=ProgramIoMode.FUNCTIONAL,
        source_manifest=manifest,
        program_artifact_sha256="cd" * 32,
        blobs=tuple({blob.id: blob for blob in blobs}.values()),
        initializations=tuple(initializations),
        output_probes=probes,
    )
    return contract, parameter, kv


def _rebuild(
    contract: ProgramIoContract,
    manifest: LinkedProgramManifest,
    *,
    initializations: tuple[ProgramSramInitialization, ...] | None = None,
    output_probes: tuple[ProgramOutputProbe, ...] | None = None,
) -> ProgramIoContract:
    actual_initializations = (
        contract.initializations if initializations is None else initializations
    )
    actual_probes = (
        contract.output_probes if output_probes is None else output_probes
    )
    used = {
        entry.blob_ref for entry in (*actual_initializations, *actual_probes)
    }
    return ProgramIoContract.create(
        producer_pass=contract.producer_pass,
        mode=contract.mode,
        source_manifest=manifest,
        program_artifact_sha256=contract.program_artifact_sha256,
        blobs=tuple(blob for blob in contract.blobs if blob.id in used),
        initializations=actual_initializations,
        output_probes=actual_probes,
    )


class ProgramIoHbmValidateAgainstTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        run = _compile_through_n6()
        entry = run.linked.entries[0]
        cls.manifest = entry.manifest
        cls.ir1 = entry.lowering_context.ir1
        cls.contract, cls.parameter, cls.kv = _stateful_contract(cls.manifest)

    def test_parameter_seed_and_kv_seed_probe_close_exactly(self) -> None:
        self.contract.validate_against(self.manifest)
        hbm_initializations = tuple(
            entry
            for entry in self.contract.initializations
            if type(entry.target) is ProgramHbmTarget
        )
        hbm_probes = tuple(
            entry
            for entry in self.contract.output_probes
            if type(entry.target) is ProgramHbmTarget
        )
        self.assertEqual(len(hbm_initializations), 2)
        self.assertEqual(len(hbm_probes), 1)
        self.assertEqual(hbm_probes[0].target.state_ref, self.kv.state_ref)
        self.assertFalse(hasattr(hbm_probes[0].target, "address"))

    def test_rejects_partial_span_and_final_symbol_tamper(self) -> None:
        state_entry = next(
            entry
            for entry in self.contract.initializations
            if type(entry.target) is ProgramHbmTarget
        )
        partial = ProgramSramInitialization.create(
            **{**state_entry._semantic_key(), "offset_bytes": 1}
        )
        forged = _rebuild(
            self.contract,
            self.manifest,
            initializations=tuple(
                partial if entry is state_entry else entry
                for entry in self.contract.initializations
            ),
        )
        with self.assertRaisesRegex(SchemaError, "whole-state"):
            forged.validate_against(self.manifest)

        assert type(state_entry.target) is ProgramHbmTarget
        wrong_target = replace(
            state_entry.target,
            expected_symbol_name=state_entry.target.expected_symbol_name + "_wrong",
        )
        wrong_symbol = ProgramSramInitialization.create(
            **{**state_entry._semantic_key(), "target": wrong_target}
        )
        forged = _rebuild(
            self.contract,
            self.manifest,
            initializations=tuple(
                wrong_symbol if entry is state_entry else entry
                for entry in self.contract.initializations
            ),
        )
        with self.assertRaisesRegex(SchemaError, "final symbol identity"):
            forged.validate_against(self.manifest)

    def test_read_only_probe_and_nonmanifest_identity_fail_closed(self) -> None:
        parameter_blob = next(
            blob
            for blob in self.contract.blobs
            if blob.length_bytes == self.parameter.size_bytes
        )
        parameter_probe = _state_probe(
            self.manifest, self.parameter, parameter_blob
        )
        forged = _rebuild(
            self.contract,
            self.manifest,
            output_probes=(*self.contract.output_probes, parameter_probe),
        )
        with self.assertRaisesRegex(SchemaError, "READ_WRITE"):
            forged.validate_against(self.manifest)

        parameter_entry = next(
            entry
            for entry in self.contract.initializations
            if type(entry.target) is ProgramHbmTarget
            and entry.target.state_ref == self.parameter.state_ref
        )
        assert type(parameter_entry.target) is ProgramHbmTarget
        wrong_target = replace(
            parameter_entry.target, state_ref="nonmanifest-state"
        )
        reserved_entry = ProgramSramInitialization.create(
            **{**parameter_entry._semantic_key(), "target": wrong_target}
        )
        forged = _rebuild(
            self.contract,
            self.manifest,
            initializations=tuple(
                reserved_entry if entry is parameter_entry else entry
                for entry in self.contract.initializations
            ),
        )
        with self.assertRaisesRegex(SchemaError, "exactly preserve StateABI"):
            forged.validate_against(self.manifest)


if __name__ == "__main__":
    unittest.main()
