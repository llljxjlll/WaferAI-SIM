from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    BufferABI,
    LinkedProgramManifest,
    ProgramSymbolKind,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.common import DType, stable_artifact_id
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.program_io import (
    PROGRAM_IO_CONTRACT_SCHEMA_VERSION,
    ProgramBlob,
    ProgramIoContract,
    ProgramIoMode,
    ProgramIoPurpose,
    ProgramIoTargetKind,
    ProgramOutputCapture,
    ProgramOutputComparison,
    ProgramOutputProbe,
    ProgramSramInitialization,
    ProgramSramTarget,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)

from test_n6_schema import _single_profile_fixture


def _abis(manifest: LinkedProgramManifest) -> tuple[BufferABI, ...]:
    by_id: dict[str, BufferABI] = {}
    for linked in manifest.fragments:
        fragment = linked.fragment if type(linked) is RegionManifest else linked
        for abi in fragment.buffer_abi:
            previous = by_id.setdefault(abi.id, abi)
            assert previous == abi
    return tuple(sorted(by_id.values(), key=lambda abi: abi.id))


def _symbol_for(
    manifest: LinkedProgramManifest,
    abi: BufferABI,
) -> tuple[int, object]:
    matches = tuple(
        (index, definition)
        for index, definition in enumerate(manifest.program_symbol_definitions)
        if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
        and definition.symbol.source_ref == abi.storage_id
        and abi.logical_core in definition.logical_cores
    )
    assert len(matches) == 1
    return matches[0]


def _runtime_core(manifest: LinkedProgramManifest, abi: BufferABI) -> int:
    return next(
        binding.runtime_core_id
        for binding in manifest.core_bindings
        if binding.logical_core == abi.logical_core
    )


def _initialization(
    manifest: LinkedProgramManifest,
    abi: BufferABI,
    blob: ProgramBlob,
    *,
    purpose: ProgramIoPurpose = ProgramIoPurpose.ACTIVATION,
    **updates: object,
) -> ProgramSramInitialization:
    symbol_index, definition = _symbol_for(manifest, abi)
    semantic_key: dict[str, object] = {
        "target": ProgramSramTarget(
            kind=ProgramIoTargetKind.SRAM,
            runtime_core_id=_runtime_core(manifest, abi),
            program_symbol_ref=definition.symbol.id,
            finalized_symbol_index=symbol_index,
            expected_symbol_name=definition.name,
            buffer_abi_id=abi.id,
            storage_id=abi.storage_id,
            value_id=abi.value_id,
            tensor_slice=abi.tensor_slice,
            dtype=abi.dtype,
            layout=abi.layout,
        ),
        "offset_bytes": 0,
        "length_bytes": abi.size_bytes,
        "blob_ref": blob.id,
        "purpose": purpose,
    }
    semantic_key.update(updates)
    return ProgramSramInitialization.create(**semantic_key)


def _probe(
    manifest: LinkedProgramManifest,
    abi: BufferABI,
    blob: ProgramBlob,
    **updates: object,
) -> ProgramOutputProbe:
    symbol_index, definition = _symbol_for(manifest, abi)
    semantic_key: dict[str, object] = {
        "target": ProgramSramTarget(
            kind=ProgramIoTargetKind.SRAM,
            runtime_core_id=_runtime_core(manifest, abi),
            program_symbol_ref=definition.symbol.id,
            finalized_symbol_index=symbol_index,
            expected_symbol_name=definition.name,
            buffer_abi_id=abi.id,
            storage_id=abi.storage_id,
            value_id=abi.value_id,
            tensor_slice=abi.tensor_slice,
            dtype=abi.dtype,
            layout=abi.layout,
        ),
        "offset_bytes": 0,
        "length_bytes": abi.size_bytes,
        "blob_ref": blob.id,
        "comparison": ProgramOutputComparison.EXACT_BYTES,
        "capture": ProgramOutputCapture.AFTER_PROGRAM,
    }
    semantic_key.update(updates)
    return ProgramOutputProbe.create(**semantic_key)


def _retarget_initialization(
    entry: ProgramSramInitialization,
    **updates: object,
) -> ProgramSramInitialization:
    assert type(entry.target) is ProgramSramTarget
    return ProgramSramInitialization.create(
        **{
            **entry._semantic_key(),
            "target": replace(entry.target, **updates),
        }
    )


def _retarget_probe(
    entry: ProgramOutputProbe,
    **updates: object,
) -> ProgramOutputProbe:
    assert type(entry.target) is ProgramSramTarget
    return ProgramOutputProbe.create(
        **{
            **entry._semantic_key(),
            "target": replace(entry.target, **updates),
        }
    )


def _contract_with(
    contract: ProgramIoContract,
    manifest: LinkedProgramManifest,
    *,
    mode: ProgramIoMode | None = None,
    initializations: tuple[ProgramSramInitialization, ...] | None = None,
    output_probes: tuple[ProgramOutputProbe, ...] | None = None,
    blobs: tuple[ProgramBlob, ...] | None = None,
) -> ProgramIoContract:
    actual_initializations = (
        contract.initializations if initializations is None else initializations
    )
    actual_probes = contract.output_probes if output_probes is None else output_probes
    if blobs is None:
        used = {
            entry.blob_ref for entry in (*actual_initializations, *actual_probes)
        }
        blobs = tuple(blob for blob in contract.blobs if blob.id in used)
    return ProgramIoContract.create(
        producer_pass=contract.producer_pass,
        mode=contract.mode if mode is None else mode,
        source_manifest=manifest,
        program_artifact_sha256=contract.program_artifact_sha256,
        blobs=blobs,
        initializations=actual_initializations,
        output_probes=actual_probes,
    )


def _restable_contract(
    contract: ProgramIoContract,
    **updates: object,
) -> ProgramIoContract:
    changed = replace(contract, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "program_io_contract",
            changed._semantic_key(),
            schema_version=PROGRAM_IO_CONTRACT_SCHEMA_VERSION,
        ),
    )


def _fixture() -> tuple[LinkedProgramManifest, ProgramIoContract]:
    manifest = _single_profile_fixture()[2].entries[0].manifest
    borrowed = tuple(
        abi for abi in _abis(manifest) if abi.ownership is BufferOwnership.BORROWED
    )
    owned = tuple(
        abi for abi in _abis(manifest) if abi.ownership is BufferOwnership.OWNED
    )
    blobs: list[ProgramBlob] = []
    initializations: list[ProgramSramInitialization] = []
    for index, abi in enumerate(borrowed):
        blob = ProgramBlob.create(bytes([0x10 + index]) * abi.size_bytes)
        blobs.append(blob)
        initializations.append(_initialization(manifest, abi, blob))
    output_blob = ProgramBlob.create(bytes([0xA5]) * owned[0].size_bytes)
    blobs.append(output_blob)
    probe = _probe(manifest, owned[0], output_blob)
    unique_blobs = tuple({blob.id: blob for blob in blobs}.values())
    contract = ProgramIoContract.create(
        producer_pass="program_io_fixture",
        mode=ProgramIoMode.FUNCTIONAL,
        source_manifest=manifest,
        program_artifact_sha256="ab" * 32,
        blobs=unique_blobs,
        initializations=tuple(initializations),
        output_probes=(probe,),
    )
    return manifest, contract


class ProgramIoSchemaTest(unittest.TestCase):
    def test_functional_contract_is_stable_strict_and_roundtrips(self) -> None:
        manifest, contract = _fixture()
        contract.validate_against(manifest)
        second_manifest, second = _fixture()
        self.assertEqual(second_manifest, manifest)
        self.assertEqual(second, contract)
        self.assertEqual(
            loads_dataclass(ProgramIoContract, canonical_json(contract)),
            contract,
        )

        raw = canonical_json(contract)
        with self.assertRaises(SchemaError):
            loads_dataclass(
                ProgramIoContract,
                raw[:-1] + ',"unknown":0}',
            )

    def test_old_v1alpha1_contract_fails_closed(self) -> None:
        _manifest, contract = _fixture()
        old = replace(
            contract,
            schema_version="wafer_frontend.program_io_contract/v1alpha1",
        )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            old.validate()

    def test_blob_rejects_noncanonical_bytes_length_hash_and_id(self) -> None:
        blob = ProgramBlob.create(b"abc")
        blob.validate()
        for changed in (
            replace(blob, bytes_base64="YWJj\n"),
            replace(blob, length_bytes=4),
            replace(blob, sha256="0" * 64),
            replace(blob, id="blob_impostor"),
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(SchemaError):
                    changed.validate()

    def test_rejects_manifest_digest_and_symbol_identity_tamper(self) -> None:
        manifest, contract = _fixture()
        wrong_digest = _restable_contract(
            contract,
            source_linked_manifest_digest="0" * 64,
        )
        with self.assertRaises(SchemaError):
            wrong_digest.validate_against(manifest)

        first = contract.initializations[0]
        cases = (
            _retarget_initialization(first, program_symbol_ref="wrong_label"),
            _retarget_initialization(
                first,
                finalized_symbol_index=first.target.finalized_symbol_index + 1,
            ),
            _retarget_initialization(
                first,
                expected_symbol_name=first.target.expected_symbol_name + "_wrong",
            ),
            _retarget_initialization(first, storage_id="wrong_storage"),
            _retarget_initialization(first, dtype=DType.FP32),
        )
        for changed in cases:
            forged = _contract_with(
                contract,
                manifest,
                initializations=(changed, *contract.initializations[1:]),
            )
            with self.subTest(changed=changed):
                with self.assertRaises(SchemaError):
                    forged.validate_against(manifest)

    def test_functional_rejects_missing_borrowed_owned_and_timing_partial(self) -> None:
        manifest, contract = _fixture()
        missing = _contract_with(
            contract,
            manifest,
            initializations=contract.initializations[:-1],
        )
        with self.assertRaises(SchemaError):
            missing.validate_against(manifest)

        owned = next(
            abi for abi in _abis(manifest) if abi.ownership is BufferOwnership.OWNED
        )
        owned_blob = ProgramBlob.create(bytes([0xCC]) * owned.size_bytes)
        owned_activation = _initialization(manifest, owned, owned_blob)
        forged = _contract_with(
            contract,
            manifest,
            initializations=(*contract.initializations, owned_activation),
            blobs=(*contract.blobs, owned_blob),
        )
        with self.assertRaises(SchemaError):
            forged.validate_against(manifest)

        timing_partial = _initialization(
            manifest,
            owned,
            owned_blob,
            purpose=ProgramIoPurpose.TIMING_PARTIAL,
        )
        forged = _contract_with(
            contract,
            manifest,
            initializations=(*contract.initializations, timing_partial),
            blobs=(*contract.blobs, owned_blob),
        )
        with self.assertRaises(SchemaError):
            forged.validate_against(manifest)

    def test_timing_explicitly_allows_owned_timing_partial(self) -> None:
        manifest, contract = _fixture()
        owned = next(
            abi for abi in _abis(manifest) if abi.ownership is BufferOwnership.OWNED
        )
        blob = ProgramBlob.create(bytes([0x5A]) * owned.size_bytes)
        timing_partial = _initialization(
            manifest,
            owned,
            blob,
            purpose=ProgramIoPurpose.TIMING_PARTIAL,
        )
        timing = _contract_with(
            contract,
            manifest,
            mode=ProgramIoMode.TIMING,
            initializations=(*contract.initializations, timing_partial),
            blobs=(*contract.blobs, blob),
        )
        timing.validate_against(manifest)

        borrowed = contract.initializations[0]
        borrowed_timing_partial = ProgramSramInitialization.create(
            **{
                **borrowed._semantic_key(),
                "purpose": ProgramIoPurpose.TIMING_PARTIAL,
            }
        )
        forged = _contract_with(
            contract,
            manifest,
            mode=ProgramIoMode.TIMING,
            initializations=(
                borrowed_timing_partial,
                *contract.initializations[1:],
            ),
        )
        with self.assertRaisesRegex(SchemaError, "OWNED"):
            forged.validate_against(manifest)

    def test_rejects_out_of_range_and_overlapping_initializations(self) -> None:
        manifest, contract = _fixture()
        first = contract.initializations[0]
        out_of_range = ProgramSramInitialization.create(
            **{**first._semantic_key(), "offset_bytes": 1}
        )
        forged = _contract_with(
            contract,
            manifest,
            initializations=(out_of_range, *contract.initializations[1:]),
        )
        with self.assertRaises(SchemaError):
            forged.validate_against(manifest)

        overlap = ProgramSramInitialization.create(
            **{**first._semantic_key(), "purpose": ProgramIoPurpose.WEIGHT}
        )
        forged = _contract_with(
            contract,
            manifest,
            initializations=(*contract.initializations, overlap),
        )
        with self.assertRaises(SchemaError):
            forged.validate_against(manifest)

    def test_output_probe_rejects_borrowed_target_and_metadata_tamper(self) -> None:
        manifest, contract = _fixture()
        borrowed_abi = next(
            abi
            for abi in _abis(manifest)
            if abi.ownership is BufferOwnership.BORROWED
        )
        borrowed_init = next(
            entry
            for entry in contract.initializations
            if type(entry.target) is ProgramSramTarget
            and entry.target.buffer_abi_id == borrowed_abi.id
        )
        blob = next(
            blob for blob in contract.blobs if blob.id == borrowed_init.blob_ref
        )
        borrowed_probe = _probe(manifest, borrowed_abi, blob)
        forged = _contract_with(
            contract,
            manifest,
            output_probes=(borrowed_probe,),
        )
        with self.assertRaises(SchemaError):
            forged.validate_against(manifest)

        probe = contract.output_probes[0]
        assert type(probe.target) is ProgramSramTarget
        wrong_layout = _retarget_probe(
            probe,
            layout=probe.target.layout + "_wrong",
        )
        forged = _contract_with(
            contract,
            manifest,
            output_probes=(wrong_layout,),
        )
        with self.assertRaises(SchemaError):
            forged.validate_against(manifest)


if __name__ == "__main__":
    unittest.main()
