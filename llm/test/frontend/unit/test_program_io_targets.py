from __future__ import annotations

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir2 import TensorSlice
from llm.frontend.wafer_frontend.schema.program_io import (
    PROGRAM_IO_CONTRACT_SCHEMA_VERSION,
    PROGRAM_OUTPUT_PROBE_SCHEMA_VERSION,
    PROGRAM_SRAM_INITIALIZATION_SCHEMA_VERSION,
    ProgramBlob,
    ProgramHbmTarget,
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


def _sram_target() -> ProgramSramTarget:
    return ProgramSramTarget(
        kind=ProgramIoTargetKind.SRAM,
        runtime_core_id=3,
        program_symbol_ref="sram-symbol",
        finalized_symbol_index=7,
        expected_symbol_name="sram_name",
        buffer_abi_id="buffer-abi",
        storage_id="storage",
        value_id="value",
        tensor_slice=TensorSlice("value", (0, 0), (2, 2)),
        dtype=DType.FP16,
        layout="row_major",
    )


def _hbm_target() -> ProgramHbmTarget:
    return ProgramHbmTarget(
        kind=ProgramIoTargetKind.HBM,
        program_symbol_ref="hbm-symbol",
        finalized_symbol_index=11,
        expected_symbol_name="hbm_name",
        state_abi_id="state-abi",
        state_ref="state",
        hbm_binding_ref="hbm-binding",
    )


def _initialization(target, purpose) -> ProgramSramInitialization:
    blob = ProgramBlob.create(bytes(8))
    return ProgramSramInitialization.create(
        target=target,
        offset_bytes=0,
        length_bytes=8,
        blob_ref=blob.id,
        purpose=purpose,
    )


class ProgramIoTaggedTargetSchemaTest(unittest.TestCase):
    def test_sram_and_hbm_entries_are_strict_and_roundtrip(self) -> None:
        sram = _initialization(
            _sram_target(), ProgramIoPurpose.ACTIVATION
        )
        hbm = _initialization(_hbm_target(), ProgramIoPurpose.STATE)
        probe = ProgramOutputProbe.create(
            target=_hbm_target(),
            offset_bytes=0,
            length_bytes=8,
            blob_ref=ProgramBlob.create(bytes(8)).id,
            comparison=ProgramOutputComparison.EXACT_BYTES,
            capture=ProgramOutputCapture.AFTER_PROGRAM,
        )
        for entry_type, entry in (
            (ProgramSramInitialization, sram),
            (ProgramSramInitialization, hbm),
            (ProgramOutputProbe, probe),
        ):
            with self.subTest(entry=entry):
                entry.validate()
                self.assertEqual(
                    loads_dataclass(entry_type, canonical_json(entry)), entry
                )
        self.assertEqual(
            PROGRAM_SRAM_INITIALIZATION_SCHEMA_VERSION,
            "wafer_frontend.program_sram_initialization/v1alpha2",
        )
        self.assertEqual(
            PROGRAM_OUTPUT_PROBE_SCHEMA_VERSION,
            "wafer_frontend.program_output_probe/v1alpha2",
        )
        self.assertEqual(
            PROGRAM_IO_CONTRACT_SCHEMA_VERSION,
            "wafer_frontend.program_io_contract/v1alpha2",
        )

    def test_old_flat_entry_and_physical_hbm_address_fail_closed(self) -> None:
        entry = _initialization(
            _sram_target(), ProgramIoPurpose.ACTIVATION
        )
        old = json.loads(canonical_json(entry))
        target = old.pop("target")
        old.update(target)
        with self.assertRaises(SchemaError):
            loads_dataclass(
                ProgramSramInitialization,
                json.dumps(old, sort_keys=True, separators=(",", ":")),
            )

        hbm = _initialization(_hbm_target(), ProgramIoPurpose.STATE)
        physical = json.loads(canonical_json(hbm))
        physical["target"]["address"] = 0x1000
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(
                ProgramSramInitialization,
                json.dumps(physical, sort_keys=True, separators=(",", ":")),
            )

    def test_tag_and_state_purpose_must_match_exactly(self) -> None:
        with self.assertRaisesRegex(SchemaError, "must be HBM"):
            replace(
                _hbm_target(), kind=ProgramIoTargetKind.SRAM
            ).validate()
        with self.assertRaisesRegex(SchemaError, "STATE purpose"):
            _initialization(
                _hbm_target(), ProgramIoPurpose.ACTIVATION
            ).validate()
        with self.assertRaisesRegex(SchemaError, "STATE purpose"):
            _initialization(
                _sram_target(), ProgramIoPurpose.STATE
            ).validate()


if __name__ == "__main__":
    unittest.main()
