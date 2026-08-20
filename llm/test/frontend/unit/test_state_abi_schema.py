from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    AddressRelocation,
    CommandFragment,
    CoreFragmentStream,
    FragmentKind,
    LinkedProgramManifest,
    ProgramSymbol,
    ProgramSymbolKind,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    SemanticOperandId,
    StateABI,
    StateOperandBinding,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.global_action import LogicalCoreRef
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateLifetime,
    StateKind,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
    to_primitive,
)


def _state_abi() -> StateABI:
    return StateABI.create(
        state_ref="state.parameter.w_o.rank0",
        hbm_binding_ref="hbm.binding.w_o.rank0",
        kind=StateKind.PARAMETER,
        lifetime=PersistentStateLifetime.PERSISTENT,
        access=PersistentStateAccess.READ_ONLY,
        shape=(8, 8),
        dtype=DType.FP16,
        layout="row_major",
        die_id=0,
        address=0x1000,
        size_bytes=128,
        alignment_bytes=64,
    )


def _load_fragment() -> CommandFragment:
    abi = _state_abi()
    hbm_symbol = ProgramSymbol(
        "symbol.hbm.w_o.rank0",
        ProgramSymbolKind.ABSOLUTE_ADDRESS,
        abi.hbm_binding_ref,
    )
    sram_symbol = ProgramSymbol(
        "symbol.sram.w_o.rank0",
        ProgramSymbolKind.ABSOLUTE_ADDRESS,
        "buffer.binding.w_o.rank0",
    )
    record = RelocatableRecord(
        "global.action.dma_in.w_o.rank0",
        RecordOpcode.LSU_LOAD,
        (
            RecordOperand.address(
                "hbm_address",
                SemanticOperandId.HBM_ADDRESS,
                hbm_symbol.id,
            ),
            RecordOperand.literal("size_bytes", abi.size_bytes),
            RecordOperand.address(
                "destination_address",
                SemanticOperandId.DESTINATION_ADDRESS,
                sram_symbol.id,
            ),
        ),
    )
    stream = CoreFragmentStream(
        LogicalCoreRef(0, 0),
        (record,),
        (),
        (
            AddressRelocation(
                0,
                SemanticOperandId.DESTINATION_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                sram_symbol.id,
                0,
            ),
            AddressRelocation(
                0,
                SemanticOperandId.HBM_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                hbm_symbol.id,
                0,
            ),
        ),
    )
    return CommandFragment.create(
        producer_pass="state_dma_lowering",
        source_global_dag_id="global.action.dag.state.tp1",
        kind=FragmentKind.STATE_IO,
        claimed_action_ids=(record.source_global_action_id,),
        core_streams=(stream,),
        runtime_symbols=(),
        program_symbols=tuple(sorted((hbm_symbol, sram_symbol), key=lambda item: item.id)),
        buffer_abi=(),
        state_abi=(abi,),
    )


class StateABISchemaTest(unittest.TestCase):
    def test_trainable_parameter_is_persistent_read_write_only(self) -> None:
        trainable = StateABI.create(
            state_ref="state.trainable.w_o.rank0",
            hbm_binding_ref="hbm.binding.trainable.w_o.rank0",
            kind=StateKind.TRAINABLE_PARAMETER,
            lifetime=PersistentStateLifetime.PERSISTENT,
            access=PersistentStateAccess.READ_WRITE,
            shape=(8, 8),
            dtype=DType.FP16,
            layout="row_major",
            die_id=0,
            address=0x2000,
            size_bytes=128,
            alignment_bytes=64,
        )
        trainable.validate()
        with self.assertRaisesRegex(SchemaError, "trainable parameter"):
            replace(trainable, access=PersistentStateAccess.READ_ONLY).validate()
        with self.assertRaisesRegex(SchemaError, "parameter must"):
            replace(_state_abi(), access=PersistentStateAccess.READ_WRITE).validate()

    def test_load_fragment_round_trip_and_stable_id(self) -> None:
        fragment = _load_fragment()
        fragment.validate()
        decoded = loads_dataclass(CommandFragment, canonical_json(fragment))
        self.assertEqual(decoded, fragment)
        self.assertEqual(decoded.id, fragment.id)
        self.assertEqual(decoded.state_abi, (_state_abi(),))

    def test_state_and_hbm_witnesses_fail_closed(self) -> None:
        fragment = _load_fragment()
        abi = fragment.state_abi[0]
        with self.subTest("stale_state_abi_id"):
            with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
                replace(
                    fragment,
                    state_abi=(replace(abi, address=abi.address + 64),),
                ).validate()

        with self.subTest("nonzero_hbm_addend"):
            stream = fragment.core_streams[0]
            relocations = tuple(
                replace(relocation, addend=1)
                if relocation.operand_id is SemanticOperandId.HBM_ADDRESS
                else relocation
                for relocation in stream.address_relocations
            )
            with self.assertRaisesRegex(SchemaError, "contained"):
                replace(
                    fragment,
                    core_streams=(replace(stream, address_relocations=relocations),),
                ).validate()

        with self.subTest("missing_required_wire_field"):
            payload = to_primitive(fragment)
            del payload["state_abi"]
            with self.assertRaises(SchemaError):
                loads_dataclass(CommandFragment, canonical_json(payload))

    def test_state_operand_binding_and_linked_wire_are_strict(self) -> None:
        binding = StateOperandBinding(
            "fragment.state.load",
            LogicalCoreRef(0, 0),
            0,
            SemanticOperandId.HBM_ADDRESS,
            _state_abi().id,
        )
        binding.validate("state_operand_binding")
        with self.assertRaisesRegex(SchemaError, "only supports HBM_ADDRESS"):
            replace(
                binding,
                operand_id=SemanticOperandId.DESTINATION_ADDRESS,
            ).validate("state_operand_binding")

        from test_linked_program_manifest_schema import valid_linked_manifest

        manifest = valid_linked_manifest()[4]
        self.assertEqual(manifest.state_operand_bindings, ())
        decoded = loads_dataclass(
            LinkedProgramManifest,
            canonical_json(manifest),
        )
        self.assertEqual(decoded, manifest)
        payload = to_primitive(manifest)
        del payload["state_operand_bindings"]
        with self.assertRaises(SchemaError):
            loads_dataclass(
                LinkedProgramManifest,
                canonical_json(payload),
            )


if __name__ == "__main__":
    unittest.main()
