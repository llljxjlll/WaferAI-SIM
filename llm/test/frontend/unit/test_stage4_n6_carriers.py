from __future__ import annotations

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    link_stage4 as public_link_stage4,
    lower_stage4 as public_lower_stage4,
)
from llm.frontend.wafer_frontend.passes.global_action_dag import (
    build_stage4_global_action,
)
from llm.frontend.wafer_frontend.passes.link_program import link_stage4
from llm.frontend.wafer_frontend.passes.lower_program import lower_stage4
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
)
from llm.frontend.wafer_frontend.schema import (
    Stage4LinkedProgram as PublicStage4LinkedProgram,
    Stage4LoweredProgram as PublicStage4LoweredProgram,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    FragmentKind,
    RecordOpcode,
    RuntimeOperandField,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.n6 import (
    STAGE4_LINKED_PROGRAM_SCHEMA_VERSION,
    STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION,
    Stage4LinkedProgram,
    Stage4LoweredProgram,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass

from test_stage4_global_action_carrier import _scheduled, _scheduled_pdr


def _global(*, fused: bool):
    return build_stage4_global_action(_scheduled(fused=fused))


def _global_pdr():
    return build_stage4_global_action(_scheduled_pdr())


class Stage4N6CarrierTest(unittest.TestCase):
    def test_public_api_fused_tp1_round_trip_and_counts(self) -> None:
        self.assertIs(public_lower_stage4, lower_stage4)
        self.assertIs(public_link_stage4, link_stage4)
        self.assertIs(PublicStage4LoweredProgram, Stage4LoweredProgram)
        self.assertIs(PublicStage4LinkedProgram, Stage4LinkedProgram)
        source = _global(fused=True)
        lowered = lower_stage4(source)
        linked = link_stage4(lowered)
        lowered.validate_against(source)
        linked.validate_against(lowered)
        self.assertEqual(
            STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION,
            "wafer_frontend.stage4_lowered_program/v1alpha4",
        )
        self.assertEqual(
            STAGE4_LINKED_PROGRAM_SCHEMA_VERSION,
            "wafer_frontend.stage4_linked_program/v1alpha4",
        )
        self.assertEqual(len(lowered.fragments), 92)
        self.assertFalse(
            any(
                type(fragment) is CommandFragment
                and fragment.kind is FragmentKind.STATE_TRANSFER
                for fragment in lowered.fragments
            )
        )
        manifest = linked.manifest
        self.assertEqual(
            (
                len(manifest.fragments),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
                len(manifest.fragment_interfaces),
                len(manifest.core_streams),
            ),
            (92, 560, 42, 1, 200, 92, 1),
        )
        self.assertEqual(lowered.source_global_action_carrier_id, source.id)
        self.assertEqual(linked.source_lowered_carrier_id, lowered.id)
        self.assertEqual(linked.lowering_context, lowered.lowering_context)
        self.assertEqual(
            loads_dataclass(
                Stage4LoweredProgram,
                canonical_json(lowered),
                path="lowered",
            ),
            lowered,
        )
        self.assertEqual(
            loads_dataclass(
                Stage4LinkedProgram,
                canonical_json(linked),
                path="linked",
            ),
            linked,
        )
        self.assertEqual(lower_stage4(source), lowered)
        self.assertEqual(link_stage4(lowered), linked)

    def test_pds_tp1_has_96_fragments_eight_endpoints_and_exact_link(self) -> None:
        source = _global(fused=False)
        lowered = lower_stage4(source)
        linked = link_stage4(lowered)
        endpoints = tuple(
            fragment
            for fragment in lowered.fragments
            if type(fragment) is CommandFragment
            and fragment.kind is FragmentKind.STATE_TRANSFER
        )
        self.assertEqual((len(lowered.fragments), len(endpoints)), (96, 8))
        self.assertEqual(
            sum(len(fragment.claimed_action_ids) for fragment in endpoints),
            12,
        )
        manifest = linked.manifest
        self.assertEqual(
            (
                len(manifest.fragments),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
                len(manifest.fragment_interfaces),
                len(manifest.core_streams),
            ),
            (96, 564, 38, 18, 219, 96, 2),
        )
        self.assertEqual(
            tuple(fragment.id for fragment in linked.leaf_fragments),
            tuple(sorted(fragment.id for fragment in linked.leaf_fragments)),
        )

    def test_pdr_tp2_to_tp1_alpha2_wrappers_and_segment_relocations(self) -> None:
        source = _global_pdr()
        lowered = lower_stage4(source)
        linked = link_stage4(lowered)
        lowered.validate_against(source)
        linked.validate_against(lowered)
        self.assertEqual(
            loads_dataclass(
                Stage4LoweredProgram,
                canonical_json(lowered),
                path="lowered",
            ),
            lowered,
        )
        self.assertEqual(
            loads_dataclass(
                Stage4LinkedProgram,
                canonical_json(linked),
                path="linked",
            ),
            linked,
        )

        endpoints = tuple(
            fragment
            for fragment in lowered.fragments
            if type(fragment) is CommandFragment
            and fragment.kind is FragmentKind.STATE_TRANSFER
        )
        self.assertEqual(
            (
                len(source.projection.state_transfers),
                sum(
                    len(contract.segments)
                    for contract in source.projection.state_transfers
                ),
                len(lowered.fragments),
                len(endpoints),
            ),
            (8, 64, 152, 16),
        )
        self.assertTrue(
            all(
                len(fragment.buffer_abi) == 1
                and fragment.state_abi == ()
                for fragment in endpoints
            )
        )
        transfer_records = tuple(
            record
            for fragment in endpoints
            for stream in fragment.core_streams
            for record in stream.records
            if record.opcode
            in (
                RecordOpcode.DTE_SEND,
                RecordOpcode.DTE_RECV,
                RecordOpcode.DTE_WAIT,
            )
        )
        transport_records = tuple(
            record
            for record in transfer_records
            if record.opcode in (RecordOpcode.DTE_SEND, RecordOpcode.DTE_RECV)
        )
        self.assertEqual(
            (
                sum(
                    record.opcode is RecordOpcode.DTE_SEND
                    for record in transfer_records
                ),
                sum(
                    record.opcode is RecordOpcode.DTE_RECV
                    for record in transfer_records
                ),
                sum(
                    record.opcode is RecordOpcode.DTE_WAIT
                    for record in transfer_records
                ),
                sum(
                    len(stream.address_relocations)
                    for fragment in endpoints
                    for stream in fragment.core_streams
                ),
                sum(
                    len(stream.runtime_relocations)
                    for fragment in endpoints
                    for stream in fragment.core_streams
                ),
            ),
            # The 55 destination-safe wave edges each lower to one SET and
            # one WAIT with three runtime relocations per event record.
            (64, 64, 64, 152, 384 + 55 * 2 * 3),
        )
        self.assertEqual(
            sum(
                next(
                    operand.literal_value
                    for operand in record.operands
                    if operand.name == "length_bytes"
                )
                for record in transport_records
            ),
            2048,
        )
        self.assertEqual(
            sum(
                relocation.operand_id
                in (
                    SemanticOperandId.SOURCE_ADDRESS,
                    SemanticOperandId.DESTINATION_ADDRESS,
                )
                for fragment in endpoints
                for stream in fragment.core_streams
                for relocation in stream.address_relocations
            ),
            128,
        )
        self.assertEqual(
            sum(
                relocation.field is RuntimeOperandField.DTE_TOKEN
                for fragment in endpoints
                for stream in fragment.core_streams
                for relocation in stream.runtime_relocations
            ),
            128,
        )
        manifest = linked.manifest
        self.assertEqual(
            (
                len(manifest.fragments),
                len(manifest.address_operand_bindings),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
            ),
            # Every wave edge contributes three canonical event runtime
            # symbol definitions in the linked manifest.
            (152, 1210, 339 + 55 * 3, 432),
        )
        seeds, expected = build_deterministic_timing_state_overrides(linked)
        self.assertIs(type(seeds), dict)
        self.assertIs(type(expected), dict)
        self.assertEqual(
            build_deterministic_timing_state_overrides(linked),
            (seeds, expected),
        )

    def test_strict_serde_source_context_fragment_and_manifest_tamper_reject(self) -> None:
        source = _global(fused=False)
        lowered = lower_stage4(source)
        linked = link_stage4(lowered)
        for artifact_type, artifact, required_field in (
            (Stage4LoweredProgram, lowered, "fragments"),
            (Stage4LinkedProgram, linked, "manifest"),
        ):
            raw = json.loads(canonical_json(artifact))
            raw["unexpected"] = True
            with self.subTest(artifact=artifact_type.__name__, mode="unknown"):
                with self.assertRaises(SchemaError):
                    loads_dataclass(
                        artifact_type,
                        canonical_json(raw),
                        path="artifact",
                    )
            del raw["unexpected"]
            del raw[required_field]
            with self.subTest(artifact=artifact_type.__name__, mode="missing"):
                with self.assertRaises(SchemaError):
                    loads_dataclass(
                        artifact_type,
                        canonical_json(raw),
                        path="artifact",
                    )

        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                lowered,
                schema_version="wafer_frontend.stage4_lowered_program/v1alpha2",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                linked,
                schema_version="wafer_frontend.stage4_linked_program/v1alpha2",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "global-action carrier"):
            lowered.validate_against(_global(fused=True))
        with self.assertRaisesRegex(SchemaError, "lowered carrier"):
            linked.validate_against(lower_stage4(_global(fused=True)))
        with self.assertRaisesRegex(SchemaError, "cover every executable"):
            replace(lowered, fragments=lowered.fragments[:-1]).validate()
        with self.assertRaisesRegex(SchemaError, "manifest_linker"):
            replace(
                linked,
                manifest=replace(linked.manifest, producer_pass="wrong"),
            ).validate()
        with self.assertRaises(SchemaError):
            replace(
                lowered,
                lowering_context=replace(
                    lowered.lowering_context,
                    global_dag=replace(
                        lowered.lowering_context.global_dag,
                        source_schedule_set_id="wrong",
                    ),
                ),
            ).validate()
        with self.assertRaises(SchemaError):
            replace(
                linked,
                leaf_fragments=linked.leaf_fragments[:-1],
            ).validate()


if __name__ == "__main__":
    unittest.main()
