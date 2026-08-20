from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.context import LoweringContext
from llm.frontend.wafer_frontend.passes.program_io import (
    _resolved_abis,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.global_action import GlobalActionDAG
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferAccess,
    BufferOwnership,
    BufferUseRole,
)
from llm.frontend.wafer_frontend.schema.n6 import LinkedProgramProfile
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramIoMode,
    ProgramIoPurpose,
    ProgramSramTarget,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from test_n6_schema import _single_profile_fixture


_ARTIFACT_SHA256 = "cd" * 32


def _source() -> LinkedProgramProfile:
    return _single_profile_fixture()[2].entries[0]


def _forged_actions(
    source: LinkedProgramProfile,
    actions: tuple,
) -> LinkedProgramProfile:
    dag = replace(source.lowering_context.global_dag, actions=actions)
    context = replace(source.lowering_context, global_dag=dag)
    return replace(source, lowering_context=context)


class ProgramIoPassTest(unittest.TestCase):
    def test_lightweight_profile_builds_exact_timing_contract(self) -> None:
        source = _source()
        contract = build_timing_program_io(source, _ARTIFACT_SHA256)
        self.assertIs(contract.mode, ProgramIoMode.TIMING)
        self.assertEqual(contract.program_artifact_sha256, _ARTIFACT_SHA256)
        self.assertEqual(len(contract.initializations), 6)
        self.assertEqual(
            Counter(entry.purpose for entry in contract.initializations),
            Counter(
                {
                    ProgramIoPurpose.ACTIVATION: 2,
                    ProgramIoPurpose.WEIGHT: 2,
                    ProgramIoPurpose.TIMING_PARTIAL: 2,
                }
            ),
        )
        self.assertEqual(len(contract.output_probes), 2)
        terminal_values = {
            value.id
            for value in source.lowering_context.ir1.values
            if not value.consumers
        }
        self.assertEqual(
            {
                probe.target.value_id
                for probe in contract.output_probes
                if type(probe.target) is ProgramSramTarget
            },
            terminal_values,
        )
        self.assertTrue(
            all(not any(blob.payload()) for blob in contract.blobs)
        )
        contract.validate_against(source.manifest)

    def test_is_deterministic_and_preserves_input(self) -> None:
        source = _source()
        before = canonical_digest(source)
        first = build_timing_program_io(source, _ARTIFACT_SHA256)
        second = build_timing_program_io(source, _ARTIFACT_SHA256)
        self.assertEqual(first, second)
        self.assertEqual(canonical_digest(source), before)
        order = tuple(
            (
                entry.target.runtime_core_id,
                entry.target.finalized_symbol_index,
                entry.offset_bytes,
                entry.length_bytes,
                entry.id,
            )
            for entry in first.initializations
        )
        self.assertEqual(order, tuple(sorted(order)))

    def test_rejects_invalid_sha_and_non_profile_source(self) -> None:
        with self.assertRaises(SchemaError):
            build_timing_program_io(_source(), "not-a-sha")
        with self.assertRaises(SchemaError):
            build_timing_program_io(object(), _ARTIFACT_SHA256)  # type: ignore[arg-type]

    def test_fails_closed_on_missing_use_and_access_role_conflict(self) -> None:
        source = _source()
        action = source.lowering_context.global_dag.actions[0]
        missing_action = replace(action, buffer_uses=action.buffer_uses[1:])
        forged = _forged_actions(
            source,
            (missing_action, *source.lowering_context.global_dag.actions[1:]),
        )
        with patch.object(LinkedProgramProfile, "validate", return_value=None):
            with self.assertRaises(SchemaError):
                build_timing_program_io(forged, _ARTIFACT_SHA256)

        first_use = action.buffer_uses[0]
        conflict = replace(
            first_use,
            access=BufferAccess.WRITE,
            role=BufferUseRole.COMP_INPUT,
        )
        conflict_action = replace(
            action,
            buffer_uses=(conflict, *action.buffer_uses[1:]),
        )
        forged = _forged_actions(
            source,
            (conflict_action, *source.lowering_context.global_dag.actions[1:]),
        )
        with patch.object(LinkedProgramProfile, "validate", return_value=None):
            with self.assertRaises(SchemaError):
                build_timing_program_io(forged, _ARTIFACT_SHA256)

    def test_fails_closed_on_owned_first_read(self) -> None:
        source = _source()
        actions = list(source.lowering_context.global_dag.actions)
        target_index = next(
            index
            for index, action in enumerate(actions)
            if any(use.role is BufferUseRole.COMP_OUTPUT for use in action.buffer_uses)
        )
        action = actions[target_index]
        uses = list(action.buffer_uses)
        output_index = next(
            index
            for index, use in enumerate(uses)
            if use.role is BufferUseRole.COMP_OUTPUT
        )
        uses[output_index] = replace(
            uses[output_index],
            access=BufferAccess.READ,
            role=BufferUseRole.COMP_INPUT,
            operand_index=0,
        )
        actions[target_index] = replace(action, buffer_uses=tuple(uses))
        forged = _forged_actions(source, tuple(actions))
        with patch.object(LinkedProgramProfile, "validate", return_value=None):
            with self.assertRaisesRegex(SchemaError, "first use"):
                build_timing_program_io(forged, _ARTIFACT_SHA256)

    def test_fails_closed_on_missing_or_nonowned_terminal_mapping(self) -> None:
        source = _source()
        no_terminal_ir1 = replace(
            source.lowering_context.ir1,
            values=tuple(
                replace(value, consumers=("synthetic_consumer",))
                for value in source.lowering_context.ir1.values
            ),
        )
        no_terminal = replace(
            source,
            lowering_context=replace(
                source.lowering_context,
                ir1=no_terminal_ir1,
            ),
        )
        with patch.object(LinkedProgramProfile, "validate", return_value=None):
            with self.assertRaises(SchemaError):
                build_timing_program_io(no_terminal, _ARTIFACT_SHA256)

        borrowed_value_ids = {
            binding.value_id
            for schedule in source.lowering_context.schedule_set.schedules
            for binding in schedule.buffer_bindings
            if binding.ownership is BufferOwnership.BORROWED
        }
        borrowed_terminal_ir1 = replace(
            source.lowering_context.ir1,
            values=tuple(
                replace(value, consumers=())
                if value.id in borrowed_value_ids
                else value
                for value in source.lowering_context.ir1.values
            ),
        )
        borrowed_terminal = replace(
            source,
            lowering_context=replace(
                source.lowering_context,
                ir1=borrowed_terminal_ir1,
            ),
        )
        with patch.object(LinkedProgramProfile, "validate", return_value=None):
            with self.assertRaises(SchemaError):
                build_timing_program_io(borrowed_terminal, _ARTIFACT_SHA256)

    def test_sram_overrides_seed_borrowed_and_dedupe_terminal_probe(self) -> None:
        source = _source()
        resolved = _resolved_abis(source)
        borrowed = next(
            item
            for item in resolved
            if item.abi.ownership is BufferOwnership.BORROWED
        )
        owned = next(
            item
            for item in resolved
            if item.abi.ownership is BufferOwnership.OWNED
        )
        seed = bytes([0x5A]) * borrowed.abi.size_bytes
        expected = bytes(owned.abi.size_bytes)
        contract = build_timing_program_io(
            source,
            _ARTIFACT_SHA256,
            sram_seed_overrides={borrowed.abi.id: seed},
            sram_expected_overrides={owned.abi.id: expected},
        )
        blobs = {blob.id: blob.payload() for blob in contract.blobs}
        initialization = next(
            entry
            for entry in contract.initializations
            if type(entry.target) is ProgramSramTarget
            and entry.target.buffer_abi_id == borrowed.abi.id
        )
        self.assertEqual(blobs[initialization.blob_ref], seed)
        probes = tuple(
            entry
            for entry in contract.output_probes
            if type(entry.target) is ProgramSramTarget
            and entry.target.buffer_abi_id == owned.abi.id
        )
        self.assertEqual(len(probes), 1)
        self.assertEqual(blobs[probes[0].blob_ref], expected)
        contract.validate_against(source.manifest)

    def test_sram_overrides_fail_closed(self) -> None:
        source = _source()
        resolved = _resolved_abis(source)
        borrowed = next(
            item
            for item in resolved
            if item.abi.ownership is BufferOwnership.BORROWED
        )
        owned = next(
            item
            for item in resolved
            if item.abi.ownership is BufferOwnership.OWNED
        )
        seed = bytes(borrowed.abi.size_bytes)
        expected = bytes(owned.abi.size_bytes)
        cases = (
            (
                "unknown seed",
                {"unknown-buffer-abi": seed},
                None,
                "manifested BufferABI",
            ),
            ("owned seed", {owned.abi.id: expected}, None, "borrowed"),
            ("short seed", {borrowed.abi.id: seed[:-1]}, None, "whole"),
            (
                "mutable seed",
                {borrowed.abi.id: bytearray(seed)},
                None,
                "immutable bytes",
            ),
            (
                "unknown expected",
                None,
                {"unknown-buffer-abi": expected},
                "manifested BufferABI",
            ),
            ("borrowed expected", None, {borrowed.abi.id: seed}, "owned"),
            ("short expected", None, {owned.abi.id: expected[:-1]}, "whole"),
            (
                "mutable expected",
                None,
                {owned.abi.id: bytearray(expected)},
                "immutable bytes",
            ),
            (
                "terminal conflict",
                None,
                {owned.abi.id: bytes([1]) * owned.abi.size_bytes},
                "conflicting expected payloads",
            ),
        )
        for name, seeds, expected_values, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(SchemaError, message):
                    build_timing_program_io(
                        source,
                        _ARTIFACT_SHA256,
                        sram_seed_overrides=seeds,  # type: ignore[arg-type]
                        sram_expected_overrides=expected_values,  # type: ignore[arg-type]
                    )


if __name__ == "__main__":
    unittest.main()
