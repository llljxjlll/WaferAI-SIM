from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.program_io import (
    _deterministic_timing_state_overrides,
    _resolved_state_abis,
    _state_access_is_permitted,
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.ir2 import StateUseAccess
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess,
)
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramHbmTarget,
    ProgramIoPurpose,
    ProgramSramTarget,
)

from test_n6_pipeline import _compile_through_n6


_ARTIFACT_SHA256 = "ef" * 32


def _payloads(source):
    resolved = _resolved_state_abis(source)
    seeds = {
        item.abi.state_ref: bytes([index + 1]) * item.abi.size_bytes
        for index, item in enumerate(resolved)
    }
    expected = {
        item.abi.state_ref: bytes([0x80 + index]) * item.abi.size_bytes
        for index, item in enumerate(resolved)
        if item.abi.access is PersistentStateAccess.READ_WRITE
    }
    return resolved, seeds, expected


class ProgramIoStatePassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = _compile_through_n6().linked.entries[0]
        cls.resolved, cls.seeds, cls.expected = _payloads(cls.source)

    def test_declared_access_is_a_permission_not_a_required_trace(self) -> None:
        read = {StateUseAccess.READ}
        write = {StateUseAccess.WRITE}
        both = {StateUseAccess.READ, StateUseAccess.WRITE}
        self.assertTrue(
            _state_access_is_permitted(
                PersistentStateAccess.READ_ONLY, read
            )
        )
        self.assertFalse(
            _state_access_is_permitted(
                PersistentStateAccess.READ_ONLY, write
            )
        )
        for actual in (read, write, both):
            with self.subTest(actual=actual):
                self.assertTrue(
                    _state_access_is_permitted(
                        PersistentStateAccess.READ_WRITE, actual
                    )
                )
        self.assertFalse(
            _state_access_is_permitted(
                PersistentStateAccess.READ_WRITE, set()
            )
        )
        self.assertFalse(
            _state_access_is_permitted(
                PersistentStateAccess.RESERVED, read
            )
        )

    def test_tp2_builds_exact_whole_state_entries(self) -> None:
        contract = build_timing_program_io(
            self.source,
            _ARTIFACT_SHA256,
            state_seed_overrides=self.seeds,
            state_expected_overrides=self.expected,
        )
        hbm_initializations = tuple(
            entry
            for entry in contract.initializations
            if type(entry.target) is ProgramHbmTarget
        )
        hbm_probes = tuple(
            entry
            for entry in contract.output_probes
            if type(entry.target) is ProgramHbmTarget
        )
        self.assertEqual(len(self.resolved), 12)
        self.assertEqual(len(hbm_initializations), 12)
        self.assertEqual(len(hbm_probes), 4)
        self.assertEqual(
            Counter(item.abi.access for item in self.resolved),
            Counter(
                {
                    PersistentStateAccess.READ_ONLY: 8,
                    PersistentStateAccess.READ_WRITE: 4,
                }
            ),
        )
        self.assertTrue(
            all(
                entry.purpose is ProgramIoPurpose.STATE
                and entry.offset_bytes == 0
                and entry.length_bytes
                == next(
                    item.abi.size_bytes
                    for item in self.resolved
                    if item.abi.state_ref == entry.target.state_ref
                )
                and not hasattr(entry.target, "address")
                for entry in hbm_initializations
            )
        )
        blobs = {blob.id: blob.payload() for blob in contract.blobs}
        self.assertEqual(
            {
                entry.target.state_ref: blobs[entry.blob_ref]
                for entry in hbm_initializations
            },
            self.seeds,
        )
        self.assertEqual(
            {
                entry.target.state_ref: blobs[entry.blob_ref]
                for entry in hbm_probes
            },
            self.expected,
        )
        self.assertTrue(
            all(
                type(entry.target) in (ProgramSramTarget, ProgramHbmTarget)
                for entry in (*contract.initializations, *contract.output_probes)
            )
        )
        contract.validate_against(self.source.manifest)

    def test_production_overrides_are_canonical_nonzero_and_distinct(self) -> None:
        first = build_deterministic_timing_state_overrides(self.source)
        second = build_deterministic_timing_state_overrides(self.source)
        self.assertEqual(first, second)
        seeds, expected = first
        self.assertEqual(tuple(seeds), tuple(sorted(seeds)))
        self.assertEqual(tuple(expected), tuple(sorted(expected)))
        required = {
            item.abi.state_ref
            for item in self.resolved
            if item.first_access is StateUseAccess.READ
        }
        stored_read_write = {
            item.abi.state_ref
            for item in self.resolved
            if item.first_access is StateUseAccess.READ
            and item.abi.access is PersistentStateAccess.READ_WRITE
            and any(
                access is StateUseAccess.WRITE
                for _action_index, access in item.uses
            )
        }
        self.assertEqual(set(seeds), required)
        self.assertEqual(set(expected), stored_read_write)
        self.assertEqual(len(set(seeds.values())), len(seeds))
        for state_ref, payload in seeds.items():
            with self.subTest(state_ref=state_ref):
                self.assertTrue(payload)
                self.assertTrue(all(byte != 0 for byte in payload))
                self.assertEqual(
                    len(payload),
                    next(
                        item.abi.size_bytes
                        for item in self.resolved
                        if item.abi.state_ref == state_ref
                    ),
                )
        self.assertTrue(
            all(
                payload == seeds[state_ref]
                for state_ref, payload in expected.items()
            )
        )
        self.assertTrue(
            all(
                item.abi.state_ref not in expected
                for item in self.resolved
                if item.abi.access is PersistentStateAccess.READ_ONLY
            )
        )

    def test_write_first_state_gets_no_seed_or_pseudo_expected(self) -> None:
        read_write = next(
            item
            for item in self.resolved
            if item.abi.access is PersistentStateAccess.READ_WRITE
        )
        write_use = next(
            use
            for use in read_write.uses
            if use[1] is StateUseAccess.WRITE
        )
        read_use = next(
            use for use in read_write.uses if use[1] is StateUseAccess.READ
        )
        write_first = replace(read_write, uses=(write_use, read_use))
        seeds, expected = _deterministic_timing_state_overrides((write_first,))
        self.assertEqual(seeds, {})
        self.assertEqual(expected, {})

    def test_missing_seed_and_unused_override_fail_closed(self) -> None:
        with self.assertRaisesRegex(SchemaError, "load-before-store"):
            build_timing_program_io(self.source, _ARTIFACT_SHA256)

        missing = dict(self.seeds)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            build_timing_program_io(
                self.source,
                _ARTIFACT_SHA256,
                state_seed_overrides=missing,
            )

        unused = {**self.seeds, "unused-state": b""}
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            build_timing_program_io(
                self.source,
                _ARTIFACT_SHA256,
                state_seed_overrides=unused,
            )

    def test_read_only_expected_and_bad_payloads_fail_closed(self) -> None:
        read_only = next(
            item
            for item in self.resolved
            if item.abi.access is PersistentStateAccess.READ_ONLY
        )
        with self.assertRaisesRegex(SchemaError, "READ_WRITE"):
            build_timing_program_io(
                self.source,
                _ARTIFACT_SHA256,
                state_seed_overrides=self.seeds,
                state_expected_overrides={
                    read_only.abi.state_ref: bytes(read_only.abi.size_bytes)
                },
            )

        short = dict(self.seeds)
        state_ref = next(iter(short))
        short[state_ref] = short[state_ref][:-1]
        with self.assertRaisesRegex(SchemaError, "whole state"):
            build_timing_program_io(
                self.source,
                _ARTIFACT_SHA256,
                state_seed_overrides=short,
            )

        mutable = dict(self.seeds)
        mutable[state_ref] = bytearray(mutable[state_ref])  # type: ignore[assignment]
        with self.assertRaisesRegex(SchemaError, "immutable bytes"):
            build_timing_program_io(
                self.source,
                _ARTIFACT_SHA256,
                state_seed_overrides=mutable,
            )


if __name__ == "__main__":
    unittest.main()
