from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
    link_train,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    _resolved_abis,
    _resolved_state_abis,
    _state_abi_groups,
    _train_label_seed_overrides,
    _validate_train_terminal_abis,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramHbmTarget,
    ProgramIoMode,
    ProgramSramTarget,
)
from llm.frontend.wafer_frontend.schema.train_n6 import TrainLinkedProgram
from llm.test.frontend.integration.train_forward_cases import (
    build_train_forward_case,
)


_ARTIFACT_SHA256 = "ab" * 32


class TrainForwardProgramIoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = link_train(build_train_forward_case().lowered)
        cls.resolved = _resolved_abis(cls.source)
        cls.resolved_state = _resolved_state_abis(cls.source)
        cls.label_seeds = _train_label_seed_overrides(
            cls.source,
            cls.resolved,
        )
        cls.state_seeds, cls.state_expected = (
            build_deterministic_timing_state_overrides(cls.source)
        )
        cls.contract = build_timing_program_io(
            cls.source,
            _ARTIFACT_SHA256,
            state_seed_overrides=cls.state_seeds,
            state_expected_overrides=cls.state_expected,
        )

    def test_dp2_tp2_exact_labels_states_and_loss_timing_probes(self) -> None:
        contract = self.contract
        self.assertIs(contract.mode, ProgramIoMode.TIMING)
        self.assertEqual(len(self.state_seeds), 30)
        self.assertEqual(self.state_expected, {})
        self.assertEqual(
            Counter(type(item.target) for item in contract.initializations),
            Counter({ProgramSramTarget: 268, ProgramHbmTarget: 60}),
        )
        self.assertEqual(
            Counter(type(item.target) for item in contract.output_probes),
            Counter({ProgramSramTarget: 4}),
        )
        self.assertEqual(len(contract.initializations), 328)
        self.assertEqual(len(contract.output_probes), 4)

        blobs = {item.id: item.payload() for item in contract.blobs}
        initialized_by_abi = {
            item.target.buffer_abi_id: blobs[item.blob_ref]
            for item in contract.initializations
            if type(item.target) is ProgramSramTarget
        }
        self.assertEqual(
            tuple(
                sorted(
                    tuple(
                        int.from_bytes(payload[index : index + 4], "little")
                        for index in range(0, len(payload), 4)
                    )
                    for payload in self.label_seeds.values()
                )
            ),
            ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)),
        )
        self.assertEqual(
            {
                abi_id: initialized_by_abi[abi_id]
                for abi_id in self.label_seeds
            },
            self.label_seeds,
        )
        self.assertTrue(
            all(
                type(probe.target) is ProgramSramTarget
                and probe.target.dtype is DType.FP32
                and probe.length_bytes == 16
                and not any(blobs[probe.blob_ref])
                for probe in contract.output_probes
            )
        )
        physical_state_targets = tuple(
            item.target
            for item in contract.initializations
            if type(item.target) is ProgramHbmTarget
        )
        self.assertEqual(len(physical_state_targets), 60)
        self.assertEqual(
            Counter(item.state_ref for item in physical_state_targets),
            Counter({state_ref: 2 for state_ref in self.state_seeds}),
        )
        self.assertEqual(
            len({item.hbm_binding_ref for item in physical_state_targets}),
            60,
        )
        contract.validate_against(self.source.manifest)

    def test_label_overrides_are_idempotent_but_conflicts_fail_closed(self) -> None:
        with patch.object(TrainLinkedProgram, "validate", return_value=None):
            explicit = build_timing_program_io(
                self.source,
                _ARTIFACT_SHA256,
                sram_seed_overrides=self.label_seeds,
                state_seed_overrides=self.state_seeds,
            )
        self.assertEqual(explicit, self.contract)

        abi_id, payload = next(iter(self.label_seeds.items()))
        conflict = bytes([payload[0] ^ 0xFF]) + payload[1:]
        with patch.object(TrainLinkedProgram, "validate", return_value=None):
            with self.assertRaisesRegex(SchemaError, "label override conflicts"):
                build_timing_program_io(
                    self.source,
                    _ARTIFACT_SHA256,
                    sram_seed_overrides={abi_id: conflict},
                    state_seed_overrides=self.state_seeds,
                )

    def test_missing_logical_state_and_numeric_loss_expected_fail_closed(self) -> None:
        missing = dict(self.state_seeds)
        missing.pop(next(iter(missing)))
        with patch.object(TrainLinkedProgram, "validate", return_value=None):
            with self.assertRaisesRegex(SchemaError, "exactly cover"):
                build_timing_program_io(
                    self.source,
                    _ARTIFACT_SHA256,
                    state_seed_overrides=missing,
                )

        terminal = next(
            item
            for item in self.resolved
            if item.abi.dtype is DType.FP32
            and item.abi.ownership is BufferOwnership.OWNED
        )
        with patch.object(TrainLinkedProgram, "validate", return_value=None):
            with self.assertRaisesRegex(SchemaError, "numeric loss"):
                build_timing_program_io(
                    self.source,
                    _ARTIFACT_SHA256,
                    sram_expected_overrides={
                        terminal.abi.id: bytes(terminal.abi.size_bytes)
                    },
                    state_seed_overrides=self.state_seeds,
                )

    def test_duplicate_state_semantics_and_replica_loss_coverage_fail_closed(
        self,
    ) -> None:
        group = next(
            items
            for items in _state_abi_groups(self.resolved_state).values()
            if len(items) == 2
        )
        for field_name, changed_abi in (
            ("shape", replace(group[1].abi, shape=(group[1].abi.shape[0] + 1,))),
            ("size", replace(group[1].abi, size_bytes=group[1].abi.size_bytes + 2)),
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(SchemaError, "conflicting physical"):
                    _state_abi_groups((group[0], replace(group[1], abi=changed_abi)))

        terminal_values = {
            value.id
            for replica in self.source.source.replicas
            for value in replica.lowering_context.ir1.values
            if not value.consumers
        }
        resolved_terminal = tuple(
            item
            for item in self.resolved
            if item.abi.value_id in terminal_values
        )
        with self.assertRaisesRegex(SchemaError, "every replica TP shard"):
            _validate_train_terminal_abis(
                self.source,
                terminal_values,
                resolved_terminal[:-1],
            )

        target = resolved_terminal[-1]
        crossed = replace(
            target,
            uses=tuple(
                replace(use, replica_index=0)
                for use in target.uses
            ),
        )
        with self.assertRaises(SchemaError):
            _validate_train_terminal_abis(
                self.source,
                terminal_values,
                (*resolved_terminal[:-1], crossed),
            )


if __name__ == "__main__":
    unittest.main()
