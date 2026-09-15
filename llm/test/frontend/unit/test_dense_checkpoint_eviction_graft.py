"""Structural regression for the signed bounded L2 activation eviction carrier."""
from __future__ import annotations

import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_checkpoint_eviction_graft import (
    graft_dense_checkpoint_eviction,
)
from llm.frontend.wafer_frontend.passes.dense_checkpoint_physical_source import (
    derive_dense_checkpoint_activation_tape,
    derive_dense_checkpoint_physical_cut,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.test.frontend.integration.run_bounded_dense_seeded_ce_canary import build_case


class DenseCheckpointEvictionGraftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = build_case().manifest
        cls.cut = derive_dense_checkpoint_physical_cut(
            cls.source,
            replay_action_id="global_action_bc57cb5c877160c6",
            backward_action_id="global_action_85f0379642a64773")

    def test_low_hbm_rejects_full_tape_and_admits_checkpoint(self) -> None:
        with self.assertRaisesRegex(SchemaError, "activation tape exceeds"):
            derive_dense_checkpoint_activation_tape(
                self.cut, checkpoint_enabled=False, hbm_capacity_bytes=1408)
        tape = derive_dense_checkpoint_activation_tape(
            self.cut, checkpoint_enabled=True, hbm_capacity_bytes=1408)
        self.assertEqual((tape.activation_hbm_address, tape.size_bytes), (1344, 32))

    def test_save_evict_restore_replay_signed_abi(self) -> None:
        tape = derive_dense_checkpoint_activation_tape(
            self.cut, checkpoint_enabled=True, hbm_capacity_bytes=1408)
        manifest = graft_dense_checkpoint_eviction(self.source, self.cut, tape)
        manifest.validate("checkpoint_eviction_test")
        activation = [state for fragment in manifest.fragments
                      for state in fragment.state_abi
                      if state.kind is StateKind.ACTIVATION]
        self.assertEqual(len(activation), 1)
        self.assertEqual((activation[0].address, activation[0].size_bytes), (1344, 32))
        records = [
            (ref, next(fragment.core_streams[0].records[ref.fragment_record_index]
                       for fragment in manifest.fragments if fragment.id == ref.fragment_id))
            for ref in manifest.core_streams[0].records]
        opcodes = [record.opcode for _, record in records]
        self.assertEqual(opcodes.count(RecordOpcode.MATMUL),
                         sum(r.opcode is RecordOpcode.MATMUL
                             for f in self.source.fragments for s in f.core_streams
                             for r in s.records) + 1)
        save = next(i for i, (_, record) in enumerate(records)
                    if record.opcode is RecordOpcode.LSU_STORE
                    and record.source_global_action_id.startswith("dense_checkpoint_save_action_"))
        restore = next(i for i, (_, record) in enumerate(records)
                       if record.opcode is RecordOpcode.LSU_LOAD
                       and record.source_global_action_id.startswith("dense_checkpoint_restore_action_"))
        replay = next(i for i, (_, record) in enumerate(records)
                      if record.opcode is RecordOpcode.MATMUL
                      and record.source_global_action_id.startswith("dense_checkpoint_replay_action_"))
        self.assertEqual(opcodes[save + 1], RecordOpcode.SRAM_FREE)
        self.assertEqual(opcodes[restore - 1], RecordOpcode.SRAM_ALLOC_AT)
        self.assertLess(save, restore)
        self.assertEqual(opcodes[replay - 1], RecordOpcode.SRAM_BIND)
        self.assertEqual(opcodes[replay + 1:replay + 3],
                         [RecordOpcode.SRAM_FREE, RecordOpcode.SRAM_FREE])

    def test_source_digest_drift_rejects_eviction(self) -> None:
        tape = derive_dense_checkpoint_activation_tape(
            self.cut, checkpoint_enabled=True, hbm_capacity_bytes=1408)
        with self.assertRaisesRegex(SchemaError, "source manifest changed"):
            graft_dense_checkpoint_eviction(
                self.source, replace(self.cut, manifest_digest="0" * 64), tape)


if __name__ == "__main__":
    unittest.main()
