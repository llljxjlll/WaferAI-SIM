"""Real fixed-L2 linked source cut and fail-closed checkpoint negatives."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_checkpoint_physical_source import (
    _bound_operand, _opcode, derive_dense_checkpoint_physical_cut,
    derive_dense_checkpoint_activation_tape,
    derive_dense_checkpoint_replay_template,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode, SemanticOperandId, StateABI,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess, PersistentStateLifetime, StateKind,
)
from llm.test.frontend.integration.run_bounded_dense_seeded_ce_canary import build_case
from llm.test.frontend.unit.test_dense_training_ce_seeded_phase import DenseSeededCePhaseTest


class DenseCheckpointPhysicalSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_case()
        cls.backward_action = DenseSeededCePhaseTest.backward.id
        logits = DenseSeededCePhaseTest.tape.logits
        candidates = []
        manifest = cls.case.manifest
        for stream in manifest.core_streams:
            for position, ref in enumerate(stream.records):
                if _opcode(manifest, stream.logical_core, ref) is not RecordOpcode.MATMUL:
                    continue
                operand = _bound_operand(
                    manifest, stream.logical_core, ref, position,
                    SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
                if operand.buffer.value_id == logits.value_id:
                    candidates.append(ref.source_global_action_id)
        if len(candidates) != 1:
            raise AssertionError("real L2 source has no unique LM-head logits producer")
        cls.replay_action = candidates[0]

    def test_real_linked_l2_forward_to_native_ce_backward_cut(self) -> None:
        cut = derive_dense_checkpoint_physical_cut(
            self.case.manifest, replay_action_id=self.replay_action,
            backward_action_id=self.backward_action)
        self.assertEqual(cut.manifest_id, self.case.manifest.id)
        self.assertEqual(len(cut.persistent_parameter_states), 15)
        self.assertIn(cut.replay_parameter_state.id,
                      {state.id for state in cut.persistent_parameter_states})
        self.assertEqual(cut.no_checkpoint_saved_bytes, 128)
        self.assertLess(cut.checkpoint_saved_bytes, 128)
        self.assertEqual(cut.replay_output.buffer.value_id,
                         cut.backward_input.buffer.value_id)
        self.assertEqual(cut.replay_output.buffer.storage_id,
                         cut.backward_input.buffer.storage_id)
        self.assertLess(cut.replay_output.linked_position,
                        cut.backward_input.linked_position)
        template = derive_dense_checkpoint_replay_template(
            self.case.manifest, cut)
        self.assertEqual(template.record.opcode, RecordOpcode.MATMUL)
        self.assertEqual({r.operand_id for r in template.address_relocations}, {
            SemanticOperandId.COMPUTE_INPUT_ADDRESS,
            SemanticOperandId.COMPUTE_DATA_ADDRESS,
            SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
        })
        with self.assertRaisesRegex(SchemaError, "differs from original"):
            replace(template, source_record_digest="0" * 64).validate(cut)

    def test_same_physical_hbm_capacity_accepts_checkpoint_rejects_full_tape(self) -> None:
        cut = derive_dense_checkpoint_physical_cut(
            self.case.manifest, replay_action_id=self.replay_action,
            backward_action_id=self.backward_action)
        base = (cut.highest_parameter_end_bytes + 63) & -64
        capacity = (base + cut.checkpoint_saved_bytes + 63) & -64
        checkpoint = derive_dense_checkpoint_activation_tape(
            cut, checkpoint_enabled=True, hbm_capacity_bytes=capacity)
        self.assertEqual(checkpoint.activation_hbm_address, base)
        self.assertEqual(checkpoint.saved_buffer_abi_id, cut.replay_input.buffer.id)
        self.assertEqual(checkpoint.state_kind, "activation")
        self.assertEqual(checkpoint.state_lifetime, "step")
        self.assertEqual(checkpoint.state_access, "read_write")
        self.assertNotIn(checkpoint.hbm_binding_ref,
                         {state.hbm_binding_ref for state in
                          cut.persistent_parameter_states})
        with self.assertRaisesRegex(SchemaError, "shape/dtype disagrees"):
            replace(checkpoint, dtype="int32").validate(cut)
        with self.assertRaisesRegex(SchemaError, "exceeds the real HBM"):
            derive_dense_checkpoint_activation_tape(
                cut, checkpoint_enabled=False, hbm_capacity_bytes=capacity)
        full_capacity = (base + cut.no_checkpoint_saved_bytes + 63) & -64
        full = derive_dense_checkpoint_activation_tape(
            cut, checkpoint_enabled=False, hbm_capacity_bytes=full_capacity)
        self.assertEqual(full.saved_buffer_abi_id, cut.replay_output.buffer.id)
        self.assertEqual(full.size_bytes, 128)

    def test_production_activation_state_abi_exact_step_and_negative_lifetime(self) -> None:
        cut = derive_dense_checkpoint_physical_cut(
            self.case.manifest, replay_action_id=self.replay_action,
            backward_action_id=self.backward_action)
        base = (cut.highest_parameter_end_bytes + 63) & -64
        capacity = (base + cut.checkpoint_saved_bytes + 63) & -64
        tape = derive_dense_checkpoint_activation_tape(
            cut, checkpoint_enabled=True, hbm_capacity_bytes=capacity)
        abi = StateABI.create(
            state_ref="activation.checkpoint.real_l2_hidden",
            hbm_binding_ref=tape.hbm_binding_ref,
            kind=StateKind.ACTIVATION,
            lifetime=PersistentStateLifetime.STEP,
            access=PersistentStateAccess.READ_WRITE,
            shape=tape.shape, dtype=DType.FP16, layout=tape.layout,
            die_id=tape.die_id, address=tape.activation_hbm_address,
            size_bytes=tape.size_bytes, alignment_bytes=64)
        abi.validate("real_l2_checkpoint_activation")
        self.assertEqual(abi.size_bytes, 32)
        with self.assertRaisesRegex(SchemaError, "checkpoint activation"):
            replace(abi, lifetime=PersistentStateLifetime.PERSISTENT).validate(
                "invalid_persistent_checkpoint_activation")
        with self.assertRaisesRegex(SchemaError, "checkpoint activation"):
            replace(abi, dtype=DType.FP32,
                    size_bytes=abi.size_bytes * 2).validate(
                "invalid_dtype_checkpoint_activation")

    def test_wrong_backward_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "exactly one native source record"):
            derive_dense_checkpoint_physical_cut(
                self.case.manifest, replay_action_id=self.replay_action,
                backward_action_id=self.replay_action)

    def test_unsigned_replay_weight_operand_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "one exact BufferABI operand"):
            derive_dense_checkpoint_physical_cut(
                self.case.manifest, replay_action_id=self.replay_action,
                backward_action_id=self.backward_action,
                replay_weight_operand=SemanticOperandId.HBM_ADDRESS)

    def test_hidden_and_logit_extent_must_be_physical_saving(self) -> None:
        cut = derive_dense_checkpoint_physical_cut(
            self.case.manifest, replay_action_id=self.replay_action,
            backward_action_id=self.backward_action)
        with self.assertRaisesRegex(SchemaError, "checkpoint input extent drifted"):
            replace(cut, checkpoint_saved_bytes=cut.no_checkpoint_saved_bytes).validate()


if __name__ == "__main__":
    unittest.main()
