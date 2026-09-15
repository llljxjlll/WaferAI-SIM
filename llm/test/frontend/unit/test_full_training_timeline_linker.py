"""Physical work, parameter home, and loss tape contract before TRAIN E2E linking."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.full_training_timeline_linker import (
    cut_moe_training_unit, locate_forward_cross_entropy_tape,
    reconcile_training_parameter_homes, require_physical_operation_coverage,
)
from llm.frontend.wafer_frontend.lowering.moe_full_training_namespace import (
    namespace_moe_training_unit,
)
from llm.frontend.wafer_frontend.passes.moe_compile_sequence import compile_moe_sequence
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.artifact_manifest import ProgramSymbolKind
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.workload_run import WorkloadFamily
from llm.test.frontend.unit.test_moe_compile_sequence import _manifest
from llm.test.frontend.unit.test_flexible_dense_train import _spec
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.moe_full_model_compile_sequence import _rank_local_fabric
from llm.frontend.wafer_frontend.passes.flexible_dense_train import materialize_flexible_dense_train_forward
from llm.frontend.wafer_frontend.passes.flexible_dense_backward import materialize_flexible_dense_backward


class FullTrainingTimelineLinkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fabric = physical_fabric_from_data(minimal_hardware(2, 1, sram_bytes=65536))
        spaces = valid_hbm_address_spaces(fabric)
        base = _spec(1, 1)
        spec = replace(
            base,
            model=replace(base.model, V=16, NH=2, KVH=2, DH=2,
                          rotary_dim=2, max_position_embeddings=32),
            workload=replace(base.workload,
                             train=replace(base.workload.train, seq_len=4)),
        )
        spec.validate("real_shared_training_model")
        forward = materialize_flexible_dense_train_forward(
            spec, RectMeshSpec(1, 1), _rank_local_fabric(fabric), (spaces[0],),
            producer_pass="moe_full_model_shared_train_audit",
        )
        cls.forward = forward
        cls.backward = materialize_flexible_dense_backward(
            forward, _rank_local_fabric(fabric), (spaces[0],),
        )
        cls.ce_action = next(
            action.id for action in forward.linked_forward.source.replicas[0]
            .lowering_context.global_dag.actions
            if getattr(action.origin_ref, "op_id", None) == "T0.cross_entropy__dp0"
        )
        cls.moe = compile_moe_sequence(
            _manifest(WorkloadFamily.MOE_TRAINING),
            source_rank_policy="rank0_shared_spine",
        )

    def test_all_real_parameter_homes_have_one_load_and_store(self) -> None:
        homes = reconcile_training_parameter_homes(
            self.forward.linked_forward.manifest, self.backward.manifest,
            self.forward.plan.parameter_templates,
        )
        self.assertEqual(len(homes), 15)
        self.assertEqual(sum(home.trainable_abi.size_bytes for home in homes), 936)
        self.assertEqual(max(home.trainable_abi.address
                             + home.trainable_abi.size_bytes for home in homes), 1344)

    def test_parameter_size_mismatch_rejected(self) -> None:
        templates = list(self.forward.plan.parameter_templates)
        templates[0] = replace(templates[0], weight_bytes=templates[0].weight_bytes * 2)
        with self.assertRaisesRegex(SchemaError, "exact HBM home"):
            reconcile_training_parameter_homes(
                self.forward.linked_forward.manifest, self.backward.manifest,
                tuple(templates),
            )

    def test_ce_tape_has_three_distinct_late_frees(self) -> None:
        tape = locate_forward_cross_entropy_tape(
            self.forward.linked_forward.manifest, action_id=self.ce_action,
        )
        self.assertEqual([(abi.size_bytes, abi.dtype.value) for abi in
                          (tape.logits, tape.labels, tape.per_row_loss)],
                         [(128, "fp16"), (16, "int32"), (16, "fp32")])
        self.assertEqual(len(set(tape.terminal_frees)), 3)
        self.assertEqual([definition.value for definition
                          in tape.original_operand_definitions], [3328, 3456, 3520])
        self.assertEqual(tape.terminal_free_core_positions, (152, 151, 150))

    def test_missing_ce_native_tape_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "native CE producer"):
            locate_forward_cross_entropy_tape(
                self.forward.linked_forward.manifest,
                action_id="wrong_ce_action_id",
            )

    def test_each_moe_physical_cut_preserves_all_records(self) -> None:
        self.assertEqual(len(self.moe.units), 4)
        for unit in self.moe.units:
            cut = cut_moe_training_unit(unit.plan, unit.linked_manifest)
            self.assertEqual(sorted(n for _, n in cut.first_backward_index), [26, 32])
            original = {stream.logical_core: stream.records
                        for stream in unit.linked_manifest.core_streams}
            self.assertEqual(set(original), {stream.logical_core for stream in cut.forward})
            for early, late in zip(cut.forward, cut.backward):
                self.assertEqual(early.records + late.records,
                                 original[early.logical_core])

    def test_two_step_moe_namespace_has_distinct_actions_and_stable_hbm(self) -> None:
        fabric = physical_fabric_from_data(minimal_hardware(2, 1, sram_bytes=131072))
        homes = valid_hbm_address_spaces(fabric)
        named = [namespace_moe_training_unit(
            unit, timeline_source_id="one_true_two_step_training_timeline",
            hbm_homes=homes, sram_capacity_bytes=131072,
        ) for unit in self.moe.units]
        for step in (0, 1):
            same_layer = [item for item in named if item.layer == 0 and item.step == step]
            self.assertEqual(len(same_layer), 1)
        state_by_unit = [{abi.id: (abi.address, abi.size_bytes)
                          for fragment in item.fragments for abi in fragment.state_abi}
                         for item in named]
        self.assertEqual(state_by_unit[0], state_by_unit[2])
        self.assertEqual(state_by_unit[1], state_by_unit[3])
        self.assertFalse(set(state_by_unit[0]) & set(state_by_unit[1]))
        self.assertFalse(set(dict(named[0].action_ids).values()) &
                         set(dict(named[2].action_ids).values()))
        for item in named:
            original = self.moe.units[item.step * 2 + item.layer].linked_manifest
            self.assertEqual(sum(len(s.records) for s in item.forward + item.backward),
                             sum(len(s.records) for s in original.core_streams))
            region_values = {definition.name: definition.value for definition
                             in item.program_definitions if
                             definition.symbol.kind is ProgramSymbolKind.SRAM_REGION}
            self.assertEqual(region_values, {"input": 8192, "comm": 45056})
            local_program = {symbol.id: symbol for fragment in item.fragments
                             for symbol in fragment.program_symbols}
            local_runtime = {symbol.id: symbol for fragment in item.fragments
                             for symbol in fragment.runtime_symbols}
            program_def = {definition.symbol.id: definition.symbol
                           for definition in item.program_definitions}
            runtime_def = {definition.symbol.id: definition.symbol
                           for definition in item.runtime_definitions}
            self.assertEqual(local_program, program_def)
            self.assertEqual(local_runtime, runtime_def)
            by_fragment = {fragment.id: fragment for fragment in item.fragments}
            for binding in item.address_bindings:
                self.assertIn(binding.fragment_id, by_fragment)
                declared = {abi.id for abi in
                            by_fragment[binding.fragment_id].buffer_abi}
                self.assertLessEqual(set(binding.buffer_abi_ids), declared)
            for binding in item.state_bindings:
                self.assertIn(binding.state_abi_id,
                              {abi.id for abi in
                               by_fragment[binding.fragment_id].state_abi})

    def test_reject_missing_named_layer_or_native_loss(self) -> None:
        original = self.forward.linked_forward
        by_action = {
            action.id: getattr(action.origin_ref, "op_id", "")
            for action in original.source.replicas[0]
            .lowering_context.global_dag.actions
        }
        requirements = {
            "T0.layer0.norm2__dp0": {RecordOpcode.RMSNORM: 1},
            "T0.layer1.norm2__dp0": {RecordOpcode.RMSNORM: 1},
            "T0.layer0.attention__dp0": {RecordOpcode.ATTENTION_EXACT: 1},
            "T0.layer1.attention__dp0": {RecordOpcode.ATTENTION_EXACT: 1},
            "T0.lm_head__dp0": {RecordOpcode.MATMUL: 1},
            "T0.cross_entropy__dp0": {RecordOpcode.CROSS_ENTROPY_FORWARD: 1},
        }
        require_physical_operation_coverage(
            original.manifest, operation_by_action=by_action,
            required_by_operation=requirements,
        )
        for corrupted in (
            {key: value for key, value in by_action.items()
             if value != "T0.layer1.norm2__dp0"},
            by_action,
        ):
            broken = dict(requirements)
            if corrupted is by_action:
                broken["T0.cross_entropy__dp0"] = {
                    RecordOpcode.CROSS_ENTROPY_BACKWARD: 1,
                }
            with self.assertRaisesRegex(SchemaError, "lacks physical work"):
                require_physical_operation_coverage(
                    original.manifest, operation_by_action=corrupted,
                    required_by_operation=broken,
                )


if __name__ == "__main__":
    unittest.main()
