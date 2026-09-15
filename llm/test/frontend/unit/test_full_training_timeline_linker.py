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
    derive_moe_training_state_version_edges,
    derive_moe_training_transport_edges, namespace_moe_training_unit,
)
from llm.frontend.wafer_frontend.schema.full_training_physical_dag import (
    build_full_training_physical_dag, full_training_physical_dag_source_id,
)
from llm.frontend.wafer_frontend.lowering.full_training_program_merger import (
    require_full_training_opcode_matrix,
)
from llm.frontend.wafer_frontend.lowering.full_training_ce_tape_graft import (
    graft_native_ce_backward_onto_forward_tape,
    graft_seeded_ce_backward_onto_forward_tape,
)
from llm.frontend.wafer_frontend.lowering.full_training_ce_tape_program import (
    build_bounded_seeded_ce_physical_manifest,
)
from llm.frontend.wafer_frontend.schema.global_action import GlobalActionDAG
from llm.frontend.wafer_frontend.passes.moe_compile_sequence import compile_moe_sequence
from llm.frontend.wafer_frontend.passes.moe_training_shared_gradient_bridge import (
    locate_shared_backward_gradient_producer,
)
from llm.frontend.wafer_frontend.passes.full_training_ce_seed_program_io import (
    build_bounded_seeded_ce_program_io,
)
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

    def test_native_ce_timing_surrogate_graft_requires_new_dag_and_retains_tape(self) -> None:
        from llm.test.frontend.unit.test_dense_training_ce_phase_carrier import (
            DenseTrainingCePhaseCarrierTest,
        )
        from llm.frontend.wafer_frontend.passes.dense_training_ce_phase_carrier import (
            build_dense_training_ce_phase_carrier,
        )
        fixture = DenseTrainingCePhaseCarrierTest
        fixture.setUpClass()
        real = self.forward.linked_forward.source.replicas[0]
        old_global = real.lowering_context.global_dag
        global_args = old_global._semantic_key()
        global_args["actions"] = old_global.actions + (fixture.backward,)
        new_global = GlobalActionDAG.create(
            producer_pass="full_training_native_ce_global_action_dag",
            **global_args,
        )
        new_global.validate("real_forward_and_ce_backward_global_dag")
        native = build_dense_training_ce_phase_carrier(
            fixture.manifest, fixture.tape,
            backward_action=fixture.backward, gradient=fixture.grad,
            gradient_address=fixture.address,
            gradient_label=fixture.label, region=fixture.region,
            source_global_dag_id=new_global.id,
        )
        native_record = next(record for record in native.fragment.core_streams[0].records
                             if record.opcode is RecordOpcode.CROSS_ENTROPY_BACKWARD)
        upstream = next(operand.symbol_ref for operand in native_record.operands
                        if operand.name == "upstream_address")
        self.assertEqual(upstream, fixture.tape.original_operand_definitions[2].symbol.id)
        # This source-stage tape graft has a physical native opcode, but the
        # forward loss value is a mathematical invalid surrogate for dLoss.
        # It cannot pass the full-training independent-seed oracle.
        forward = graft_native_ce_backward_onto_forward_tape(
            fixture.manifest, fixture.tape, native,
            source_global_dag_id=new_global.id,
        )
        original = fixture.manifest.core_streams[0].records
        retained = forward.core_streams[0].records
        self.assertEqual(len(retained), len(original) - 3 + 7)
        self.assertEqual(forward.global_dag_id, new_global.id)
        self.assertTrue(all(fragment.source_global_dag_id == new_global.id
                            for fragment in forward.fragments))
        self.assertEqual(tuple(old for old, _ in
                               forward.original_to_backward_frees),
                         fixture.tape.terminal_frees)
        self.assertTrue(all(old not in retained for old
                            in fixture.tape.terminal_frees))
        self.assertTrue(all(new in retained for _, new
                            in forward.original_to_backward_frees))
        self.assertEqual([abi.lifetime_end_exclusive for abi in
                          native.extended_forward_buffers], [42] * 3)
        with self.assertRaisesRegex(SchemaError, "newly re-signed"):
            graft_native_ce_backward_onto_forward_tape(
                fixture.manifest, fixture.tape, native,
                source_global_dag_id=fixture.manifest.source_global_dag_id,
            )
        with self.assertRaisesRegex(SchemaError, "exactly its consumed BufferABI"):
            graft_native_ce_backward_onto_forward_tape(
                fixture.manifest, fixture.tape,
                replace(native, old_to_extended_abi_ids=
                        native.old_to_extended_abi_ids[:2]),
                source_global_dag_id=new_global.id,
            )

    def test_seeded_ce_graft_keeps_forward_loss_free_and_binds_real_dloss(self) -> None:
        from llm.test.frontend.unit.test_dense_training_ce_seeded_phase import (
            DenseSeededCePhaseTest,
        )
        from llm.frontend.wafer_frontend.passes.dense_training_ce_seeded_phase import (
            build_dense_training_ce_seeded_phase,
        )
        fixture = DenseSeededCePhaseTest
        fixture.setUpClass()
        real = self.forward.linked_forward.source.replicas[0]
        old_global = real.lowering_context.global_dag
        args = old_global._semantic_key()
        args["actions"] = old_global.actions + (fixture.backward,)
        new_global = GlobalActionDAG.create(
            producer_pass="full_training_true_dloss_ce_global_dag", **args,
        )
        new_global.validate("native_seeded_ce_graft_source")
        seeded = build_dense_training_ce_seeded_phase(
            fixture.forward, fixture.tape, backward_action=fixture.backward,
            source_global_dag_id=new_global.id,
            loss_gradient=fixture.loss_gradient,
            loss_gradient_address=fixture.loss_address,
            loss_gradient_label=fixture.loss_label,
            logits_gradient=fixture.logits_gradient,
            logits_gradient_address=fixture.grad_address,
            logits_gradient_label=fixture.grad_label,
            region=fixture.region,
        )
        merged = graft_seeded_ce_backward_onto_forward_tape(
            fixture.forward, fixture.tape, seeded,
            source_global_dag_id=new_global.id,
        )
        old = fixture.forward.core_streams[0].records
        actual = merged.core_streams[0].records
        self.assertEqual(len(actual), len(old) - 2 + 8)
        self.assertEqual(merged.loss_gradient_seed, b"\x00\x00\x80\x3f" * 4)
        self.assertEqual(merged.loss_gradient_abi_id, fixture.loss_gradient.id)
        self.assertEqual(merged.retained_forward_loss_free.source_global_action_id,
                         fixture.tape.terminal_frees[2].source_global_action_id)
        self.assertIn(merged.retained_forward_loss_free, actual)
        self.assertNotIn(fixture.tape.terminal_frees[2], actual)
        self.assertTrue(all(old not in actual and new in actual for old, new
                            in merged.original_to_backward_frees))
        self.assertEqual(len(merged.old_to_extended_tape_abi_ids), 2)
        native = next(record for record in seeded.fragment.core_streams[0].records
                      if record.opcode is RecordOpcode.CROSS_ENTROPY_BACKWARD)
        aux = next(item.symbol_ref for item in native.operands
                   if item.name == "upstream_address")
        self.assertEqual(aux, fixture.loss_address.symbol.id)
        self.assertNotEqual(aux, fixture.tape.original_operand_definitions[2].symbol.id)
        with self.assertRaisesRegex(SchemaError, "retain forward loss FREE"):
            graft_seeded_ce_backward_onto_forward_tape(
                fixture.forward, fixture.tape,
                replace(seeded, retained_forward_loss_free=
                        fixture.tape.terminal_frees[0]),
                source_global_dag_id=new_global.id,
            )
        with self.assertRaisesRegex(SchemaError, "nonzero dLoss"):
            graft_seeded_ce_backward_onto_forward_tape(
                fixture.forward, fixture.tape,
                replace(seeded, loss_gradient_seed=b"\x00" * 16),
                source_global_dag_id=new_global.id,
            )
        from llm.test.frontend.unit.test_dense_training_ce_phase_carrier import (
            DenseTrainingCePhaseCarrierTest as TimingPhase,
        )
        from llm.frontend.wafer_frontend.passes.dense_training_ce_phase_carrier import (
            build_dense_training_ce_phase_carrier,
        )
        timing = build_dense_training_ce_phase_carrier(
            TimingPhase.manifest, TimingPhase.tape,
            backward_action=TimingPhase.backward,
            gradient=TimingPhase.grad,
            gradient_address=TimingPhase.address,
            gradient_label=TimingPhase.label,
            region=TimingPhase.region,
            source_global_dag_id=new_global.id,
        )
        with self.assertRaisesRegex(SchemaError, "typed independent seeded phase"):
            graft_seeded_ce_backward_onto_forward_tape(
                fixture.forward, fixture.tape, timing,
                source_global_dag_id=new_global.id,
            )
        bounded = build_bounded_seeded_ce_physical_manifest(
            fixture.forward, graft=merged, new_global_dag=new_global,
        )
        self.assertEqual(sum(len(stream.records) for stream in bounded.core_streams),
                         len(old) - 2 + 8)
        sidecar = build_bounded_seeded_ce_program_io(
            self.forward.linked_forward, bounded, merged, "a" * 64,
        )
        seed_entries = [
            entry for entry in sidecar.initializations
            if getattr(entry.target, "buffer_abi_id", None) ==
               fixture.loss_gradient.id
        ]
        self.assertEqual(len(seed_entries), 1)
        seed_blob = next(blob for blob in sidecar.blobs
                         if blob.id == seed_entries[0].blob_ref)
        self.assertEqual(seed_blob.payload(), b"\x00\x00\x80\x3f" * 4)
        with self.assertRaisesRegex(SchemaError, "nonzero physical seed"):
            build_bounded_seeded_ce_program_io(
                self.forward.linked_forward, bounded,
                replace(merged, loss_gradient_seed=bytes(16)), "a" * 64,
            )
        context = real.lowering_context
        with self.assertRaisesRegex(
            SchemaError, "actions must exactly preserve projection DAG/task order",
        ):
            bounded.validate_against(
                context.ir1, context.fusion_plans, context.standalone_plans,
                context.projection, context.schedule_set,
                new_global, bounded.fragments,
            )

    def test_missing_ce_native_tape_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "native CE producer"):
            locate_forward_cross_entropy_tape(
                self.forward.linked_forward.manifest,
                action_id="wrong_ce_action_id",
            )

    def test_existing_shared_backward_motif_cannot_feed_moe_gradient(self) -> None:
        source = self.backward.manifest
        fragments = {fragment.id: fragment for fragment in source.fragments}
        matmul_refs = [
            ref for stream in source.core_streams
            for ref in stream.records
            if next(local.records[ref.fragment_record_index]
                    for local in fragments[ref.fragment_id].core_streams
                    if local.logical_core == stream.logical_core).opcode
               is RecordOpcode.MATMUL
        ]
        self.assertTrue(matmul_refs)
        with self.assertRaisesRegex(SchemaError, "not owned FP16 m×H SRAM"):
            locate_shared_backward_gradient_producer(
                source, source_action_id=matmul_refs[0].source_global_action_id,
                expected_opcode=RecordOpcode.MATMUL, tokens=4, hidden_size=4,
            )

    def test_each_moe_physical_cut_preserves_all_records(self) -> None:
        self.assertEqual(len(self.moe.units), 4)
        for unit in self.moe.units:
            cut = cut_moe_training_unit(unit.plan, unit.linked_manifest)
            first_backward = dict(cut.first_backward_index)
            original = {stream.logical_core: stream.records
                        for stream in unit.linked_manifest.core_streams}
            self.assertEqual(set(original), {stream.logical_core for stream in cut.forward})
            self.assertEqual(set(first_backward), set(original))
            for early, late in zip(cut.forward, cut.backward):
                self.assertGreater(len(early.records), 0)
                self.assertGreater(len(late.records), 0)
                self.assertEqual(first_backward[early.logical_core],
                                 len(early.records))
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
        self.assertEqual(len(derive_moe_training_state_version_edges(named[0], named[2])), 4)
        self.assertEqual(len(derive_moe_training_state_version_edges(named[1], named[3])), 4)
        with self.assertRaisesRegex(SchemaError, "same MoE layer"):
            derive_moe_training_state_version_edges(named[0], named[3])
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

    def test_physical_dag_signs_true_moe_transport_and_rejects_cycles(self) -> None:
        fabric = physical_fabric_from_data(minimal_hardware(2, 1, sram_bytes=131072))
        unit = self.moe.units[0]
        named = namespace_moe_training_unit(
            unit, timeline_source_id="typed_physical_moe_only_receipt",
            hbm_homes=valid_hbm_address_spaces(fabric), sram_capacity_bytes=131072,
        )
        streams = tuple(replace(front, records=front.records + back.records)
                        for front, back in zip(named.forward, named.backward))
        in_backward = {ref.source_global_action_id for stream in named.backward
                       for ref in stream.records}
        metadata = {new: (old, f"moe-only.step0.layer0.{old}", 0, 0,
                          "backward" if new in in_backward else "forward")
                    for old, new in named.action_ids}
        required = tuple(sorted({value[1] for value in metadata.values()}))
        edges = derive_moe_training_transport_edges(unit, named)
        self.assertTrue(edges)
        receipt = build_full_training_physical_dag(
            fragments=named.fragments, streams=streams,
            source_artifact_ids=(unit.id, unit.linked_manifest.id),
            operation_by_action=metadata, transport_edges=edges,
            required_operation_ids=required,
        )
        self.assertEqual(receipt.transport_edges, edges)
        self.assertEqual(sum(len(action.executable_records)
                             for action in receipt.actions),
                         sum(len(stream.records) for stream in streams))
        with self.assertRaisesRegex(SchemaError, "two steps and two layers"):
            require_full_training_opcode_matrix(receipt)
        victim = next(old for old, new in named.action_ids if new == edges[0][0])
        with self.assertRaisesRegex(SchemaError, "no actual cloned DTE"):
            derive_moe_training_transport_edges(
                unit, replace(named, action_ids=tuple((old, new) for old, new
                     in named.action_ids if old != victim)),
            )
        first = streams[0].records[0].source_global_action_id
        last = streams[0].records[-1].source_global_action_id
        with self.assertRaisesRegex(SchemaError, "dependency cycle"):
            build_full_training_physical_dag(
                fragments=named.fragments, streams=streams,
                source_artifact_ids=(unit.id, unit.linked_manifest.id),
                operation_by_action=metadata, transport_edges=edges,
                state_version_edges=((last, first),),
                required_operation_ids=required,
            )

    def test_four_moe_blocks_form_one_source_dag_but_reject_missing_dense_ce(self) -> None:
        fabric = physical_fabric_from_data(minimal_hardware(2, 1, sram_bytes=131072))
        anchors = tuple(sorted({self.moe.id,
                                *(unit.id for unit in self.moe.units),
                                *(unit.linked_manifest.id for unit in self.moe.units)}))
        source = full_training_physical_dag_source_id(anchors)
        named = {(unit.step, unit.layer): namespace_moe_training_unit(
            unit, timeline_source_id=source,
            hbm_homes=valid_hbm_address_spaces(fabric),
            sram_capacity_bytes=131072,
        ) for unit in self.moe.units}
        transport, versions, metadata = [], [], {}
        for unit in self.moe.units:
            item = named[(unit.step, unit.layer)]
            transport.extend(derive_moe_training_transport_edges(unit, item))
            backward = {ref.source_global_action_id for stream in item.backward
                        for ref in stream.records}
            metadata.update({new: (
                old, f"moe-only.step{unit.step}.layer{unit.layer}.{old}",
                unit.step, unit.layer,
                "backward" if new in backward else "forward",
            ) for old, new in item.action_ids})
        for layer in (0, 1):
            versions.extend(derive_moe_training_state_version_edges(
                named[(0, layer)], named[(1, layer)],
            ))
        first = named[(0, 0)]
        streams = []
        for early in first.forward:
            core = early.logical_core
            records = tuple(ref for step in (0, 1)
                            for layer, phase in ((0, "forward"), (1, "forward"),
                                                 (1, "backward"), (0, "backward"))
                            for segment in getattr(named[(step, layer)], phase)
                            if segment.logical_core == core for ref in segment.records)
            streams.append(replace(early, records=records))
        fragments = tuple(fragment for item in named.values()
                          for fragment in item.fragments)
        receipt = build_full_training_physical_dag(
            fragments=fragments, streams=tuple(streams),
            source_artifact_ids=anchors, operation_by_action=metadata,
            transport_edges=tuple(sorted(set(transport))),
            state_version_edges=tuple(sorted(set(versions))),
            required_operation_ids=tuple(sorted(value[1] for value in metadata.values())),
        )
        self.assertEqual(receipt.id, source)
        self.assertEqual(len(receipt.state_version_edges), 8)
        self.assertEqual(sum(len(action.executable_records) for action in receipt.actions),
                         sum(len(stream.records) for stream in streams))
        with self.assertRaisesRegex(SchemaError, "native loss/backward"):
            require_full_training_opcode_matrix(receipt)

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
