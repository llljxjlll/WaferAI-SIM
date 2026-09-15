from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.moe_full_model_compile_sequence import (
    compile_moe_full_model_inference_sequence,
)
from llm.frontend.wafer_frontend.schema.e2e_workload_graph import E2EOperationKind
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION, RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.flexible_moe import MoeRectActionKind
from llm.frontend.wafer_frontend.schema.moe_full_model_compile_sequence import (
    MoeFullModelCompileSegment,
    MoeFullModelCompileSequence,
    MoeFullModelCoverage,
    MoeFullModelLowering,
    MoeFullModelOperationBinding,
    MoeFullModelRuntimeStatus,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces, valid_spec
from llm.test.frontend.unit.test_moe_compile_sequence import (
    _manifest,
)
from llm.frontend.wafer_frontend.schema.workload_run import WorkloadFamily


def _legacy_template() -> ExperimentSpec:
    data = deepcopy(valid_spec())
    data["model"].update({
        "V": 16,
        "H": 4,
        "I": 8,
        "L": 2,
        "NH": 2,
        "KVH": 2,
        "DH": 2,
        "rotary_dim": 2,
        "max_position_embeddings": 32,
    })
    data["parallel"]["instances"][0].update({
        "role": "prefill",
        "tp": 1,
        "dp": 1,
        "ep": 1,
        "pp": 1,
        "replicas": 1,
        "sp": False,
    })
    profile = data["workload"]["infer"]["profile"]
    profile.update({
        "prefill_tokens": 4,
        "decode_tokens": 0,
        "num_seqs": 2,
        "context_sum": 4,
        "context_max": 2,
        "kv_pages": 2,
    })
    return from_data(ExperimentSpec, data, path="legacy_template")


class MoeFullModelCompileSequenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _manifest(WorkloadFamily.MOE_INFERENCE)
        cls.fabric = physical_fabric_from_data(
            minimal_hardware(2, 1, sram_bytes=65536)
        )
        cls.sequence = compile_moe_full_model_inference_sequence(
            cls.manifest,
            _legacy_template(),
            cls.fabric,
            hbm_address_spaces=valid_hbm_address_spaces(cls.fabric),
        )

    def test_full_graph_is_bound_in_exact_step_and_operation_order(self) -> None:
        sequence = self.sequence
        self.assertIs(sequence.coverage, MoeFullModelCoverage.FULL_MODEL)
        self.assertIs(
            sequence.runtime_status,
            MoeFullModelRuntimeStatus.EXECUTABLE_MANIFEST_MATERIALIZED,
        )
        self.assertEqual(
            tuple((segment.phase, segment.step) for segment in sequence.segments),
            (("prefill", 0), ("decode", 1), ("decode", 2)),
        )
        observed = tuple(
            binding.operation_ref
            for segment in sequence.segments
            for binding in segment.operation_bindings
        )
        self.assertEqual(
            observed,
            tuple(item.id for item in self.manifest.logical_graph.operations),
        )
        expected_kinds = set(E2EOperationKind) - {
            E2EOperationKind.PARAMETER_LOAD,
            E2EOperationKind.MLP_UP,
            E2EOperationKind.MLP_ACTIVATION,
            E2EOperationKind.MLP_DOWN,
            E2EOperationKind.LOSS,
            E2EOperationKind.DENSE_BACKWARD,
            E2EOperationKind.SHARED_BACKWARD,
            E2EOperationKind.GRAD_DISPATCH,
            E2EOperationKind.EXPERT_BACKWARD,
            E2EOperationKind.DX_COMBINE,
            E2EOperationKind.WGRAD,
            E2EOperationKind.ROUTER_GRADIENT,
            E2EOperationKind.EXPERT_GRADIENT,
            E2EOperationKind.GRADIENT_SYNC,
            E2EOperationKind.SGD_UPDATE,
            E2EOperationKind.PARAMETER_STORE,
            E2EOperationKind.STEP_COMMIT,
            E2EOperationKind.OPTIMIZER_LOAD,
            E2EOperationKind.OPTIMIZER_STORE,
            E2EOperationKind.ADAMW_UPDATE,
        }
        self.assertEqual(
            {binding.kind for segment in sequence.segments for binding in segment.operation_bindings},
            expected_kinds,
        )

    def test_dense_mlp_is_explicitly_replaced_by_every_layer_moe_unit(self) -> None:
        for segment in self.sequence.segments:
            self.assertEqual(len(segment.replaced_dense_mlp_node_refs), 6)
            self.assertEqual(len(segment.moe_unit_refs), 2)
            self.assertTrue(all(
                ref.endswith((".gate_up", ".swiglu", ".down"))
                for ref in segment.replaced_dense_mlp_node_refs
            ))
            lowerings = {binding.lowering for binding in segment.operation_bindings}
            self.assertTrue({
                MoeFullModelLowering.SHARED_SPINE_NODE,
                MoeFullModelLowering.SHARED_SPINE_STATE_ACCESS,
                MoeFullModelLowering.SHARED_SPINE_VALUE,
                MoeFullModelLowering.FLEXIBLE_MOE_ACTION,
                MoeFullModelLowering.FLEXIBLE_MOE_TRACE,
            }.issubset(lowerings))
            self.assertEqual(segment.replica_die_ids, (0, 1))

    def test_executable_manifest_replaces_dense_mlp_with_real_layer_blocks(self) -> None:
        for segment in self.sequence.segments:
            manifest = segment.executable_manifest
            records = {
                (fragment.id, index): record
                for fragment in manifest.fragments
                for stream in fragment.core_streams
                for index, record in enumerate(stream.records)
            }
            dense_actions = {
                action.id: getattr(
                    action.origin_ref,
                    "op_id",
                    getattr(action.origin_ref, "node_ref", None),
                )
                for action in segment.shared_spine_profile.lowering_context.global_dag.actions
            }
            linked_actions = {
                ref.source_global_action_id
                for stream in manifest.core_streams for ref in stream.records
            }
            self.assertFalse({
                action_id for action_id, node_ref in dense_actions.items()
                if node_ref in segment.replaced_dense_mlp_node_refs
            } & linked_actions)
            core0 = manifest.core_streams[0]
            opcodes = tuple(
                records[(ref.fragment_id, ref.fragment_record_index)].opcode
                for ref in core0.records
            )
            self.assertEqual(opcodes.count(RecordOpcode.DTE_ISSUE), 4)
            self.assertEqual(opcodes.count(RecordOpcode.DTE_SEND), 2)
            self.assertIn(RecordOpcode.EMBEDDING_LOOKUP, opcodes)
            self.assertIn(RecordOpcode.ATTENTION_EXACT, opcodes)
            self.assertEqual(tuple(stream.logical_core.die_id for stream in manifest.core_streams), (0, 1))
            self.assertEqual(segment.executable_manifest_digest, canonical_digest(manifest))

    def test_missing_bridge_gate_or_last_layer_fails_physical_dataflow_oracle(self) -> None:
        from llm.test.frontend.integration.run_moe_full_model_sequence_runtime_canary import (
            prove_full_model_dataflow,
        )
        segment = self.sequence.segments[0]
        manifest = segment.executable_manifest
        units = {item.id: item for item in self.sequence.moe_blocks.units}
        layer_units = tuple(units[ref] for ref in segment.moe_unit_refs)
        prove_full_model_dataflow(segment, layer_units)
        source = manifest.source_global_dag_id
        schema = "wafer_frontend.moe_full_model_region_linker/v1alpha1"
        gate = next(action for action in layer_units[0].plan.actions if action.kind is MoeRectActionKind.GATE and action.rank == 0)
        gate_id = stable_artifact_id("moe_full_model_action", {"source": source, "layer": 0, "action": gate.id}, schema_version=schema)
        last_layer_actions = {
            stable_artifact_id("moe_full_model_action", {"source": source, "layer": 1, "action": action.id}, schema_version=schema)
            for action in layer_units[1].plan.actions
        }
        bridge_id = stable_artifact_id("moe_full_model_bridge_action", {"source": source, "layer": 0, "role": "input"}, schema_version=schema)
        for dropped, message in (
            ({bridge_id}, "MoE dataflow lacks exact action record"),
            ({gate_id}, "MoE dataflow lacks exact action record"),
            (last_layer_actions, "MoE dataflow lacks exact action record"),
        ):
            broken_streams = tuple(replace(
                core, records=tuple(ref for ref in core.records if ref.source_global_action_id not in dropped)
            ) for core in manifest.core_streams)
            broken = replace(segment, executable_manifest=replace(manifest, core_streams=broken_streams))
            with self.assertRaisesRegex(RuntimeError, message):
                prove_full_model_dataflow(broken, layer_units)

    def test_missing_expert_projection_or_swiglu_fails_physical_work_oracle(self) -> None:
        from llm.test.frontend.integration.run_moe_full_model_sequence_runtime_canary import (
            prove_full_model_dataflow,
        )
        segment = self.sequence.segments[0]
        manifest = segment.executable_manifest
        unit_by_id = {item.id: item for item in self.sequence.moe_blocks.units}
        units = tuple(unit_by_id[ref] for ref in segment.moe_unit_refs)
        expert = next(action for action in units[0].plan.actions
                      if action.kind is MoeRectActionKind.EXPERT_FORWARD and action.rank == 1)
        expert_id = stable_artifact_id(
            "moe_full_model_action",
            {"source": manifest.source_global_dag_id, "layer": 0, "action": expert.id},
            schema_version="wafer_frontend.moe_full_model_region_linker/v1alpha1",
        )
        fragments = {fragment.id: fragment for fragment in manifest.fragments}
        core = next(item for item in manifest.core_streams if item.logical_core.die_id == 1)
        expert_ids = {expert_id, *(stable_artifact_id(
            "moe_full_model_action",
            {"source": manifest.source_global_dag_id, "layer": 0, "action": stable_artifact_id(
                "flexible_moe_expert_projection_action",
                {"plan": units[0].plan.id, "expert": expert.id, "stage": stage},
                schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
            )}, schema_version="wafer_frontend.moe_full_model_region_linker/v1alpha1",
        ) for stage in ("up", "swiglu", "down"))}
        expert_refs = [ref for ref in core.records if ref.source_global_action_id in expert_ids]
        projection_refs = [ref for ref in expert_refs if
                           next(stream.records[ref.fragment_record_index]
                                for stream in fragments[ref.fragment_id].core_streams
                                if stream.logical_core == core.logical_core).opcode is RecordOpcode.MATMUL]
        activation_ref = next(ref for ref in expert_refs if
                              next(stream.records[ref.fragment_record_index]
                                   for stream in fragments[ref.fragment_id].core_streams
                                   if stream.logical_core == core.logical_core).opcode is RecordOpcode.SWIGLU)
        self.assertEqual(len(projection_refs), 3)
        for removed in (*projection_refs, activation_ref):
            broken = replace(segment, executable_manifest=replace(
                manifest, core_streams=tuple(replace(
                    stream, records=tuple(ref for ref in stream.records if ref != removed)
                ) if stream.logical_core == core.logical_core else stream
                    for stream in manifest.core_streams),
            ))
            with self.assertRaisesRegex(RuntimeError, "expert lacks exact gate/up/down MATMUL and SwiGLU"):
                prove_full_model_dataflow(broken, units)

    def test_canonical_artifact_round_trip_preserves_full_cover(self) -> None:
        rebuilt = loads_dataclass(
            MoeFullModelCompileSequence,
            canonical_json(self.sequence),
        )
        self.assertEqual(rebuilt, self.sequence)
        self.assertEqual(rebuilt.digest, self.sequence.digest)

    def test_unknown_shared_spine_lineage_fails_closed(self) -> None:
        original = self.sequence.segments[0]
        bindings = list(original.operation_bindings)
        target_index = next(
            index
            for index, binding in enumerate(bindings)
            if binding.lowering is MoeFullModelLowering.SHARED_SPINE_NODE
        )
        semantic = bindings[target_index]._semantic()
        semantic["production_refs"] = ("forged.shared.node",)
        bindings[target_index] = MoeFullModelOperationBinding.create(**semantic)
        segment_semantic = original._semantic()
        segment_semantic["operation_bindings"] = tuple(bindings)
        segments = (
            MoeFullModelCompileSegment.create(**segment_semantic),
            *self.sequence.segments[1:],
        )
        with self.assertRaisesRegex(SchemaError, "production ref is unknown"):
            MoeFullModelCompileSequence.create(
                moe_blocks=self.sequence.moe_blocks,
                segments=segments,
            )

    def test_known_but_wrong_node_and_moe_action_fail_oracle(self) -> None:
        original = self.sequence.segments[0]
        for target_lowering, wrong_ref in (
            (
                MoeFullModelLowering.SHARED_SPINE_NODE,
                next(
                    ref
                    for binding in original.operation_bindings
                    if binding.kind is E2EOperationKind.QKV
                    for ref in binding.production_refs
                ),
            ),
            (
                MoeFullModelLowering.FLEXIBLE_MOE_ACTION,
                next(
                    ref
                    for binding in original.operation_bindings
                    if binding.kind is E2EOperationKind.COMBINE
                    for ref in binding.production_refs
                ),
            ),
        ):
            bindings = list(original.operation_bindings)
            target_index = next(
                index
                for index, binding in enumerate(bindings)
                if binding.lowering is target_lowering
                and wrong_ref not in binding.production_refs
            )
            semantic = bindings[target_index]._semantic()
            semantic["production_refs"] = (wrong_ref,)
            bindings[target_index] = MoeFullModelOperationBinding.create(**semantic)
            segment_semantic = original._semantic()
            segment_semantic["operation_bindings"] = tuple(bindings)
            forged = MoeFullModelCompileSegment.create(**segment_semantic)
            with self.assertRaisesRegex(SchemaError, "exactly lower"):
                MoeFullModelCompileSequence.create(
                    moe_blocks=self.sequence.moe_blocks,
                    segments=(forged, *self.sequence.segments[1:]),
                )

    def test_operation_omission_and_runtime_overclaim_fail_closed(self) -> None:
        original = self.sequence.segments[0]
        segment_semantic = original._semantic()
        segment_semantic["operation_bindings"] = original.operation_bindings[:-1]
        shortened = MoeFullModelCompileSegment.create(**segment_semantic)
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            MoeFullModelCompileSequence.create(
                moe_blocks=self.sequence.moe_blocks,
                segments=(shortened, *self.sequence.segments[1:]),
            )
        with self.assertRaisesRegex(SchemaError, "claim drifted"):
            replace(self.sequence, runtime_status="complete").validate()


if __name__ == "__main__":
    unittest.main()
