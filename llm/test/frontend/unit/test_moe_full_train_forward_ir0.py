"""Two-layer MoE full forward source and physical EP ownership regressions."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import (
    SchemaError, UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.passes.moe_compile_sequence import (
    compile_moe_sequence,
)
from llm.frontend.wafer_frontend.schema.workload_run import WorkloadFamily
from llm.test.frontend.unit.test_moe_compile_sequence import _manifest
from llm.frontend.wafer_frontend.passes.moe_full_train_forward_ir0 import (
    _source_moe_operation_workload,
    build_moe_full_train_forward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_forward_validator import (
    MoeFullTrainForwardValidator,
)
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.ir0 import EdgeKind,GraphEdge,IR0
from llm.frontend.wafer_frontend.schema.moe_full_training_block_workload import (
    MoeForwardBlockKind,
)
from llm.test.frontend.unit.test_full_training_timeline_linker import (
    FullTrainingTimelineLinkerTest,
)


class MoeFullTrainForwardIr0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FullTrainingTimelineLinkerTest.setUpClass()
        cls.forward = FullTrainingTimelineLinkerTest.forward.plan.forward_graph
        cls.sequence = FullTrainingTimelineLinkerTest.moe

    def test_true_two_layer_shared_spine_and_ep2_forward_source(self):
        phase = build_moe_full_train_forward_ir0(self.forward,self.sequence)
        MoeFullTrainForwardValidator.validate(
            phase, original_dense=self.forward, sequence=self.sequence,
        )
        self.assertEqual(len(phase.graph.nodes),32)
        self.assertEqual(len(phase.graph.persistent_states),27)
        self.assertEqual(len(phase.ep_state_owners),16)
        self.assertEqual(len(phase.shared_source_state_refs),11)
        self.assertEqual(len(phase.removed_dense_op_refs),6)
        with self.assertRaisesRegex(UnsupportedFeatureError,
                                    "typed source phase"):
            DenseIR0Validator.validate(phase.graph)

    def test_true_one_die_two_layer_ep1_forward_source(self):
        sequence = compile_moe_sequence(
            _manifest(WorkloadFamily.MOE_TRAINING, rows=1, columns=1),
            source_rank_policy="rank0_shared_spine",
        )
        self.assertEqual(len(sequence.units), 4)
        phase = build_moe_full_train_forward_ir0(self.forward, sequence)
        MoeFullTrainForwardValidator.validate(
            phase, original_dense=self.forward, sequence=sequence,
        )
        self.assertEqual(phase.graph.instances[0].parallel.ep, 1)
        self.assertEqual(len(phase.ep_state_owners), 8)
        self.assertEqual({owner.ep_owner for owner in phase.ep_state_owners}, {0})
        self.assertEqual(len(phase.removed_dense_op_refs), 6)
        self.assertEqual(len(phase.shared_source_state_refs), 11)
        self.assertEqual(len(phase.graph.persistent_states), 19)
        for layer in (0, 1):
            prefix = f"T0.layer{layer}."
            nodes = {node.id: node for node in phase.graph.nodes}
            self.assertNotIn(prefix + "gate_up", nodes)
            self.assertNotIn(prefix + "swiglu", nodes)
            self.assertNotIn(prefix + "down", nodes)
            self.assertEqual(
                nodes[prefix + "residual2"].inputs[1],
                prefix + "moe.combine_out",
            )
            self.assertEqual(
                nodes[prefix + "moe.router"].inputs,
                (prefix + "norm2_out", prefix + "moe.router.weight.ep0"),
            )
        with self.assertRaisesRegex(UnsupportedFeatureError, "typed source phase"):
            DenseIR0Validator.validate(phase.graph)

    def test_ep1_router_rejects_missing_or_phantom_gate_replica(self):
        sequence = compile_moe_sequence(
            _manifest(WorkloadFamily.MOE_TRAINING, rows=1, columns=1),
            source_rank_policy="rank0_shared_spine",
        )
        phase = build_moe_full_train_forward_ir0(self.forward, sequence)
        router = next(node for node in phase.graph.nodes
                      if node.id == "T0.layer0.moe.router")
        for inputs in (router.inputs[:1], (*router.inputs, router.inputs[1] + ".phantom")):
            with self.subTest(inputs=inputs), self.assertRaisesRegex(
                    SchemaError, "operand arity"):
                replace(router, inputs=inputs).validate("ep1_router")

    def test_dropping_any_layer_or_expert_owner_fails(self):
        phase = build_moe_full_train_forward_ir0(self.forward,self.sequence)
        for position in (0,1,15):
            broken = replace(phase,ep_state_owners=phase.ep_state_owners[:position]
                             + phase.ep_state_owners[position+1:])
            with self.subTest(position=position),self.assertRaises(SchemaError):
                MoeFullTrainForwardValidator.validate(
                    broken,original_dense=self.forward,sequence=self.sequence,
                )

    def test_expert1_cannot_be_relabelled_as_expert0_home(self):
        phase = build_moe_full_train_forward_ir0(self.forward,self.sequence)
        position = next(i for i,owner in enumerate(phase.ep_state_owners)
                        if owner.ep_owner == 1)
        tampered = replace(phase.ep_state_owners[position],ep_owner=0)
        broken = replace(phase,ep_state_owners=phase.ep_state_owners[:position]
                         + (tampered,) + phase.ep_state_owners[position+1:])
        with self.assertRaisesRegex(SchemaError,"actual source E2E tensor home"):
            broken.validate_against(self.forward,self.sequence)

    def test_layer_route_source_identity_cannot_drift(self):
        phase = build_moe_full_train_forward_ir0(self.forward,self.sequence)
        for layer in (0,1):
            refs = list(phase.source_route_trace_refs)
            refs[layer] = refs[1-layer]
            with self.subTest(layer=layer),self.assertRaisesRegex(
                    SchemaError,"identity or route digest drifted"):
                replace(phase,source_route_trace_refs=tuple(refs)).validate_against(
                    self.forward,self.sequence,
                )

    def test_layer_residual_must_really_consume_moe_combine(self):
        phase = build_moe_full_train_forward_ir0(self.forward,self.sequence)
        for layer in (0,1):
            prefix = f"T0.layer{layer}."
            nodes = tuple(replace(node,inputs=(node.inputs[0],prefix+"norm2_out"))
                          if node.id == prefix+"residual2" else node
                          for node in phase.graph.nodes)
            values = tuple(
                replace(value,consumers=())
                if value.id == prefix+"moe.combine_out" else
                replace(value,consumers=(*value.consumers,prefix+"residual2"))
                if value.id == prefix+"norm2_out" else value
                for value in phase.graph.values
            )
            data = tuple(GraphEdge(
                f"moe_data::{value.id}::{consumer}",EdgeKind.DATA,
                value.producer,consumer,value.id,
            ) for value in values if value.producer is not None
                for consumer in value.consumers)
            controls = tuple(edge for edge in phase.graph.edges
                             if edge.kind is EdgeKind.CONTROL)
            semantic = phase.graph._semantic_key()
            semantic.update(nodes=nodes,values=values,edges=(*data,*controls))
            broken = IR0.create(producer_pass=phase.graph.producer_pass,**semantic)
            broken.validate()
            with self.subTest(layer=layer),self.assertRaisesRegex(
                    SchemaError,"real hidden→router/dispatch→combine→residual DATA chain broke"):
                replace(phase,graph=broken).validate_against(
                    self.forward,self.sequence,
                )

    def test_expert_projection_source_route_digest_cannot_be_forged(self):
        phase = build_moe_full_train_forward_ir0(self.forward,self.sequence)
        for layer in (0,1):
            target = f"T0.layer{layer}.moe.expert1"
            nodes = tuple(replace(node,workload=replace(
                node.workload,source_route_trace_digest="0"*64))
                if node.id == target else node for node in phase.graph.nodes)
            semantic = phase.graph._semantic_key()
            semantic["nodes"] = nodes
            broken = IR0.create(producer_pass=phase.graph.producer_pass,**semantic)
            broken.validate()
            with self.subTest(layer=layer),self.assertRaisesRegex(
                    SchemaError,"physical source action lost original route"):
                replace(phase,graph=broken).validate_against(
                    self.forward,self.sequence,
                )

    def test_step1_requires_version1_real_parameter_reads(self):
        with self.assertRaisesRegex(SchemaError,
                                    "two-layer step0 TRAIN"):
            build_moe_full_train_forward_ir0(
                self.forward,self.sequence,step=1,
            )

    def test_both_layers_require_exact_frozen_route_operation_identity(self):
        for layer in (0,1):
            for expert in (0,1):
                source = _source_moe_operation_workload(
                    self.sequence,step=0,layer=layer,
                    kind=MoeForwardBlockKind.EXPERT,expert=expert,
                )
                self.assertEqual(source.projected_matmul_flops,384)
                self.assertEqual(source.expert_projection_parameter_bytes,192)


if __name__ == "__main__":
    unittest.main()
