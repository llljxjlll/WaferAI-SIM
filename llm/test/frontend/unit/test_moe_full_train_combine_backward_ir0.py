"""0x28 must consume the real layer1 score/expert tape and shared dCombined."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_ce_backward_ir0 import (
    append_moe_full_train_ce_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_head_backward_ir0 import (
    append_moe_full_train_head_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_shared_reverse_ir0 import (
    append_moe_full_train_shared_reverse_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_combine_backward_ir0 import (
    append_moe_full_train_combine_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_forward_ir0 import (
    build_moe_full_train_forward_ir0,
)
from llm.frontend.wafer_frontend.schema.action import (
    canonical_compute_operand_roles,
)
from llm.frontend.wafer_frontend.schema.ir0 import IR0, OpKind
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeCombineBackwardSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.sources = tuple(append_moe_full_train_shared_reverse_ir0(
            append_moe_full_train_head_backward_ir0(
                append_moe_full_train_ce_backward_ir0(
                    build_single_die_moe_train_physical_source(
                        Fixture, step=step)[0])))
            for step in (0, 1))

    def test_two_steps_use_exact_route_score_expert_and_shared_dcombined(self):
        graphs = tuple(append_moe_full_train_combine_backward_ir0(source)
                       for source in self.sources)
        self.assertNotEqual(graphs[0].id, graphs[1].id)
        for graph in graphs:
            nodes = {node.id: node for node in graph.nodes}
            combine = nodes["T0.layer1.moe.combine"]
            residual = nodes["backward::T0.layer1.residual2"]
            backward = nodes["backward::T0.layer1.moe.combine"]
            self.assertEqual(backward.kind, OpKind.MOE_COMBINE_BACKWARD)
            self.assertEqual(backward.inputs,
                             (*combine.inputs[-2:], combine.inputs[0],
                              residual.outputs[1]))
            self.assertEqual(backward.workload.source_forward_op_ref,
                             combine.id)
            self.assertEqual(canonical_compute_operand_roles(
                backward.kind, backward.workload, tiled=False),
                (("route_ids", "route_scores", "expert_output",
                  "dcombined_gradient"),
                 ("router_score_gradient", "expert_output_gradient")))
            self.assertEqual(len(graph.nodes), 37)
            graph.validate()

    def test_ep2_two_steps_scatter_distinct_expert_gradients(self):
        phases = (
            Fixture.phase,
            build_moe_full_train_forward_ir0(Fixture.dense,
                                             Fixture.sequence, step=1),
        )
        for step, phase in enumerate(phases):
            with self.subTest(step=step):
                source = append_moe_full_train_shared_reverse_ir0(
                    append_moe_full_train_head_backward_ir0(
                        append_moe_full_train_ce_backward_ir0(phase)))
                graph = append_moe_full_train_combine_backward_ir0(source)
                nodes = {node.id: node for node in graph.nodes}
                values = {value.id: value for value in graph.values}
                forward = nodes["T0.layer1.moe.combine"]
                residual = nodes["backward::T0.layer1.residual2"]
                backward = nodes["backward::T0.layer1.moe.combine"]
                self.assertEqual(backward.workload.expert_count, 2)
                self.assertEqual(backward.inputs,
                                 (*forward.inputs[-2:], *forward.inputs[:-2],
                                  residual.outputs[1]))
                self.assertEqual(len(backward.outputs), 3)
                self.assertEqual(canonical_compute_operand_roles(
                    backward.kind, backward.workload, tiled=False),
                    (("route_ids", "route_scores", "expert0_output",
                      "expert1_output", "dcombined_gradient"),
                     ("router_score_gradient",
                      "expert0_output_gradient",
                      "expert1_output_gradient")))
                self.assertEqual(values[backward.outputs[0]].shape, (4, 2))
                for expert in (0, 1):
                    self.assertEqual(values[backward.outputs[expert+1]].shape,
                                     values[forward.inputs[expert]].shape)
                    self.assertNotEqual(backward.outputs[expert+1],
                                        forward.inputs[expert])
                graph.validate()
                forged = replace(backward, inputs=(
                    *backward.inputs[:3], backward.inputs[2],
                    backward.inputs[4],
                ))
                semantic = graph._semantic_key()
                semantic["nodes"] = (*graph.nodes[:-1], forged)
                with self.assertRaises(SchemaError):
                    IR0.create(producer_pass=graph.producer_pass,
                               **semantic).validate()

    def test_wrong_route_digest_and_no_dcombined_fail_closed(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_combine_backward_ir0(
                append_moe_full_train_head_backward_ir0(
                    append_moe_full_train_ce_backward_ir0(
                        build_single_die_moe_train_physical_source(
                            Fixture, step=0)[0])))
        graph = append_moe_full_train_combine_backward_ir0(self.sources[0])
        backward = graph.nodes[-1]
        with self.assertRaises(SchemaError):
            semantic = graph._semantic_key()
            semantic["nodes"] = (*graph.nodes[:-1], replace(
                backward, workload=replace(
                    backward.workload,
                    source_route_trace_digest="0"*64)))
            IR0.create(producer_pass=graph.producer_pass, **semantic).validate()
        with self.assertRaises(SchemaError):
            append_moe_full_train_combine_backward_ir0(graph)


if __name__ == "__main__":
    unittest.main()
