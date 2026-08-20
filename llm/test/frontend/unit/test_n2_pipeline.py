from __future__ import annotations

import unittest
from math import prod

from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.passes.pass_manager import (
    PassManager,
    PipelinePhase,
)
from llm.frontend.wafer_frontend.passes.validate_fusion import (
    FusionSemanticValidator,
)
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.passes.validate_logical_bundle import (
    DenseLogicalBundleValidator,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import (
    AttentionWorkload,
    CollectiveWorkload,
    GemmWorkload,
    OpKind,
)
from llm.frontend.wafer_frontend.schema.logical import (
    ExpandedIR0Bundle,
    IR0Template,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, from_data

from _fixtures import valid_spec


def _spec() -> ExperimentSpec:
    return from_data(ExperimentSpec, valid_spec(), path="spec")


def _compile(
    spec: ExperimentSpec,
) -> tuple[PassManager, IR0Template, ExpandedIR0Bundle]:
    manager = PassManager()
    template = manager.run_pass("build_ir0", spec, build_ir0)
    bundle = manager.run_pass("logical_expand", template, logical_expand)
    return manager, template, bundle


class N2PipelineTest(unittest.TestCase):
    def test_real_pass_pipeline_is_continuous_validated_and_reproducible(self) -> None:
        spec = _spec()
        input_digest = canonical_digest(spec)
        manager, template, bundle = _compile(spec)

        self.assertEqual(manager.snapshot.phase, PipelinePhase.LOGICAL_EXPANDED)
        self.assertEqual(len(manager.snapshot.receipts), 2)
        first, second = manager.snapshot.receipts
        self.assertEqual(first.pass_name, "build_ir0")
        self.assertEqual(second.pass_name, "logical_expand")
        self.assertEqual(first.input_digest, input_digest)
        self.assertEqual(first.output_digest, canonical_digest(template))
        self.assertEqual(second.input_digest, first.output_digest)
        self.assertEqual(second.output_digest, canonical_digest(bundle))
        self.assertEqual(canonical_digest(spec), input_digest)

        DenseLogicalBundleValidator.validate(template, bundle)
        for entry in bundle.entries:
            DenseIR0Validator.validate(entry.graph)
            FusionSemanticValidator.validate(entry.graph)

        second_spec = _spec()
        second_input_digest = canonical_digest(second_spec)
        second_manager, second_template, second_bundle = _compile(second_spec)
        self.assertEqual(canonical_digest(second_spec), second_input_digest)
        self.assertEqual(canonical_digest(second_template), canonical_digest(template))
        self.assertEqual(canonical_digest(second_bundle), canonical_digest(bundle))
        self.assertEqual(second_manager.snapshot, manager.snapshot)

    def test_tp2_l1_sample_numeric_aggregation_is_exact(self) -> None:
        _manager, _template, bundle = _compile(_spec())
        graph = bundle.entries[0].graph

        weights = tuple(
            value
            for value in graph.values
            if value.producer is None and ".w_" in value.id
        )
        self.assertEqual(len(weights), 6)
        parameters = sum(prod(value.shape) for value in weights)
        self.assertEqual(parameters, 590336)

        gemms = tuple(
            node.workload for node in graph.nodes if node.kind is OpKind.GEMM
        )
        self.assertTrue(all(isinstance(work, GemmWorkload) for work in gemms))
        logical_flops = sum(
            2 * m * n * k
            for work in gemms
            for m, n, k in (work.logical_shape,)
        )
        rank_flops = sum(
            2 * m * n * k
            for work in gemms
            for m, n, k in (work.rank_shape,)
        )
        self.assertEqual(logical_flops, 46137344)
        self.assertEqual(rank_flops, 23068672)

        collectives = tuple(
            node.workload
            for node in graph.nodes
            if node.kind is OpKind.COLLECTIVE
        )
        self.assertEqual(len(collectives), 4)
        self.assertTrue(
            all(isinstance(work, CollectiveWorkload) for work in collectives)
        )
        self.assertEqual(
            sum(work.rank_logical_payload_bytes for work in collectives),
            32768,
        )
        self.assertEqual(
            sum(work.group_logical_payload_bytes for work in collectives),
            65536,
        )

        attention = next(
            node.workload
            for node in graph.nodes
            if node.kind is OpKind.ATTENTION
        )
        self.assertIsInstance(attention, AttentionWorkload)
        self.assertEqual(attention.query_key_pairs, 528)
        global_attention_matmul_flops = (
            4
            * attention.query_key_pairs
            * attention.num_heads
            * attention.head_dim
        )
        rank_attention_matmul_flops = (
            4
            * attention.query_key_pairs
            * attention.rank_num_heads
            * attention.head_dim
        )
        self.assertEqual(global_attention_matmul_flops, 540672)
        self.assertEqual(rank_attention_matmul_flops, 270336)
        self.assertEqual(
            attention.query_key_pairs * attention.num_heads,
            2112,
        )
        self.assertEqual(
            attention.query_key_pairs * attention.rank_num_heads,
            1056,
        )


if __name__ == "__main__":
    unittest.main()
