from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.lite_moe_dp4 import (
    S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID,
    S3_LITE_MOE_DP4_INFER_CASE_ID,
    S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID,
)
from llm.test.frontend.integration.lite_moe_dp4_cases import (
    LiteMoeDp4Mode,
    build_lite_moe_dp4_case,
    build_lite_moe_dp4_cases,
    build_lite_moe_dp4_infer_program_io_case,
    build_lite_moe_dp4_backward_program_io_case,
    build_lite_moe_dp4_train_forward_program_io_case,
)


class LiteMoeDp4CaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_lite_moe_dp4_cases()

    def test_three_real_pre_n6_cases_and_determinism(self) -> None:
        self.assertEqual(
            tuple(case.case_id for case in self.cases),
            (
                S3_LITE_MOE_DP4_INFER_CASE_ID,
                S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID,
                S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID,
            ),
        )
        self.assertEqual(tuple(case.mode for case in self.cases), tuple(LiteMoeDp4Mode))
        for case in self.cases:
            case.validate()
        self.assertEqual(build_lite_moe_dp4_case(LiteMoeDp4Mode.DOWN_WGRAD), self.cases[-1])

    def test_forward_topology_work_and_routes_are_exact(self) -> None:
        case = self.cases[0]
        self.assertEqual(case.topology.die_grid, (2, 2))
        self.assertEqual(case.topology.expert_home_die_ids, (0, 1, 2, 3))
        self.assertEqual(case.topology.token_source_die_ids, (0, 1, 2, 3, 0, 1, 2, 3))
        self.assertEqual(case.topology.remote_token_indices, (1, 2, 3, 4, 5, 6))
        self.assertEqual(
            (
                len(case.adapter.graph.nodes),
                len(case.adapter.graph.values),
                len(case.adapter.graph.edges),
                len(case.adapter.p2p_bindings),
            ),
            (44, 64, 42, 12),
        )
        self.assertEqual(tuple(len(die.tasks) for die in case.forward.projection.dies), (20, 26, 26, 20))
        self.assertEqual(
            (
                len(case.forward.projection.flows),
                sum(flow.bytes for flow in case.forward.projection.flows),
                case.oracle.data_packets,
                case.oracle.total_expert_gemm_flops,
            ),
            (12, 384, 24, 24576),
        )

    def test_training_forward_and_backward_exact(self) -> None:
        train = self.cases[1].train_forward
        backward = self.cases[2].backward
        assert train is not None and backward is not None
        self.assertEqual((len(train.tape_buffers), len(train.tape_copies), train.total_tape_bytes), (8, 8, 512))
        self.assertEqual(tuple(item.die_id for item in train.tape_buffers), (0, 0, 1, 1, 2, 2, 3, 3))
        self.assertEqual(
            tuple(map(len, (
                backward.remote_gradients,
                backward.token_wgrads,
                backward.expert_reduces,
                backward.sgd_stores,
            ))),
            (6, 8, 4, 4),
        )
        self.assertEqual(sum(item.bytes for item in backward.remote_gradients), 192)
        self.assertTrue(
            all(item.dtype is DType.FP32 and item.size_bytes == 2048 for item in backward.token_wgrads)
        )
        self.assertEqual(sum(item.state_store_bytes for item in backward.sgd_stores), 4096)

    def test_tamper_fails_closed(self) -> None:
        with self.assertRaisesRegex(SchemaError, "case id/mode"):
            replace(self.cases[0], case_id="case.forged").validate()
        with self.assertRaises(SchemaError):
            replace(self.cases[0].topology, token_source_die_ids=(0,) * 8).validate()
        train = self.cases[1].train_forward
        backward = self.cases[2].backward
        assert train is not None and backward is not None
        with self.assertRaises(SchemaError):
            replace(train.tape_buffers[0], size_bytes=32).validate("tape")
        with self.assertRaises(SchemaError):
            replace(backward.token_wgrads[0], dtype=DType.FP16).validate("wgrad")

    def test_infer_lower_link_actual_sha_program_io(self) -> None:
        artifact_sha = "7" * 64
        result = build_lite_moe_dp4_infer_program_io_case(artifact_sha)
        result.validate()
        self.assertEqual(result.program_io.program_artifact_sha256, artifact_sha)
        manifest = result.linked.manifest
        self.assertEqual(
            (
                len(manifest.fragments),
                len(manifest.core_streams),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
            ),
            (80, 4, 404, 24),
        )
        self.assertEqual(
            (
                len(result.program_io.blobs),
                len(result.program_io.initializations),
                len(result.program_io.output_probes),
            ),
            (16, 80, 8),
        )
        with self.assertRaises(SchemaError):
            replace(result, linked=replace(result.linked, manifest=replace(
                manifest,
                core_streams=manifest.core_streams[::-1],
            ))).validate()

    def test_train_forward_actual_sha_program_io_has_exact_tape_terminals(self) -> None:
        artifact_sha = "8" * 64
        result = build_lite_moe_dp4_train_forward_program_io_case(artifact_sha)
        result.validate()
        self.assertEqual(result.program_io.program_artifact_sha256, artifact_sha)
        probes = result.program_io.output_probes
        self.assertEqual(
            tuple(sorted(item.length_bytes for item in probes)),
            (32,) * 8 + (64,) * 8,
        )
        self.assertEqual(
            sum(item.length_bytes for item in probes),
            768,
        )
        tampered = type(result.program_io).create(
            producer_pass=result.program_io.producer_pass,
            mode=result.program_io.mode,
            source_manifest=result.linked.manifest,
            program_artifact_sha256=result.program_io.program_artifact_sha256,
            blobs=result.program_io.blobs,
            initializations=result.program_io.initializations,
            output_probes=probes[:-1],
        )
        with self.assertRaisesRegex(SchemaError, "eight combined and eight tape"):
            replace(
                result,
                program_io=tampered,
            ).validate()

    def test_backward_actual_sha_program_io_has_only_updated_hbm_probes(self) -> None:
        artifact_sha = "9" * 64
        result = build_lite_moe_dp4_backward_program_io_case(artifact_sha)
        result.validate()
        self.assertEqual(result.program_io.program_artifact_sha256, artifact_sha)
        self.assertEqual(
            tuple(item.length_bytes for item in result.program_io.output_probes),
            (1024,) * 4,
        )
        self.assertTrue(all(
            type(item.target).__name__ == "ProgramHbmTarget"
            for item in result.program_io.output_probes
        ))
        tampered = type(result.program_io).create(
            producer_pass=result.program_io.producer_pass,
            mode=result.program_io.mode,
            source_manifest=result.linked.manifest,
            program_artifact_sha256=result.program_io.program_artifact_sha256,
            blobs=result.program_io.blobs,
            initializations=result.program_io.initializations,
            output_probes=result.program_io.output_probes[:-1],
        )
        with self.assertRaisesRegex(SchemaError, "four updated HBM"):
            replace(result, program_io=tampered).validate()


if __name__ == "__main__":
    unittest.main()
