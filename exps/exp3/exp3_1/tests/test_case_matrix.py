from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest


EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import case_matrix


class CaseMatrixTests(unittest.TestCase):
    def test_frozen_logical_matrix_has_48_points(self) -> None:
        cases = case_matrix.build_logical_cases()
        self.assertEqual(len(cases), 48)
        self.assertEqual(len({case.case_id for case in cases}), 48)
        self.assertEqual(
            Counter(case.operator_family for case in cases),
            {"gemm_rs": 24, "dispatch_gemm": 24},
        )
        self.assertEqual({case.D for case in cases}, {6, 9, 36})
        self.assertEqual({case.S for case in cases}, {2304, 36864})

    def test_dense_padding_is_exactly_exp1_1_gemm_rs_padding(self) -> None:
        case = next(
            item
            for item in case_matrix.build_logical_cases()
            if item.D == 6
            and item.S == 2304
            and item.model == "LLaMA-2-7B"
            and item.stage == "down_proj"
        )
        self.assertEqual(case.logical_shape, (2304, 4096, 11008))
        self.assertEqual(case.runtime_shape, (2304, 4128, 11010))
        self.assertEqual(case.coarse_key, (2304, 4128, 1835))
        self.assertEqual(case.ring_key, (2304, 688, 1835))
        self.assertEqual(case.rc_key, (2304, 1376, 5505))
        self.assertEqual(case.padding["rule"], "exp1_1_gemm_rs_semantic_padding")
        self.assertEqual(case.padded_flops, 2 * 2304 * 4128 * 11010)

    def test_real_moe_profiles_and_two_stages(self) -> None:
        cases = case_matrix.build_logical_cases()
        mixtral_up = next(
            item
            for item in cases
            if item.D == 6
            and item.S == 2304
            and item.model == "Mixtral-8x7B"
            and item.stage == "up_gate"
        )
        self.assertEqual((mixtral_up.hidden_size, mixtral_up.intermediate_size), (4096, 14336))
        self.assertEqual((mixtral_up.experts, mixtral_up.topk), (8, 2))
        self.assertEqual(mixtral_up.coarse_key, (768, 14336, 4096))
        self.assertEqual(mixtral_up.chunk_key, (96, 14336, 4096))
        self.assertEqual(mixtral_up.gemm_execution_count, 2)

        mixtral_down = next(
            item
            for item in cases
            if item.D == 6
            and item.S == 2304
            and item.model == "Mixtral-8x7B"
            and item.stage == "down"
        )
        self.assertEqual(mixtral_down.coarse_key, (768, 4096, 14336))
        self.assertEqual(mixtral_down.chunk_key, (96, 4096, 14336))
        self.assertEqual(mixtral_down.gemm_execution_count, 1)

        deepseek = next(
            item
            for item in cases
            if item.D == 36
            and item.S == 36864
            and item.model == "DeepSeek-V3"
            and item.stage == "up_gate"
        )
        self.assertEqual((deepseek.hidden_size, deepseek.intermediate_size), (7168, 2048))
        self.assertEqual((deepseek.experts, deepseek.topk), (256, 8))
        self.assertEqual(deepseek.coarse_key, (8192, 2048, 7168))
        self.assertEqual(deepseek.chunk_key, (32, 2048, 7168))

    def test_five_groups_each_have_24_semantic_and_unique_shapes(self) -> None:
        references = case_matrix.required_shape_references()
        counts = Counter(reference.group for reference in references)
        self.assertEqual(counts, {group: 24 for group in case_matrix.SHAPE_GROUPS})
        self.assertEqual(len(references), 120)
        self.assertEqual(len({reference.shape for reference in references}), 120)
        up_gate = [ref for ref in references if "_up_gate_" in ref.case_id]
        self.assertTrue(up_gate)
        self.assertTrue(all(ref.gemm_execution_count == 2 for ref in up_gate))

    def test_artifact_emission_is_deterministic_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_paths = case_matrix.emit_artifacts(first)
            second_paths = case_matrix.emit_artifacts(second)
            for name in first_paths:
                self.assertEqual(
                    first_paths[name].read_bytes(), second_paths[name].read_bytes()
                )
            logical = json.loads(first_paths["logical_cases"].read_text())
            coverage = json.loads(first_paths["shape_coverage_report"].read_text())
        self.assertEqual(logical["case_count"], 48)
        self.assertEqual(coverage["semantic_entry_count"], 120)
        self.assertEqual(coverage["unique_shape_count"], 120)


if __name__ == "__main__":
    unittest.main()
