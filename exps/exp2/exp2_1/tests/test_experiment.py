from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CLOCK_HZ = 500_000_000.0


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def digest_without_result(record: dict) -> str:
    value = dict(record)
    value.pop("result_digest")
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class CheckedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training = load("training_e2e.json")
        cls.amdahl = load("training_operator_amdahl.json")
        cls.prefill_amdahl = load("prefill_operator_amdahl.json")
        cls.decode = load("inference_decode_e2e.json")
        cls.prefill = load("inference_prefill_pd_breakdown.json")
        cls.request = load("inference_request_e2e.json")
        cls.skew = load("moe_skew_sensitivity.json")
        cls.shared = load("deepseek_shared_expert_sensitivity.json")
        cls.capacity = load("capacity_audit.json")
        cls.calibration = load("calibration_summary.json")

    def test_exact_primary_case_matrix(self) -> None:
        self.assertEqual(len(self.training), 12)
        self.assertEqual(len(self.decode), 12)
        self.assertEqual(len(self.prefill), 12)
        self.assertEqual(len(self.request), 12)
        models = {row["model_id"] for row in self.training}
        self.assertEqual(len(models), 6)
        self.assertEqual(
            {(row["model_id"], row["seq_len"]) for row in self.training},
            {(model, seq) for model in models for seq in (2304, 36864)},
        )
        self.assertEqual(
            {(row["model_id"], row["seq_len"]) for row in self.prefill},
            {(model, seq) for model in models for seq in (2304, 36864)},
        )
        self.assertEqual(
            {(row["model_id"], row["batch_size"]) for row in self.decode},
            {(model, batch) for model in models for batch in (64, 512)},
        )
        self.assertEqual(
            {(row["model_id"], row["batch_size"]) for row in self.request},
            {(model, batch) for model in models for batch in (64, 512)},
        )

    def test_request_composite_formula_and_source_links(self) -> None:
        decode = {
            (row["model_id"], row["batch_size"]): row for row in self.decode
        }
        prefill = {
            row["model_id"]: row
            for row in self.prefill if row["seq_len"] == 2304
        }
        for row in self.request:
            source_decode = decode[(row["model_id"], row["batch_size"])]
            source_prefill = prefill[row["model_id"]]
            self.assertEqual(row["output_tokens"], 512)
            self.assertEqual(
                row["prefill_result_digest"], source_prefill["result_digest"]
            )
            self.assertEqual(
                row["decode_result_digest"], source_decode["result_digest"]
            )
            expected_base = source_prefill["T_base_cycles"] + 512 * source_decode["T_base_cycles"]
            expected_overlap = source_prefill["T_overlap_cycles"] + 512 * source_decode["T_overlap_cycles"]
            self.assertTrue(math.isclose(row["T_base_cycles"], expected_base, rel_tol=1e-12))
            self.assertTrue(math.isclose(row["T_overlap_cycles"], expected_overlap, rel_tol=1e-12))
            self.assertTrue(math.isclose(row["speedup"], expected_base / expected_overlap, rel_tol=1e-12))
            self.assertTrue(math.isclose(row["ttft_fraction_overlap"] + row["decode_fraction_overlap"], 1.0, rel_tol=1e-12))
            expected_rate = 512 * CLOCK_HZ / expected_overlap
            self.assertTrue(math.isclose(row["request_output_tokens_per_s_overlap"], expected_rate, rel_tol=1e-12))
            self.assertIn("constant_local_tpot_extrapolated_over_512_tokens", row["limitation_tags"])
            self.assertIn("decode_kv_growth_within_generation_not_replayed", row["limitation_tags"])


    def test_decode_primary_excludes_prefill_and_uses_two_d_instances(self) -> None:
        for row in self.decode:
            self.assertEqual(row["workload"], "inference_decode_steady_step")
            self.assertEqual(row["T_base_cycles"], row["base_phase_cycles"]["decode"])
            self.assertEqual(
                row["T_overlap_cycles"], row["overlap_phase_cycles"]["decode"]
            )
            self.assertGreater(row["composite_T_base_cycles"], row["T_base_cycles"])
            expected = 2 * row["batch_size"] * CLOCK_HZ / row["T_overlap_cycles"]
            self.assertTrue(
                math.isclose(
                    row["system_decode_tokens_per_s_overlap"],
                    expected,
                    rel_tol=1e-12,
                )
            )
            self.assertEqual(
                row["TTFT_cycles_overlap"],
                row["prefill_cycles_overlap"]
                + row["handoff_cycles_overlap"]
                + row["handoff_wait_cycles_overlap"],
            )

    def test_training_formula_and_same_work(self) -> None:
        for row in self.training:
            self.assertTrue(row["same_work_invariant"])
            self.assertEqual(row["training_tokens_per_step"], 4 * row["seq_len"])
            expected = (
                row["training_tokens_per_step"] * CLOCK_HZ / row["T_overlap_cycles"]
            )
            self.assertTrue(
                math.isclose(
                    row["training_tokens_per_s_overlap"], expected, rel_tol=1e-12
                )
            )
            self.assertTrue(
                math.isclose(
                    row["speedup"],
                    row["T_base_cycles"] / row["T_overlap_cycles"],
                    rel_tol=1e-12,
                )
            )
    def test_full_train_overlap_accelerates_forward_and_backward(self) -> None:
        for row in self.training:
            self.assertEqual(row["primary_training_state"], "full_train_overlap")
            self.assertTrue(row["same_work_invariant"])
            self.assertTrue(row["training_schedule_ordering_passed"])
            self.assertLessEqual(
                row["T_full_train_overlap_cycles"],
                row["T_forward_only_overlap_cycles"],
            )
            self.assertLessEqual(
                row["T_forward_only_overlap_cycles"], row["T_base_cycles"]
            )
            self.assertTrue(
                math.isclose(
                    row["speedup_full_train"],
                    row["T_base_cycles"] / row["T_full_train_overlap_cycles"],
                    rel_tol=1e-12,
                )
            )
            expected_tps = (
                row["training_tokens_per_step"]
                * CLOCK_HZ
                / row["T_full_train_overlap_cycles"]
            )
            self.assertTrue(
                math.isclose(
                    row["training_tokens_per_s_full_train"],
                    expected_tps,
                    rel_tol=1e-12,
                )
            )
            shorter = row[
                "full_train_phase_strictly_shorter_than_forward_only"
            ]
            self.assertFalse(shorter["forward"])
            self.assertTrue(shorter["backward"])
            self.assertTrue(shorter["wgrad"])
            self.assertEqual(
                row["full_train_phase_cycles"]["forward"],
                row["overlap_phase_cycles"]["forward"],
            )
            for phase in ("forward", "backward", "wgrad"):
                self.assertLess(
                    row["full_train_phase_cycles"][phase],
                    row["base_phase_cycles"][phase],
                    msg=(
                        f"{row['case_id']} full_train did not accelerate {phase}"
                    ),
                )
            self.assertLess(
                row["full_train_phase_cycles"]["backward"],
                row["overlap_phase_cycles"]["backward"],
            )
            self.assertLess(
                row["full_train_phase_cycles"]["wgrad"],
                row["overlap_phase_cycles"]["wgrad"],
            )

    def test_operator_amdahl_mapping_and_unweighted_means(self) -> None:
        self.assertEqual(len(self.amdahl), 12)
        training = {row["case_id"]: row for row in self.training}
        self.assertEqual(
            {row["training_case_id"] for row in self.amdahl}, set(training)
        )
        expected_counts = {
            "mixtral_8x7b": 6,
            "deepseek_v3": 4,
        }
        exps_root = ROOT.parents[1]
        for row in self.amdahl:
            selected = row["selected_operators"]
            expected_count = expected_counts.get(row["model_id"], 4)
            self.assertEqual(row["operator_speedup_count"], expected_count)
            self.assertEqual(len(selected), expected_count)
            expected_mean = sum(item["speedup"] for item in selected) / len(selected)
            self.assertTrue(
                math.isclose(
                    row["operator_speedup_mean"], expected_mean, rel_tol=1e-12
                )
            )
            source_training = training[row["training_case_id"]]
            self.assertEqual(
                row["training_result_digest"], source_training["result_digest"]
            )
            self.assertTrue(
                math.isclose(
                    row["system_speedup_full_train"],
                    source_training["speedup_full_train"],
                    rel_tol=1e-12,
                )
            )
            self.assertGreater(
                row["operator_speedup_mean"], row["system_speedup_full_train"]
            )
            for item in selected:
                source_path = exps_root / item["source_path"]
                self.assertEqual(
                    item["source_file_digest"],
                    hashlib.sha256(source_path.read_bytes()).hexdigest(),
                )
                if item["source_experiment"] == "exp1-1":
                    self.assertEqual(item["source_mesh"], "3x3")
                    self.assertEqual(item["target_mesh"], "3x3")
                    self.assertEqual(item["mesh_match"], "exact")
                    expected_source_seq = {2304: 2048, 36864: 32768}[
                        row["seq_len"]
                    ]
                    self.assertEqual(item["source_seq_len"], expected_source_seq)
                    self.assertEqual(
                        item["target_over_source_seq_ratio"], 1.125
                    )
                else:
                    self.assertEqual(item["source_experiment"], "exp1-2")
                    self.assertEqual(item["source_profile"], "H128")
                    self.assertEqual(item["source_placement"], "noncompact")
                    self.assertEqual(item["seq_match"], "exact")
                    self.assertEqual(item["source_seq_len"], row["seq_len"])

    def test_prefill_operator_mapping_and_unweighted_means(self) -> None:
        self.assertEqual(len(self.prefill_amdahl), 12)
        prefill = {row["case_id"]: row for row in self.prefill}
        self.assertEqual(
            {row["prefill_case_id"] for row in self.prefill_amdahl}, set(prefill)
        )
        expected_counts = {"mixtral_8x7b": 6, "deepseek_v3": 4}
        exps_root = ROOT.parents[1]
        for row in self.prefill_amdahl:
            selected = row["selected_operators"]
            expected_count = expected_counts.get(row["model_id"], 4)
            self.assertEqual(row["operator_speedup_count"], expected_count)
            self.assertEqual(len(selected), expected_count)
            expected_mean = sum(item["speedup"] for item in selected) / len(selected)
            self.assertTrue(math.isclose(
                row["operator_speedup_mean"], expected_mean, rel_tol=1e-12
            ))
            source_prefill = prefill[row["prefill_case_id"]]
            self.assertEqual(
                row["prefill_result_digest"], source_prefill["result_digest"]
            )
            expected_system = (
                source_prefill["prefill_cycles_base"]
                / source_prefill["prefill_cycles_overlap"]
            )
            self.assertTrue(math.isclose(
                row["system_prefill_speedup"], expected_system, rel_tol=1e-12
            ))
            self.assertGreater(
                row["operator_speedup_mean"], row["system_prefill_speedup"]
            )
            for item in selected:
                source_path = exps_root / item["source_path"]
                self.assertEqual(
                    item["source_file_digest"],
                    hashlib.sha256(source_path.read_bytes()).hexdigest(),
                )
                if item["source_experiment"] == "exp1-1":
                    self.assertEqual(item["source_mesh"], "2x3")
                    self.assertEqual(item["target_mesh"], "2x3")
                    self.assertEqual(item["mesh_match"], "exact")
                    expected_source_seq = {2304: 2048, 36864: 32768}[
                        row["seq_len"]
                    ]
                    self.assertEqual(item["source_seq_len"], expected_source_seq)
                    self.assertEqual(item["target_seq_len"], row["seq_len"])
                    self.assertEqual(item["seq_match"], "nearest_same_mesh_proxy")
                    self.assertEqual(item["target_over_source_seq_ratio"], 1.125)
                else:
                    self.assertEqual(item["source_experiment"], "exp1-2")
                    self.assertEqual(item["source_profile"], "H128")
                    self.assertEqual(item["source_placement"], "compact")
                    self.assertEqual(item["seq_match"], "exact")
                    self.assertEqual(item["source_seq_len"], row["seq_len"])
                    self.assertEqual(
                        item["placement_match"],
                        "compact_four_rank_proxy_for_contiguous_2x3_instance",
                    )


    def test_capacity_is_a_hard_status_not_a_footnote(self) -> None:
        self.assertEqual(self.capacity["training_case_count"], 12)
        self.assertEqual(self.capacity["inference_case_count"], 24)
        self.assertTrue(
            all(
                row["capacity_status"] == "capacity_infeasible_projection"
                for row in self.training + self.decode
            )
        )
        feasible_prefill = {
            row["model_id"]
            for row in self.prefill
            if row["capacity_status"] == "capacity_feasible"
        }
        self.assertTrue(
            all(row["capacity_status"] == "capacity_infeasible_projection"
                for row in self.request)
        )
        self.assertEqual(feasible_prefill, {"llama2_7b", "llama3_8b"})

    def test_calibration_gate_and_structural_evidence_provenance(self) -> None:
        self.assertFalse(self.calibration["target_unit_closure_passed"])
        self.assertFalse(self.calibration["direct_validation_passed"])
        self.assertTrue(self.calibration["repeatability_passed"])
        self.assertEqual(
            self.calibration["publish_status"],
            "analytical_only_target_binding_unclosed",
        )
        signatures = self.calibration["evidence_signatures"]
        self.assertTrue(any(value.startswith("stage4_pds:") for value in signatures))
        self.assertTrue(
            any(value.startswith("flexible_moe_train_1x2:") for value in signatures)
        )
        for row in self.training + self.decode + self.prefill + self.request:
            self.assertFalse(row["target_unit_closure"])
            self.assertEqual(row["evidence_signatures"], signatures)
            self.assertIn(
                "target_hardware_unit_closure_failed", row["limitation_tags"]
            )

    def test_mla_and_moe_skew_are_explicit(self) -> None:
        mla = [
            row
            for row in self.training + self.decode + self.prefill + self.request
            if row["model_id"] == "deepseek_v3"
        ]
        self.assertTrue(mla)
        self.assertTrue(
            all(row["estimate_source"] == "analytical_only_mla" for row in mla)
        )
        self.assertEqual(len(self.skew), 24)
        self.assertEqual(
            {row["routing_skew"] for row in self.skew}, {1.0, 1.25, 1.5}
        )
        self.assertEqual(
            {row["model_id"] for row in self.skew},
            {"mixtral_8x7b", "deepseek_v3"},
        )
        self.assertEqual(
            {row["sensitivity_workload"] for row in self.skew},
            {"training", "inference_decode"},
        )

    def test_model_limitations_survive_capacity_merge(self) -> None:
        required = {
            "analytical_vector_rate_prior",
            "representative_collective_route_abstraction",
            "local_noc_die_level_abstraction",
            "hbm_ingress_anchor_abstraction",
        }
        for row in self.training + self.decode:
            self.assertTrue(required.issubset(row["limitation_tags"]))
            self.assertIn("capacity_infeasible_projection", row["limitation_tags"])
            self.assertFalse(row["include_shared_experts"])
        for row in self.prefill:
            if row["capacity_status"] == "capacity_feasible":
                self.assertNotIn(
                    "capacity_infeasible_projection", row["limitation_tags"]
                )

    def test_deepseek_shared_expert_is_separate_sensitivity(self) -> None:
        self.assertEqual(len(self.shared), 8)
        self.assertEqual(
            {row["shared_expert_mode"] for row in self.shared},
            {"routed_only", "routed_plus_shared"},
        )
        self.assertEqual(
            {row["include_shared_experts"] for row in self.shared}, {False, True}
        )
        for workload in ("training", "inference_decode"):
            rows = [
                row for row in self.shared
                if row["sensitivity_workload"] == workload
            ]
            key = "seq_len" if workload == "training" else "batch_size"
            for condition in {row[key] for row in rows}:
                pair = {
                    row["include_shared_experts"]: row
                    for row in rows if row[key] == condition
                }
                self.assertGreaterEqual(
                    pair[True]["T_overlap_cycles"],
                    pair[False]["T_overlap_cycles"],
                )

    def test_sensitivity_training_schema_and_current_tool_digests(self) -> None:
        for row in self.skew:
            if row["sensitivity_workload"] == "training":
                for key in (
                    "workload",
                    "model_family",
                    "attention_type",
                    "training_tokens_per_step",
                    "training_tokens_per_s_overlap",
                ):
                    self.assertIn(key, row)
        expected = {
            "tool_digest": ROOT / "run_experiment.py",
            "e2e_replay_tool_digest": ROOT / "e2e_replay.py",
            "capacity_tool_digest": ROOT / "capacity_model.py",
            "hardware_digest": ROOT / "configs" / "target_hardware.json",
            "simulation_digest": ROOT / "configs" / "target_simulation.json",
        }
        for field, path in expected.items():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertTrue(
                all(row[field] == actual
                    for row in self.training + self.decode + self.request)
            )

    def test_result_digests_are_self_independent_and_valid(self) -> None:
        rows = (
            self.training + self.amdahl + self.prefill_amdahl + self.decode
            + self.prefill + self.request + self.skew + self.shared
        )
        self.assertEqual(len({row["case_id"] for row in rows}), len(rows))
        for row in rows:
            self.assertEqual(row["result_digest"], digest_without_result(row))

    def test_five_svg_figures_expose_required_encodings(self) -> None:
        expected = {
            "training_e2e.svg": 24,
            "inference_request_e2e.svg": 24,
            "inference_decode_e2e.svg": 24,
            "inference_prefill_e2e.svg": 48,
            "inference_prefill_pd_breakdown.svg": 72,
        }
        for name, minimum_bars in expected.items():
            text = (ROOT / "figures" / name).read_text(encoding="utf-8")
            self.assertIn('data-role="one-x-line"', text)
            self.assertIn("capacity-hatch", text)
            self.assertIn("mla-hatch", text)
            self.assertIn("normalized to figure maximum", text)
            if name in {
                "training_e2e.svg",
                "inference_prefill_e2e.svg",
                "inference_prefill_pd_breakdown.svg",
            }:
                self.assertNotIn("axis maximum 1.1", text)
                self.assertNotIn(">1.10</text>", text)
                self.assertNotIn(">1.1</text>", text)
                for label in ("0", "0.2", "0.4", "0.6", "0.8", "1"):
                    self.assertIn(f">{label}</text>", text)
                left_one = re.search(
                    r'<text x="80" y="([0-9.]+)"[^>]*>1</text>', text
                )
                right_max = re.search(
                    r'<text x="1503" y="([0-9.]+)"[^>]*>4.5×</text>', text
                )
                self.assertIsNotNone(left_one)
                self.assertIsNotNone(right_max)
                self.assertEqual(left_one.group(1), right_max.group(1))
            else:
                self.assertIn("axis maximum 1.1", text)
                self.assertIn(">1.10</text>", text)
            self.assertIn('stroke="#B5B5B5"', text)
            self.assertNotIn('stroke="#D7D7D7"', text)
            self.assertIn(">normalized bar height</text>", text)
            self.assertGreaterEqual(text.count('data-role="'), minimum_bars)
            bar_lines = [
                line for line in text.splitlines()
                if line.startswith('<rect data-role=') and '-texture"' not in line
            ]
            self.assertTrue(bar_lines)
            self.assertTrue(
                all('stroke="#222" stroke-width="3"' in line for line in bar_lines)
            )
            if name == "training_e2e.svg":
                self.assertNotIn('data-role="forward-only-bar"', text)
                self.assertNotIn('data-role="overlap-bar"', text)
                self.assertNotIn('data-role="uncertainty-band"', text)
                self.assertEqual(text.count('data-role="system-speedup-line"'), 1)
                self.assertEqual(text.count('data-role="operator-speedup-line"'), 1)
                self.assertEqual(text.count('data-role="base-bar"'), 12)
                self.assertEqual(text.count('data-role="full-train-bar"'), 12)
                self.assertEqual(text.count('data-role="system-speedup-point"'), 12)
                self.assertEqual(text.count('data-role="operator-speedup-point"'), 12)
                self.assertIn('stroke="#A23B72" stroke-width="10.125"', text)
                self.assertIn('r="13.02" fill="white" stroke="#A23B72" stroke-width="7.65"', text)
                self.assertIn('stroke="#4B3F99" stroke-width="10.125"', text)
                self.assertIn('r="13.02" fill="white" stroke="#4B3F99" stroke-width="7.65"', text)
                self.assertIn(">4.5×</text>", text)
                self.assertIn('fill="#75635B"', text)
                self.assertIn('fill="#D9822B"', text)
                self.assertNotIn('fill="#2F9E73"', text)
                base_geometry = []
                full_geometry = []
                for line in text.splitlines():
                    if 'data-role="base-bar"' in line:
                        base_geometry.append(self._rect_geometry(line))
                    if 'data-role="full-train-bar"' in line:
                        full_geometry.append(self._rect_geometry(line))
                for base, full in zip(base_geometry, full_geometry):
                    self.assertAlmostEqual(base[0] + base[1], full[0], delta=0.011)
                    self.assertGreater(base[1], 35.0)
                base_x = [item[0] for item in base_geometry]
                for group_start in range(0, 10, 2):
                    within_model = base_x[group_start + 1] - base_x[group_start]
                    between_models = base_x[group_start + 2] - base_x[group_start + 1]
                    self.assertGreater(between_models, 1.35 * within_model)
            elif name == "inference_prefill_e2e.svg":
                self.assertNotIn('data-role="uncertainty-band"', text)
                self.assertEqual(text.count('data-role="system-speedup-line"'), 1)
                self.assertEqual(text.count('data-role="operator-speedup-line"'), 1)
                self.assertEqual(text.count('data-role="base-bar"'), 12)
                self.assertEqual(text.count('data-role="overlap-bar"'), 12)
                self.assertEqual(text.count('data-role="system-speedup-point"'), 12)
                self.assertEqual(text.count('data-role="operator-speedup-point"'), 12)
                self.assertIn('stroke="#C23B22" stroke-width="10.125"', text)
                self.assertIn(
                    'r="13.02" fill="white" stroke="#C23B22" stroke-width="7.65"',
                    text,
                )
                self.assertIn('stroke="#2F4B7C" stroke-width="10.125"', text)
                self.assertIn(
                    'r="13.02" fill="white" stroke="#2F4B7C" stroke-width="7.65"',
                    text,
                )
                self.assertIn(">4.5×</text>", text)
                self.assertIn('fill="#6B5B95"', text)
                self.assertIn('fill="#D9A441"', text)
                base_geometry = []
                overlap_geometry = []
                for line in text.splitlines():
                    if 'data-role="base-bar"' in line:
                        base_geometry.append(self._rect_geometry(line))
                    if 'data-role="overlap-bar"' in line:
                        overlap_geometry.append(self._rect_geometry(line))
                for base, overlap in zip(base_geometry, overlap_geometry):
                    self.assertAlmostEqual(
                        base[0] + base[1], overlap[0], delta=0.011
                    )
                    self.assertGreater(base[1], 35.0)
                base_x = [item[0] for item in base_geometry]
                for group_start in range(0, 10, 2):
                    within_model = base_x[group_start + 1] - base_x[group_start]
                    between_models = base_x[group_start + 2] - base_x[group_start + 1]
                    self.assertGreater(between_models, 1.35 * within_model)
            else:
                self.assertIn('data-role="uncertainty-band"', text)
                self.assertIn('data-role="speedup-line"', text)
                self.assertIn('stroke="#007C83" stroke-width="4.5"', text)
                point_lines = [
                    line for line in text.splitlines()
                    if 'data-role="speedup-point"' in line
                ]
                self.assertTrue(
                    all('r="6.3"' in line and 'stroke-width="3.3"' in line
                        for line in point_lines)
                )

    @staticmethod
    def _rect_geometry(line: str) -> tuple[float, float]:
        attributes = {}
        for token in line.replace("<rect ", "").replace("/>", "").split():
            if "=" in token:
                key, value = token.split("=", 1)
                attributes[key] = value.strip('"')
        return float(attributes["x"]), float(attributes["width"])


if __name__ == "__main__":
    unittest.main()

