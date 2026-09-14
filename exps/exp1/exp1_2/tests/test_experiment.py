from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from plot_results import build_chart, read_rows, write_svg
from run_experiment import (
    DIE_CORES,
    DTYPE_BYTES,
    EP_SIZE,
    HARDWARE_PROFILES,
    SRAM_CAPACITY_BYTES,
    TILE_K,
    TILE_M,
    TILE_N,
    estimate_case,
    iter_cases,
    normalize,
    tile_live_bytes,
)


BASE = Path(__file__).resolve().parents[1]
PROFILES = ("H128", "H2000")


class Exp12Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = list(iter_cases())
        cls.arch = {
            (profile, case.case_id): estimate_case(
                case, hardware_profile=profile, include_hbm=True,
                network_scenario="isolated_group",
            )
            for profile in PROFILES for case in cls.cases
        }

    def test_case_matrix_profiles_assignment_and_work(self) -> None:
        self.assertEqual(len(self.cases), 16)
        self.assertEqual(len({case.case_id for case in self.cases}), 16)
        self.assertEqual(HARDWARE_PROFILES["H128"].die_tensor_flops, 128e12)
        self.assertEqual(HARDWARE_PROFILES["H2000"].die_tensor_flops, 2000e12)
        self.assertEqual(
            HARDWARE_PROFILES["H128"].die_vector_flops,
            HARDWARE_PROFILES["H2000"].die_vector_flops,
        )
        for case in self.cases:
            shape = normalize(case)
            matrix = shape["assignment_matrix"]
            logical = case.seq_len * case.model.top_k
            self.assertEqual(sum(map(sum, matrix)), logical)
            self.assertEqual(
                shape["per_expert_logical_M"],
                [sum(matrix[s][e] for s in range(EP_SIZE))
                 for e in range(case.model.expert_count)],
            )
            for logical_m, runtime_m in zip(
                shape["per_expert_logical_M"],
                shape["per_expert_runtime_M"], strict=True,
            ):
                self.assertGreaterEqual(runtime_m, logical_m)
                self.assertEqual(runtime_m % TILE_M, 0)
            record = self.arch[("H128", case.case_id)]
            factor = 4 if case.operator == "DISPATCH_GEMM" else 2
            expected_flops = (
                factor * case.seq_len * case.model.top_k
                * case.model.hidden_size * case.model.expert_intermediate_size
            )
            self.assertEqual(record["logical_flops"], expected_flops)
            if factor == 4:
                self.assertEqual(record["logical_gate_flops"], expected_flops // 2)
                self.assertEqual(record["logical_up_flops"], expected_flops // 2)
            else:
                self.assertEqual(record["logical_down_flops"], expected_flops)
            homes = shape["expert_home_rank"]
            expected_remote = sum(
                matrix[s][e] for s in range(EP_SIZE)
                for e, home in enumerate(homes) if s != home
            )
            self.assertEqual(record["remote_assignment_count"], expected_remote)
            self.assertEqual(
                record["local_assignment_count"] + expected_remote, logical
            )
            self.assertEqual(
                record["remote_payload_bytes"],
                expected_remote * case.model.hidden_size * DTYPE_BYTES,
            )
            self.assertEqual(
                record["physical_model_manifest"]["conservation"][
                    "remote_flow_assignments"
                ],
                expected_remote,
            )

    def test_grouped_schedule_noc_k_loop_sram_and_four_states(self) -> None:
        for record in self.arch.values():
            self.assertLessEqual(
                tile_live_bytes(operator=record["operator"]), SRAM_CAPACITY_BYTES
            )
            active = (
                record["intra_pe"] * record["intra_pm"]
                * record["intra_pn"] * record["intra_pk"]
            )
            baseline_active = (
                record["baseline_intra_pe"] * record["baseline_intra_pm"]
                * record["baseline_intra_pn"] * record["baseline_intra_pk"]
            )
            self.assertEqual(active, record["active_cores"])
            self.assertEqual(baseline_active, record["baseline_active_cores"])
            self.assertLessEqual(active, DIE_CORES)
            self.assertLessEqual(baseline_active, DIE_CORES)
            self.assertEqual(record["underfilled_cores"], DIE_CORES - active)
            local_ids = record["per_rank_expert_ids"][record["bottleneck_rank"]]
            expert_tm = record["per_expert_Tm"]
            tn, tk = record["Tn"], record["Tk"]
            pm, pn, pk = record["intra_pm"], record["intra_pn"], record["intra_pk"]
            a_bytes = b_bytes = reduction_bytes = 0
            for expert in local_ids:
                output_tiles = expert_tm[expert] * tn
                a_bytes += (
                    output_tiles * tk * (min(pn, tn) - 1)
                    * TILE_M * TILE_K * DTYPE_BYTES
                )
                b_bytes += (
                    output_tiles * tk * (min(pm, expert_tm[expert]) - 1)
                    * TILE_K * TILE_N * DTYPE_BYTES
                )
                reduction_bytes += (
                    output_tiles * (min(pk, tk) - 1)
                    * TILE_M * TILE_N * 4
                )
            self.assertEqual(record["a_broadcast_bytes"], a_bytes)
            self.assertEqual(record["b_broadcast_bytes"], b_bytes)
            self.assertEqual(record["reduction_bytes"], reduction_bytes)
            self.assertEqual(
                record["local_transport_bytes"],
                a_bytes + b_bytes + reduction_bytes,
            )
            for state in ("T00_cycles", "T10_cycles", "T01_cycles", "T11_cycles"):
                self.assertGreater(record[state], 0)
            self.assertAlmostEqual(
                record["actual_speedup"],
                record["T00_cycles"] / record["T11_cycles"],
            )
            self.assertLessEqual(
                record["architecture_lower_bound_cycles"], record["T11_cycles"]
            )
            self.assertEqual(
                record["estimate_source"], "analytical_physical_resource_replay"
            )
            self.assertFalse(record["simulator_unit_closure"])
            self.assertEqual(
                record["calibration_status"], "cycle_accurate_calibration_pending"
            )

    def test_four_stacks_capacity_and_unique_address_owner(self) -> None:
        expected_attachments = [[1, 0], [4, 0], [1, 5], [4, 5]]
        for case in self.cases:
            record = self.arch[("H128", case.case_id)]
            stacks = record["hbm_stacks"]
            self.assertEqual(
                [item["attachment"] for item in stacks], expected_attachments
            )
            ranges = [item["address_range"] for item in stacks]
            self.assertTrue(all(
                left[1] <= right[0]
                for left, right in zip(ranges, ranges[1:])
            ))
            self.assertEqual(
                sum(item["read_bytes"] for item in stacks),
                record["hbm_total_read_bytes"],
            )
            self.assertTrue(record["hbm_address_owner_unique"])
            self.assertTrue(record["hbm_capacity_feasible"])
            bindings = record["physical_model_manifest"][
                "weight_allocation"
            ]["bindings"]
            self.assertEqual(len(bindings), case.model.expert_count * 3)
            self.assertEqual(
                {item["matrix"] for item in bindings}, {"gate", "up", "down"}
            )
            intervals = []
            for binding in bindings:
                stack_range = ranges[binding["stack_id"]]
                begin, end = (
                    binding["address"],
                    binding["address"] + binding["size_bytes"],
                )
                self.assertGreaterEqual(begin, stack_range[0])
                self.assertLessEqual(end, stack_range[1])
                intervals.append((begin, end))
            intervals.sort()
            self.assertTrue(all(
                left[1] <= right[0]
                for left, right in zip(intervals, intervals[1:])
            ))

    def test_profiles_and_hbm_free_are_paired(self) -> None:
        for case in self.cases:
            low = self.arch[("H128", case.case_id)]
            high = self.arch[("H2000", case.case_id)]
            for key in (
                "workload_digest", "physical_model_digest",
                "runtime_flops", "remote_payload_bytes", "hbm_total_read_bytes",
            ):
                self.assertEqual(low[key], high[key])
            self.assertNotEqual(low["hardware_digest"], high["hardware_digest"])
            self.assertAlmostEqual(
                low["compute_ideal_cycles"] / high["compute_ideal_cycles"],
                2000 / 128,
            )
            self.assertEqual(low["vector_rate_status"], "provisional_vector_rate")
            ablation = estimate_case(
                case, hardware_profile="H2000", include_hbm=False,
                network_scenario="isolated_group",
            )
            self.assertEqual(high["workload_digest"], ablation["workload_digest"])
            self.assertEqual(
                high["hbm_total_read_bytes"], ablation["hbm_total_read_bytes"]
            )
            self.assertGreater(ablation["hbm_cycles"], 0)
            self.assertEqual(ablation["modeled_hbm_cycles"], 0)
            self.assertFalse(ablation["hbm_included"])
            self.assertEqual(
                ablation["estimate_source"],
                "analytical_physical_resource_replay_hbm_free",
            )
            self.assertLessEqual(ablation["T11_cycles"], high["T11_cycles"])

    def test_checked_results_loaded_sensitivity_and_grouped_plot(self) -> None:
        for profile in ("h128", "h2000"):
            for mode in ("architecture", "hbm_free_compute_comm"):
                rows = read_rows(
                    BASE / "results" / profile / mode / "results.csv"
                )
                self.assertEqual(len(rows), 16)
                self.assertEqual(
                    {row["tensor_profile"].lower() for row in rows}, {profile}
                )
                for operator in ("DISPATCH_GEMM", "GEMM_COMBINE"):
                    chart = build_chart(rows, operator)
                    self.assertEqual(len(chart.points), 8)
                    with tempfile.TemporaryDirectory() as directory:
                        output = Path(directory) / "chart.svg"
                        write_svg(chart, output)
                        root = ET.parse(output).getroot()
                    ns = "{http://www.w3.org/2000/svg}"
                    rects = list(root.iter(f"{ns}rect"))
                    self.assertEqual(
                        sum(x.get("data-role") == "T00-bar" for x in rects), 8
                    )
                    self.assertEqual(
                        sum(x.get("data-role") == "T11-bar" for x in rects), 8
                    )
                synthetic = copy.deepcopy(rows)
                synthetic[0]["optimized_tflops"] = str(
                    float(synthetic[0]["naive_tflops"]) / 2
                )
                synthetic[0]["actual_speedup"] = "0.5"
                chart = build_chart(synthetic, synthetic[0]["operator"])
                changed = next(
                    x for x in chart.points
                    if x.case_id == synthetic[0]["profile_case_id"]
                )
                self.assertLess(changed.optimized_tflops, changed.naive_tflops)
                self.assertEqual(changed.actual_speedup, 0.5)

        csv.field_size_limit(16 * 1024 * 1024)
        loaded_rows = []
        for profile in ("h128", "h2000"):
            path = (
                BASE / "results" / "loaded_groups" / profile
                / "architecture" / "results.csv"
            )
            with path.open(newline="", encoding="utf-8") as stream:
                loaded_rows.extend(csv.DictReader(stream))
        self.assertEqual(len(loaded_rows), 8)
        self.assertEqual({row["model"] for row in loaded_rows}, {"DeepSeek-V3"})
        self.assertEqual({int(row["seq_len"]) for row in loaded_rows}, {36864})
        self.assertEqual(
            {row["network_scenario"] for row in loaded_rows}, {"loaded_groups"}
        )
        self.assertEqual(
            {int(row["scenario_group_count"]) for row in loaded_rows}, {9}
        )


class CalibrationEvidenceTest(unittest.TestCase):
    def test_cycle_smoke_is_structural_prior_only(self) -> None:
        evidence = json.loads(
            (BASE / "calibration" / "source_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        encoded = json.dumps(evidence, ensure_ascii=False)
        self.assertIn("2727", encoded)
        self.assertIn("3566", encoded)
        self.assertIn("target_unit_closure", encoded)
        self.assertIn("false", encoded.lower())


if __name__ == "__main__":
    unittest.main()
