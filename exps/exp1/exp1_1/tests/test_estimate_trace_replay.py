from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from estimate_trace_replay import (
    DTYPE_BYTES, HBM_TRAFFIC_MODEL, SRAM_CAPACITY, estimate_case, iter_cases,
    load_calibrations, normalize, tile_live_bytes,
)


class TraceReplayEstimateTests(unittest.TestCase):
    def test_all_cases_are_sram_safe_and_have_ordered_speedups(self) -> None:
        records = [estimate_case(case, {}) for case in iter_cases()]
        self.assertEqual(len(records), 176)
        self.assertLessEqual(tile_live_bytes(), SRAM_CAPACITY)
        for record in records:
            self.assertGreaterEqual(record["runtime_M"], record["logical_M"])
            self.assertGreaterEqual(record["runtime_N"], record["logical_N"])
            self.assertGreaterEqual(record["runtime_K"], record["logical_K"])
            self.assertGreater(record["T00_cycles"], record["T10_cycles"])
            self.assertGreater(record["T00_cycles"], record["T01_cycles"])
            self.assertGreater(record["T10_cycles"], record["T11_cycles"])
            self.assertGreater(record["T01_cycles"], record["T11_cycles"])
            self.assertEqual(record["hbm_traffic_model"], HBM_TRAFFIC_MODEL)
            self.assertGreaterEqual(record["congested_upper_bound_cycles"],
                                    record["normal_cycles"])

    def test_hbm_uses_fused_boundary_tile_replay(self) -> None:
        from estimate_trace_replay import _analytical_stages

        for case in iter_cases():
            shape = normalize(case)
            stages = _analytical_stages(case, shape)
            if case.operator == "AG_GEMM":
                expected = DTYPE_BYTES * (
                    shape["Tm"] * shape["runtime_K"] * shape["rank_N"]
                )
            else:
                expected = DTYPE_BYTES * (
                    shape["Tn"] * shape["runtime_M"] * shape["rank_K"]
                    + shape["Tm"] * shape["rank_K"] * shape["runtime_N"]
                )
            self.assertAlmostEqual(stages["hbm_bytes"], expected)

    def test_normalization_shards_before_per_die_tiling(self) -> None:
        for case in iter_cases():
            shape = normalize(case)
            self.assertEqual(shape["runtime_M"] % 128, 0)
            self.assertEqual(shape["rank_N"] % 512, 0)
            self.assertEqual(shape["rank_K"] % 256, 0)
            if case.operator == "AG_GEMM":
                self.assertEqual(shape["runtime_N"],
                                 case.mesh.dies * shape["rank_N"])
                self.assertEqual(shape["runtime_K"], shape["rank_K"])
            else:
                self.assertEqual(shape["runtime_N"], shape["rank_N"])
                self.assertEqual(shape["runtime_K"],
                                 case.mesh.dies * shape["rank_K"])


    def test_adaptive_intra_schedule_uses_mn_and_only_needed_split_k(self) -> None:
        from estimate_trace_replay import _analytical_stages

        schedules = set()
        for case in iter_cases():
            shape = normalize(case)
            stages = _analytical_stages(case, shape)
            pm = int(stages["intra_pm"])
            pn = int(stages["intra_pn"])
            pk = int(stages["intra_pk"])
            self.assertEqual(pm * pn * pk, 16)
            self.assertLessEqual(pm, shape["Tm"])
            self.assertLessEqual(pn, shape["Tn"])
            self.assertGreater(stages["spatial_utilization"], 0.0)
            self.assertLessEqual(stages["spatial_utilization"], 1.0)
            self.assertLessEqual(pk, max(1, shape["rank_K"] // 256))
            if shape["rank_K"] == 256:
                self.assertEqual(pk, 1)
            self.assertTrue(pm > 1 or pn > 1)
            schedules.add((pm, pn, pk))
        self.assertGreater(len(schedules), 1)

    def test_large_mesh_attainment_varies_with_schedule_and_workload(self) -> None:
        records = [
            estimate_case(case, {}) for case in iter_cases()
            if case.mesh.name == "6x6"
        ]
        for operator in ("AG_GEMM", "GEMM_RS"):
            selected = [
                record for record in records
                if record["operator"] == operator
            ]
            rates = [record["theory_attainment_rate"] for record in selected]
            self.assertGreater(max(rates) - min(rates), 0.005)
            self.assertGreater(
                len({round(record["core_schedule_efficiency"], 6)
                     for record in selected}),
                1,
            )
            self.assertTrue(all(
                0 < record["t11_contention_tail"] < 0.10
                for record in selected
            ))

    def test_theory_uses_the_same_padded_runtime_shape(self) -> None:
        from estimate_trace_replay import CLOCK_HZ, _analytical_stages
        from run_experiment import calculate_theory

        for case in iter_cases():
            record = estimate_case(case, {})
            shape = normalize(case)
            stages = _analytical_stages(case, shape)
            architecture_naive_cycles = (
                stages["compute_naive"] + stages["communication"]
                + stages["hbm"] + stages["local_transport"]
            )
            ideal_intra = max(
                stages["compute_ideal"],
                stages["hbm"], stages["local_transport"],
            )
            qfull = shape["Tm"] * shape["Tn"]
            architecture_upper_cycles = (
                max(ideal_intra, stages["communication"])
                + min(ideal_intra, stages["communication"]) / qfull
            )
            self.assertAlmostEqual(
                record["theory_naive_time"], architecture_naive_cycles / CLOCK_HZ
            )
            self.assertAlmostEqual(
                record["theory_time"], architecture_upper_cycles / CLOCK_HZ
            )
            self.assertAlmostEqual(
                record["theory_speedup"],
                architecture_naive_cycles / architecture_upper_cycles,
            )
            self.assertGreaterEqual(record["theory_attainment_rate"], 0.70)
            self.assertLessEqual(record["theory_attainment_rate"], 0.85)
            algorithmic = calculate_theory(
                case,
                runtime_mnk=(shape["runtime_M"], shape["runtime_N"],
                             shape["runtime_K"]),
            )
            self.assertAlmostEqual(record["algorithmic_theory_time"],
                                   algorithmic.time_seconds)

    def test_raw_trace_summary_is_accepted_and_used(self) -> None:
        payload = [{
            "mesh": "1x4", "operator": "AG_GEMM",
            "tile_completion_cycles": [100, 180, 255, 330],
            "program_done_cycles": 350,
            "congested_ii_cycles": 90,
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = load_calibrations(path)
        case = next(iter(iter_cases()))
        record = estimate_case(case, calibration)
        self.assertEqual(record["estimate_source"],
                         "cycle_accurate_trace_calibrated")
        self.assertAlmostEqual(record["congestion_factor"], 90 / 75)


if __name__ == "__main__":
    unittest.main()
