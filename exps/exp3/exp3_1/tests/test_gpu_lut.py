from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
import warnings


EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import case_matrix
import gpu_lut


def template() -> dict[str, object]:
    return copy.deepcopy(case_matrix.build_artifacts()["required_gpu_shapes"])


def measured_document(latency_ns: float = 1234.0) -> dict[str, object]:
    document = template()
    document["gpu"] = {
        "name": "test-gpu",
        "count": 1,
        "clock_policy": "locked",
    }
    document["software"] = {
        "driver": "test-driver",
        "cuda": "test-cuda",
        "cublas_or_backend": "test-backend",
    }
    document["measurement"]["warmup_iterations"] = 5
    document["measurement"]["measured_iterations"] = 20
    for entries in document["lookup"].values():
        for entry in entries:
            entry[1] = latency_ns
    return document


class GpuLutTests(unittest.TestCase):
    def test_placeholder_mode_fills_nulls_and_marks_evidence(self) -> None:
        document = template()
        lut = gpu_lut.load_gpu_lut(document, allow_placeholder=True)
        self.assertEqual(len(lut.latencies_ns), 120)
        self.assertEqual(len(lut.placeholder_shapes), 120)
        self.assertEqual(lut.evidence, gpu_lut.PLACEHOLDER_EVIDENCE)
        shape = (768, 14336, 4096)
        self.assertEqual(lut.lookup(shape), gpu_lut.placeholder_latency_ns(shape))
        self.assertTrue(math.isfinite(lut.lookup(shape)))
        self.assertGreater(lut.lookup(shape), 0)
        self.assertEqual(
            lut.lookup_entry(shape).evidence, gpu_lut.PLACEHOLDER_EVIDENCE
        )
        self.assertEqual(
            lut.group_latency_ns("dispatch_gemm_coarse", shape, 2),
            2 * lut.lookup(shape),
        )

    def test_placeholder_model_and_digest_are_deterministic(self) -> None:
        document = template()
        first = gpu_lut.load_gpu_lut(document, allow_placeholder=True)
        second = gpu_lut.load_gpu_lut(document, allow_placeholder=True)
        self.assertEqual(first.latencies_ns, second.latencies_ns)
        self.assertEqual(first.normalized_sha256, second.normalized_sha256)
        self.assertEqual(first.metadata["placeholder_model"], dict(gpu_lut.PLACEHOLDER_ROOFLINE))

    def test_placeholder_roofline_matches_configured_single_card(self) -> None:
        self.assertEqual(
            gpu_lut.PLACEHOLDER_ROOFLINE["peak_flops_per_second"], 360.0e12
        )

    def test_materialized_numeric_placeholder_requires_explicit_opt_in(self) -> None:
        required = template()
        materialized = gpu_lut.materialize_placeholder_document(required)
        self.assertEqual(
            materialized["data_status"], gpu_lut.PLACEHOLDER_DATA_STATUS
        )
        self.assertFalse(materialized["evidence"]["is_gpu_measurement"])
        self.assertTrue(
            all(
                type(entry[1]) is float and entry[1] > 0
                for entries in materialized["lookup"].values()
                for entry in entries
            )
        )
        with self.assertRaisesRegex(gpu_lut.GpuLutError, "allow_placeholder"):
            gpu_lut.load_gpu_lut(required, materialized)
        lut = gpu_lut.load_gpu_lut(
            required, materialized, allow_placeholder=True
        )
        self.assertEqual(len(lut.placeholder_shapes), 120)
        self.assertEqual(lut.evidence, gpu_lut.PLACEHOLDER_EVIDENCE)
        self.assertEqual(
            set(lut.evidence_by_shape.values()), {gpu_lut.PLACEHOLDER_EVIDENCE}
        )

    def test_materialized_placeholder_yaml_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "placeholder.yaml"
            gpu_lut.write_placeholder_yaml(template(), output)
            lut = gpu_lut.load_gpu_lut(
                template(), output, allow_placeholder=True
            )
        self.assertEqual(len(lut.latencies_ns), 120)
        self.assertTrue(lut.uses_placeholders)

    def test_strict_loader_accepts_complete_positive_measurements(self) -> None:
        required = template()
        measured = measured_document()
        lut = gpu_lut.load_gpu_lut(required, measured)
        self.assertFalse(lut.uses_placeholders)
        self.assertEqual(lut.evidence, gpu_lut.MEASURED_EVIDENCE)
        self.assertEqual(set(lut.latencies_ns.values()), {1234.0})

    def test_strict_loader_rejects_template_nulls_and_metadata(self) -> None:
        with self.assertRaises(gpu_lut.GpuLutError):
            gpu_lut.load_gpu_lut(template())

    def test_missing_shape_reports_failure_even_in_placeholder_mode(self) -> None:
        required = template()
        incomplete = template()
        incomplete["lookup"]["dispatch_gemm_coarse"].pop()
        with self.assertRaisesRegex(gpu_lut.GpuLutError, "missing required exact shapes"):
            gpu_lut.load_gpu_lut(
                required, incomplete, allow_placeholder=True
            )

    def test_duplicate_shape_must_have_identical_latency(self) -> None:
        required = template()
        measured = measured_document()
        first = measured["lookup"]["gemm_rs_coarse"][0]
        measured["lookup"]["gemm_rs_1d_ring_c_eq_d"].append(
            [copy.deepcopy(first[0]), first[1] + 1]
        )
        with self.assertRaisesRegex(gpu_lut.GpuLutError, "conflicting latencies"):
            gpu_lut.load_gpu_lut(required, measured)

    def test_extra_shape_warns_or_is_strict_error(self) -> None:
        required = template()
        measured = measured_document()
        measured["lookup"]["gemm_rs_coarse"].append([[1, 2, 3], 4.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lut = gpu_lut.load_gpu_lut(required, measured)
        self.assertEqual(len(caught), 1)
        self.assertEqual(len(lut.warnings), 1)
        with self.assertRaisesRegex(gpu_lut.GpuLutError, "extra shapes"):
            gpu_lut.load_gpu_lut(
                required, measured, strict_extra_shapes=True
            )

    def test_invalid_latency_types_and_values_are_rejected(self) -> None:
        for invalid in ("123", 0, -1, float("inf"), float("nan"), True):
            with self.subTest(invalid=invalid):
                measured = measured_document()
                measured["lookup"]["gemm_rs_coarse"][0][1] = invalid
                with self.assertRaises(gpu_lut.GpuLutError):
                    gpu_lut.load_gpu_lut(template(), measured)


if __name__ == "__main__":
    unittest.main()
