from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema import (
    CapabilityStatus,
    LiteRuntimeCapabilityMatrix,
    S2LiteRuntimeReport,
    S3LiteRuntimeReport,
)
from llm.frontend.wafer_frontend.schema.serde import loads_dataclass


_ROOT = Path(__file__).resolve().parents[4]
_BASELINE = _ROOT / "notes/frontend/baselines/lite-runtime-v1"


class LiteRuntimeEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s2 = loads_dataclass(
            S2LiteRuntimeReport,
            (_BASELINE / "s2_runtime_report.json").read_text(encoding="utf-8"),
            path="s2_report",
        )
        cls.s3 = loads_dataclass(
            S3LiteRuntimeReport,
            (_BASELINE / "s3_runtime_report.json").read_text(encoding="utf-8"),
            path="s3_report",
        )
        cls.matrix = loads_dataclass(
            LiteRuntimeCapabilityMatrix,
            (_BASELINE / "capability_matrix.json").read_text(encoding="utf-8"),
            path="matrix",
        )

    def test_approved_scope_reloads_exactly(self) -> None:
        self.s2.validate()
        self.s3.validate()
        self.matrix.validate_against(self.s2, self.s3)
        self.assertEqual(self.s2.makespan_cycles, 6164)
        self.assertEqual(self.s3.makespan_cycles, 8342)
        self.assertIs(self.matrix.s2_status, CapabilityStatus.E2E_TIMING)
        self.assertIs(self.matrix.s3_status, CapabilityStatus.E2E_TIMING)
        self.assertIs(self.matrix.full_training_status, CapabilityStatus.UNSUPPORTED)
        self.assertIs(self.matrix.dynamic_moe_status, CapabilityStatus.UNSUPPORTED)

    def test_functional_and_scope_promotion_fail_closed(self) -> None:
        for forged in (
            replace(self.s2, compute_functional=True),
            replace(self.s2, model_functional=True),
            replace(self.s3, compute_functional=True),
            replace(self.s3, routing_functional=True),
            replace(self.s3, model_functional=True),
            replace(self.matrix, full_training_status=CapabilityStatus.E2E_TIMING),
            replace(self.matrix, dynamic_moe_status=CapabilityStatus.E2E_TIMING),
        ):
            with self.subTest(type=type(forged).__name__):
                with self.assertRaises(SchemaError):
                    forged.validate()

    def test_counts_digests_and_file_matrix_fail_closed(self) -> None:
        for forged in (
            replace(self.s2, record_count=170),
            replace(self.s3, d2d_logical_bytes=255),
            replace(self.s3, runtime_marker_digest="0" * 64),
            replace(self.matrix, s2_report_id=self.s3.id),
        ):
            with self.subTest(type=type(forged).__name__):
                with self.assertRaises(SchemaError):
                    forged.validate()
        self.assertEqual(
            tuple(sorted(path.name for path in _BASELINE.iterdir())),
            (
                "capability_matrix.json",
                "input_digests.json",
                "s2.runtime.0.log",
                "s2.runtime.1.log",
                "s2_runtime_report.json",
                "s3.runtime.0.log",
                "s3.runtime.1.log",
                "s3_runtime_report.json",
            ),
        )
        self.assertFalse(tuple(_BASELINE.rglob("*.npup")))


if __name__ == "__main__":
    unittest.main()
