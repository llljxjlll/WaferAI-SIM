from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.workload_shape_matrix import (
    build_workload_shape_matrix,
)
from llm.frontend.wafer_frontend.schema.workload_materialization import (
    WorkloadMaterializationStatus,
)
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadCapabilityLevel,
    WorkloadFamily,
)
from llm.frontend.wafer_frontend.schema.workload_shape_matrix import (
    WORKLOAD_SHAPE_MATRIX_SCHEMA_VERSION,
    WorkloadShapeMappingMode,
    WorkloadShapeMatrix,
)


class WorkloadShapeMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = build_workload_shape_matrix()
        cls.by_key = {report.matrix_key: report for report in cls.matrix.reports}

    def test_exact_100_shapes_times_four_families(self) -> None:
        self.assertEqual(len(self.matrix.reports), 400)
        self.assertEqual(len(self.by_key), 400)
        expected = {
            (row, column, family)
            for row in range(1, 11)
            for column in range(1, 11)
            for family in WorkloadFamily
        }
        self.assertEqual(set(self.by_key), expected)
        self.assertEqual(
            len({item.request.case_id for item in self.matrix.reports}),
            400,
        )
        self.assertEqual(
            len({item.request_digest for item in self.matrix.reports}),
            400,
        )

    def test_transpose_and_non_contiguous_idle_mapping(self) -> None:
        first = self.by_key[(3, 4, WorkloadFamily.MOE_TRAINING)]
        transposed = self.by_key[(4, 3, WorkloadFamily.MOE_TRAINING)]
        for report in (first, transposed):
            self.assertIs(
                report.mapping_mode,
                WorkloadShapeMappingMode.FIXED_FOUR_RANK_IDLE_DIES,
            )
            self.assertEqual(len(report.active_die_ids), 4)
            self.assertTrue(report.idle_die_ids)
            self.assertGreater(
                max(
                    right - left
                    for left, right in zip(
                        report.active_die_ids,
                        report.active_die_ids[1:],
                    )
                ),
                1,
            )
        self.assertEqual(first.logical_work, transposed.logical_work)
        self.assertNotEqual(first.request.case_id, transposed.request.case_id)
        self.assertNotEqual(first.placement_digest, transposed.placement_digest)

        all_dies = self.by_key[(2, 4, WorkloadFamily.MOE_INFERENCE)]
        self.assertIs(
            all_dies.mapping_mode,
            WorkloadShapeMappingMode.MESH_SCALED_ALL_DIES,
        )
        self.assertEqual(all_dies.active_die_ids, tuple(range(8)))
        self.assertEqual(all_dies.idle_die_ids, ())
        self.assertGreater(all_dies.request.parallel.tp, 1)
        self.assertGreater(all_dies.request.parallel.ep, 1)

    def test_stable_matrix_id_and_canonical_order(self) -> None:
        rebuilt = WorkloadShapeMatrix.create(
            reports=tuple(reversed(self.matrix.reports))
        )
        self.assertEqual(rebuilt, self.matrix)
        self.assertEqual(
            rebuilt.schema_version,
            WORKLOAD_SHAPE_MATRIX_SCHEMA_VERSION,
        )
        with self.assertRaisesRegex(SchemaError, "unstable matrix id"):
            replace(self.matrix, id="workload_shape_matrix_forged").validate()

    def test_missing_duplicate_and_extra_cases_fail_closed(self) -> None:
        with self.assertRaisesRegex(SchemaError, "exactly once"):
            WorkloadShapeMatrix.create(reports=self.matrix.reports[:-1])
        duplicated = (*self.matrix.reports[:-1], self.matrix.reports[0])
        with self.assertRaisesRegex(SchemaError, "exactly once"):
            WorkloadShapeMatrix.create(reports=duplicated)
        with self.assertRaisesRegex(SchemaError, "exactly once"):
            WorkloadShapeMatrix.create(
                reports=(*self.matrix.reports, self.matrix.reports[0])
            )

    def test_preflight_never_promotes_runtime_capability(self) -> None:
        self.assertIs(
            self.matrix.runtime_status,
            WorkloadCapabilityLevel.NOT_MEASURED,
        )
        for report in self.matrix.reports:
            self.assertIs(
                report.materialization_status,
                WorkloadMaterializationStatus.UNSUPPORTED,
            )
            self.assertIs(
                report.runtime_status,
                WorkloadCapabilityLevel.NOT_MEASURED,
            )
            self.assertIn("family.runtime", report.unsupported_requirements)
            self.assertGreater(report.logical_work.operation_count, 0)
            self.assertGreater(report.logical_work.tensor_value_count, 0)
            self.assertGreater(report.logical_work.memory_reserved_bytes, 0)

    def test_four_family_10x10_large_mesh_canaries(self) -> None:
        canaries = [
            report
            for report in self.matrix.reports
            if report.is_large_mesh_canary
        ]
        self.assertEqual(len(canaries), 4)
        self.assertEqual(
            {item.request.family for item in canaries},
            set(WorkloadFamily),
        )
        self.assertEqual(
            tuple(item.request.case_id for item in canaries),
            self.matrix.large_mesh_canary_case_ids,
        )
        for report in canaries:
            self.assertEqual(len(report.active_die_ids), 100)
            self.assertEqual(report.idle_die_ids, ())
            self.assertEqual(report.logical_work.logical_rank_count, 100)
            self.assertEqual(len(report.placement_digest), 64)
            self.assertEqual(len(report.logical_graph_digest), 64)
            self.assertEqual(len(report.transport_requests_digest), 64)
            self.assertEqual(len(report.memory_plan_digest), 64)


if __name__ == "__main__":
    unittest.main()
