from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError

from flexible_mesh_cases import (
    all_flexible_mesh_cases,
    FLEXIBLE_MESH_REPRESENTATIVE_CASES,
    FlexibleMeshCase,
    validate_flexible_mesh_case_matrix,
)
from flexible_mesh_runtime_provider import FlexibleMeshRuntimeProvider
from flexible_mesh_runtime_report import (
    FlexibleMeshCapabilityStatus,
    FlexibleMeshRuntimeReport,
)


class FlexibleMeshCaseMatrixTest(unittest.TestCase):
    def test_representative_matrix_covers_required_shape_classes(self) -> None:
        self.assertEqual(
            tuple(
                (case.mesh.rows, case.mesh.columns)
                for case in FLEXIBLE_MESH_REPRESENTATIVE_CASES
            ),
            ((1, 1), (1, 10), (10, 1), (2, 3), (3, 2), (3, 3), (10, 10)),
        )
        by_shape = {
            (case.mesh.rows, case.mesh.columns): case
            for case in FLEXIBLE_MESH_REPRESENTATIVE_CASES
        }
        self.assertTrue(by_shape[(1, 10)].is_linear)
        self.assertTrue(by_shape[(10, 1)].is_linear)
        self.assertTrue(by_shape[(1, 1)].meshslice_eligible_shape)
        self.assertTrue(by_shape[(2, 3)].meshslice_eligible_shape)
        self.assertTrue(by_shape[(3, 2)].meshslice_eligible_shape)
        self.assertFalse(by_shape[(3, 3)].mesh.has_hamiltonian_cycle)

    def test_full_envelope_is_generated_without_golden_files(self) -> None:
        cases = all_flexible_mesh_cases()
        validate_flexible_mesh_case_matrix(cases)
        self.assertEqual(len(cases), 100)
        self.assertEqual(cases[0].name, "mesh_1x1")
        self.assertEqual(cases[-1].name, "mesh_10x10")

    def test_duplicate_and_drifted_cases_fail_closed(self) -> None:
        case = FlexibleMeshCase.create(2, 3, "rectangular")
        with self.assertRaisesRegex(SchemaError, "unique"):
            validate_flexible_mesh_case_matrix((case, case))
        with self.assertRaisesRegex(SchemaError, "derived"):
            replace(case, name="mesh_3x2").validate()


class FlexibleMeshProviderAndReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provider = FlexibleMeshRuntimeProvider()
        cls.prepared = cls.provider.prepare_matrix(FLEXIBLE_MESH_REPRESENTATIVE_CASES)

    def test_provider_uses_real_fabric_and_bounded_topology(self) -> None:
        by_shape = {
            (item.case.mesh.rows, item.case.mesh.columns): item
            for item in self.prepared
        }
        maximum = by_shape[(10, 10)]
        self.assertEqual(
            (
                maximum.die_count,
                maximum.directed_link_count,
                maximum.expected_ordered_pair_route_count,
                maximum.max_hop_count,
            ),
            (100, 360, 9900, 18),
        )
        self.assertEqual(
            self.provider.prepare_matrix(FLEXIBLE_MESH_REPRESENTATIVE_CASES),
            self.prepared,
        )
        left = by_shape[(2, 3)]
        right = by_shape[(3, 2)]
        self.assertEqual(left.directed_link_count, right.directed_link_count)
        self.assertEqual(left.max_hop_count, right.max_hop_count)
        self.assertNotEqual(left.row_rank_orders, right.row_rank_orders)

    def test_report_is_deterministic_and_does_not_overclaim_runtime(self) -> None:
        report = FlexibleMeshRuntimeReport.create(self.prepared)
        self.assertEqual(report, FlexibleMeshRuntimeReport.create(self.prepared))
        self.assertEqual(report.measurement_scope, "fixture_foundation_only")
        self.assertTrue(
            all(
                status is FlexibleMeshCapabilityStatus.NOT_MEASURED
                for _, status in report.completion_states
            )
        )
        self.assertTrue(
            all(
                status is FlexibleMeshCapabilityStatus.OUT_OF_SCOPE
                for _, status in report.f7_capabilities
            )
        )
        linear = next(row for row in report.cases if row.case_name == "mesh_1x10")
        rectangle = next(row for row in report.cases if row.case_name == "mesh_2x3")
        self.assertIs(
            linear.meshslice_execution,
            FlexibleMeshCapabilityStatus.NOT_MEASURED,
        )
        self.assertEqual(linear.meshslice_mode, "row_only")
        self.assertIs(
            rectangle.meshslice_execution,
            FlexibleMeshCapabilityStatus.NOT_MEASURED,
        )

    def test_report_tampering_fails_closed(self) -> None:
        report = FlexibleMeshRuntimeReport.create(self.prepared)
        forged = replace(
            report,
            completion_states=(
                ("mesh_foundation_complete", FlexibleMeshCapabilityStatus.VERIFIED),
            )
            + report.completion_states[1:],
        )
        with self.assertRaisesRegex(SchemaError, "cannot claim"):
            forged.validate()


if __name__ == "__main__":
    unittest.main()
