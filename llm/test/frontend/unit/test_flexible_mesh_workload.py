from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.flexible_mesh_compiler import (
    FlexibleMeshCapabilityStatus,
    FlexibleMeshStageKind,
    compile_flexible_mesh_workload,
)
from llm.frontend.wafer_frontend.passes.flexible_moe import (
    build_round_robin_flexible_moe_spec,
)
from llm.frontend.wafer_frontend.policies.swizzle.meshslice_2d import (
    meshslice_execution_mode,
)
from llm.frontend.wafer_frontend.schema.flexible_moe import FlexibleMoeMode
from llm.frontend.wafer_frontend.schema.flexible_mesh_capacity import (
    FlexibleMeshCapacityProfile,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_groups import (
    FlexibleMeshGroupKind,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_workload import (
    FlexibleMeshSliceMode,
    FlexibleMeshWorkloadSpec,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)


class FlexibleMeshWorkloadTest(unittest.TestCase):
    def test_dense_infer_and_train_cover_all_one_hundred_rectangles(self) -> None:
        modes: set[FlexibleMeshSliceMode] = set()
        for rows in range(1, 11):
            for columns in range(1, 11):
                mesh = RectMeshSpec(rows, columns)
                for workload in (
                    FlexibleMeshWorkloadSpec.dense_infer(mesh),
                    FlexibleMeshWorkloadSpec.dense_train(mesh),
                ):
                    with self.subTest(
                        rows=rows,
                        columns=columns,
                        kind=workload.workload_kind.value,
                    ):
                        result = compile_flexible_mesh_workload(workload)
                        result.validate()
                        self.assertEqual(result.plan.demand.ranks, rows * columns)
                        self.assertLessEqual(
                            result.plan.demand.peak_sessions_per_core_per_wave, 3
                        )
                        self.assertIs(
                            result.capability_report.mesh_foundation,
                            FlexibleMeshCapabilityStatus.VERIFIED,
                        )
                        self.assertIs(
                            result.capability_report.baseline_plan,
                            FlexibleMeshCapabilityStatus.VERIFIED,
                        )
                        self.assertIs(
                            result.capability_report.runtime,
                            FlexibleMeshCapabilityStatus.NOT_MEASURED,
                        )
                        modes.add(result.plan.meshslice_mode)
        self.assertEqual(modes, set(FlexibleMeshSliceMode))

    def test_dense_train_has_closed_dp_tp_groups_and_ordered_step(self) -> None:
        result = compile_flexible_mesh_workload(
            FlexibleMeshWorkloadSpec.dense_train(RectMeshSpec(3, 4))
        )
        groups = result.plan.groups
        self.assertEqual(
            tuple(group.ranks for group in groups.select(FlexibleMeshGroupKind.TP)),
            ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)),
        )
        self.assertEqual(
            tuple(group.ranks for group in groups.select(FlexibleMeshGroupKind.DP)),
            ((0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11)),
        )
        self.assertEqual(
            result.plan.stages,
            (
                FlexibleMeshStageKind.PARAMETER_LOAD,
                FlexibleMeshStageKind.FORWARD,
                FlexibleMeshStageKind.LOSS,
                FlexibleMeshStageKind.BACKWARD,
                FlexibleMeshStageKind.GRADIENT_SYNC,
                FlexibleMeshStageKind.SGD,
                FlexibleMeshStageKind.STATE_STORE,
            ),
        )
        self.assertTrue(result.plan.waves)
        for wave in result.plan.waves:
            self.assertLessEqual(wave.peak_sessions_per_rank, 3)

    def test_moe_infer_and_train_cross_the_unified_one_hundred_shape_bridge(self) -> None:
        for rows in range(1, 11):
            for columns in range(1, 11):
                mesh = RectMeshSpec(rows, columns)
                inference = build_round_robin_flexible_moe_spec(
                    mesh,
                    FlexibleMoeMode.INFERENCE,
                    routing_shift=0 if mesh.rank_count == 1 else 1,
                )
                training = build_round_robin_flexible_moe_spec(
                    mesh,
                    FlexibleMoeMode.TRAIN,
                    routing_shift=0 if mesh.rank_count == 1 else 1,
                )
                for workload in (
                    FlexibleMeshWorkloadSpec.moe_infer(inference),
                    FlexibleMeshWorkloadSpec.moe_train(training),
                ):
                    with self.subTest(
                        rows=rows,
                        columns=columns,
                        kind=workload.workload_kind.value,
                    ):
                        result = compile_flexible_mesh_workload(workload)
                        result.validate()
                        self.assertEqual(
                            result.plan.demand.ranks, mesh.rank_count
                        )
                        self.assertLessEqual(
                            result.plan.demand.peak_sessions_per_core_per_wave, 3
                        )
                        self.assertEqual(
                            len(result.plan.groups.select(FlexibleMeshGroupKind.EP)),
                            1,
                        )

    def test_degenerate_meshes_emit_no_fake_transport(self) -> None:
        local = compile_flexible_mesh_workload(
            FlexibleMeshWorkloadSpec.dense_train(RectMeshSpec(1, 1))
        )
        self.assertIs(local.plan.meshslice_mode, FlexibleMeshSliceMode.LOCAL)
        self.assertEqual(local.plan.waves, ())
        self.assertEqual(local.plan.demand.peak_sessions_per_core_per_wave, 0)

        row = compile_flexible_mesh_workload(
            FlexibleMeshWorkloadSpec.dense_train(RectMeshSpec(1, 4))
        )
        self.assertIs(row.plan.meshslice_mode, FlexibleMeshSliceMode.ROW_ONLY)
        self.assertTrue(row.plan.groups.select(FlexibleMeshGroupKind.TP))
        self.assertTrue(all(wave.sends for wave in row.plan.waves))

        column = compile_flexible_mesh_workload(
            FlexibleMeshWorkloadSpec.dense_train(RectMeshSpec(4, 1))
        )
        self.assertIs(
            column.plan.meshslice_mode, FlexibleMeshSliceMode.COLUMN_ONLY
        )
        self.assertTrue(column.plan.groups.select(FlexibleMeshGroupKind.DP))
        self.assertTrue(all(wave.sends for wave in column.plan.waves))

    def test_capacity_fails_before_plan_materialization(self) -> None:
        source = FlexibleMeshWorkloadSpec.dense_train(RectMeshSpec(2, 2))
        constrained = replace(
            source,
            capacity=replace(FlexibleMeshCapacityProfile(), max_actions=1),
        )
        with self.assertRaises(SchemaError) as raised:
            compile_flexible_mesh_workload(constrained)
        self.assertEqual(raised.exception.code, "flexible_mesh_capacity_exceeded")

    def test_ids_are_deterministic_and_transpose_is_explicit(self) -> None:
        mesh = RectMeshSpec(2, 3)
        normal = compile_flexible_mesh_workload(
            FlexibleMeshWorkloadSpec.dense_train(mesh)
        )
        repeated = compile_flexible_mesh_workload(
            FlexibleMeshWorkloadSpec.dense_train(mesh)
        )
        transposed = compile_flexible_mesh_workload(
            FlexibleMeshWorkloadSpec.dense_train(mesh, transposed=True)
        )
        self.assertEqual(normal.id, repeated.id)
        self.assertNotEqual(normal.id, transposed.id)
        self.assertEqual(
            len(transposed.plan.groups.select(FlexibleMeshGroupKind.TP)), 3
        )
        self.assertEqual(
            len(transposed.plan.groups.select(FlexibleMeshGroupKind.DP)), 2
        )

    def test_v2_schema_roundtrip_and_production_meshslice_mode_agree(self) -> None:
        dense = FlexibleMeshWorkloadSpec.dense_train(RectMeshSpec(3, 2))
        moe = FlexibleMeshWorkloadSpec.moe_train(
            build_round_robin_flexible_moe_spec(
                RectMeshSpec(2, 3), FlexibleMoeMode.TRAIN
            )
        )
        for workload in (dense, moe):
            self.assertEqual(
                loads_dataclass(
                    FlexibleMeshWorkloadSpec,
                    canonical_json(workload),
                ),
                workload,
            )
        for rows in range(1, 11):
            for columns in range(1, 11):
                schema_mode = FlexibleMeshWorkloadSpec.dense_infer(
                    RectMeshSpec(rows, columns)
                ).meshslice_mode
                self.assertEqual(
                    schema_mode.value,
                    meshslice_execution_mode(rows, columns).value,
                )


if __name__ == "__main__":
    unittest.main()
