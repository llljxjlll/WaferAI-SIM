from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
from unittest.mock import patch
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_moe import compile_flexible_moe_baseline
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseFamily,
)
from llm.frontend.wafer_frontend.schema.flexible_moe import FlexibleMoeMode
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec

from flexible_mesh_release_moe import FlexibleMoeReleaseAdapter
from flexible_mesh_release_supplemental_coverage import (
    REPRESENTATIVE_RELEASE_SHAPES,
    audit_primary_supplemental_coverage,
    supplemental_coverage_catalog,
)
from flexible_moe_supplemental_profiles import (
    FlexibleMoeSupplementalTrace,
    REPRESENTATIVE_SHAPES,
    SUPPLEMENTAL_PROFILES,
    build_supplemental_spec,
    trusted_spec_builders,
)
from run_flexible_mesh_release_moe_supplemental import (
    _cases,
    _parse_args,
    _parse_shapes,
    _supplemental_report,
)
from run_flexible_mesh_release_dense import _cases as _dense_primary_cases
from run_flexible_mesh_release_meshslice import _cases as _meshslice_primary_cases
from run_flexible_mesh_release_moe import _cases as _moe_primary_cases


_ROOT = Path(__file__).resolve().parents[4]


class FlexibleMoeSupplementalTest(unittest.TestCase):
    def test_cli_defaults_bind_one_shot_release_profile(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run_flexible_mesh_release_moe_supplemental.py"],
        ):
            args = _parse_args()
        self.assertEqual(
            args.runtime_profile_version,
            "flexible-mesh-timing-v3-one-shot",
        )

    def test_catalog_and_cases_are_exact_and_independent(self) -> None:
        self.assertEqual(len(SUPPLEMENTAL_PROFILES), 6)
        self.assertEqual(len({item.digest for item in SUPPLEMENTAL_PROFILES}), 6)
        cases = _cases(Namespace(
            family="both",
            trace="all",
            shapes=REPRESENTATIVE_SHAPES,
            runtime_profile_version="flexible-mesh-timing-v2",
        ))
        self.assertEqual(REPRESENTATIVE_SHAPES, REPRESENTATIVE_RELEASE_SHAPES)
        self.assertEqual(len(REPRESENTATIVE_SHAPES), 17)
        self.assertEqual(len(cases), 102)
        self.assertEqual(len({item.id for item in cases}), 102)
        self.assertEqual(
            {item.trace_model_digest for item in cases},
            {item.digest for item in SUPPLEMENTAL_PROFILES},
        )
        self.assertEqual(
            {(item.mesh.rows, item.mesh.columns) for item in cases},
            set(REPRESENTATIVE_SHAPES),
        )
        required = _cases(Namespace(
            family="both",
            trace="required",
            shapes=REPRESENTATIVE_SHAPES,
            runtime_profile_version="flexible-mesh-timing-v2",
        ))
        self.assertEqual(len(required), 68)
        self.assertEqual(len({item.id for item in required}), 68)
        self.assertEqual(
            {item.trace_model_digest for item in required},
            {
                item.digest for item in SUPPLEMENTAL_PROFILES
                if item.trace is not FlexibleMoeSupplementalTrace.BALANCED
            },
        )

    def test_catalog_is_witnessed_by_real_primary_case_generators(self) -> None:
        runtime = Namespace(runtime_profile_version="flexible-mesh-timing-v2")
        primary = (
            *_dense_primary_cases(runtime),
            *_moe_primary_cases(Namespace(
                family="both",
                runtime_profile_version="flexible-mesh-timing-v2",
            )),
            *_meshslice_primary_cases(runtime),
        )
        self.assertEqual(len(primary), 600)
        audit_primary_supplemental_coverage(primary)
        catalog = supplemental_coverage_catalog()
        self.assertEqual(
            catalog["dense_primary_witnesses"],
            (
                ("compute_only", (1, 1)),
                ("dp_only", (2, 1)),
                ("dp_x_tp", (2, 2)),
                ("tp_only", (1, 2)),
            ),
        )
        self.assertEqual(
            catalog["meshslice_primary_witnesses"]["capacity_boundary_shape"],
            (10, 10),
        )
        self.assertEqual(
            catalog["meshslice_primary_witnesses"]
            ["optimized_auto_performance_selection"],
            "out_of_scope",
        )

        with self.assertRaisesRegex(SchemaError, "witnesses are missing"):
            audit_primary_supplemental_coverage(primary[:-1])
        with self.assertRaisesRegex(SchemaError, "duplicated"):
            audit_primary_supplemental_coverage((*primary, primary[0]))

    def test_empty_shard_report_keeps_scope_without_claiming_completion(self) -> None:
        report = _supplemental_report(
            (),
            binding=Namespace(id="binding0", digest="digest0"),
            shard_index=2,
            shard_count=4,
        )
        self.assertFalse(report["primary_completion_eligible"])
        self.assertTrue(report["timing_only"])
        self.assertEqual(report["case_count"], 0)
        self.assertEqual(report["execution_count"], 0)
        self.assertEqual(report["failed_case_ids"], [])
        self.assertEqual(report["runtime_verified_case_ids"], [])
        self.assertEqual(report["repeatability_verified_case_ids"], [])
        self.assertEqual(
            report["planned_coverage"], supplemental_coverage_catalog()
        )

    def test_trace_builders_cover_local_balanced_and_hot_empty(self) -> None:
        mesh = RectMeshSpec(2, 3)
        local = build_supplemental_spec(
            mesh, FlexibleMoeMode.INFERENCE, FlexibleMoeSupplementalTrace.ALL_LOCAL,
        )
        balanced = build_supplemental_spec(
            mesh, FlexibleMoeMode.INFERENCE, FlexibleMoeSupplementalTrace.BALANCED,
        )
        hot = build_supplemental_spec(
            mesh, FlexibleMoeMode.INFERENCE, FlexibleMoeSupplementalTrace.HOT_EMPTY,
        )
        self.assertEqual(compile_flexible_moe_baseline(local).flows, ())
        self.assertTrue(compile_flexible_moe_baseline(balanced).flows)
        self.assertEqual(balanced.trace.expert_histogram, (1,) * 6)
        self.assertEqual(hot.trace.expert_histogram, (6, 0, 0, 0, 0, 0))

    def test_adapter_rejects_untrusted_digest_and_shape_parser_is_closed(self) -> None:
        family = FlexibleMeshReleaseFamily.MOE_INFERENCE
        adapter = FlexibleMoeReleaseAdapter(
            family,
            hardware_template_json=(
                _ROOT / "llm/test/program/p5_large_hardware.json"
            ).read_text(encoding="utf-8"),
            mapping_text=(
                _ROOT / "llm/test/default/mapping.spec"
            ).read_text(encoding="utf-8"),
            trusted_spec_builders=trusted_spec_builders(family),
        )
        forged = FlexibleMeshReleaseCase.create(
            family=family,
            mesh=RectMeshSpec(1, 2),
            trace_model_digest="a" * 64,
            runtime_profile_version="flexible-mesh-timing-v2",
        )
        with self.assertRaisesRegex(SchemaError, "profile drifted"):
            adapter.materialize(forged)
        self.assertEqual(_parse_shapes("1x1,2x3"), ((1, 1), (2, 3)))
        with self.assertRaisesRegex(Exception, "representative allowlist"):
            _parse_shapes("4x4")


if __name__ == "__main__":
    unittest.main()
