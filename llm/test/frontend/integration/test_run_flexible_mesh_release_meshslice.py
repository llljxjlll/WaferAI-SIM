from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
import sys
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseFamily,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec

from flexible_mesh_release_meshslice import FlexibleMeshSliceReleaseAdapter
from flexible_mesh_release_profiles import (
    release_family_profile,
    release_trace_model_digest,
)
from run_flexible_mesh_release_meshslice import (
    _cases,
    _parse_args,
    _parse_mesh,
    _selected_cases,
)
import run_flexible_mesh_release_dense as dense_cli
import run_flexible_mesh_release_meshslice as meshslice_cli
import run_flexible_mesh_release_moe as moe_cli


_ARTIFACT_SHA = "0" * 64


class FlexibleMeshSliceReleaseCliTest(unittest.TestCase):
    def test_v5_profiles_bind_piecewise_tiny_gemm_and_reject_v4_digest(self) -> None:
        old_rule = "runtime_problem=m4_per_row_n4_per_column_k2_per_rank"
        new_rule = (
            "runtime_problem=m4_per_row_n4_per_column_"
            "k_factor=(4_if_rows_times_columns_eq_1_else_2)"
        )
        families = (
            FlexibleMeshReleaseFamily.MESHSLICE_AG,
            FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK,
            FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK,
        )
        for family in families:
            with self.subTest(family=family):
                profile = release_family_profile(family)
                self.assertTrue(profile.profile_version.endswith("/v5"))
                self.assertIn(new_rule, profile.adapter_inputs)
                old_profile = replace(
                    profile,
                    profile_version=profile.profile_version[:-1] + "4",
                    adapter_inputs=tuple(
                        old_rule if item == new_rule else item
                        for item in profile.adapter_inputs
                    ),
                )
                self.assertNotEqual(old_profile.digest, profile.digest)
                adapter = FlexibleMeshSliceReleaseAdapter(
                    family, mapping_text="0:0\n"
                )
                current_case = FlexibleMeshReleaseCase.create(
                    family=family,
                    mesh=RectMeshSpec(1, 1),
                    trace_model_digest=profile.digest,
                    runtime_profile_version="flexible-mesh-timing-v3-one-shot",
                )
                self.assertEqual(
                    len(adapter.expected_hardware_sha256(current_case)), 64
                )
                old_case = FlexibleMeshReleaseCase.create(
                    family=family,
                    mesh=RectMeshSpec(1, 1),
                    trace_model_digest=old_profile.digest,
                    runtime_profile_version="flexible-mesh-timing-v3-one-shot",
                )
                with self.assertRaisesRegex(SchemaError, "profile drifted"):
                    adapter.expected_hardware_sha256(old_case)

    def test_all_family_clis_share_exact_binding_and_cache_helpers(self) -> None:
        self.assertIs(moe_cli._binding, dense_cli._binding)
        self.assertIs(meshslice_cli._binding, dense_cli._binding)
        self.assertIs(moe_cli._validate_cached, dense_cli._validate_cached)
        self.assertIs(meshslice_cli._validate_cached, dense_cli._validate_cached)
        self.assertIs(moe_cli._write_or_validate_binding, dense_cli._write_or_validate_binding)
        self.assertIs(
            meshslice_cli._write_or_validate_binding,
            dense_cli._write_or_validate_binding,
        )

    def test_cli_defaults_bind_release_tools(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run_flexible_mesh_release_meshslice.py", "--shard-index", "0",
             "--shard-count", "1"],
        ):
            args = _parse_args()
        self.assertEqual(
            args.runtime_profile_version,
            "flexible-mesh-timing-v3-one-shot",
        )
        self.assertEqual(args.tool_version, "build-release-final")
        self.assertEqual(args.family, "all")
        self.assertIsNone(args.mesh)
        for name in ("finalizer", "resolver", "npusim"):
            self.assertEqual(
                getattr(args, name).parent.name, "build-release-final"
            )


    def test_cli_matrix_covers_three_families_and_all_100_shapes(self) -> None:
        cases = _cases(Namespace(runtime_profile_version="flexible-mesh-timing-v2"))
        self.assertEqual(len(cases), 300)
        self.assertEqual(len({case.id for case in cases}), 300)
        self.assertEqual(
            {case.family for case in cases},
            {
                FlexibleMeshReleaseFamily.MESHSLICE_AG,
                FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK,
                FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK,
            },
        )
        selected_by_name = {
            name: _cases(Namespace(
                runtime_profile_version="flexible-mesh-timing-v2",
                family=name,
            ))
            for name in ("ag", "rs", "ar")
        }
        for name, family in (
            ("ag", FlexibleMeshReleaseFamily.MESHSLICE_AG),
            ("rs", FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK),
            ("ar", FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK),
        ):
            selected = selected_by_name[name]
            self.assertEqual(len(selected), 100)
            self.assertEqual(len({case.id for case in selected}), 100)
            self.assertEqual({case.family for case in selected}, {family})
        self.assertEqual(
            {case.id for selected in selected_by_name.values() for case in selected},
            {case.id for case in cases},
        )
        for shard_count in (1, 7, 32):
            selected = tuple(
                tuple(
                    case for index, case in enumerate(cases)
                    if index % shard_count == shard_index
                )
                for shard_index in range(shard_count)
            )
            flattened = tuple(case for shard in selected for case in shard)
            self.assertEqual(len(flattened), 300)
            self.assertEqual({case.id for case in flattened}, {case.id for case in cases})

    def test_repeatable_mesh_filter_selects_exact_representative_scope(self) -> None:
        meshes = (
            (1, 1), (1, 4), (4, 1), (2, 2),
            (2, 3), (3, 2), (3, 3), (10, 10),
        )
        args = Namespace(
            runtime_profile_version="flexible-mesh-timing-v3-one-shot",
            family="all",
            mesh=list(reversed(meshes)),
            shard_index=0,
            shard_count=1,
        )
        selected = _selected_cases(args)
        self.assertEqual(len(selected), 24)
        self.assertEqual(
            {(case.mesh.rows, case.mesh.columns) for case in selected},
            set(meshes),
        )
        self.assertEqual(
            {case.family for case in selected},
            {
                FlexibleMeshReleaseFamily.MESHSLICE_AG,
                FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK,
                FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK,
            },
        )
        self.assertEqual(_parse_mesh("10x1"), (10, 1))
        for invalid in ("0x1", "1x11", "01x1", "1X1", "1x1x1", "axb"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(Exception):
                    _parse_mesh(invalid)

    def test_duplicate_mesh_cli_is_rejected(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run_flexible_mesh_release_meshslice.py",
                "--shard-index", "0", "--shard-count", "1",
                "--mesh", "2x3", "--mesh", "2x3",
            ],
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_p5_physical_cores_are_verified_before_logical_normalization(self) -> None:
        family = FlexibleMeshReleaseFamily.MESHSLICE_AG
        case = FlexibleMeshReleaseCase.create(
            family=family,
            mesh=RectMeshSpec(1, 3),
            trace_model_digest=release_trace_model_digest(family),
            runtime_profile_version="flexible-mesh-timing-v2",
        )
        adapter = FlexibleMeshSliceReleaseAdapter(family, mapping_text="0:0\n")
        materialized = adapter.materialize(case)
        contract = adapter.build_program_io(materialized, _ARTIFACT_SHA)
        output = self._output(materialized.manifest, contract)
        observation = adapter.observe(
            case, materialized, contract, _ARTIFACT_SHA, output,
        )
        self.assertEqual(observation.rank_coverage, (0, 1, 2))
        self.assertEqual(observation.core_coverage, (0, 1, 2))

        physical = tuple(
            sorted(binding.runtime_core_id for binding in materialized.manifest.core_bindings)
        )
        forged = output.replace(
            f"{physical[-1]}:1,", f"{physical[-1] + 1}:1,", 1,
        )
        with self.assertRaisesRegex(SchemaError, "core coverage differs"):
            adapter.observe(case, materialized, contract, _ARTIFACT_SHA, forged)

    @staticmethod
    def _output(manifest, contract) -> str:
        cores = tuple(sorted(
            binding.runtime_core_id for binding in manifest.core_bindings
        ))
        lines = [
            (
                f"[PROGRAM_IO] phase={phase} mode=timing "
                f"initializations={len(contract.initializations)} "
                f"probes={len(contract.output_probes)} "
                f"checksum={_ARTIFACT_SHA} pass=1"
            )
            for phase in ("resolved", "applied", "verify")
        ]
        blobs = {item.id: item for item in contract.blobs}
        lines.extend(
            (
                f"[PROGRAM_IO_PROBE] id={probe.id} "
                f"expected_checksum={blobs[probe.blob_ref].sha256} "
                f"checksum={blobs[probe.blob_ref].sha256} "
                "valid=1 exact=1 pass=1"
            )
            for probe in contract.output_probes
        )
        lines.extend((
            "[SIM_RESULT] makespan_cycles=1",
            "[HOSTLANE] mismatch=0",
            "[HOSTSIG] done=" + "".join(f"{core}:1," for core in cores),
            (
                "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
                "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0 event=0"
            ),
            "[DRAIN] router_residual=0",
            "[DRAIN] d2d_link_residual=0",
            "[P5 P2P TIMING DRAIN] residual=0",
        ))
        lines.extend(
            f"[PROGRAM_MEMORY] core={core} lsu_residual=0 dte_residual=0"
            for core in cores
        )
        lines.extend((
            "[D2D_TYPE] data_out=2",
            "[D2D_LINK] 0->1 data_out=1",
            "[D2D_LINK] 1->2 data_out=1",
            "[CREDIT] data_balanced=1 ctrl_balanced=1",
            "End DONE reception",
        ))
        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
