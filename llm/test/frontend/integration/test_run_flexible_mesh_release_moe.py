from __future__ import annotations

from argparse import ArgumentTypeError, Namespace
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseBinding,
    FlexibleMeshReleaseFamily,
    FlexibleMeshReleaseTool,
    FlexibleMeshReleaseToolKind,
)

from flexible_mesh_release_profiles import release_trace_model_digest
from run_flexible_mesh_release_moe import (
    _cases,
    _checkpoint_name,
    _parse_mesh,
    _parse_args,
    _write_or_validate_binding,
)


def _binding(environment: str) -> FlexibleMeshReleaseBinding:
    digest = hashlib.sha256(b"moe-cli-tool").hexdigest()
    return FlexibleMeshReleaseBinding.create(
        runtime_profile_version="flexible-mesh-timing-v2",
        environment_profile_version=environment,
        tools=tuple(
            FlexibleMeshReleaseTool(
                kind=kind,
                binary_path=f"/tool/{kind.value}",
                version="test",
                sha256=digest,
                allowlisted_sha256=(digest,),
            )
            for kind in FlexibleMeshReleaseToolKind
        ),
        hardware_config_sha256="1" * 64,
        simulation_config_sha256="2" * 64,
        mapping_config_sha256="3" * 64,
    )


class FlexibleMoeReleaseCliTest(unittest.TestCase):
    def test_cli_defaults_bind_release_tools(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run_flexible_mesh_release_moe.py", "--shard-index", "0",
             "--shard-count", "1"],
        ):
            args = _parse_args()
        self.assertEqual(
            args.runtime_profile_version,
            "flexible-mesh-timing-v3-one-shot",
        )
        self.assertEqual(args.tool_version, "build-release-final")
        for name in ("finalizer", "resolver", "npusim"):
            self.assertEqual(getattr(args, name).parent.name, "build-release-final")

    def test_cases_are_exact_trusted_family_rectangles(self) -> None:
        for selected, expected_families, expected_count in (
            ("inference", (FlexibleMeshReleaseFamily.MOE_INFERENCE,), 100),
            ("train", (FlexibleMeshReleaseFamily.MOE_TRAIN,), 100),
            ("both", (
                FlexibleMeshReleaseFamily.MOE_INFERENCE,
                FlexibleMeshReleaseFamily.MOE_TRAIN,
            ), 200),
        ):
            cases = _cases(Namespace(
                family=selected,
                runtime_profile_version="flexible-mesh-timing-v2",
            ))
            self.assertEqual(len(cases), expected_count)
            self.assertEqual({case.family for case in cases}, set(expected_families))
            self.assertEqual(
                {case.trace_model_digest for case in cases},
                {release_trace_model_digest(family) for family in expected_families},
            )
            for shard_count in (1, 4, 7, 32):
                sharded = tuple(
                    case for shard_index in range(shard_count)
                    for index, case in enumerate(cases)
                    if index % shard_count == shard_index
                )
                self.assertEqual(len(sharded), expected_count)
                self.assertEqual({case.id for case in sharded}, {case.id for case in cases})

    def test_repeatable_mesh_filter_is_exact_and_canonical(self) -> None:
        selected = (
            (10, 10),
            (3, 3),
            (3, 2),
            (2, 3),
            (2, 2),
            (4, 1),
            (1, 4),
            (1, 1),
        )
        cases = _cases(Namespace(
            family="both",
            mesh=selected,
            runtime_profile_version="flexible-mesh-timing-v3-one-shot",
        ))
        expected_meshes = (
            (1, 1), (1, 4), (2, 2), (2, 3), (3, 2), (3, 3), (4, 1), (10, 10),
        )
        self.assertEqual(len(cases), 2 * len(expected_meshes))
        for family_index, family in enumerate((
            FlexibleMeshReleaseFamily.MOE_INFERENCE,
            FlexibleMeshReleaseFamily.MOE_TRAIN,
        )):
            family_cases = cases[
                family_index * len(expected_meshes):(family_index + 1) * len(expected_meshes)
            ]
            self.assertEqual(
                tuple(case.family for case in family_cases),
                (family,) * len(expected_meshes),
            )
            self.assertEqual(
                tuple((case.mesh.rows, case.mesh.columns) for case in family_cases),
                expected_meshes,
            )
        all_cases = _cases(Namespace(
            family="both",
            runtime_profile_version="flexible-mesh-timing-v3-one-shot",
        ))
        expected_ids = {
            case.id for case in all_cases
            if (case.mesh.rows, case.mesh.columns) in set(selected)
        }
        self.assertEqual({case.id for case in cases}, expected_ids)
        filtered_name = _checkpoint_name(Namespace(
            mesh=selected, shard_index=0, shard_count=1,
        ))
        full_name = _checkpoint_name(Namespace(
            mesh=(), shard_index=0, shard_count=1,
        ))
        self.assertIn("mesh_filter_", filtered_name)
        self.assertNotEqual(filtered_name, full_name)
        self.assertEqual(
            filtered_name,
            _checkpoint_name(Namespace(
                mesh=tuple(reversed(selected)), shard_index=0, shard_count=1,
            )),
        )

    def test_mesh_parser_rejects_duplicate_and_illegal_values(self) -> None:
        self.assertEqual(_parse_mesh("10x1"), (10, 1))
        for invalid in (
            "0x1", "1x0", "11x1", "1x11", "01x2", "1X2", "1x2x3", "axb",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ArgumentTypeError, "mesh"):
                    _parse_mesh(invalid)
        with patch.object(
            sys,
            "argv",
            [
                "run_flexible_mesh_release_moe.py",
                "--shard-index", "0",
                "--shard-count", "1",
                "--mesh", "2x3",
                "--mesh", "2x3",
            ],
        ), self.assertRaises(SystemExit):
            _parse_args()

    def test_root_binding_is_not_overwritten_on_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release_binding.json"
            binding = _binding("p5-rect-release-v1")
            _write_or_validate_binding(path, binding)
            original = path.read_bytes()
            _write_or_validate_binding(path, binding)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(SchemaError, "binding drifted"):
                _write_or_validate_binding(path, _binding("forged"))
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
