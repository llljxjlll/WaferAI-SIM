from __future__ import annotations

import argparse
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

from flexible_mesh_release_dense import _validate_dense_p2p_closure
from flexible_mesh_release_profiles import release_family_profile
from run_flexible_mesh_release_dense import (
    _cases,
    _checkpoint_name,
    _parse_args,
    _parse_mesh,
    _selected_cases,
    _write_or_validate_binding,
)


class FlexibleDenseReleaseP2PObserverTest(unittest.TestCase):
    def test_no_transport_requires_absent_core_state_and_exact_global_drain(
        self,
    ) -> None:
        timing_drain = "[P5 P2P TIMING DRAIN] residual=0\n"
        _validate_dense_p2p_closure(
            timing_drain,
            (0,),
            {0: 0},
            {0: 0},
        )
        forged = (
            "[P5 P2P STATS] core=0 tx_local_completions=0 "
            "rx_local_completions=0\n" + timing_drain,
            "[P5 P2P DRAIN] core=0 residual=0\n" + timing_drain,
            "",
            "[P5 P2P TIMING DRAIN] residual=1\n",
            timing_drain + timing_drain,
        )
        for output in forged:
            with self.subTest(output=output):
                with self.assertRaises(SchemaError):
                    _validate_dense_p2p_closure(
                        output,
                        (0,),
                        {0: 0},
                        {0: 0},
                    )

    def test_transport_requires_exact_completions_and_core_drains(self) -> None:
        output = (
            "[P5 P2P STATS] core=0 tx_local_completions=1 "
            "rx_local_completions=0\n"
            "[P5 P2P STATS] core=1 tx_local_completions=0 "
            "rx_local_completions=1\n"
            "[P5 P2P DRAIN] core=0 residual=0\n"
            "[P5 P2P DRAIN] core=1 residual=0\n"
            "[P5 P2P TIMING DRAIN] residual=0\n"
        )
        _validate_dense_p2p_closure(
            output, (0, 1), {0: 1, 1: 0}, {0: 0, 1: 1},
        )
        with self.assertRaisesRegex(SchemaError, "completions"):
            _validate_dense_p2p_closure(
                output.replace("tx_local_completions=1", "tx_local_completions=2"),
                (0, 1),
                {0: 1, 1: 0},
                {0: 0, 1: 1},
            )


class FlexibleDenseReleaseCliTest(unittest.TestCase):
    def test_cli_defaults_bind_release_tools(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run_flexible_mesh_release_dense.py", "--shard-index", "0",
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

    def _binding(self, environment: str) -> FlexibleMeshReleaseBinding:
        digest = hashlib.sha256(b"tool").hexdigest()
        tools = tuple(
            FlexibleMeshReleaseTool(
                kind=kind,
                binary_path=f"/tool/{kind.value}",
                version="test",
                sha256=digest,
                allowlisted_sha256=(digest,),
            )
            for kind in FlexibleMeshReleaseToolKind
        )
        return FlexibleMeshReleaseBinding.create(
            runtime_profile_version="flexible-mesh-timing-v2",
            environment_profile_version=environment,
            tools=tools,
            hardware_config_sha256="1" * 64,
            simulation_config_sha256="2" * 64,
            mapping_config_sha256="3" * 64,
        )

    def test_cases_are_exact_100_dense_rectangles(self) -> None:
        cases = _cases(argparse.Namespace(
            runtime_profile_version="flexible-mesh-timing-v2",
        ))
        self.assertEqual(len(cases), 100)
        self.assertEqual(
            {(case.mesh.rows, case.mesh.columns) for case in cases},
            {(rows, columns) for rows in range(1, 11) for columns in range(1, 11)},
        )
        self.assertEqual(
            {case.family for case in cases},
            {FlexibleMeshReleaseFamily.DENSE_TRAIN},
        )
        for shard_count in (1, 4, 7, 32):
            selected = tuple(
                case for shard_index in range(shard_count)
                for index, case in enumerate(cases)
                if index % shard_count == shard_index
            )
            self.assertEqual(len(selected), 100)
            self.assertEqual({case.id for case in selected}, {case.id for case in cases})

    def test_repeatable_mesh_filter_is_exact_and_checkpoint_isolated(self) -> None:
        args = argparse.Namespace(
            runtime_profile_version="flexible-mesh-timing-v2",
            shard_index=0,
            shard_count=1,
            mesh=((1, 1), (2, 3), (10, 10)),
        )
        selected = _selected_cases(args)
        self.assertEqual(
            tuple((case.mesh.rows, case.mesh.columns) for case in selected),
            ((1, 1), (2, 3), (10, 10)),
        )
        filtered_name = _checkpoint_name(args)
        full_name = _checkpoint_name(argparse.Namespace(
            shard_index=0,
            shard_count=1,
            mesh=None,
        ))
        self.assertIn("mesh_filter_", filtered_name)
        self.assertNotEqual(filtered_name, full_name)

    def test_mesh_parser_rejects_noncanonical_duplicate_and_out_of_range(self) -> None:
        self.assertEqual(_parse_mesh("2x3"), (2, 3))
        for value in ("2X3", "02x3", "0x1", "11x1", "2x", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _parse_mesh(value)
        with patch.object(
            sys,
            "argv",
            [
                "run_flexible_mesh_release_dense.py",
                "--shard-index", "0",
                "--shard-count", "1",
                "--mesh", "2x3",
                "--mesh", "2x3",
            ],
        ), self.assertRaises(SystemExit):
            _parse_args()

    def test_dense_profile_binds_tree_algorithm_v3(self) -> None:
        profile = release_family_profile(FlexibleMeshReleaseFamily.DENSE_TRAIN)
        self.assertEqual(profile.profile_version, "tiny_dense_dp_rows_tp_columns/v3")
        self.assertIn(
            "dp_sync=row_major_binary_tree_reduce_broadcast_root0",
            profile.adapter_inputs,
        )
        self.assertIn(
            "release_layers=1",
            profile.adapter_inputs,
        )

    def test_root_binding_is_canonical_and_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release_binding.json"
            binding = self._binding("p5-rect-release-v1")
            _write_or_validate_binding(path, binding)
            first = path.read_bytes()
            _write_or_validate_binding(path, binding)
            self.assertEqual(path.read_bytes(), first)
            with self.assertRaisesRegex(SchemaError, "binding drifted"):
                _write_or_validate_binding(
                    path, self._binding("forged-environment"),
                )
            self.assertEqual(path.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
