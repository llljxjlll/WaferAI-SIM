"""Complete-shape and immutable binding gates for native release aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm.test.frontend.integration.aggregate_native_mesh_release import aggregate
from llm.test.frontend.integration import run_dense_native_mesh_matrix as dense


def _shard(root: Path, index: int, count: int, *, tool: str = "same") -> Path:
    folder = root / f"shard{index}"
    folder.mkdir()
    shapes = dense.select_shapes(dense.RELEASE_SHAPES, shard_index=index, shard_count=count)
    binding = {"shard_index": index, "shard_count": count, "shapes": shapes,
               "driver_sha256": hashlib.sha256(Path(dense.__file__).read_bytes()).hexdigest(),
               "runner_sha256": hashlib.sha256((Path(dense.__file__).parent / "run_dense_sequence_runtime_canary.py").read_bytes()).hexdigest(),
               "dram_config_sha256": hashlib.sha256((Path(dense.__file__).resolve().parents[4] / "DRAMSys/configs/hbm2-example.json").read_bytes()).hexdigest(),
               "tool_sha256": {"npusim": tool}, "simulation_sha256": "fixed"}
    path = folder / "matrix_binding.json"
    path.write_text(json.dumps(binding))
    receipt = {"binding_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "shapes": shapes, "completed_shapes": shapes, "status": "verified"}
    (folder / "matrix_receipt.json").write_text(json.dumps(receipt))
    return folder


class NativeReleaseAggregationTest(unittest.TestCase):
    def test_requires_all_100_shapes_and_reaudits_each_case(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first, second = (_shard(root, index, 2) for index in range(2))
            with patch.object(dense, "audit_cached_case", return_value={}) as audit:
                report = aggregate("dense", (first, second))
            self.assertEqual(report["physical_shapes"], 100)
            self.assertEqual(report["independent_native_executions"], 200)
            self.assertEqual(audit.call_count, 100)
            with patch.object(dense, "audit_cached_case", return_value={}):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    aggregate("dense", (first,))

    def test_rejects_duplicate_shard_and_tool_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = _shard(root, 0, 2)
            second = _shard(root, 1, 2, tool="different")
            with patch.object(dense, "audit_cached_case", return_value={}):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    aggregate("dense", (first, first))
                with self.assertRaisesRegex(ValueError, "different runner"):
                    aggregate("dense", (first, second))

    def test_rejects_missing_case_and_mutated_binding_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first, second = (_shard(root, index, 2) for index in range(2))
            with self.assertRaisesRegex(ValueError, "case_evidence.json"):
                aggregate("dense", (first, second))
            binding_path = second / "matrix_binding.json"
            binding = json.loads(binding_path.read_text())
            binding["shapes"] = binding["shapes"][:-1]
            binding_path.write_text(json.dumps(binding))
            with patch.object(dense, "audit_cached_case", return_value={}):
                with self.assertRaisesRegex(ValueError, "binding bytes drifted"):
                    aggregate("dense", (first, second))

    def test_source_drift_is_rejected_before_case_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            first = _shard(Path(raw), 0, 1)
            binding_path = first / "matrix_binding.json"
            binding = json.loads(binding_path.read_text())
            binding["runner_sha256"] = "old"
            binding_path.write_text(json.dumps(binding))
            with patch.object(dense, "audit_cached_case", return_value={}) as audit:
                with self.assertRaisesRegex(ValueError, "runner source drifted"):
                    aggregate("dense", (first,))
                audit.assert_not_called()

    def test_family_and_empty_shards_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "family must"):
            aggregate("other", ())
        with self.assertRaisesRegex(ValueError, "at least one"):
            aggregate("moe", ())


if __name__ == "__main__":
    unittest.main()
