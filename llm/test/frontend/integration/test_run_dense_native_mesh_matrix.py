"""Regression for native physical Die placement and fresh-run release auditing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from llm.test.frontend.integration.run_dense_native_mesh_matrix import (
    audit_fresh,
    compare_fresh,
)


_DIGEST = "a" * 64


def _fixture(root: Path) -> Path:
    directory = root / "fresh"
    directory.mkdir()
    binding = {
        "runtime_status": "verified",
        "sequence_digest": _DIGEST,
        "kv_boundaries_bytes": [10, 12, 14],
    }
    for key, suffix in (("linked_manifest_sha256", "linked.json"), ("npup_sha256", "npup"), ("program_io_sha256", "program_io.json")):
        digests = []
        for index in range(3):
            file = directory / f"segment_{index}.{suffix}"
            file.write_bytes(f"{suffix}-{index}".encode())
            digests.append(hashlib.sha256(file.read_bytes()).hexdigest())
        binding[key] = digests
    hardware_text = json.dumps({"x": 2, "y": 2, "die": {"x": 4, "y": 1}})
    (directory / "hardware.json").write_text(hardware_text)
    binding["hardware_sha256"] = hashlib.sha256(hardware_text.encode()).hexdigest()
    (directory / "source_tool_binding.json").write_text(json.dumps(binding))
    receipt = {
        "runtime_status": "verified",
        "mesh": "1x4",
        "sequence_digest": _DIGEST,
        "active_die_ids": [0, 1, 2, 3],
        "compiled_core_die_ids": [[0, 1, 2, 3]] * 3,
        "frontend_core_grid": [2, 2],
        "native_core_grid": [2, 2],
        "frontend_cores_per_die": 4,
        "native_cores_per_die": 4,
        "source_request_sha256": _DIGEST,
        "workload_case_id": "fixture",
    }
    (directory / "compiled_receipt.json").write_text(json.dumps(receipt))
    lines = [
        *(f"[PROGRAM_MEMORY] core={core} lsu_issued=1" for core in (0, 4, 8, 12)),
        *(f"[DENSE_SEQUENCE_KV] index={index} bytes={size} digest={_DIGEST} pass=1" for index, size in enumerate((10, 12, 14))),
        "[SIM_RESULT] makespan_cycles=10",
        "[D2D_DATA] in_pkts=3 out_pkts=3",
        *(f"[D2D_LINK] idx={index} die{source}->die{target} dir={direction} req_in=1 data_in=1 data_out=1" for index, (source, target, direction) in enumerate(((0, 1, "E"), (1, 0, "W"), (1, 2, "E"), (2, 1, "W"), (2, 3, "E"), (3, 2, "W")))),
    ]
    (directory / "npusim.stdout.txt").write_text("\n".join(lines))
    return directory


class NativeMatrixAuditTest(unittest.TestCase):
    def test_exact_physical_die_coverage_and_two_fresh_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = _fixture(Path(raw))
            first = audit_fresh(directory, "1x4")
            self.assertEqual(first["physical_die_ids"], (0, 1, 2, 3))
            second = dict(first, phase_wall_seconds={"native_npusim": 2.0})
            compare_fresh(first, second)
            with self.assertRaisesRegex(ValueError, "independent"):
                compare_fresh(first, dict(second, makespan_cycles=11))

    def test_old_stride16_profile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = _fixture(Path(raw))
            receipt = json.loads((directory / "compiled_receipt.json").read_text())
            receipt["native_core_grid"] = [4, 4]
            receipt["native_cores_per_die"] = 16
            (directory / "compiled_receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "core grids disagree"):
                audit_fresh(directory, "1x4")

    def test_logical_four_dies_collapsing_to_one_physical_die_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = _fixture(Path(raw))
            path = directory / "npusim.stdout.txt"
            text = path.read_text()
            for old, new in (("core=4", "core=1"), ("core=8", "core=2"), ("core=12", "core=3")):
                text = text.replace(old, new)
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "actual NpuSim physical Dies"):
                audit_fresh(directory, "1x4")

    def test_nonadjacent_native_link_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = _fixture(Path(raw))
            path = directory / "npusim.stdout.txt"
            path.write_text(path.read_text().replace("die0->die1 dir=E", "die0->die2 dir=E"))
            with self.assertRaisesRegex(ValueError, "nonadjacent"):
                audit_fresh(directory, "1x4")

    def test_actual_native_vertical_north_south_convention(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = _fixture(Path(raw))
            receipt = json.loads((directory / "compiled_receipt.json").read_text())
            receipt["mesh"] = "4x1"
            (directory / "compiled_receipt.json").write_text(json.dumps(receipt))
            hardware = json.loads((directory / "hardware.json").read_text())
            hardware["die"] = {"x": 1, "y": 4}
            hardware_text = json.dumps(hardware)
            (directory / "hardware.json").write_text(hardware_text)
            binding = json.loads((directory / "source_tool_binding.json").read_text())
            binding["hardware_sha256"] = hashlib.sha256(hardware_text.encode()).hexdigest()
            (directory / "source_tool_binding.json").write_text(json.dumps(binding))
            path = directory / "npusim.stdout.txt"
            path.write_text(path.read_text().replace(" dir=E ", " dir=N ").replace(" dir=W ", " dir=S "))
            observation = audit_fresh(directory, "4x1")
            self.assertEqual(observation["physical_die_ids"], (0, 1, 2, 3))

    def test_hardware_die_grid_wrong_despite_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = _fixture(Path(raw))
            hardware = json.loads((directory / "hardware.json").read_text())
            hardware["die"] = {"x": 2, "y": 2}
            hardware_text = json.dumps(hardware)
            (directory / "hardware.json").write_text(hardware_text)
            binding = json.loads((directory / "source_tool_binding.json").read_text())
            binding["hardware_sha256"] = hashlib.sha256(hardware_text.encode()).hexdigest()
            (directory / "source_tool_binding.json").write_text(json.dumps(binding))
            with self.assertRaisesRegex(ValueError, "physical Die mesh"):
                audit_fresh(directory, "1x4")

    def test_artifact_byte_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = _fixture(Path(raw))
            (directory / "segment_2.npup").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "npup_sha256 bytes drifted"):
                audit_fresh(directory, "1x4")


if __name__ == "__main__":
    unittest.main()
