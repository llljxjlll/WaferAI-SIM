"""Regression for the strict MoE full-model two-fresh matrix audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

from llm.test.frontend.integration.run_moe_full_model_native_mesh_matrix import (
    M1_SHAPES,
    RELEASE_SHAPES,
    _require_matching_matrix_binding,
    audit_cached_case,
    audit_fresh,
    compare_fresh,
    matrix_binding,
    select_shapes,
)


_DIGEST = "a" * 64
_RUNNER_KEY = (
    "llm/test/frontend/integration/"
    "run_moe_full_model_sequence_runtime_canary.py"
)
_ROOT = Path(__file__).resolve().parents[4]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path, shape: str = "1x2") -> Path:
    rows, columns = (int(item) for item in shape.split("x"))
    ranks = rows * columns
    directory = root / "fresh"
    directory.mkdir(parents=True)
    artifact_names = []
    for index in range(3):
        manifest_path = directory / f"segment_{index}.linked.json"
        artifact_path = directory / f"segment_{index}.npup"
        report_path = directory / f"segment_{index}.finalizer.json"
        sidecar_path = directory / f"segment_{index}.program_io.json"
        resolver_path = directory / f"segment_{index}.resolver.stdout.txt"
        manifest = {"id": f"manifest_{index}"}
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        )
        artifact_path.write_bytes(f"npup-{index}".encode())
        report_path.write_text(json.dumps({
            "artifact_sha256": _sha(artifact_path),
            "linked_manifest_id": manifest["id"],
            "linked_manifest_digest": _sha(manifest_path),
        }))
        sidecar = {
            "id": f"program_io_{index}",
            "program_artifact_sha256": _sha(artifact_path),
            "source_linked_manifest_id": manifest["id"],
            "source_linked_manifest_digest": _sha(manifest_path),
            "initializations": [{}],
            "output_probes": [{}],
        }
        sidecar_path.write_text(json.dumps(sidecar))
        resolver_path.write_text(
            f"ProgramIo resolved id={sidecar['id']} "
            "initializations=1 probes=1\n"
        )
        artifact_names.extend((
            manifest_path.name,
            artifact_path.name,
            report_path.name,
            sidecar_path.name,
            resolver_path.name,
        ))

    hardware = {
        "x": 2,
        "y": 2,
        "die": {"x": columns, "y": rows},
        "memory_system": {
            "hbm_stacks": [
                {"compute_die_id": rank} for rank in range(ranks)
            ],
            "address_policy": {
                "home_ranges": [{"die_id": rank} for rank in range(ranks)]
            },
        },
    }
    (directory / "hardware.json").write_text(json.dumps(hardware))
    (directory / "mapping.spec").write_text("0:0\n")
    artifact_names.extend(("hardware.json", "mapping.spec"))

    if ranks == 1:
        expected_links = []
        link_lines = []
        request_hops = packet_hops = 0
    elif shape == "1x2":
        expected_links = [
            {
                "source_die": 0,
                "destination_die": 1,
                "direction": "E",
                "request_hops": 1,
                "packet_hops": 1,
            },
            {
                "source_die": 1,
                "destination_die": 0,
                "direction": "W",
                "request_hops": 1,
                "packet_hops": 1,
            },
        ]
        link_lines = [
            "[D2D_LINK] idx=0 die0->die1 dir=E "
            "req_in=1 req_out=1 ack_in=2 ack_out=2 data_in=1 data_out=1.",
            "[D2D_LINK] idx=1 die1->die0 dir=W "
            "req_in=1 req_out=1 ack_in=2 ack_out=2 data_in=1 data_out=1.",
        ]
        request_hops = packet_hops = 2
    else:
        raise ValueError("unit fixture only supports 1x1 and 1x2")

    runtime_lines = [
        *(
            f"[PROGRAM_MEMORY] core={rank * 4} lsu_issued=1 "
            "lsu_completed=1 x=0 lsu_residual=0 dte_residual=0"
            for rank in range(ranks)
        ),
        *(
            f"[P5 P2P DRAIN] core={rank * 4} residual=0"
            for rank in range(ranks) if ranks > 1
        ),
        "[DENSE_SEQUENCE_SEGMENT] index=0 status=done final=0",
        "[DENSE_SEQUENCE_SEGMENT] index=1 status=done final=0",
        "[DENSE_SEQUENCE_SEGMENT] index=2 status=done final=1",
        "[DENSE_SEQUENCE_PROGRAM_IO] index=0 probes=1 pass=1",
        "[DENSE_SEQUENCE_PROGRAM_IO] index=1 probes=1 pass=1",
        "[DENSE_SEQUENCE_PROGRAM_IO] index=2 probes=1 pass=1",
        "[DENSE_SEQUENCE_DRAIN] segments=3 one_shot=1",
        *(
            f"[DENSE_SEQUENCE_KV] index={index} bytes={size} "
            f"digest={_DIGEST} pass=1"
            for index, size in enumerate((128, 192, 256))
        ),
        "[SIM_RESULT] makespan_cycles=10",
        f"[D2D_TYPE] request_in={request_hops} "
        f"request_out={request_hops} ack_in={2 * request_hops} "
        f"ack_out={2 * request_hops} data_in={packet_hops} "
        f"data_out={packet_hops}",
        f"[D2D_DATA] in_pkts={packet_hops} out_pkts={packet_hops}",
        *link_lines,
    ]
    runtime_path = directory / "npusim.stdout.txt"
    runtime_path.write_text("\n".join(runtime_lines) + "\n")

    binding = {
        "schema_version": "moe-full-model-source-tool-binding-v1",
        "source_tool_at_entry": {
            "imported_python_sha256": {
                _RUNNER_KEY: _sha(_ROOT / _RUNNER_KEY),
            },
            "tool_sha256": {
                **{
                    name: _DIGEST
                    for name in ("finalizer", "resolver", "simulation")
                },
                "npusim": _sha(Path(sys.executable).resolve()),
            },
        },
        "npusim_execution": {
            "executable": str(Path(sys.executable).resolve()),
            "cwd": str(Path(sys.executable).resolve().parent),
        },
        "additional_imported_python_sha256": {},
        "artifact_files_sha256": {
            name: _sha(directory / name) for name in sorted(artifact_names)
        },
        "sequence_digest": _DIGEST,
        "workload_case_id": "fixture",
        "source_request_sha256": _DIGEST,
        "kv_boundaries_bytes": [128, 192, 256],
        "executable_core_bindings": [
            {"rank": rank, "runtime_core_id": rank * 4}
            for rank in range(ranks)
        ],
        "expected_d2d_links": expected_links,
        "runtime_status": "verified",
    }
    binding_path = directory / "source_tool_binding.json"
    binding_path.write_text(json.dumps(binding, sort_keys=True))
    receipt = {
        "schema_version": "moe-full-model-runtime-receipt-v1",
        "mesh": shape,
        "active_die_ids": list(range(ranks)),
        "compiled_core_die_ids": [list(range(ranks))] * 3,
        "frontend_core_grid": [2, 2],
        "native_core_grid": [2, 2],
        "frontend_cores_per_die": 4,
        "native_cores_per_die": 4,
        "workload_case_id": "fixture",
        "source_request_sha256": _DIGEST,
        "sequence_digest": _DIGEST,
        "source_tool_binding_sha256": _sha(binding_path),
        "runtime_log_sha256": _sha(runtime_path),
        "runtime_status": "verified",
    }
    (directory / "compiled_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True)
    )
    return directory


def _rebind_runtime(directory: Path) -> None:
    receipt_path = directory / "compiled_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["runtime_log_sha256"] = _sha(directory / "npusim.stdout.txt")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))


def _rebind_source_tool(directory: Path) -> None:
    receipt_path = directory / "compiled_receipt.json"
    binding_path = directory / "source_tool_binding.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["source_tool_binding_sha256"] = _sha(binding_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))


class MoeFullModelNativeMatrixAuditTest(unittest.TestCase):
    def test_reopens_artifacts_and_exact_physical_cores(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            observed = audit_fresh(_fixture(Path(raw)), "1x2")
            self.assertEqual(observed["core_bindings"], ((0, 0), (1, 4)))
            self.assertEqual(len(observed["artifact_files_sha256"]), 17)
            self.assertEqual(len(observed["resolver_evidence"]), 3)

    def test_single_die_requires_zero_d2d(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            observed = audit_fresh(_fixture(Path(raw), "1x1"), "1x1")
            self.assertEqual(observed["d2d_links"], ())
            self.assertEqual(observed["core_bindings"], ((0, 0),))

    def test_artifact_byte_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = _fixture(Path(raw))
            (directory / "segment_2.npup").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact bytes drifted"):
                audit_fresh(directory, "1x2")

    def test_current_worktree_source_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = _fixture(Path(raw))
            binding_path = directory / "source_tool_binding.json"
            binding = json.loads(binding_path.read_text())
            binding["source_tool_at_entry"]["imported_python_sha256"][
                _RUNNER_KEY
            ] = _DIGEST
            binding_path.write_text(json.dumps(binding, sort_keys=True))
            _rebind_source_tool(directory)
            with self.assertRaisesRegex(ValueError, "current worktree"):
                audit_fresh(directory, "1x2")

    def test_wrong_native_core_stride_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = _fixture(Path(raw))
            receipt_path = directory / "compiled_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["native_core_grid"] = [4, 4]
            receipt["native_cores_per_die"] = 16
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "core grids disagree"):
                audit_fresh(directory, "1x2")

    def test_misdirected_link_is_rejected_after_log_rebind(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = _fixture(Path(raw))
            runtime = directory / "npusim.stdout.txt"
            runtime.write_text(
                runtime.read_text().replace(
                    "die0->die1 dir=E", "die0->die1 dir=W"
                )
            )
            _rebind_runtime(directory)
            with self.assertRaisesRegex(ValueError, "misdirected"):
                audit_fresh(directory, "1x2")

    def test_fresh_comparison_allows_metrics_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            first = audit_fresh(_fixture(Path(raw)), "1x2")
            second = dict(first, total_wall_seconds=2.0)
            compare_fresh(first, second)
            with self.assertRaisesRegex(ValueError, "independent"):
                compare_fresh(first, dict(second, makespan_cycles=11))

    def test_canonical_m1_and_all_release_shards_are_exact(self) -> None:
        self.assertEqual(
            M1_SHAPES,
            ("1x1", "1x4", "4x1", "2x2",
             "2x3", "3x2", "3x3", "10x10"),
        )
        self.assertEqual(len(RELEASE_SHAPES), 100)
        shards = tuple(
            select_shapes(
                RELEASE_SHAPES, shard_index=index, shard_count=7
            )
            for index in range(7)
        )
        self.assertEqual(
            set().union(*(set(shard) for shard in shards)),
            set(RELEASE_SHAPES),
        )
        self.assertEqual(sum(map(len, shards)), 100)
        with self.assertRaisesRegex(ValueError, "shard_index"):
            select_shapes(
                RELEASE_SHAPES, shard_index=7, shard_count=7
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            select_shapes(
                ("1x1", "1x1"), shard_index=0, shard_count=1
            )

    def test_matrix_binding_survives_exact_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = {}
            for name in ("finalizer", "resolver", "npusim", "simulation"):
                paths[name] = root / name
                paths[name].write_bytes(name.encode())
            args = SimpleNamespace(
                **paths, shard_index=0, shard_count=1,
            )
            binding = matrix_binding(args, ("1x1", "1x3"))
            persisted = root / "matrix_binding.json"
            persisted.write_text(json.dumps(binding))
            self.assertEqual(json.loads(json.dumps(binding)), binding)
            self.assertIsInstance(binding["shapes"], list)
            _require_matching_matrix_binding(persisted, binding)
            forged = dict(binding, shapes=("1x1", "1x3"))
            with self.assertRaisesRegex(ValueError, "JSON-native"):
                _require_matching_matrix_binding(persisted, forged)

    def test_resume_reopens_both_fresh_trees_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            case = root / "case"
            case.mkdir()
            for fresh in (0, 1):
                source = _fixture(root / f"source{fresh}")
                shutil.copytree(source, case / f"fresh{fresh}")
            observed = [
                audit_fresh(case / f"fresh{fresh}", "1x2")
                for fresh in (0, 1)
            ]
            (case / "case_evidence.json").write_text(json.dumps({
                "shape": "1x2",
                "fresh": observed,
            }, indent=2, sort_keys=True))
            audit_cached_case(case, "1x2")
            report_path = case / "case_evidence.json"
            report = json.loads(report_path.read_text())
            report["fresh"][1]["makespan_cycles"] = 11
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "evidence"):
                audit_cached_case(case, "1x2")


if __name__ == "__main__":
    unittest.main()
