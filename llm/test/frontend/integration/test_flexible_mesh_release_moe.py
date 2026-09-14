from __future__ import annotations

from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseFamily,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec

from flexible_mesh_release_moe import FlexibleMoeReleaseAdapter
from flexible_mesh_release_profiles import release_trace_model_digest


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "llm/test/program/p5_large_hardware.json"


class FlexibleMeshReleaseMoeAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = FlexibleMeshReleaseCase.create(
            family=FlexibleMeshReleaseFamily.MOE_INFERENCE,
            mesh=RectMeshSpec(1, 1),
            trace_model_digest=release_trace_model_digest(
                FlexibleMeshReleaseFamily.MOE_INFERENCE
            ),
            runtime_profile_version="flexible-moe-timing-v2",
        )
        cls.adapter = FlexibleMoeReleaseAdapter(
            FlexibleMeshReleaseFamily.MOE_INFERENCE,
            hardware_template_json=_HARDWARE.read_text(encoding="utf-8"),
            mapping_text="0:0\n",
        )
        cls.materialized = cls.adapter.materialize(cls.case)
        cls.artifact_sha = "a" * 64
        cls.contract = cls.adapter.build_program_io(
            cls.materialized, cls.artifact_sha,
        )

    def _output(self) -> str:
        init_count = len(self.contract.initializations)
        probe_count = len(self.contract.output_probes)
        lines = [
            f"[PROGRAM_IO] phase=resolved mode=timing initializations={init_count} "
            f"probes={probe_count} checksum={self.artifact_sha} pass=1",
            f"[PROGRAM_IO] phase=applied mode=timing initializations={init_count} "
            f"probes={probe_count} checksum={self.artifact_sha} pass=1",
        ]
        blobs = {blob.id: blob for blob in self.contract.blobs}
        for probe in self.contract.output_probes:
            checksum = blobs[probe.blob_ref].sha256
            lines.append(
                f"[PROGRAM_IO_PROBE] id={probe.id} expected_checksum={checksum} "
                f"checksum={checksum} valid=1 exact=1 pass=1"
            )
        lines.extend((
            f"[PROGRAM_IO] phase=verify mode=timing initializations={init_count} "
            f"probes={probe_count} checksum={'b' * 64} pass=1",
            "[SIM_RESULT] makespan_cycles=373",
            "[PROGRAM_MEMORY] core=0 lsu_issued=3 lsu_completed=3 "
            "lsu_residual=0 dte_residual=0",
            "[P5 P2P TIMING DRAIN] residual=0",
            "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 gather=0 "
            "reduce_rx=0 endpoints=0 dte_tokens=0 event=0",
            "[DRAIN] router_residual=0",
            "[DRAIN] d2d_link_residual=0",
            "[CREDIT] data_balanced=1 ctrl_balanced=1",
            "[HOSTLANE] done_total=1 ack_total=2 mismatch=0 per_lane_done=1,0",
            "[HOSTSIG] done=0:1, ack=0:0:2",
        ))
        return "\n".join(lines)

    def _observe(self, output: str):
        return self.adapter.observe(
            self.case,
            self.materialized,
            self.contract,
            self.artifact_sha,
            output,
        )

    def test_exact_timing_markers_close(self) -> None:
        observation = self._observe(self._output())
        self.assertEqual(observation.makespan_cycles, 373)

    def test_program_io_phase_and_probe_corruption_fail_closed(self) -> None:
        with self.assertRaisesRegex(SchemaError, "ProgramIO phases"):
            self._observe(self._output().replace("phase=applied", "phase=missing"))
        with self.assertRaisesRegex(SchemaError, "ProgramIO probe"):
            self._observe(self._output().replace("exact=1 pass=1", "exact=0 pass=1", 1))

    def test_drain_and_host_corruption_fail_closed(self) -> None:
        with self.assertRaisesRegex(SchemaError, "collective drain"):
            self._observe(self._output().replace("tree_entries=0", "tree_entries=1"))
        with self.assertRaisesRegex(SchemaError, "P5 timing drain"):
            self._observe(self._output().replace(
                "[P5 P2P TIMING DRAIN] residual=0",
                "[P5 P2P TIMING DRAIN] residual=1",
            ))
        with self.assertRaisesRegex(SchemaError, "P5 drain coverage"):
            self._observe(self._output().replace(
                "[P5 P2P TIMING DRAIN] residual=0",
                "[P5 P2P DRAIN] core=0 residual=0\n"
                "[P5 P2P TIMING DRAIN] residual=0",
            ))
        with self.assertRaisesRegex(SchemaError, "HOSTLANE"):
            self._observe(self._output().replace("mismatch=0", "mismatch=1"))
        with self.assertRaisesRegex(SchemaError, "credit marker"):
            self._observe(self._output().replace(
                "[CREDIT] data_balanced=1 ctrl_balanced=1", "",
            ))
        with self.assertRaisesRegex(SchemaError, "credit marker"):
            self._observe(self._output().replace("data_balanced=1", "data_balanced=0"))

    def test_family_drift_fails_closed(self) -> None:
        train = FlexibleMeshReleaseCase.create(
            family=FlexibleMeshReleaseFamily.MOE_TRAIN,
            mesh=RectMeshSpec(1, 1),
            trace_model_digest=release_trace_model_digest(
                FlexibleMeshReleaseFamily.MOE_TRAIN
            ),
            runtime_profile_version="flexible-moe-timing-v2",
        )
        with self.assertRaisesRegex(SchemaError, "family drifted"):
            self.adapter.materialize(train)

    def test_trace_model_profile_cannot_be_caller_overridden(self) -> None:
        drifted = FlexibleMeshReleaseCase.create(
            family=FlexibleMeshReleaseFamily.MOE_INFERENCE,
            mesh=RectMeshSpec(1, 1),
            trace_model_digest="0" * 64,
            runtime_profile_version="flexible-moe-timing-v2",
        )
        with self.assertRaisesRegex(SchemaError, "trace/model profile drifted"):
            self.adapter.materialize(drifted)


if __name__ == "__main__":
    unittest.main()
