from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.program_io import (
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_runtime import (
    FlexibleMeshRuntimeCase,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_workload import (
    FlexibleMeshSliceOperation,
    FlexibleMeshWorkloadSpec,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec

from flexible_mesh_runtime_markers import parse_flexible_mesh_runtime_marker
from flexible_mesh_runtime_meshslice import (
    ProductionMeshSliceRuntimeMaterializer,
)


_ARTIFACT_SHA = "0" * 64


class FlexibleMeshRuntimeMarkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = FlexibleMeshRuntimeCase.create(
            FlexibleMeshWorkloadSpec.dense_infer(RectMeshSpec(1, 1)),
            FlexibleMeshSliceOperation.AG_GEMM,
        )
        cls.executable = ProductionMeshSliceRuntimeMaterializer(
            mapping_text="0:0\n"
        ).materialize(cls.case)
        cls.contract = build_timing_program_io(
            cls.executable.linked_source,
            _ARTIFACT_SHA,
        )
        cls.output = cls._output()

    @classmethod
    def _output(cls) -> str:
        contract = cls.contract
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
            "[HOSTSIG] done=0:1,",
            (
                "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
                "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0 event=0"
            ),
            "[DRAIN] router_residual=0",
            "[DRAIN] d2d_link_residual=0",
            "[P5 P2P TIMING DRAIN] residual=0",
            "[PROGRAM_MEMORY] core=0 lsu_residual=0 dte_residual=0",
            "[D2D_TYPE] data_out=0",
            "[CREDIT] data_balanced=1 ctrl_balanced=1",
            "End DONE reception",
        ))
        return "\n".join(lines) + "\n"

    def _parse(self, output: str):
        return parse_flexible_mesh_runtime_marker(
            output,
            case=self.case,
            manifest=self.executable.manifest,
            contract=self.contract,
            artifact_sha256=_ARTIFACT_SHA,
        )

    def test_real_manifest_contract_accepts_exact_zero_residual_markers(self) -> None:
        marker = self._parse(self.output)
        self.assertTrue(marker.residual.is_zero)
        self.assertEqual(marker.rank_coverage, (0,))

    def test_missing_or_hidden_residual_markers_fail_closed(self) -> None:
        missing_link_drain = self.output.replace(
            "[DRAIN] d2d_link_residual=0\n", ""
        )
        with self.assertRaisesRegex(SchemaError, "each appear exactly once"):
            self._parse(missing_link_drain)

        nonzero_collective = self.output.replace(
            "tree_entries=0", "tree_entries=1"
        )
        with self.assertRaisesRegex(SchemaError, "residual is non-zero"):
            self._parse(nonzero_collective)

        nonzero_memory = self.output.replace(
            "lsu_residual=0", "lsu_residual=1"
        )
        with self.assertRaisesRegex(SchemaError, "residual is non-zero"):
            self._parse(nonzero_memory)

        duplicate_typed_summary = self.output + "[D2D_TYPE] data_out=0\n"
        with self.assertRaisesRegex(SchemaError, "appear exactly once"):
            self._parse(duplicate_typed_summary)

    def test_credit_balance_marker_is_exact_and_digest_bound(self) -> None:
        marker = self._parse(self.output)
        missing = self.output.replace(
            "[CREDIT] data_balanced=1 ctrl_balanced=1\n", ""
        )
        with self.assertRaisesRegex(SchemaError, "credit balance marker"):
            self._parse(missing)
        unbalanced = self.output.replace(
            "data_balanced=1 ctrl_balanced=1",
            "data_balanced=1 ctrl_balanced=0",
        )
        with self.assertRaisesRegex(SchemaError, "credit balance marker"):
            self._parse(unbalanced)
        reordered = self.output.replace(
            "data_balanced=1 ctrl_balanced=1",
            "ctrl_balanced=1 data_balanced=1",
        )
        self.assertNotEqual(self._parse(reordered).marker_digest, marker.marker_digest)



if __name__ == "__main__":
    unittest.main()
