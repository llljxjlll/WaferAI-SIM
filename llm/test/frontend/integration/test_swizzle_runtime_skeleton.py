from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from llm.frontend.wafer_frontend.errors import (
    SchemaError,
    StageNotImplementedError,
)
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoMode
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.swizzle_evidence import (
    SWIZZLE_RUNTIME_MARKER_SCHEMA_VERSION,
    SwizzleBranchRuntimeEvidence,
    SwizzleCaseRuntimeEvidence,
    SwizzleComparisonBranch,
    SwizzleComparisonSuitePlan,
    SwizzleCoreCount,
    SwizzleNamedCount,
    SwizzleRuntimeArtifactEvidence,
    SwizzleRuntimeComparisonReport,
    SwizzleRuntimeControlEvidence,
    SwizzleRuntimeMetrics,
    SwizzleRuntimeProgramIoEvidence,
    SwizzleRuntimeRepeatEvidence,
    SwizzleRuntimeToolEvidence,
)

from run_swizzle_runtime import (
    run_swizzle_runtime_suite,
    validate_runtime_config_paths,
)
from swizzle_comparison import build_swizzle_comparison_suite


_DIGEST = "a" * 64
_ARTIFACT = "b" * 64


def _branch(case, branch, makespan: int) -> SwizzleBranchRuntimeEvidence:
    cost = case.baseline_cost if branch is SwizzleComparisonBranch.NAIVE else case.selected_cost
    control = SwizzleRuntimeControlEvidence(
        ack_counts=(SwizzleCoreCount(0, 1),),
        done_counts=(SwizzleCoreCount(0, 1),),
        drain_residuals=tuple(
            SwizzleNamedCount(name, 0)
            for name in ("collective", "global", "p2p", "timing")
        ),
        proto_wait_count=0,
        all_done_boundary_reached=True,
    )
    metrics = SwizzleRuntimeMetrics(
        logical_bytes=cost.logical_bytes,
        byte_hops=cost.byte_hops,
        packet_count=1,
        direction_port_utilization=cost.direction_port_utilization,
        sram_high_water_bytes=cost.sram_high_water_bytes,
        control_action_count=cost.control_action_count,
    )
    repeats = tuple(
        SwizzleRuntimeRepeatEvidence(
            run_index=index,
            makespan_cycles=makespan,
            marker_digest=_DIGEST,
            control_digest=_DIGEST,
            metrics_digest=_DIGEST,
        )
        for index in range(2)
    )
    result = SwizzleBranchRuntimeEvidence(
        branch=branch,
        tools=SwizzleRuntimeToolEvidence(_DIGEST, _DIGEST, _DIGEST),
        artifact=SwizzleRuntimeArtifactEvidence(
            linked_manifest_id=f"manifest.{branch.value}",
            linked_manifest_digest=_DIGEST,
            program_artifact_sha256=_ARTIFACT,
            artifact_size_bytes=16,
            finalizer_artifact_sha256s=(_ARTIFACT, _ARTIFACT),
            finalizer_report_digests=(_DIGEST, _DIGEST),
        ),
        program_io=SwizzleRuntimeProgramIoEvidence(
            contract_id=f"program_io.{branch.value}",
            contract_digest=_DIGEST,
            program_artifact_sha256=_ARTIFACT,
            mode=ProgramIoMode.TIMING,
            initialization_count=1,
            probe_count=1,
            all_probes_passed=True,
        ),
        control=control,
        metrics=metrics,
        marker_schema_version=SWIZZLE_RUNTIME_MARKER_SCHEMA_VERSION,
        repeats=repeats,
    )
    result.validate()
    return result


def _report() -> SwizzleRuntimeComparisonReport:
    suite = build_swizzle_comparison_suite()
    cases = tuple(
        SwizzleCaseRuntimeEvidence(
            case_plan_ref=case.id,
            naive=_branch(case, SwizzleComparisonBranch.NAIVE, 10),
            # Deliberately slower: performance is not a schema correctness gate.
            swizzle=_branch(case, SwizzleComparisonBranch.SWIZZLE, 20),
        )
        for case in suite.cases
    )
    return SwizzleRuntimeComparisonReport.create(
        suite=suite,
        cases=cases,
        timing_execution=True,
        functional_execution=False,
    )


class SwizzleRuntimeSkeletonTest(unittest.TestCase):
    def test_three_case_plan_is_stable_strict_and_self_contained(self) -> None:
        first = build_swizzle_comparison_suite()
        second = build_swizzle_comparison_suite()
        self.assertEqual(first, second)
        self.assertEqual(
            loads_dataclass(
                SwizzleComparisonSuitePlan,
                canonical_json(first),
                path="suite",
            ),
            first,
        )
        self.assertTrue(all(case.fusion_candidate_refs for case in first.cases))
        self.assertTrue(all(case.swizzle_plan_ref for case in first.cases))
        self.assertTrue(all(case.projection_ref for case in first.cases))

    def test_missing_lower_link_fails_before_files_or_tool_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            runtime_root = Path(raw) / "must_not_be_created"
            with self.assertRaisesRegex(
                StageNotImplementedError,
                "refuses to fabricate a manifest",
            ):
                run_swizzle_runtime_suite(
                    provider=None,
                    finalizer=Path("/missing/finalizer"),
                    resolver=Path("/missing/resolver"),
                    npusim=Path("/missing/npusim"),
                    simulation=Path("/missing/simulation"),
                    runtime_root=runtime_root,
                )
            self.assertFalse(runtime_root.exists())

    def test_runtime_root_must_resolve_real_dramsys_inputs(self) -> None:
        root = Path(__file__).resolve().parents[4]
        hardware_json = (
            root / "notes/frontend/examples/hardware_2x1.json"
        ).read_text(encoding="utf-8")
        simulation_json = (
            root / "llm/test/sram/simulation.json"
        ).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as raw:
            runtime_root = Path(raw) / "program-runtime"
            runtime_root.mkdir()
            with self.assertRaisesRegex(
                SchemaError, "runtime cwd does not resolve"
            ):
                validate_runtime_config_paths(
                    hardware_json=hardware_json,
                    simulation_json=simulation_json,
                    runtime_root=runtime_root,
                )
            config = Path(raw) / "DRAMSys/configs/hbm2-example.json"
            config.parent.mkdir(parents=True)
            config.write_text("{}", encoding="utf-8")
            self.assertEqual(
                validate_runtime_config_paths(
                    hardware_json=hardware_json,
                    simulation_json=simulation_json,
                    runtime_root=runtime_root,
                ),
                (config.resolve(),) * 3,
            )

    def test_report_requires_two_exact_repeats_actual_sha_and_timing_only(self) -> None:
        report = _report()
        report.validate()
        self.assertTrue(report.timing_execution)
        self.assertFalse(report.functional_execution)
        self.assertTrue(
            all(
                case.swizzle.repeats[0].makespan_cycles
                > case.naive.repeats[0].makespan_cycles
                for case in report.cases
            )
        )
        with self.assertRaisesRegex(SchemaError, "timing=true"):
            replace(report, functional_execution=True).validate()
        branch = report.cases[0].swizzle
        with self.assertRaisesRegex(SchemaError, "actual artifact SHA"):
            replace(
                branch,
                program_io=replace(
                    branch.program_io,
                    program_artifact_sha256="c" * 64,
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "npusim repeats"):
            replace(
                branch,
                repeats=(
                    branch.repeats[0],
                    replace(branch.repeats[1], makespan_cycles=21),
                ),
            ).validate()


if __name__ == "__main__":
    unittest.main()
