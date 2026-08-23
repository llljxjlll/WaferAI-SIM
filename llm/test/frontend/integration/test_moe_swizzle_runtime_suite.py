from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_moe_scale_swizzle_overlay import (
    build_moe_scale_swizzle_overlay,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_pair_feasibility import (
    build_moe_swizzle_pair_feasibility_witness,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_workload_state_abi import (
    build_moe_swizzle_workload_state_abi,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_cost import (
    build_moe_swizzle_pair_cost_context,
    estimate_moe_swizzle_materialized_pair_cycles,
    select_moe_swizzle_workload_deployment,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleAlgorithm
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES,
    MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionMode,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_plan import (
    MoeSwizzleDeploymentMode,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_placement import (
    MoeWholePairPlacementReason,
)

from moe_swizzle_runtime_evidence import (
    MoeSwizzleNamedCount,
    MoeSwizzleRuntimeBenefitReport,
    MoeSwizzleRuntimeControlClosure,
    MoeSwizzleRuntimeObservation,
    MoeSwizzleRuntimePairEvidence,
)
from moe_swizzle_runtime_markers import parse_moe_swizzle_runtime_markers
from moe_swizzle_runtime_provider import (
    PendingW9MoeSwizzleRuntimeProvider,
    moe_swizzle_calibration_run_digest,
    preflight_moe_swizzle_runtime_provider,
)
from moe_swizzle_runtime_suite import (
    MoeSwizzleRuntimeBranch,
    MoeSwizzleRuntimeCasePlan,
    MoeSwizzleRuntimeScope,
    MoeSwizzleRuntimeSuitePlan,
    build_moe_swizzle_runtime_case_plan,
)
from moe_swizzle_scale_cases import build_moe_swizzle_scale_cases
from run_moe_swizzle_calibration import (
    MoeSwizzleCalibrationArtifactEvidence,
    MoeSwizzleCalibrationRunEvidence,
    canonical_moe_swizzle_calibration_keys,
)

from test_swizzle_moe_calibration import _complete_runtime
from test_swizzle_moe_cost import _measured_profile


_DRAIN_NAMES = (
    "lsu", "dte", "p2p_endpoint", "p2p_timing", "collective", "router", "d2d_link",
)


def _observation(case, branch, naive_cycles=110, auto_cycles=100):
    marker = parse_moe_swizzle_runtime_markers(_complete_runtime())
    cycles = auto_cycles if branch.branch is MoeSwizzleRuntimeBranch.SWIZZLE_AUTO else naive_cycles
    control = MoeSwizzleRuntimeControlClosure(
        expected_probe_count=case.same_work.combined_terminal_count + case.same_work.tape_terminal_count,
        observed_probe_count=case.same_work.combined_terminal_count + case.same_work.tape_terminal_count,
        expected_probe_bytes=case.same_work.combined_terminal_bytes + case.same_work.tape_terminal_bytes,
        observed_probe_bytes=case.same_work.combined_terminal_bytes + case.same_work.tape_terminal_bytes,
        expected_ack_count=4,
        observed_ack_count=4,
        expected_done_count=4,
        observed_done_count=4,
        expected_physical_root_count=marker.physical_root_count,
        residuals=tuple(MoeSwizzleNamedCount(name, 0) for name in _DRAIN_NAMES),
        proto_wait_count=0,
        program_io_digest="e" * 64,
    )
    return MoeSwizzleRuntimeObservation.create(
        case_ref=case.id,
        branch=branch.branch,
        same_work_digest=canonical_digest(case.same_work),
        deployment_digest=canonical_digest(branch),
        workload_selection_ref=branch.workload_selection_ref,
        source_workload_selection_id=branch.workload_selection_ref,
        repeat_makespans=(cycles, cycles),
        repeat_marker_digests=(marker.marker_digest, marker.marker_digest),
        repeat_markers=(marker, marker),
        finalizer_artifact_sha256=("f" * 64, "f" * 64),
        raw_stdout_sha256=("1" * 64, "2" * 64),
        tool_sha256="a" * 64,
        hardware_sha256="b" * 64,
        simulation_sha256="c" * 64,
        mapping_sha256="d" * 64,
        control=control,
        timing_execution=True,
        functional_execution=False,
    )


class MoeSwizzleRuntimeSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = _measured_profile()
        cls.scale_case = build_moe_swizzle_scale_cases()[2]
        cls.preflight = build_moe_swizzle_runtime_case_plan(
            cls.scale_case,
            MoeScaleExecutionMode.INFER_FORWARD,
            cls.profile,
            scope=MoeSwizzleRuntimeScope.REGION_PREFLIGHT,
            target_pattern=FusionPattern.MOE_DISPATCH_GEMM,
        )
        dispatch, combine = cls.preflight.economic_decisions
        argmin = lambda decision: min(
            decision.ranked_candidates,
            key=lambda candidate: (candidate.cost.estimated_cycles, candidate.id),
        ).id
        cls.baseline_pair = (dispatch.baseline.id, combine.baseline.id)
        cls.fused_pair = (argmin(dispatch), argmin(combine))
        mode_pairs = {
            (dispatch_ref, combine_ref)
            for dispatch_ref in (cls.baseline_pair[0], cls.fused_pair[0])
            for combine_ref in (cls.baseline_pair[1], cls.fused_pair[1])
        }
        pair_context = build_moe_swizzle_pair_cost_context(
            dispatch, combine,
        )
        pair_bounds = pair_context.bounds
        ir1 = build_moe_swizzle_scale_cases()[0].c0_production_case.forward.n4.graph
        state_abi = build_moe_swizzle_workload_state_abi(
            ir1, cls.preflight.execution, cls.scale_case.spec, cls.scale_case.oracle,
        )
        witness_by_pair = {}
        frontier = []
        best_corrected = float("inf")
        for pair, lower_bound in pair_bounds:
            if lower_bound >= best_corrected:
                break
            witness = build_moe_swizzle_pair_feasibility_witness(
                ir1, cls.preflight.execution, cls.scale_case.spec,
                cls.preflight.economic_decisions, state_abi, pair,
            )
            witness_by_pair[pair] = witness
            frontier.append(pair)
            if witness.feasible:
                best_corrected = min(
                    best_corrected,
                    estimate_moe_swizzle_materialized_pair_cycles(
                        dispatch, combine, witness,
                        context=pair_context,
                    ),
                )
        for pair in sorted(mode_pairs - set(frontier)):
            witness_by_pair[pair] = build_moe_swizzle_pair_feasibility_witness(
                ir1, cls.preflight.execution, cls.scale_case.spec,
                cls.preflight.economic_decisions, state_abi, pair,
            )
        witnesses = tuple(
            witness_by_pair[pair] for pair in sorted(witness_by_pair)
        )
        cls.pair_witnesses = witnesses
        cls.frontier_pairs = tuple(frontier)
        cls.mode_pairs = mode_pairs
        cls.economic_decisions = (dispatch, combine)
        cls.selection = select_moe_swizzle_workload_deployment(
            dispatch, combine, witnesses,
        )
        cls.workload = build_moe_swizzle_runtime_case_plan(
            cls.scale_case,
            MoeScaleExecutionMode.INFER_FORWARD,
            cls.profile,
            scope=MoeSwizzleRuntimeScope.WORKLOAD,
            workload_selections=(cls.selection,),
        )
        cls.overlay = build_moe_scale_swizzle_overlay(
            cls.workload.execution, cls.workload.economic_decisions,
            cls.workload.workload_selection,
        )

    def test_region_preflight_isolates_target_and_keeps_economic_decisions(self) -> None:
        case = self.preflight
        self.assertEqual(case.spec.name, "C2")
        self.assertEqual(tuple(item.branch for item in case.branches), tuple(MoeSwizzleRuntimeBranch))
        naive, auto, forced = case.branches
        self.assertTrue(all(item.algorithm is SwizzleAlgorithm.UNFUSED for item in naive.deployments))
        self.assertTrue(auto.deployments[0].economic_selected)
        self.assertIs(auto.deployments[1].algorithm, SwizzleAlgorithm.UNFUSED)
        self.assertIs(auto.deployments[1].mode, MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE)
        self.assertIs(forced.deployments[0].mode, MoeSwizzleDeploymentMode.FUSED_FORCED)
        self.assertFalse(forced.deployments[0].economic_selected)
        self.assertIs(forced.deployments[1].algorithm, SwizzleAlgorithm.UNFUSED)
        self.assertEqual(len({item.same_work_digest for item in case.branches}), 1)
        self.assertTrue(
            set(case.deployed_comp_shapes).issubset(
                MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES
            )
        )
        self.assertIn((1, 16, 32), case.deployed_comp_shapes)
        self.assertTrue(
            set(case.deployed_swiglu_group_shapes).issubset(
                MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES
            )
        )
        self.assertIn((1, 32, 32), case.deployed_swiglu_group_shapes)
        self.assertIsNone(case.workload_selection)
        self.assertTrue(all(
            branch.workload_selection_ref is None for branch in case.branches
        ))
        self.assertEqual(
            loads_dataclass(MoeSwizzleRuntimeCasePlan, canonical_json(case), path="case"),
            case,
        )


    def test_workload_auto_consumes_joint_selection_and_overlay_lineage(self) -> None:
        case = self.workload
        selection = self.selection
        naive, auto, forced = case.branches
        self.assertIs(case.workload_selection, selection)
        self.assertIsNone(naive.workload_selection_ref)
        self.assertEqual(auto.workload_selection_ref, selection.id)
        self.assertIsNone(forced.workload_selection_ref)
        witness_by_pair = {
            item.candidate_refs: item for item in self.pair_witnesses
        }
        self.assertEqual(
            set(witness_by_pair),
            self.mode_pairs | set(self.frontier_pairs),
        )
        for witness in witness_by_pair.values():
            if witness.placement.feasible:
                self.assertIs(
                    witness.placement.reason,
                    MoeWholePairPlacementReason.ADMITTED,
                )
                self.assertIsNotNone(witness.endpoint)
                self.assertIsNotNone(witness.workload_abi_id)
                self.assertTrue(witness.dynamic_root_keys)
                self.assertGreater(witness.dynamic_sram_high_water_bytes, 0)
            else:
                self.assertFalse(witness.feasible)
                self.assertIs(
                    witness.placement.reason,
                    MoeWholePairPlacementReason.ORDINARY_VALUE_CROSS_CORE,
                )
                self.assertIsNone(witness.endpoint)
                self.assertIsNone(witness.workload_abi_id)
                self.assertEqual(witness.dynamic_root_keys, ())
        self.assertEqual(
            self.frontier_pairs,
            (("moe_swizzle_candidate_2280d4074534aa1c",
              "moe_swizzle_candidate_d3d505ad34c029a5"),),
        )
        self.assertEqual(selection.frontier_stop_lower_bound, 309.0)
        self.assertFalse(witness_by_pair[self.fused_pair].feasible)
        self.assertIs(
            witness_by_pair[self.fused_pair].placement.reason,
            MoeWholePairPlacementReason.ORDINARY_VALUE_CROSS_CORE,
        )
        selected_witness = witness_by_pair[self.frontier_pairs[0]]
        self.assertTrue(selected_witness.feasible)
        self.assertTrue(all(
            item.depth == 1 for item in selected_witness.storage_color_depths
        ))
        self.assertEqual(
            (selected_witness.whole_physical_root_count,
             selected_witness.whole_alloc_count,
             selected_witness.whole_free_count),
            (108, 100, 92),
        )
        self.assertEqual(
            tuple(
                (item.runtime_core_id, item.alloc_count,
                 item.bind_count, item.free_count)
                for item in selected_witness.core_lifecycle_counts
            ),
            (
                (0, 6, 1, 5), (1, 6, 1, 5),
                (2, 6, 1, 6), (3, 6, 1, 6),
                (4, 6, 1, 6), (5, 6, 1, 6),
                (6, 7, 1, 6), (7, 7, 1, 6),
                (8, 6, 1, 6), (9, 6, 1, 6),
                (10, 7, 1, 6), (11, 7, 1, 6),
                (12, 6, 1, 5), (13, 6, 1, 5),
                (14, 6, 1, 6), (15, 6, 1, 6),
            ),
        )
        self.assertEqual(
            selected_witness.id,
            "moe_whole_pair_feasibility_f72385e6e14e8c6e",
        )
        self.assertEqual(
            selection.id,
            "moe_swizzle_workload_selection_92d37473f032c3bf",
        )
        self.assertEqual(
            (selection.baseline_estimated_cycles,
             selection.selected_estimated_cycles),
            (556.0, 308.0),
        )
        self.assertAlmostEqual(
            selection.baseline_estimated_cycles
            / selection.selected_estimated_cycles,
            1.8051948051948052,
        )
        self.assertEqual(
            self.frontier_pairs[-1],
            (
                selection.selected_dispatch_candidate_ref,
                selection.selected_combine_candidate_ref,
            ),
        )
        self.assertEqual(
            (
                selection.selected_dispatch_candidate_ref,
                selection.selected_combine_candidate_ref,
            ),
            self.frontier_pairs[-1],
        )
        candidates = {
            item.id: item
            for decision in self.economic_decisions
            for item in decision.ranked_candidates
        }
        self.assertEqual(
            tuple(
                (
                    candidates[ref].algorithm,
                    candidates[ref].token_block_size,
                    candidates[ref].unroll_degree,
                    candidates[ref].compute_output_block_count,
                    candidates[ref].transport_output_block_count,
                )
                for ref in self.frontier_pairs[0]
            ),
            (
                (SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A, 4, 1, 1, 1),
                (SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A, 4, 1, 1, 1),
            ),
        )
        self.assertGreater(
            selection.baseline_estimated_cycles
            / selection.selected_estimated_cycles,
            1.10,
        )
        self.assertEqual(
            tuple(item.candidate_ref for item in auto.deployments),
            self.frontier_pairs[-1],
        )
        self.assertTrue(all(item.economic_selected for item in auto.deployments))
        self.assertTrue(all(
            item.candidate_ref != selected_ref
            for item, selected_ref in zip(
                forced.deployments,
                (
                    selection.selected_dispatch_candidate_ref,
                    selection.selected_combine_candidate_ref,
                ),
                strict=True,
            )
        ))
        self.assertEqual(
            self.overlay.source_workload_selection_id, selection.id,
        )
        observation = _observation(case, auto)
        observation.validate_against(case)
        self.assertEqual(observation.source_workload_selection_id, selection.id)
        tampered_semantic = observation._semantic()
        tampered_semantic["source_workload_selection_id"] = "forged-selection"
        tampered = MoeSwizzleRuntimeObservation.create(**tampered_semantic)
        with self.assertRaisesRegex(SchemaError, "does not belong"):
            tampered.validate_against(case)
        with self.assertRaisesRegex(SchemaError, "joint selection lineage"):
            replace(
                auto, workload_selection_ref="forged-selection",
            ).validate_against(case)

    def test_same_work_and_forced_tampers_fail_closed(self) -> None:
        branch = self.preflight.branches[0]
        with self.assertRaisesRegex(SchemaError, "same-work"):
            replace(branch, same_work_digest="0" * 64).validate_against(self.preflight)
        with self.assertRaisesRegex(SchemaError, "COMP shape set"):
            replace(self.preflight, deployed_comp_shapes=((3, 7, 11),)).validate()
        forced = self.preflight.branches[2]
        bad = replace(
            forced.deployments[0],
            economic_selected=True,
        )
        with self.assertRaisesRegex(SchemaError, "forced"):
            replace(forced, deployments=(bad, forced.deployments[1])).validate_against(self.preflight)
        with self.assertRaisesRegex(SchemaError, "forbids workload joint selection"):
            replace(
                self.preflight,
                workload_selection=object(),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "forbids workload selection lineage"):
            replace(
                self.preflight.branches[1],
                workload_selection_ref="forged-selection",
            ).validate_against(self.preflight)
        with self.assertRaisesRegex(SchemaError, "requires exact typed joint selection"):
            replace(
                self.preflight,
                scope=MoeSwizzleRuntimeScope.WORKLOAD,
                target_pattern=None,
                workload_selection=None,
            ).validate()

    def test_threshold_and_no_benefit_report_remain_incomplete(self) -> None:
        observations = tuple(
            _observation(self.preflight, branch)
            for branch in self.preflight.branches
        )
        pair = MoeSwizzleRuntimePairEvidence.create(
            self.preflight, observations[0], observations[1]
        )
        self.assertEqual(pair.speedup, 1.10)
        self.assertTrue(pair.threshold_met)
        self.assertTrue(pair.runtime_overlap_met)
        self.assertTrue(pair.benefit)
        suite = MoeSwizzleRuntimeSuitePlan.create(
            calibration_profile=self.profile,
            calibration_run_digest="9" * 64,
            cases=(self.preflight,),
        )
        report = MoeSwizzleRuntimeBenefitReport.create(suite, observations)
        self.assertTrue(report.correctness_complete)
        self.assertTrue(report.measurement_complete)
        self.assertFalse(report.performance_benefit)
        self.assertFalse(report.performance_complete)
        self.assertFalse(report.v2_complete)
        slow_auto = _observation(
            self.preflight,
            self.preflight.branches[1],
            naive_cycles=110,
            auto_cycles=101,
        )
        slow_pair = MoeSwizzleRuntimePairEvidence.create(
            self.preflight, observations[0], slow_auto
        )
        self.assertFalse(slow_pair.threshold_met)
        self.assertFalse(slow_pair.benefit)

    def test_actual_calibration_precedes_provider_and_w9_is_failclosed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.stdout"
            raw.write_text("dedicated calibration evidence\n", encoding="utf-8")
            calibration = MoeSwizzleCalibrationRunEvidence(
                self.profile,
                "a" * 64,
                "b" * 64,
                "c" * 64,
                "d" * 64,
                168,
                (raw,) * 168,
                tuple(
                    MoeSwizzleCalibrationArtifactEvidence(
                        key,
                        "production_suite_test_provider",
                        "e" * 64,
                        "f" * 64,
                        "1" * 64,
                    )
                    for key in canonical_moe_swizzle_calibration_keys()
                ),
            )
            digest = moe_swizzle_calibration_run_digest(calibration)
            first_family = (
                calibration.artifacts[0].key.kind,
                calibration.artifacts[0].key.shape,
            )
            for field, changed_value in (
                ("production_source_ref", "another_production_source"),
                ("program_sha256", "2" * 64),
            ):
                changed = replace(
                    calibration,
                    artifacts=tuple(
                        replace(item, **{field: changed_value})
                        if (item.key.kind, item.key.shape) == first_family
                        else item
                        for item in calibration.artifacts
                    ),
                )
                changed.validate()
                self.assertNotEqual(
                    digest, moe_swizzle_calibration_run_digest(changed)
                )
            suite = MoeSwizzleRuntimeSuitePlan.create(
                calibration_profile=self.profile,
                calibration_run_digest=digest,
                cases=(self.preflight,),
            )
            with self.assertRaisesRegex(SchemaError, "W9 production linked"):
                preflight_moe_swizzle_runtime_provider(
                    calibration=calibration,
                    suite=suite,
                    provider=PendingW9MoeSwizzleRuntimeProvider(),
                    output_root=root / "runtime",
                )
            wrong = MoeSwizzleRuntimeSuitePlan.create(
                calibration_profile=self.profile,
                calibration_run_digest="0" * 64,
                cases=(self.preflight,),
            )
            with self.assertRaisesRegex(SchemaError, "actual 168-run"):
                preflight_moe_swizzle_runtime_provider(
                    calibration=calibration,
                    suite=wrong,
                    provider=PendingW9MoeSwizzleRuntimeProvider(),
                    output_root=root / "wrong",
                )


if __name__ == "__main__":
    unittest.main()
