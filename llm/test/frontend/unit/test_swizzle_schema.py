from __future__ import annotations

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.swizzle.semantics import (
    analyze_ag_gemm,
    analyze_gemm_ar,
    analyze_gemm_rs,
    validate_semantic_witness,
)
from llm.frontend.wafer_frontend.schema.common import DType, MeshAxisName
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    FusionPattern,
    GemmPartition,
    ReduceOp,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle import (
    SWIZZLE_PROBLEM_SCHEMA_VERSION,
    SwizzleAlgorithm,
    SwizzleCollectiveDescriptor,
    SwizzleCollectivePosition,
    SwizzleConstraints,
    SwizzleDecision,
    SwizzleDecisionReason,
    SwizzleEfficiencyPoint,
    SwizzleFeasibilityCheck,
    SwizzleFeasibilityWitness,
    SwizzleGemmDescriptor,
    SwizzleGroupView,
    SwizzleHardwareProfile,
    SwizzleOperand,
    SwizzleRankPlacement,
    SwizzleRouteView,
    SwizzleSemanticWitness,
    SwizzleTensorAxisRole,
    SwizzleTensorView,
    SwizzleTopologyKind,
    SwizzleTopologyWitness,
    SwizzleUpdateKind,
    SwizzleCandidate,
    SwizzleCost,
    SwizzleProblem,
)


TP = MeshAxisName.TP


def _view(
    ref: str,
    shape: tuple[int, ...],
    roles: tuple[SwizzleTensorAxisRole, ...],
    *,
    dim_map: tuple[MeshAxisName | None, ...] | None = None,
    partial: tuple[MeshAxisName, ...] = (),
) -> SwizzleTensorView:
    return SwizzleTensorView(
        value_ref=ref,
        shape=shape,
        layout="row_major",
        axis_roles=roles,
        sharding_dim_map=dim_map or (None,) * len(shape),
        partial_mesh_axes=partial,
    )


def _gemm(
    *,
    batch: tuple[int, ...] = (),
    partition: GemmPartition = GemmPartition.ROW_PARALLEL,
    lhs: SwizzleTensorView | None = None,
    rhs: SwizzleTensorView | None = None,
    output: SwizzleTensorView | None = None,
    boundary_input_refs: tuple[str, ...] | None = None,
    local_operand_refs: tuple[str, ...] = (),
) -> SwizzleGemmDescriptor:
    batch_roles = (SwizzleTensorAxisRole.BATCH,) * len(batch)
    lhs = lhs or _view("lhs", batch + (4, 4), batch_roles + (SwizzleTensorAxisRole.FREE_LHS, SwizzleTensorAxisRole.CONTRACT))
    rhs = rhs or _view("rhs", batch + (4, 8), batch_roles + (SwizzleTensorAxisRole.CONTRACT, SwizzleTensorAxisRole.FREE_RHS))
    output = output or _view(
        "partial",
        batch + (4, 8),
        batch_roles + (SwizzleTensorAxisRole.FREE_LHS, SwizzleTensorAxisRole.FREE_RHS),
        partial=(TP,) if partition is GemmPartition.ROW_PARALLEL else (),
    )
    batch_product = 1
    for extent in batch:
        batch_product *= extent
    return SwizzleGemmDescriptor(
        node_ref="gemm",
        partition=partition,
        m=4,
        n=8,
        k=4,
        batch_shape=batch,
        lhs=lhs,
        rhs=rhs,
        output=output,
        boundary_input_refs=boundary_input_refs or (lhs.value_ref, rhs.value_ref),
        local_operand_refs=local_operand_refs,
        dtype=DType.FP16,
        accumulation_dtype=DType.FP32,
        flops=2 * batch_product * 4 * 8 * 4,
    )


def _post_collective(kind: CollectiveKind) -> tuple[SwizzleGemmDescriptor, SwizzleCollectiveDescriptor]:
    gemm = _gemm()
    if kind is CollectiveKind.REDUCE_SCATTER:
        output = _view(
            "result",
            (2, 8),
            (SwizzleTensorAxisRole.FREE_LHS, SwizzleTensorAxisRole.FREE_RHS),
            dim_map=(TP, None),
        )
        descriptor = SwizzleCollectiveDescriptor(
            node_ref="collective",
            kind=kind,
            reduce_op=ReduceOp.SUM,
            position=SwizzleCollectivePosition.AFTER_GEMM,
            mesh_axes=(TP,),
            participant_ranks=(0, 1),
            gather_tensor_axis=None,
            scatter_tensor_axis=0,
            logical_bytes=64,
            rank_input_bytes=64,
            rank_output_bytes=32,
            input=gemm.output,
            output=output,
        )
    else:
        output = _view(
            "result",
            (4, 8),
            (SwizzleTensorAxisRole.FREE_LHS, SwizzleTensorAxisRole.FREE_RHS),
        )
        descriptor = SwizzleCollectiveDescriptor(
            node_ref="collective",
            kind=kind,
            reduce_op=ReduceOp.SUM,
            position=SwizzleCollectivePosition.AFTER_GEMM,
            mesh_axes=(TP,),
            participant_ranks=(0, 1),
            gather_tensor_axis=None,
            scatter_tensor_axis=None,
            logical_bytes=64,
            rank_input_bytes=64,
            rank_output_bytes=64,
            input=gemm.output,
            output=output,
        )
    return gemm, descriptor


def _ag_case(role: SwizzleTensorAxisRole) -> tuple[SwizzleGemmDescriptor, SwizzleCollectiveDescriptor, tuple[str, ...]]:
    batch = (4,) if role is SwizzleTensorAxisRole.BATCH else ()
    batch_roles = (SwizzleTensorAxisRole.BATCH,) * len(batch)
    lhs_shape = batch + (4, 4)
    rhs_shape = batch + (4, 8)
    lhs_roles = batch_roles + (SwizzleTensorAxisRole.FREE_LHS, SwizzleTensorAxisRole.CONTRACT)
    rhs_roles = batch_roles + (SwizzleTensorAxisRole.CONTRACT, SwizzleTensorAxisRole.FREE_RHS)
    operand = SwizzleOperand.RHS if role is SwizzleTensorAxisRole.FREE_RHS else SwizzleOperand.LHS
    gathered_shape = lhs_shape if operand is SwizzleOperand.LHS else rhs_shape
    gathered_roles = lhs_roles if operand is SwizzleOperand.LHS else rhs_roles
    axis = gathered_roles.index(role)
    local_shape = list(gathered_shape)
    local_shape[axis] //= 2
    local_map: list[MeshAxisName | None] = [None] * len(gathered_shape)
    local_map[axis] = TP
    gathered = _view("gathered", gathered_shape, gathered_roles)
    local = _view("local_shard", tuple(local_shape), gathered_roles, dim_map=tuple(local_map))
    lhs = gathered if operand is SwizzleOperand.LHS else _view("lhs", lhs_shape, lhs_roles)
    rhs = gathered if operand is SwizzleOperand.RHS else _view("rhs", rhs_shape, rhs_roles)
    boundaries = (
        (local.value_ref, rhs.value_ref)
        if operand is SwizzleOperand.LHS
        else (lhs.value_ref, local.value_ref)
    )
    gemm = _gemm(batch=batch, partition=GemmPartition.COLUMN_PARALLEL, lhs=lhs, rhs=rhs, boundary_input_refs=boundaries)
    collective = SwizzleCollectiveDescriptor(
        node_ref="ag",
        kind=CollectiveKind.ALL_GATHER,
        reduce_op=None,
        position=SwizzleCollectivePosition.BEFORE_GEMM,
        mesh_axes=(TP,),
        participant_ranks=(0, 1),
        gather_tensor_axis=axis,
        scatter_tensor_axis=None,
        logical_bytes=64,
        rank_input_bytes=32,
        rank_output_bytes=64,
        input=local,
        output=gathered,
    )
    return gemm, collective, boundaries


def _profile() -> SwizzleHardwareProfile:
    return SwizzleHardwareProfile.create(
        peak_flops_per_cycle=1024.0,
        confidence_fraction=0.1,
        efficiency_points=(SwizzleEfficiencyPoint(4, 8, 4, 0.75),),
        dte_launch_cycles=2,
        dte_sync_cycles=1,
        hop_latency_cycles=1,
        lane_bytes_per_cycle=32.0,
        max_inflight_dte=4,
        min_transfer_bytes=16,
        efficient_tile_floor=(1, 1, 1),
        sram_budget_bytes=4096,
        double_buffer_supported=True,
    )


class SwizzleSemanticTest(unittest.TestCase):
    def test_ag_dimension_roles_select_exact_update_semantics(self) -> None:
        for role in (
            SwizzleTensorAxisRole.BATCH,
            SwizzleTensorAxisRole.FREE_LHS,
            SwizzleTensorAxisRole.FREE_RHS,
            SwizzleTensorAxisRole.CONTRACT,
        ):
            with self.subTest(role=role):
                gemm, collective, boundaries = _ag_case(role)
                witness = analyze_ag_gemm(
                    gemm,
                    collective,
                    boundary_input_refs=boundaries,
                    boundary_output_refs=(gemm.output.value_ref,),
                )
                self.assertEqual(witness.split_axis.role, role)
                self.assertEqual(
                    witness.update_kind,
                    SwizzleUpdateKind.PARTIAL_ACCUMULATION
                    if role is SwizzleTensorAxisRole.CONTRACT
                    else SwizzleUpdateKind.OUTPUT_SLICE,
                )
                validate_semantic_witness(witness, gemm, collective)

    def test_rs_and_ar_have_distinct_phase_witnesses(self) -> None:
        gemm, rs = _post_collective(CollectiveKind.REDUCE_SCATTER)
        rs_witness = analyze_gemm_rs(
            gemm,
            rs,
            boundary_input_refs=(gemm.lhs.value_ref, gemm.rhs.value_ref),
            boundary_output_refs=(rs.output.value_ref,),
        )
        self.assertTrue(rs_witness.has_reduction_phase)
        self.assertFalse(rs_witness.has_replication_phase)

        gemm, ar = _post_collective(CollectiveKind.ALL_REDUCE)
        ar_witness = analyze_gemm_ar(
            gemm,
            ar,
            boundary_input_refs=(gemm.lhs.value_ref, gemm.rhs.value_ref),
            boundary_output_refs=(ar.output.value_ref,),
        )
        self.assertTrue(ar_witness.has_reduction_phase)
        self.assertTrue(ar_witness.has_replication_phase)
        self.assertEqual(ar_witness.update_kind, SwizzleUpdateKind.REDUCE_THEN_REPLICATE)

    def test_semantics_fail_closed_on_boundary_sharding_and_tamper(self) -> None:
        gemm, collective, boundaries = _ag_case(SwizzleTensorAxisRole.CONTRACT)
        with self.assertRaisesRegex(SchemaError, "boundary input order"):
            analyze_ag_gemm(
                gemm,
                collective,
                boundary_input_refs=tuple(reversed(boundaries)),
                boundary_output_refs=(gemm.output.value_ref,),
            )
        forged = replace(
            collective,
            input=replace(collective.input, sharding_dim_map=(None,) * len(collective.input.shape)),
        )
        with self.assertRaisesRegex(SchemaError, "sharded on gather axis"):
            analyze_ag_gemm(
                gemm,
                forged,
                boundary_input_refs=boundaries,
                boundary_output_refs=(gemm.output.value_ref,),
            )
        witness = analyze_ag_gemm(
            gemm,
            collective,
            boundary_input_refs=boundaries,
            boundary_output_refs=(gemm.output.value_ref,),
        )
        with self.assertRaises(SchemaError):
            validate_semantic_witness(
                replace(witness, split_axis=replace(witness.split_axis, extent=1)),
                gemm,
                collective,
            )


class SwizzleSchemaTest(unittest.TestCase):
    def _problem(self) -> tuple[SwizzleProblem, SwizzleSemanticWitness]:
        gemm, collective, boundaries = _ag_case(SwizzleTensorAxisRole.CONTRACT)
        witness = analyze_ag_gemm(
            gemm,
            collective,
            boundary_input_refs=boundaries,
            boundary_output_refs=(gemm.output.value_ref,),
        )
        group = SwizzleGroupView(
            group_ref="group",
            logical_shape=(2, 1),
            placements=(SwizzleRankPlacement(0, 0, 0), SwizzleRankPlacement(1, 1, 0)),
            routes=(
                SwizzleRouteView("route.0.1", 0, 1, (10, 11), ("egress.10", "link.10.11")),
                SwizzleRouteView("route.1.0", 1, 0, (11, 10), ("egress.11", "link.11.10")),
            ),
        )
        problem = SwizzleProblem.create(
            source_ir1_id="ir1",
            fused_op_id="fused",
            pattern=FusionPattern.AG_GEMM,
            gemm=gemm,
            collective=collective,
            group=group,
            hardware_profile=_profile(),
            constraints=SwizzleConstraints(
                allowed_algorithms=(SwizzleAlgorithm.UNFUSED,),
                max_candidates=32,
                max_actions=512,
                max_buffers=64,
                max_chunk_count=32,
                allow_unroll_two=True,
            ),
        )
        return problem, witness

    def _cost(self) -> SwizzleCost:
        return SwizzleCost.create(
            estimated_cycles=10.0,
            lower_cycles=9.0,
            upper_cycles=11.0,
            prologue_cycles=1.0,
            steady_cycles=8.0,
            epilogue_cycles=1.0,
            logical_bytes=64,
            byte_hops=64,
            message_count=2,
            direction_port_utilization=0.5,
            control_action_count=0,
            max_inflight=0,
            sram_high_water_bytes=0,
            bottleneck_resources=("link.10.11",),
        )

    def test_problem_strict_roundtrip_old_version_and_stable_id(self) -> None:
        problem, _ = self._problem()
        self.assertEqual(
            loads_dataclass(SwizzleProblem, canonical_json(problem), path="problem"),
            problem,
        )
        raw = json.loads(canonical_json(problem))
        raw["unknown"] = True
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(SwizzleProblem, json.dumps(raw), path="problem")
        del raw["unknown"]
        del raw["constraints"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(SwizzleProblem, json.dumps(raw), path="problem")
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(problem, schema_version=SWIZZLE_PROBLEM_SCHEMA_VERSION.replace("v1alpha1", "v0")).validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(problem, id="forged").validate()

    def test_baseline_decision_is_stable_and_tamper_evident(self) -> None:
        problem, witness = self._problem()
        baseline = SwizzleCandidate.create(
            problem_ref=problem.id,
            pattern=problem.pattern,
            algorithm=SwizzleAlgorithm.UNFUSED,
            split_axis=None,
            chunk_count=0,
            unroll_degree=0,
            rank_programs=(),
            buffer_requirements=(),
            topology_witness=SwizzleTopologyWitness(
                kind=SwizzleTopologyKind.UNFUSED,
                rank_order=(),
                row_orders=(),
                column_orders=(),
                route_refs=(),
                is_complete_rectangle=False,
                has_hamiltonian_cycle=False,
            ),
            semantic_witness=witness,
            feasibility_witness=SwizzleFeasibilityWitness(
                (SwizzleFeasibilityCheck("baseline", True, "always retained"),)
            ),
            cost=self._cost(),
        )
        decision = SwizzleDecision.create(
            problem=problem,
            baseline=baseline,
            ranked_candidates=(baseline,),
            selected_candidate_ref=baseline.id,
            decision_reason=SwizzleDecisionReason.BASELINE_ONLY,
        )
        self.assertEqual(
            loads_dataclass(SwizzleDecision, canonical_json(decision), path="decision"),
            decision,
        )
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(decision, id="forged").validate()
        with self.assertRaisesRegex(SchemaError, "baseline reason"):
            SwizzleDecision.create(
                problem=problem,
                baseline=baseline,
                ranked_candidates=(baseline,),
                selected_candidate_ref=baseline.id,
                decision_reason=SwizzleDecisionReason.LOWEST_ESTIMATED_CYCLES,
            )


if __name__ == "__main__":
    unittest.main()
