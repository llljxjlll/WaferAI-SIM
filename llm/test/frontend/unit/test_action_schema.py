from __future__ import annotations

from dataclasses import replace
import math
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.action import (
    BarrierContract,
    BarrierScope,
    ChunkDim,
    ChunkSlice,
    CollectiveAlgorithm,
    ComputeContract,
    ComputeOperand,
    ComputeOperandSlice,
    ComputeTileBinding,
    ConsumerLayoutBinding,
    FusionAction,
    FusionActionKind,
    FusionPlan,
    FUSION_PLAN_SCHEMA_VERSION,
    InversePermutationEntry,
    PermutationEntry,
    RankProgram,
    ReductionContract,
    StandaloneCollectivePlan,
    STANDALONE_COLLECTIVE_PLAN_SCHEMA_VERSION,
    SyncContract,
)
from llm.frontend.wafer_frontend.schema.common import (
    DType,
    MeshAxisName,
    RoundingMode,
    Sharding,
    TensorValue,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind, FusionImpl, ReduceOp
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
    to_primitive,
)

from _fixtures import static_profile, valid_ir1 as base_valid_ir1


def valid_ir1() -> IR1:
    """Action fixture with both explicit GEMM operands required by N4a."""

    base = base_valid_ir1()
    gemm = replace(base.nodes[0], inputs=("p_v_in", "p_v_weight"))
    weight = TensorValue(
        "p_v_weight",
        (256, 128),
        DType.FP16,
        "KN",
        Sharding("mesh_tp", (MeshAxisName.TP, None), ()),
        None,
        (gemm.id,),
        None,
    )
    skeleton = replace(
        base.fused_op_skeletons[0],
        boundary_inputs=gemm.inputs,
        impl=FusionImpl.NONE,
    )
    candidate = replace(base.fusion_candidates[0], boundary_inputs=gemm.inputs)
    fields = base._semantic_key()
    fields.update(
        nodes=(gemm, base.nodes[1]),
        values=base.values + (weight,),
        fusion_candidates=(candidate,),
        fused_op_skeletons=(skeleton,),
    )
    result = IR1.create(producer_pass=base.producer_pass, **fields)
    result.validate()
    return result


def sync(action_id: str) -> SyncContract:
    return SyncContract(f"event_{action_id}", None, None)


def compute_contract(
    reads: tuple[str, ...],
    writes: tuple[str, ...],
    chunk: ChunkSlice,
    rank: int,
) -> ComputeContract:
    node = valid_ir1().nodes[0]
    full = node.workload
    chunk_workload = replace(
        full,
        logical_shape=(chunk.shape[0], full.logical_shape[1], full.logical_shape[2]),
        rank_shape=(chunk.shape[0], full.rank_shape[1], full.rank_shape[2]),
    )
    return ComputeContract(
        node.kind,
        chunk_workload,
        node.math,
        node.effects,
        node.impl_ref,
        tuple(
            ComputeOperand(value_id, role)
            for value_id, role in zip(reads, ("lhs", "rhs"), strict=True)
        ),
        tuple(
            ComputeOperand(value_id, "partial")
            for value_id in writes
        ),
        ComputeTileBinding(
            origin_workload=full,
            input_slices=(
                ComputeOperandSlice(
                    reads[0], node.inputs[0], (chunk.offset[0], 0),
                    (chunk.shape[0], full.logical_shape[2]),
                ),
                ComputeOperandSlice(
                    reads[1], node.inputs[1], (rank * full.rank_shape[2], 0),
                    (full.rank_shape[2], full.logical_shape[1]),
                ),
            ),
            output_slices=(
                ComputeOperandSlice(
                    writes[0], node.outputs[0], chunk.offset, chunk.shape,
                ),
            ),
        ),
    )


def action(
    action_id: str,
    kind: FusionActionKind,
    *,
    member_id: str | None,
    chunk_id: int | None,
    peer_rank: int | None = None,
    route: tuple[int, ...] = (),
    slice_ref: str | None = None,
    bytes: int = 0,
    dtype: DType | None = None,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    channel: str | None = None,
    compute: ComputeContract | None = None,
    reduction: ReductionContract | None = None,
    sync_contract: SyncContract | None = None,
    deps: tuple[str, ...] = (),
    step: int | None = 0,
) -> FusionAction:
    return FusionAction(
        id=action_id,
        kind=kind,
        member_id=member_id,
        chunk_id=chunk_id,
        collective_step=step if chunk_id is not None else None,
        peer_rank=peer_rank,
        expected_route=route,
        slice_ref=slice_ref,
        bytes=bytes,
        dtype=dtype,
        reads=reads,
        writes=writes,
        logical_channel=channel,
        compute=compute,
        reduction=reduction,
        sync=sync_contract or sync(action_id),
        deps=deps,
    )


def valid_plan() -> FusionPlan:
    chunks = (
        ChunkSlice("slice_0", 0, "p_v_out", (0, 0), (16, 128), 4096, 0),
        ChunkSlice("slice_1", 1, "p_v_out", (16, 0), (16, 128), 4096, 1),
    )
    contract = reduction_contract()
    reads = ("p_v_in", "p_v_weight")
    rank0_comp0 = action("r0_comp0", FusionActionKind.COMP, member_id="p_gemm_0", chunk_id=0, slice_ref="slice_0", bytes=4096, dtype=DType.FP16, reads=reads, writes=("partial_0_r0",), compute=compute_contract(reads, ("partial_0_r0",), chunks[0], 0), step=None)
    rank1_comp0 = action("r1_comp0", FusionActionKind.COMP, member_id="p_gemm_0", chunk_id=0, slice_ref="slice_0", bytes=4096, dtype=DType.FP16, reads=reads, writes=("partial_0_r1",), compute=compute_contract(reads, ("partial_0_r1",), chunks[0], 1), step=None)
    send0 = action("r1_send0", FusionActionKind.SEND, member_id="p_rs_0", chunk_id=0, peer_rank=0, route=(1, 0), slice_ref="slice_0", bytes=4096, dtype=DType.FP16, reads=("partial_0_r1",), channel="ch_0", deps=(rank1_comp0.id,))
    recv0 = action("r0_recv0", FusionActionKind.RECV, member_id="p_rs_0", chunk_id=0, peer_rank=1, route=(1, 0), slice_ref="slice_0", bytes=4096, dtype=DType.FP16, writes=("remote_0_r1",), channel="ch_0")
    wait0 = action("r0_wait0", FusionActionKind.WAIT, member_id="p_rs_0", chunk_id=0, sync_contract=SyncContract("event_r0_wait0", recv0.sync.completion_event, None), deps=(recv0.id,))
    reduce0 = action("r0_reduce0", FusionActionKind.REDUCE, member_id="p_rs_0", chunk_id=0, slice_ref="slice_0", bytes=4096, dtype=DType.FP16, reads=("partial_0_r0", "remote_0_r1"), writes=("p_v_out",), reduction=contract, deps=(rank0_comp0.id, wait0.id), step=1)
    rank0_comp1 = action("r0_comp1", FusionActionKind.COMP, member_id="p_gemm_0", chunk_id=1, slice_ref="slice_1", bytes=4096, dtype=DType.FP16, reads=reads, writes=("partial_1_r0",), compute=compute_contract(reads, ("partial_1_r0",), chunks[1], 0), step=None)
    rank1_comp1 = action("r1_comp1", FusionActionKind.COMP, member_id="p_gemm_0", chunk_id=1, slice_ref="slice_1", bytes=4096, dtype=DType.FP16, reads=reads, writes=("partial_1_r1",), compute=compute_contract(reads, ("partial_1_r1",), chunks[1], 1), step=None)
    send1 = action("r0_send1", FusionActionKind.SEND, member_id="p_rs_0", chunk_id=1, peer_rank=1, route=(0, 1), slice_ref="slice_1", bytes=4096, dtype=DType.FP16, reads=("partial_1_r0",), channel="ch_1", deps=(rank0_comp1.id,))
    recv1 = action("r1_recv1", FusionActionKind.RECV, member_id="p_rs_0", chunk_id=1, peer_rank=0, route=(0, 1), slice_ref="slice_1", bytes=4096, dtype=DType.FP16, writes=("remote_1_r0",), channel="ch_1")
    wait1 = action("r1_wait1", FusionActionKind.WAIT, member_id="p_rs_0", chunk_id=1, sync_contract=SyncContract("event_r1_wait1", recv1.sync.completion_event, None), deps=(recv1.id,))
    reduce1 = action("r1_reduce1", FusionActionKind.REDUCE, member_id="p_rs_0", chunk_id=1, slice_ref="slice_1", bytes=4096, dtype=DType.FP16, reads=("remote_1_r0", "partial_1_r1"), writes=("p_v_out",), reduction=contract, deps=(wait1.id, rank1_comp1.id), step=1)
    return FusionPlan.create(
        producer_pass="inter_die_fixture",
        source_ir1_id="ir1_fixture",
        fused_op_id="p_fusion_0",
        group_ref="group_tp",
        impl=FusionImpl.NAIVE,
        profile_key=static_profile(),
        collective_algorithm=CollectiveAlgorithm.DIRECT,
        chunk_dim=ChunkDim.M,
        chunk_count=2,
        chunk_slices=chunks,
        rank_programs=(
            RankProgram(0, (rank0_comp0, recv0, wait0, reduce0, rank0_comp1, send1)),
            RankProgram(1, (rank1_comp0, send0, rank1_comp1, recv1, wait1, reduce1)),
        ),
        input_layout="MN_partial_tp",
        logical_output_layout="MN_shard_tp",
        physical_output_layout="MN_shard_tp",
        output_permutation=(PermutationEntry(0, 0, 0), PermutationEntry(1, 1, 1)),
        inverse_permutation=(InversePermutationEntry(0, 0, 0), InversePermutationEntry(1, 1, 1)),
        consumer_layout_bindings=(),
    )


def valid_standalone_plan() -> StandaloneCollectivePlan:
    plan = valid_plan()
    chunks = plan.chunk_slices
    barrier = BarrierContract("barrier_all_gather", (0, 1), 2, BarrierScope.PLAN)
    local0 = action("ag_local0", FusionActionKind.LOCAL_COPY, member_id="p_rs_0", chunk_id=0, slice_ref="slice_0", bytes=4096, dtype=DType.FP16, reads=("p_v_partial",), writes=("p_v_out",))
    send0 = action("ag_send0", FusionActionKind.SEND, member_id="p_rs_0", chunk_id=0, peer_rank=1, route=(0, 1), slice_ref="slice_0", bytes=4096, dtype=DType.FP16, reads=("p_v_out",), channel="ag_ch_0", deps=(local0.id,))
    recv0 = action("ag_recv0", FusionActionKind.RECV, member_id="p_rs_0", chunk_id=0, peer_rank=0, route=(0, 1), slice_ref="slice_0", bytes=4096, dtype=DType.FP16, writes=("p_v_out",), channel="ag_ch_0")
    local1 = action("ag_local1", FusionActionKind.LOCAL_COPY, member_id="p_rs_0", chunk_id=1, slice_ref="slice_1", bytes=4096, dtype=DType.FP16, reads=("p_v_partial",), writes=("p_v_out",))
    send1 = action("ag_send1", FusionActionKind.SEND, member_id="p_rs_0", chunk_id=1, peer_rank=0, route=(1, 0), slice_ref="slice_1", bytes=4096, dtype=DType.FP16, reads=("p_v_out",), channel="ag_ch_1", deps=(local1.id,))
    recv1 = action("ag_recv1", FusionActionKind.RECV, member_id="p_rs_0", chunk_id=1, peer_rank=1, route=(1, 0), slice_ref="slice_1", bytes=4096, dtype=DType.FP16, writes=("p_v_out",), channel="ag_ch_1")
    barrier0 = action("ag_barrier0", FusionActionKind.BARRIER, member_id="p_rs_0", chunk_id=None, sync_contract=SyncContract("event_ag_done_r0", None, barrier), deps=(local0.id, recv1.id))
    barrier1 = action("ag_barrier1", FusionActionKind.BARRIER, member_id="p_rs_0", chunk_id=None, sync_contract=SyncContract("event_ag_done_r1", None, barrier), deps=(recv0.id, local1.id))
    return StandaloneCollectivePlan.create(
        producer_pass="collective_fixture",
        source_ir1_id=plan.source_ir1_id,
        op_id="p_rs_0",
        algorithm=CollectiveAlgorithm.DIRECT,
        group_ref=plan.group_ref,
        profile_key=plan.profile_key,
        chunk_dim=ChunkDim.M,
        chunk_slices=chunks,
        rank_programs=(
            RankProgram(0, (local0, send0, recv1, barrier0)),
            RankProgram(1, (recv0, local1, send1, barrier1)),
        ),
    )


def bound_standalone_plan() -> tuple[IR1, StandaloneCollectivePlan]:
    ir1 = valid_ir1()
    collective = ir1.nodes[1]
    input_value = replace(
        ir1.values[1],
        producer=None,
        logical_layout="MN_shard_tp",
        sharding=Sharding("mesh_tp", (MeshAxisName.TP, None), ()),
    )
    output_value = replace(
        ir1.values[2],
        logical_layout="MN",
        sharding=Sharding("mesh_tp", (None, None), ()),
    )
    all_gather = replace(
        collective,
        workload=replace(
            collective.workload,
            collective=CollectiveKind.ALL_GATHER,
            reduce_op=None,
            reduction_mesh_axes=(),
            scatter_tensor_axis=None,
            gather_tensor_axis=0,
            rank_input_bytes=4096,
            rank_output_bytes=8192,
            input_layout=input_value.logical_layout,
            output_layout=output_value.logical_layout,
        ),
    )
    ir1_fields = ir1._semantic_key()
    ir1_fields.update(
        instances=(replace(ir1.instances[0], node_ids=(all_gather.id,)),),
        nodes=(all_gather,),
        values=(input_value, output_value),
        edges=(),
        fusion_candidates=(),
        fused_op_skeletons=(),
    )
    all_gather_ir1 = IR1.create(producer_pass=ir1.producer_pass, **ir1_fields)
    template = valid_standalone_plan()
    fields = template._semantic_key()
    fields.update(source_ir1_id=all_gather_ir1.id, profile_key=all_gather_ir1.profile)
    return all_gather_ir1, StandaloneCollectivePlan.create(
        producer_pass=template.producer_pass, **fields
    )


def bound_fusion_plan() -> tuple[IR1, FusionPlan]:
    ir1 = valid_ir1()
    template = valid_plan()
    fields = template._semantic_key()
    fields.update(source_ir1_id=ir1.id, profile_key=ir1.profile)
    return ir1, FusionPlan.create(producer_pass=template.producer_pass, **fields)


def rebind_fusion_plan(ir1: IR1, template: FusionPlan | None = None) -> FusionPlan:
    template = template or valid_plan()
    fields = template._semantic_key()
    fields.update(source_ir1_id=ir1.id, profile_key=ir1.profile)
    return FusionPlan.create(producer_pass=template.producer_pass, **fields)


def recreate_ir1(ir1: IR1, **changes: object) -> IR1:
    fields = ir1._semantic_key()
    fields.update(changes)
    return IR1.create(producer_pass=ir1.producer_pass, **fields)


def replace_plan_action(
    plan: FusionPlan | StandaloneCollectivePlan,
    rank: int,
    action_id: str,
    **changes: object,
) -> FusionPlan | StandaloneCollectivePlan:
    program = plan.rank_programs[rank]
    action_index = next(
        index for index, candidate in enumerate(program.actions) if candidate.id == action_id
    )
    actions = (
        program.actions[:action_index]
        + (replace(program.actions[action_index], **changes),)
        + program.actions[action_index + 1 :]
    )
    programs = (
        plan.rank_programs[:rank]
        + (replace(program, actions=actions),)
        + plan.rank_programs[rank + 1 :]
    )
    changed = replace(plan, rank_programs=programs)
    return type(plan).create(
        producer_pass=plan.producer_pass, **changed._semantic_key()
    )


def reduction_contract(input_ranks: tuple[int, ...] = (0, 1)) -> ReductionContract:
    return ReductionContract(
        ReduceOp.SUM,
        DType.FP16,
        DType.FP32,
        DType.FP16,
        RoundingMode.RNE,
        input_ranks,
    )


class ActionSchemaTest(unittest.TestCase):
    def test_fusion_and_standalone_round_trip_with_stable_digest(self) -> None:
        self.assertEqual(
            FUSION_PLAN_SCHEMA_VERSION,
            "wafer_frontend.fusion_plan/v1alpha10",
        )
        self.assertEqual(
            STANDALONE_COLLECTIVE_PLAN_SCHEMA_VERSION,
            "wafer_frontend.standalone_collective_plan/v1alpha9",
        )
        for artifact_type, artifact in (
            (FusionPlan, valid_plan()),
            (StandaloneCollectivePlan, valid_standalone_plan()),
        ):
            artifact.validate()
            encoded = canonical_json(artifact)
            decoded = loads_dataclass(artifact_type, encoded)
            self.assertEqual(decoded, artifact)
            self.assertEqual(canonical_digest(decoded), canonical_digest(artifact))

    def test_send_recv_pairing_is_strict(self) -> None:
        plan = valid_plan()
        rank1 = plan.rank_programs[1]
        recv_index = next(
            index
            for index, candidate in enumerate(rank1.actions)
            if candidate.kind is FusionActionKind.RECV
        )
        bad_recv = replace(rank1.actions[recv_index], expected_route=(1, 0))
        actions = rank1.actions[:recv_index] + (bad_recv,) + rank1.actions[recv_index + 1 :]
        bad = replace(plan, rank_programs=(plan.rank_programs[0], replace(rank1, actions=actions)))
        with self.assertRaisesRegex(SchemaError, "payloads do not match"):
            bad.validate()

    def test_dangling_dependency_and_cycle_are_rejected(self) -> None:
        plan = valid_plan()
        actions = plan.rank_programs[0].actions
        first = actions[0]
        reduce = next(item for item in actions if item.kind is FusionActionKind.REDUCE)
        for changed in (replace(first, deps=("missing",)), replace(first, deps=(reduce.id,))):
            changed_actions = (changed,) + actions[1:]
            bad_program = replace(plan.rank_programs[0], actions=changed_actions)
            with self.assertRaises(SchemaError):
                replace(plan, rank_programs=(bad_program, plan.rank_programs[1])).validate()

    def test_permutation_and_inverse_must_be_a_bijection(self) -> None:
        plan = valid_plan()
        self.assertEqual(
            tuple(entry.logical_owner_rank for entry in plan.output_permutation),
            (0, 1),
        )
        plan.validate()
        bad = replace(
            plan,
            inverse_permutation=(InversePermutationEntry(0, 1, 0), plan.inverse_permutation[1]),
        )
        with self.assertRaisesRegex(SchemaError, "inverse|physical owner"):
            bad.validate()
        wrong_physical_owner = replace(
            plan,
            output_permutation=(
                replace(plan.output_permutation[0], physical_owner_rank=1),
                plan.output_permutation[1],
            ),
        )
        with self.assertRaisesRegex(SchemaError, "physical owner"):
            wrong_physical_owner.validate()

    def test_unknown_missing_enum_and_uint64_are_rejected(self) -> None:
        raw = to_primitive(valid_plan())
        raw["rank_programs"][0]["actions"][0]["tag_id"] = 7
        with self.assertRaisesRegex(SchemaError, "tag_id"):
            from_data(FusionPlan, raw)
        raw = to_primitive(valid_plan())
        del raw["rank_programs"][0]["actions"][0]["dtype"]
        with self.assertRaisesRegex(SchemaError, "dtype"):
            from_data(FusionPlan, raw)
        raw = to_primitive(valid_plan())
        raw["collective_algorithm"] = "RING"
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            from_data(FusionPlan, raw)
        raw = to_primitive(valid_plan())
        raw["chunk_slices"][0]["owner_rank"] = -1
        with self.assertRaises(SchemaError):
            from_data(FusionPlan, raw)

    def test_plan_cross_refs_to_ir1_are_explicit(self) -> None:
        ir1 = valid_ir1()
        template = valid_plan()
        fields = template._semantic_key()
        fields.update(source_ir1_id=ir1.id, profile_key=ir1.profile)
        plan = FusionPlan.create(producer_pass=template.producer_pass, **fields)
        plan.validate_against(ir1)
        bad_fields = plan._semantic_key()
        bad_fields["group_ref"] = "missing"
        with self.assertRaisesRegex(SchemaError, "group"):
            FusionPlan.create(producer_pass=plan.producer_pass, **bad_fields).validate_against(ir1)

        rank0 = plan.rank_programs[0]
        reduce_index = next(
            index
            for index, candidate in enumerate(rank0.actions)
            if candidate.kind is FusionActionKind.REDUCE
        )
        changed_reduce = replace(
            rank0.actions[reduce_index],
            reduction=replace(
                rank0.actions[reduce_index].reduction, reduce_op=ReduceOp.MAX
            ),
        )
        changed_actions = (
            rank0.actions[:reduce_index]
            + (changed_reduce,)
            + rank0.actions[reduce_index + 1 :]
        )
        semantic_fields = plan._semantic_key()
        semantic_fields["rank_programs"] = (
            replace(rank0, actions=changed_actions),
            plan.rank_programs[1],
        )
        semantic_mismatch = FusionPlan.create(
            producer_pass=plan.producer_pass, **semantic_fields
        )
        with self.assertRaisesRegex(SchemaError, "naive backend v1"):
            semantic_mismatch.validate_against(ir1)

    def test_standalone_chunk_owner_must_be_a_program_rank(self) -> None:
        plan = valid_standalone_plan()
        chunks = (replace(plan.chunk_slices[0], owner_rank=2), plan.chunk_slices[1])
        with self.assertRaisesRegex(SchemaError, "owner|canonical"):
            replace(plan, chunk_slices=chunks).validate()

    def test_reduction_contract_round_trip_preserves_explicit_rank_order(self) -> None:
        contract = reduction_contract((1, 0))
        reduce_action = action(
            "r0_reduce",
            FusionActionKind.REDUCE,
            member_id="p_rs_0",
            chunk_id=0,
            slice_ref="slice_0",
            bytes=2048,
            dtype=DType.FP16,
            reads=("rank_1", "rank_0"),
            writes=("p_v_out",),
            reduction=contract,
        )
        reduce_action.validate("action")
        decoded = loads_dataclass(FusionAction, canonical_json(reduce_action))
        self.assertEqual(decoded, reduce_action)
        self.assertEqual(decoded.reduction.input_ranks, (1, 0))

    def test_reduction_contract_is_required_and_rank_checked(self) -> None:
        contract = reduction_contract()
        base = action(
            "reduce",
            FusionActionKind.REDUCE,
            member_id="p_rs_0",
            chunk_id=0,
            slice_ref="slice_0",
            bytes=2048,
            dtype=DType.FP16,
            reads=("rank_0", "rank_1"),
            writes=("p_v_out",),
            reduction=contract,
        )
        for bad in (
            replace(base, reduction=None),
            replace(base, dtype=DType.FP32),
            replace(base, reduction=replace(contract, input_ranks=())),
            replace(valid_plan().rank_programs[0].actions[0], reduction=contract),
        ):
            with self.assertRaises(SchemaError):
                bad.validate("action")

        plan = valid_plan()
        rank0 = plan.rank_programs[0]
        reduce_index = next(
            index
            for index, candidate in enumerate(rank0.actions)
            if candidate.kind is FusionActionKind.REDUCE
        )
        out_of_range = replace(
            rank0.actions[reduce_index], reduction=reduction_contract((2,))
        )
        changed_actions = (
            rank0.actions[:reduce_index]
            + (out_of_range,)
            + rank0.actions[reduce_index + 1 :]
        )
        fields = plan._semantic_key()
        fields["rank_programs"] = (
            replace(rank0, actions=changed_actions),
            plan.rank_programs[1],
        )
        with self.assertRaisesRegex(SchemaError, "non-program rank"):
            FusionPlan.create(producer_pass=plan.producer_pass, **fields).validate()

    def test_only_direct_algorithms_are_structurally_accepted(self) -> None:
        with self.assertRaisesRegex(SchemaError, "only for DIRECT"):
            replace(
                valid_plan(), collective_algorithm=CollectiveAlgorithm.RING
            ).validate()
        with self.assertRaisesRegex(SchemaError, "only for DIRECT"):
            replace(
                valid_standalone_plan(), algorithm=CollectiveAlgorithm.TREE
            ).validate()

    def test_direct_fusion_requires_complete_contributions_and_reduction(self) -> None:
        plan = valid_plan()
        rank0 = plan.rank_programs[0]
        actions = tuple(
            candidate
            for candidate in rank0.actions
            if candidate.kind is not FusionActionKind.REDUCE
        )
        with self.assertRaisesRegex(SchemaError, "reduce"):
            replace(
                plan,
                rank_programs=(replace(rank0, actions=actions), plan.rank_programs[1]),
            ).validate()

    def test_direct_fusion_enforces_exact_contribution_lineage(self) -> None:
        plan = valid_plan()
        bad_send = replace_plan_action(
            plan, 1, "r1_send0", reads=("unrelated_contribution",)
        )
        with self.assertRaisesRegex(SchemaError, "SEND must read and depend"):
            bad_send.validate()

        bad_recv = replace_plan_action(
            plan, 0, "r0_recv0", writes=("remote_0_r1", "extra_contribution")
        )
        with self.assertRaisesRegex(SchemaError, "single-write RECV"):
            bad_recv.validate()

        bad_reduce = replace_plan_action(
            plan,
            0,
            "r0_reduce0",
            reads=("remote_0_r1", "partial_0_r0"),
        )
        with self.assertRaisesRegex(SchemaError, "exact contribution lineage"):
            bad_reduce.validate()

    def test_action_value_arity_and_sync_payload_are_strict(self) -> None:
        standalone = valid_standalone_plan()
        local = standalone.rank_programs[0].actions[0]
        with self.assertRaisesRegex(SchemaError, "exactly one input and one output"):
            replace(local, writes=("p_v_out", "extra")).validate("action")

        barrier = standalone.rank_programs[0].actions[-1]
        with self.assertRaisesRegex(SchemaError, "cannot read or write"):
            replace(barrier, reads=("p_v_out",)).validate("action")

        wait = action(
            "wait",
            FusionActionKind.WAIT,
            member_id="p_rs_0",
            chunk_id=None,
            writes=("p_v_out",),
            sync_contract=SyncContract("event_wait_done", "event_ready", None),
        )
        with self.assertRaisesRegex(SchemaError, "cannot read or write"):
            wait.validate("action")

    def test_plan_tuple_order_is_canonical(self) -> None:
        fusion = valid_plan()
        for bad in (
            replace(fusion, chunk_slices=tuple(reversed(fusion.chunk_slices))),
            replace(fusion, rank_programs=tuple(reversed(fusion.rank_programs))),
        ):
            with self.assertRaisesRegex(SchemaError, "canonically ordered"):
                bad.validate()
        standalone = valid_standalone_plan()
        for bad in (
            replace(standalone, chunk_slices=tuple(reversed(standalone.chunk_slices))),
            replace(standalone, rank_programs=tuple(reversed(standalone.rank_programs))),
        ):
            with self.assertRaisesRegex(SchemaError, "canonically ordered"):
                bad.validate()

    def test_ir1_binding_checks_dtype_inputs_and_collective_bytes(self) -> None:
        ir1, plan = bound_fusion_plan()
        plan.validate_against(ir1)

        wrong_dtype = replace_plan_action(plan, 0, "r0_comp0", dtype=DType.FP32)
        with self.assertRaisesRegex(SchemaError, "dtype must equal chunk value dtype"):
            wrong_dtype.validate_against(ir1)

        comp = plan.rank_programs[0].actions[0]
        assert comp.compute is not None and comp.compute.tile is not None
        wrong_input_contract = replace(
            comp.compute,
            inputs=(
                replace(comp.compute.inputs[0], value_id="wrong_input"),
                comp.compute.inputs[1],
            ),
            tile=replace(
                comp.compute.tile,
                input_slices=(
                    replace(
                        comp.compute.tile.input_slices[0],
                        operand_id="wrong_input",
                        source_value_id="wrong_input",
                    ),
                    comp.compute.tile.input_slices[1],
                ),
            ),
        )
        wrong_inputs = replace_plan_action(
            plan,
            0,
            comp.id,
            reads=("wrong_input", comp.reads[1]),
            compute=wrong_input_contract,
        )
        with self.assertRaisesRegex(SchemaError, "chunk-local GEMM"):
            wrong_inputs.validate_against(ir1)

        wrong_impl = replace_plan_action(
            plan,
            0,
            comp.id,
            compute=replace(comp.compute, impl_ref="different_impl"),
        )
        with self.assertRaisesRegex(SchemaError, "compute contract disagrees"):
            wrong_impl.validate_against(ir1)

        collective = ir1.nodes[1]
        changed_collective = replace(
            collective,
            workload=replace(
                collective.workload,
                logical_tensor_bytes=8200,
                rank_input_bytes=8200,
                rank_output_bytes=4100,
                rank_logical_payload_bytes=4100,
                group_logical_payload_bytes=8200,
            ),
        )
        ir1_fields = ir1._semantic_key()
        ir1_fields["nodes"] = (ir1.nodes[0], changed_collective)
        changed_ir1 = IR1.create(producer_pass=ir1.producer_pass, **ir1_fields)
        plan_fields = plan._semantic_key()
        plan_fields.update(source_ir1_id=changed_ir1.id, profile_key=changed_ir1.profile)
        byte_mismatch = FusionPlan.create(producer_pass=plan.producer_pass, **plan_fields)
        with self.assertRaisesRegex(SchemaError, "chunk bytes must equal"):
            byte_mismatch.validate_against(changed_ir1)

    def test_fused_gemm_tiles_preserve_global_work_and_exact_logical_slices(self) -> None:
        ir1, plan = bound_fusion_plan()
        plan.validate_against(ir1)
        gemm = ir1.nodes[0]
        full_m, full_n, full_k = gemm.workload.logical_shape
        rank_k = gemm.workload.rank_shape[2]
        comps = tuple(
            (program.rank, candidate)
            for program in plan.rank_programs
            for candidate in program.actions
            if candidate.kind is FusionActionKind.COMP
        )
        self.assertEqual(len(comps), len(plan.rank_programs) * plan.chunk_count)
        self.assertEqual(
            sum(2 * math.prod(candidate.compute.workload.rank_shape) for _, candidate in comps),
            2 * full_m * full_n * full_k,
        )
        for rank, candidate in comps:
            assert candidate.chunk_id is not None
            assert candidate.compute is not None and candidate.compute.tile is not None
            chunk = plan.chunk_slices[candidate.chunk_id]
            tile = candidate.compute.tile
            self.assertEqual(tile.origin_workload, gemm.workload)
            self.assertEqual(
                candidate.compute.workload.logical_shape,
                (chunk.shape[0], full_n, full_k),
            )
            self.assertEqual(
                tuple(item.source_value_id for item in tile.input_slices),
                gemm.inputs,
            )
            self.assertEqual(
                (tile.input_slices[0].logical_offset, tile.input_slices[0].logical_shape),
                ((chunk.offset[0], 0), (chunk.shape[0], full_k)),
            )
            self.assertEqual(
                (tile.input_slices[1].logical_offset, tile.input_slices[1].logical_shape),
                ((rank * rank_k, 0), (rank_k, full_n)),
            )
            self.assertEqual(
                (
                    tile.output_slices[0].source_value_id,
                    tile.output_slices[0].logical_offset,
                    tile.output_slices[0].logical_shape,
                ),
                (gemm.outputs[0], chunk.offset, chunk.shape),
            )

        comp = plan.rank_programs[0].actions[0]
        assert comp.compute is not None and comp.compute.tile is not None
        full_workload = replace_plan_action(
            plan,
            0,
            comp.id,
            compute=replace(comp.compute, workload=gemm.workload),
        )
        with self.assertRaisesRegex(SchemaError, "exact chunk-local GEMM"):
            full_workload.validate_against(ir1)

        local_inputs = ("rank0_chunk0_a", "rank0_chunk0_b")
        local_contract = replace(
            comp.compute,
            inputs=tuple(
                replace(operand, value_id=value_id)
                for operand, value_id in zip(comp.compute.inputs, local_inputs)
            ),
            tile=replace(
                comp.compute.tile,
                input_slices=tuple(
                    replace(binding, operand_id=value_id)
                    for binding, value_id in zip(
                        comp.compute.tile.input_slices, local_inputs
                    )
                ),
            ),
        )
        distinct_operand_ids = replace_plan_action(
            plan,
            0,
            comp.id,
            reads=local_inputs,
            compute=local_contract,
        )
        distinct_operand_ids.validate_against(ir1)
        changed_comp = distinct_operand_ids.rank_programs[0].actions[0]
        assert changed_comp.compute is not None and changed_comp.compute.tile is not None
        self.assertNotEqual(
            tuple(item.operand_id for item in changed_comp.compute.tile.input_slices),
            tuple(item.source_value_id for item in changed_comp.compute.tile.input_slices),
        )

    def test_fused_direct_wait_steps_temp_ids_and_action_order_fail_closed(self) -> None:
        plan = valid_plan()
        comp = plan.rank_programs[0].actions[0]
        wait = next(
            candidate
            for candidate in plan.rank_programs[0].actions
            if candidate.kind is FusionActionKind.WAIT
        )
        wrong_wait = replace_plan_action(
            plan,
            0,
            wait.id,
            sync=replace(wait.sync, wait_event=comp.sync.completion_event),
            deps=(comp.id,),
        )
        with self.assertRaisesRegex(SchemaError, "same-rank RECV"):
            wrong_wait.validate()

        wrong_step = replace_plan_action(
            plan,
            0,
            wait.id,
            collective_step=1,
        )
        with self.assertRaisesRegex(SchemaError, "same-rank RECV"):
            wrong_step.validate()

        comp1 = next(
            candidate
            for candidate in plan.rank_programs[0].actions
            if candidate.id == "r0_comp1"
        )
        assert comp1.compute is not None and comp1.compute.tile is not None
        duplicate_temp_contract = replace(
            comp1.compute,
            outputs=(replace(comp1.compute.outputs[0], value_id="partial_0_r0"),),
            tile=replace(
                comp1.compute.tile,
                output_slices=(
                    replace(
                        comp1.compute.tile.output_slices[0],
                        operand_id="partial_0_r0",
                    ),
                ),
            ),
        )
        duplicate_temp = replace_plan_action(
            plan,
            0,
            comp1.id,
            writes=("partial_0_r0",),
            compute=duplicate_temp_contract,
        )
        with self.assertRaisesRegex(SchemaError, "plan-global unique"):
            duplicate_temp.validate()

        rank0 = plan.rank_programs[0]
        reordered = replace(
            plan,
            rank_programs=(
                replace(
                    rank0,
                    actions=(rank0.actions[1], rank0.actions[0], *rank0.actions[2:]),
                ),
                plan.rank_programs[1],
            ),
        )
        with self.assertRaisesRegex(SchemaError, "canonical chunk/kind/peer order"):
            reordered.validate()

    def test_naive_fusion_impl_owner_layout_and_inverse_are_frozen(self) -> None:
        plan = valid_plan()
        cases = (
            (replace(plan, impl=FusionImpl.NONE), "impl=naive"),
            (
                replace(
                    plan,
                    chunk_slices=(
                        replace(plan.chunk_slices[0], owner_rank=1),
                        replace(plan.chunk_slices[1], owner_rank=0),
                    ),
                ),
                "owner_rank=chunk_id",
            ),
            (replace(plan, physical_output_layout="swizzled"), "physical output layout"),
            (
                replace(
                    plan,
                    consumer_layout_bindings=(
                        ConsumerLayoutBinding(
                            "consumer",
                            "p_v_out",
                            plan.logical_output_layout,
                            True,
                        ),
                    ),
                ),
                "forbids inverse permutation",
            ),
        )
        for bad, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    bad.validate()

    def test_standalone_all_gather_axis_uniform_bytes_and_order_fail_closed(self) -> None:
        ir1, plan = bound_standalone_plan()
        plan.validate_against(ir1)

        all_gather = ir1.nodes[0]
        wrong_axis_ir1 = recreate_ir1(
            ir1,
            nodes=(
                replace(
                    all_gather,
                    workload=replace(all_gather.workload, gather_tensor_axis=1),
                ),
            ),
        )
        fields = plan._semantic_key()
        fields.update(
            source_ir1_id=wrong_axis_ir1.id,
            profile_key=wrong_axis_ir1.profile,
        )
        wrong_axis = StandaloneCollectivePlan.create(
            producer_pass=plan.producer_pass,
            **fields,
        )
        with self.assertRaisesRegex(SchemaError, "gather axis"):
            wrong_axis.validate_against(wrong_axis_ir1)

        uneven_chunks = (
            replace(plan.chunk_slices[0], shape=(15, 128), bytes=3840),
            replace(plan.chunk_slices[1], offset=(15, 0), shape=(17, 128), bytes=4352),
        )
        uneven_programs = tuple(
            replace(
                program,
                actions=tuple(
                    replace(candidate, bytes=uneven_chunks[candidate.chunk_id].bytes)
                    if candidate.chunk_id is not None
                    else candidate
                    for candidate in program.actions
                ),
            )
            for program in plan.rank_programs
        )
        uneven_fields = plan._semantic_key()
        uneven_fields.update(chunk_slices=uneven_chunks, rank_programs=uneven_programs)
        uneven = StandaloneCollectivePlan.create(
            producer_pass=plan.producer_pass,
            **uneven_fields,
        )
        with self.assertRaisesRegex(SchemaError, "uniform canonical owner input shards"):
            uneven.validate_against(ir1)

        owner_swapped = replace(
            plan,
            chunk_slices=(
                replace(plan.chunk_slices[0], owner_rank=1),
                replace(plan.chunk_slices[1], owner_rank=0),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "canonical chunk per rank"):
            owner_swapped.validate()

        rank0 = plan.rank_programs[0]
        reordered = replace(
            plan,
            rank_programs=(
                replace(
                    rank0,
                    actions=(rank0.actions[1], rank0.actions[0], *rank0.actions[2:]),
                ),
                plan.rank_programs[1],
            ),
        )
        with self.assertRaisesRegex(SchemaError, "canonical chunk/kind/peer order"):
            reordered.validate()

    def test_fused_mvp_members_edges_boundaries_and_group_are_exact(self) -> None:
        base = valid_ir1()
        gemm, reduce_scatter = base.nodes
        value_in, value_partial, value_out = base.values[:3]
        weight = base.values[3]
        skeleton = base.fused_op_skeletons[0]

        extra = replace(
            gemm,
            id="p_extra",
            origin_node_id="extra",
            inputs=("p_v_extra_in",),
            outputs=("p_v_extra_out",),
        )
        third_member_ir1 = recreate_ir1(
            base,
            instances=(
                replace(
                    base.instances[0],
                    node_ids=base.instances[0].node_ids + (extra.id,),
                ),
            ),
            nodes=base.nodes + (extra,),
            values=base.values
            + (
                replace(
                    value_in,
                    id="p_v_extra_in",
                    producer=None,
                    consumers=(extra.id,),
                ),
                replace(
                    value_out,
                    id="p_v_extra_out",
                    producer=extra.id,
                    consumers=(),
                ),
            ),
            fused_op_skeletons=(
                replace(
                    skeleton,
                    member_node_ids=skeleton.member_node_ids + (extra.id,),
                ),
            ),
        )

        rs_input = replace(
            value_partial,
            id="p_v_rs_input",
            producer=None,
            consumers=(reduce_scatter.id,),
        )
        missing_edge_ir1 = recreate_ir1(
            base,
            nodes=(
                gemm,
                replace(reduce_scatter, inputs=(rs_input.id,)),
            ),
            values=(
                value_in,
                replace(value_partial, consumers=()),
                rs_input,
                value_out,
                weight,
            ),
            edges=(),
            fusion_candidates=(),
        )
        wrong_boundary_ir1 = recreate_ir1(
            base,
            fused_op_skeletons=(
                replace(skeleton, boundary_inputs=(value_partial.id,)),
            ),
        )
        other_group = replace(
            base.groups[0],
            id="group_other",
            embedding=replace(
                base.groups[0].embedding,
                routes=tuple(
                    replace(route, id=f"{route.id}_other")
                    for route in base.groups[0].embedding.routes
                ),
            ),
        )
        wrong_group_ir1 = recreate_ir1(
            base,
            instances=(
                replace(
                    base.instances[0],
                    group_ids=base.instances[0].group_ids + (other_group.id,),
                ),
            ),
            groups=base.groups + (other_group,),
            nodes=(
                gemm,
                replace(
                    reduce_scatter,
                    execution_group_ref=other_group.id,
                ),
            ),
        )

        cases = (
            (third_member_ir1, "exactly GEMM and ReduceScatter"),
            (missing_edge_ir1, "one internal DATA edge"),
            (wrong_boundary_ir1, "boundaries must exactly"),
            (wrong_group_ir1, "both fused members"),
        )
        for ir1, message in cases:
            with self.subTest(message=message):
                ir1.validate()
                with self.assertRaisesRegex(SchemaError, message):
                    rebind_fusion_plan(ir1).validate_against(ir1)

    def test_collective_workload_dtype_and_layout_match_ir1_values(self) -> None:
        fusion_ir1 = valid_ir1()
        collective = fusion_ir1.nodes[1]
        for name, workload in (
            (
                "layout",
                replace(
                    collective.workload,
                    input_layout="wrong_input_layout",
                ),
            ),
            ("dtype", replace(collective.workload, dtype=DType.FP32)),
        ):
            with self.subTest(collective="reduce_scatter", mismatch=name):
                changed_fusion_ir1 = recreate_ir1(
                    fusion_ir1,
                    nodes=(
                        fusion_ir1.nodes[0],
                        replace(collective, workload=workload),
                    ),
                )
                with self.assertRaisesRegex(SchemaError, "workload dtype/layout"):
                    rebind_fusion_plan(changed_fusion_ir1).validate_against(
                        changed_fusion_ir1
                    )

        ag_ir1, ag_plan = bound_standalone_plan()
        ag = ag_ir1.nodes[0]
        for name, workload in (
            (
                "layout",
                replace(ag.workload, output_layout="wrong_output_layout"),
            ),
            ("dtype", replace(ag.workload, dtype=DType.FP32)),
        ):
            with self.subTest(collective="all_gather", mismatch=name):
                changed_ag_ir1 = recreate_ir1(
                    ag_ir1,
                    nodes=(replace(ag, workload=workload),),
                )
                ag_fields = ag_plan._semantic_key()
                ag_fields.update(
                    source_ir1_id=changed_ag_ir1.id,
                    profile_key=changed_ag_ir1.profile,
                )
                changed_ag_plan = StandaloneCollectivePlan.create(
                    producer_pass=ag_plan.producer_pass,
                    **ag_fields,
                )
                with self.assertRaisesRegex(SchemaError, "workload dtype/layout"):
                    changed_ag_plan.validate_against(changed_ag_ir1)

    def test_standalone_group_must_equal_collective_execution_group(self) -> None:
        ir1, plan = bound_standalone_plan()
        other_group = replace(
            ir1.groups[0],
            id="group_other",
            embedding=replace(
                ir1.groups[0].embedding,
                routes=tuple(
                    replace(route, id=f"{route.id}_other")
                    for route in ir1.groups[0].embedding.routes
                ),
            ),
        )
        changed_ir1 = recreate_ir1(
            ir1,
            instances=(
                replace(
                    ir1.instances[0],
                    group_ids=ir1.instances[0].group_ids + (other_group.id,),
                ),
            ),
            groups=ir1.groups + (other_group,),
            nodes=(replace(ir1.nodes[0], execution_group_ref=other_group.id),),
        )
        fields = plan._semantic_key()
        fields.update(
            source_ir1_id=changed_ir1.id,
            profile_key=changed_ir1.profile,
        )
        changed_plan = StandaloneCollectivePlan.create(
            producer_pass=plan.producer_pass,
            **fields,
        )
        with self.assertRaisesRegex(SchemaError, "must equal the op execution group"):
            changed_plan.validate_against(changed_ir1)

    def test_compute_and_sync_contracts_cannot_be_empty_or_implicit(self) -> None:
        comp = valid_plan().rank_programs[0].actions[0]
        for bad in (
            replace(comp, compute=None),
            replace(comp, sync=SyncContract("", None, None)),
            replace(comp, compute=replace(comp.compute, outputs=())),
            replace(comp, compute=replace(comp.compute, impl_ref="")),
        ):
            with self.assertRaises(SchemaError):
                bad.validate("action")
        barrier = BarrierContract("barrier", (0, 1), 1, BarrierScope.PLAN)
        with self.assertRaisesRegex(SchemaError, "participant_ranks"):
            barrier.validate("barrier")

    def test_chunk_cover_is_contiguous_disjoint_and_full(self) -> None:
        ir1 = valid_ir1()
        plan = valid_plan()
        fields = plan._semantic_key()
        fields.update(source_ir1_id=ir1.id, profile_key=ir1.profile)
        chunks = (
            plan.chunk_slices[0],
            replace(plan.chunk_slices[1], offset=(15, 0)),
        )
        fields["chunk_slices"] = chunks
        changed = FusionPlan.create(producer_pass=plan.producer_pass, **fields)
        with self.assertRaisesRegex(SchemaError, "contiguous and disjoint"):
            changed.validate_against(ir1)

    def test_standalone_direct_all_gather_uses_local_copy_not_fake_comp(self) -> None:
        all_gather_ir1, bound = bound_standalone_plan()
        bound.validate_against(all_gather_ir1)
        self.assertFalse(
            any(
                candidate.kind is FusionActionKind.COMP
                for program in bound.rank_programs
                for candidate in program.actions
            )
        )
        self.assertEqual(
            sum(
                candidate.kind is FusionActionKind.LOCAL_COPY
                for program in bound.rank_programs
                for candidate in program.actions
            ),
            len(bound.chunk_slices),
        )
        ir1 = valid_ir1()
        template = valid_standalone_plan()
        rs_fields = template._semantic_key()
        rs_fields.update(source_ir1_id=ir1.id, profile_key=ir1.profile)
        rs_plan = StandaloneCollectivePlan.create(
            producer_pass=template.producer_pass, **rs_fields
        )
        with self.assertRaisesRegex(SchemaError, "only AllGather"):
            rs_plan.validate_against(ir1)

    def test_standalone_value_roles_cannot_use_unowned_temporaries(self) -> None:
        ir1, plan = bound_standalone_plan()
        for bad, message in (
            (
                replace_plan_action(plan, 0, "ag_local0", writes=("placed_chunk",)),
                "LOCAL_COPY must read input and write output",
            ),
            (
                replace_plan_action(plan, 0, "ag_send0", reads=("placed_chunk",)),
                "AllGather SEND must read output value",
            ),
            (
                replace_plan_action(plan, 0, "ag_recv1", writes=("placed_chunk",)),
                "AllGather RECV must write output value",
            ),
        ):
            with self.assertRaisesRegex(SchemaError, message):
                bad.validate_against(ir1)


if __name__ == "__main__":
    unittest.main()
