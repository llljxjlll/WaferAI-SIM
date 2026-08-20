from __future__ import annotations

from dataclasses import replace
from math import prod
from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes import (
    PipelinePhase,
    build_global_bundle,
    partition_bundle,
    plan_bundle,
    project_bundle,
    schedule_bundle,
)
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.load_fabric import load_physical_fabric
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.passes.placement import (
    place_bundle,
    validate_placement_against,
)
from llm.frontend.wafer_frontend.passes.stage2_dense_forward_oracle import (
    build_stage2_dense_forward_oracle,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.passes.validate_logical_bundle import (
    DenseLogicalBundleValidator,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import (
    AttentionWorkload,
    CollectiveWorkload,
    CrossEntropyForwardWorkload,
    CrossEntropyReduction,
    EdgeKind,
    EmbeddingWorkload,
    GemmPartition,
    GemmWorkload,
    GraphEdge,
    GreedySampleWorkload,
    IR0,
    OpKind,
    ResidualWorkload,
    RmsNormWorkload,
    RopeQkWorkload,
    SwiGluWorkload,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1, SramAllocator
from llm.frontend.wafer_frontend.schema.logical import (
    ExpandedIR0Bundle,
    ExpandedProfileIR0,
    IR0Template,
)
from llm.frontend.wafer_frontend.schema.n5 import ProjectToIR2Context
from llm.frontend.wafer_frontend.schema.n4 import FusionPartitionContext
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferAccess,
    BufferUseRole,
    OrdinaryNodeOrigin,
    SemanticTaskKind,
)
from llm.frontend.wafer_frontend.schema.persistent_state import (
    HbmBinding,
    PersistentStateManifest,
    StateKind,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import from_data

from _fixtures import (
    naive_inter_die_planning_context,
    naive_intra_die_scheduling_context,
    valid_hbm_address_spaces,
    valid_spec,
)
from test_n4_pipeline import _compile_through_n4


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "llm/test/sram/hardware_numa.json"
_HARDWARE_TP4 = _ROOT / "notes/frontend/examples/hardware_2x2.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"


def _spec(*, output: str = "logits", tp: int = 1) -> ExperimentSpec:
    raw = valid_spec()
    raw["model"].update(  # type: ignore[index]
        V=32,
        H=16,
        I=32,
        NH=4,
        KVH=4,
        DH=4,
        rotary_dim=4,
        L=2,
        max_position_embeddings=128,
    )
    raw["parallel"]["instances"][0].update(tp=tp, sp=tp > 1)  # type: ignore[index]
    raw["workload"]["infer"].update(output=output)  # type: ignore[index]
    raw["workload"]["infer"]["profile"].update(  # type: ignore[index]
        prefill_tokens=8,
        decode_tokens=0,
        num_seqs=1,
        context_sum=8,
        context_max=8,
        kv_pages=1,
    )
    return from_data(ExperimentSpec, raw, path="spec")


def _template(*, output: str = "logits", tp: int = 1) -> IR0Template:
    return build_ir0(_spec(output=output, tp=tp))


def _rebuild_graph(graph: IR0, **updates: object) -> IR0:
    fields: dict[str, object] = {
        "producer_pass": graph.producer_pass,
        "job": graph.job,
        "instances": graph.instances,
        "nodes": graph.nodes,
        "values": graph.values,
        "edges": graph.edges,
        "fusion_candidates": graph.fusion_candidates,
        "persistent_states": graph.persistent_states,
        "state_accesses": graph.state_accesses,
        "profile": graph.profile,
        "train": graph.train,
    }
    fields.update(updates)
    return IR0.create(**fields)  # type: ignore[arg-type]


def _bundle(template: IR0Template, graph: IR0) -> ExpandedIR0Bundle:
    entry = ExpandedProfileIR0.create(
        source_template_id=template.id,
        weight=template.profiles[0].weight,
        graph=graph,
    )
    return ExpandedIR0Bundle.create(source_template=template, entries=(entry,))


def _rebuild_ir1(graph: IR1, **updates: object) -> IR1:
    fields: dict[str, object] = {
        "producer_pass": graph.producer_pass,
        "source_ir0_id": graph.source_ir0_id,
        "profile": graph.profile,
        "fabric": graph.fabric,
        "instances": graph.instances,
        "groups": graph.groups,
        "nodes": graph.nodes,
        "values": graph.values,
        "edges": graph.edges,
        "fusion_candidates": graph.fusion_candidates,
        "fused_op_skeletons": graph.fused_op_skeletons,
        "cross_routes": graph.cross_routes,
        "state_accesses": graph.state_accesses,
        "persistent_state_manifest": graph.persistent_state_manifest,
    }
    fields.update(updates)
    return IR1.create(**fields)  # type: ignore[arg-type]


def _project_tiny(
    tp: int,
    *,
    output: str = "logits",
    large_sram: bool = False,
    block_allocator: bool = False,
):
    spec = _spec(tp=tp, output=output)
    template = build_ir0(spec)
    expanded = logical_expand(template)
    fabric = load_physical_fabric(
        _HARDWARE_TP4 if tp == 4 else _HARDWARE,
        _MAPPING,
    )
    if large_sram:
        fabric = replace(
            fabric,
            sram_profiles=tuple(
                replace(
                    profile,
                    capacity_bytes=64 * 1024,
                    regions=tuple(
                        replace(
                            region,
                            base_bytes=(
                                0 if index == 0 else (32 + 8 * (index - 1)) * 1024
                            ),
                            size_bytes=(32 if index == 0 else 8) * 1024,
                        )
                        for index, region in enumerate(profile.regions)
                    ),
                )
                for profile in fabric.sram_profiles
            ),
        )
    if block_allocator:
        fabric = replace(
            fabric,
            sram_profiles=tuple(
                replace(
                    profile,
                    regions=tuple(
                        replace(region, allocator=SramAllocator.BLOCK)
                        for region in profile.regions
                    ),
                )
                for profile in fabric.sram_profiles
            ),
        )
    placement_context = PlacementContext.create(
        producer_pass="stage2_dense_forward_projection",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    placed = place_bundle(expanded, placement_context)
    partitioned = partition_bundle(
        placed,
        FusionPartitionContext.create(
            producer_pass="stage2_dense_forward_projection"
        ),
    )
    planning_context = naive_inter_die_planning_context(
        "stage2_dense_forward_projection"
    )
    planned = plan_bundle(partitioned, planning_context)
    projection_context = ProjectToIR2Context.create(
        producer_pass="stage2_dense_forward_projection",
        state_transfers=(),
    )
    projected = project_bundle(planned, projection_context)
    projected.validate_against(planned, projection_context)
    return planned, projected



def _schedule_tiny(tp: int, *, output: str = "logits", block_allocator: bool = False):
    planned, projected = _project_tiny(
        tp,
        output=output,
        large_sram=True,
        block_allocator=block_allocator,
    )
    context = naive_intra_die_scheduling_context(
        "stage2_dense_forward_schedule"
    )
    scheduled = schedule_bundle(projected, context)
    scheduled.validate_against(projected, context)
    global_bundle = build_global_bundle(scheduled)
    global_bundle.validate_against(scheduled)
    return planned, projected, context, scheduled, global_bundle

class Stage2DenseForwardGraphTest(unittest.TestCase):
    def test_tp1_tp2_tp4_schedule_and_global_compute_contracts(self) -> None:
        expected_roles = {
            EmbeddingWorkload: ("indices", "table", "activation"),
            RmsNormWorkload: ("activation", "weight", "normalized"),
            RopeQkWorkload: ("packed_qkv", "packed_qkv"),
            AttentionWorkload: ("packed_qkv", "attention_output"),
            SwiGluWorkload: ("gate_up", "swiglu"),
            ResidualWorkload: ("residual", "branch", "output"),
        }
        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                (
                    _planned,
                    projected,
                    context,
                    scheduled,
                    global_bundle,
                ) = _schedule_tiny(tp)
                self.assertEqual(
                    context.policy.implementation_schema_version,
                    "wafer_frontend.naive_intra_die_policy/v6",
                )
                self.assertEqual(
                    scheduled.schema_version,
                    "wafer_frontend.scheduled_ir2_bundle/v1alpha5",
                )
                schedule_set = scheduled.entries[0].schedule_set
                self.assertEqual(
                    schedule_set.schema_version,
                    "wafer_frontend.intra_die_schedule_set/v1alpha7",
                )
                self.assertTrue(
                    all(
                        schedule.schema_version
                        == "wafer_frontend.intra_die_schedule/v1alpha12"
                        for schedule in schedule_set.schedules
                    )
                )
                self.assertEqual(
                    global_bundle.schema_version,
                    "wafer_frontend.global_action_bundle/v1alpha5",
                )
                global_dag = global_bundle.entries[0].global_dag
                self.assertEqual(
                    global_dag.schema_version,
                    "wafer_frontend.global_action_dag/v1alpha8",
                )
                self.assertTrue(
                    all(
                        action.schema_version
                        == "wafer_frontend.global_action/v1alpha6"
                        for action in global_dag.actions
                    )
                )

                schedules = {
                    schedule.id: schedule for schedule in schedule_set.schedules
                }
                seen: set[type[object]] = set()
                for action in global_dag.actions:
                    if (
                        action.task_kind is not SemanticTaskKind.COMP
                        or action.compute is None
                    ):
                        continue
                    workload_type = type(action.compute.workload)
                    roles = expected_roles.get(workload_type)
                    if roles is None:
                        continue
                    seen.add(workload_type)
                    actual_roles = tuple(
                        operand.role for operand in action.compute.inputs
                    ) + tuple(
                        operand.role for operand in action.compute.outputs
                    )
                    self.assertEqual(actual_roles, roles)
                    expected_uses = tuple(
                        (BufferUseRole.COMP_INPUT, index, BufferAccess.READ)
                        for index in range(len(action.compute.inputs))
                    ) + tuple(
                        (BufferUseRole.COMP_OUTPUT, index, BufferAccess.WRITE)
                        for index in range(len(action.compute.outputs))
                    )
                    action_uses = tuple(
                        (use.role, use.operand_index, use.access)
                        for use in action.buffer_uses
                    )
                    self.assertEqual(action_uses, expected_uses)

                    schedule = schedules[action.source.schedule_id]
                    scheduled_uses = tuple(
                        (use.role, use.operand_index, use.access)
                        for use in schedule.task_buffer_uses
                        if use.task_id == action.source.task_id
                    )
                    self.assertEqual(scheduled_uses, expected_uses)

                    if workload_type is AttentionWorkload:
                        self.assertEqual(
                            tuple(
                                operand.role for operand in action.compute.inputs
                            ),
                            ("packed_qkv",),
                        )
                        self.assertEqual(len(action.compute.inputs), 1)
                    if workload_type is EmbeddingWorkload:
                        index_use = action.buffer_uses[0]
                        binding = next(
                            item
                            for item in schedule.buffer_bindings
                            if item.id == index_use.binding_id
                        )
                        self.assertIs(binding.dtype, DType.INT32)
                        self.assertEqual(
                            binding.size_bytes,
                            prod(binding.tensor_slice.shape) * 4,
                        )
                self.assertEqual(seen, set(expected_roles))

    def test_greedy_int32_schedule_global_and_tamper_are_exact(self) -> None:
        (
            _planned,
            projected,
            _context,
            scheduled,
            global_bundle,
        ) = _schedule_tiny(1, output="greedy_sample")
        global_dag = global_bundle.entries[0].global_dag
        greedy = next(
            action
            for action in global_dag.actions
            if action.member_id == "P0.greedy_sample"
        )
        assert greedy.compute is not None
        self.assertEqual(
            tuple(operand.role for operand in greedy.compute.inputs),
            ("logits",),
        )
        self.assertEqual(
            tuple(operand.role for operand in greedy.compute.outputs),
            ("sample_ids",),
        )
        self.assertEqual(
            tuple(
                (use.role, use.operand_index, use.access)
                for use in greedy.buffer_uses
            ),
            (
                (BufferUseRole.COMP_INPUT, 0, BufferAccess.READ),
                (BufferUseRole.COMP_OUTPUT, 0, BufferAccess.WRITE),
            ),
        )
        schedule_set = scheduled.entries[0].schedule_set
        schedule = next(
            item
            for item in schedule_set.schedules
            if item.id == greedy.source.schedule_id
        )
        output_binding = next(
            item
            for item in schedule.buffer_bindings
            if item.id == greedy.buffer_uses[1].binding_id
        )
        self.assertIs(output_binding.dtype, DType.INT32)
        self.assertEqual(
            output_binding.size_bytes,
            prod(output_binding.tensor_slice.shape) * 4,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "COMP buffer uses must exactly cover ComputeContract roles/arity",
        ):
            replace(
                greedy,
                buffer_uses=greedy.buffer_uses[:-1],
            ).validate("greedy")

        embedding_task = next(
            task
            for dag in projected.entries[0].projection.dags
            for task in dag.tasks
            if task.member_id == "P0.embedding"
        )
        embedding_schedule = next(
            item
            for item in schedule_set.schedules
            if item.dag_id
            == next(
                dag.id
                for dag in projected.entries[0].projection.dags
                if embedding_task in dag.tasks
            )
        )
        uses = tuple(
            replace(use, access=BufferAccess.WRITE)
            if use.task_id == embedding_task.id
            and use.role is BufferUseRole.COMP_INPUT
            and use.operand_index == 0
            else use
            for use in embedding_schedule.task_buffer_uses
        )
        semantic_key = embedding_schedule._semantic_key()
        semantic_key["task_buffer_uses"] = uses
        bad_schedule = type(embedding_schedule).create(
            producer_pass=embedding_schedule.producer_pass,
            **semantic_key,
        )
        embedding_dag = next(
            dag
            for dag in projected.entries[0].projection.dags
            if dag.id == embedding_schedule.dag_id
        )
        with self.assertRaisesRegex(
            SchemaError,
            "rank/access/value disagrees",
        ):
            bad_schedule.validate_against(
                embedding_dag,
                scheduled.entries[0].graph,
                "bad_schedule",
            )

    def test_2kib_sram_full_forward_fails_closed(self) -> None:
        _planned, projected = _project_tiny(1)
        with self.assertRaisesRegex(
            SchemaError,
            "required=3200, available=2048",
        ):
            schedule_bundle(
                projected,
                naive_intra_die_scheduling_context("stage2_small_sram"),
            )

    def test_n5_stage2_carrier_publishes_exact_projection(self) -> None:
        (
            manager,
            _placed,
            _placement_context,
            _partitioned,
            _partition_context,
            planned,
            _planning_context,
        ) = _compile_through_n4()
        context = ProjectToIR2Context.create(
            producer_pass="stage2_dense_forward_boundary",
            state_transfers=(),
        )
        projected = manager.run_pass(
            "project_to_ir2",
            planned,
            project_bundle,
            context=context,
        )
        projected.validate_against(planned, context)
        self.assertIs(manager.snapshot.phase, PipelinePhase.IR2_PROJECTED)
        self.assertEqual(len(manager.snapshot.receipts), 6)
        self.assertEqual(manager.snapshot.receipts[-1].pass_name, "project_to_ir2")

    def test_ce_remains_schema_only_and_projection_rolls_back(self) -> None:
        planned, _projected = _project_tiny(1)
        graph = planned.entries[0].graph
        lm_head = next(node for node in graph.nodes if node.id == "P0.lm_head")
        logits = next(value for value in graph.values if value.id == "P0.logits")
        token_ids = next(
            value for value in graph.values if value.id == "P0.token_ids"
        )
        ce_id = "P0.ce_forward"
        labels = replace(
            token_ids,
            id="P0.labels",
            logical_layout="M_labels",
            consumers=(ce_id,),
        )
        losses = replace(
            token_ids,
            id="P0.losses",
            dtype=DType.FP32,
            logical_layout="M_losses",
            producer=ce_id,
            consumers=(),
        )
        ce = replace(
            lm_head,
            id=ce_id,
            origin_node_id=ce_id,
            kind=OpKind.CE_FORWARD,
            inputs=(logits.id, labels.id),
            outputs=(losses.id,),
            workload=CrossEntropyForwardWorkload(
                profile=graph.profile,
                reduction=CrossEntropyReduction.NONE,
                logical_logits_shape=(8, 32),
                rank_logits_shape=(8, 32),
                logical_label_shape=(8,),
                rank_label_shape=(8,),
                logical_loss_shape=(8,),
                rank_loss_shape=(8,),
                logits_dtype=DType.FP16,
                label_dtype=DType.INT32,
                loss_dtype=DType.FP32,
            ),
            impl_ref="cross_entropy_forward",
        )
        fields = graph._semantic_key()
        fields.update(
            instances=(
                replace(
                    graph.instances[0],
                    node_ids=graph.instances[0].node_ids + (ce.id,),
                ),
            ),
            nodes=graph.nodes + (ce,),
            values=tuple(
                replace(value, consumers=(ce.id,))
                if value.id == logits.id
                else value
                for value in graph.values
            )
            + (labels, losses),
            edges=graph.edges
            + (
                GraphEdge(
                    "P0.edge.lm_head.ce",
                    EdgeKind.DATA,
                    lm_head.id,
                    ce.id,
                    logits.id,
                ),
            ),
        )
        ce_graph = IR1.create(producer_pass=graph.producer_pass, **fields)
        ce_graph.validate()

        def reject_ce(_source, context):
            context.validate()
            return NaiveProjectToIR2().run(
                ce_graph, (), (), state_transfers=()
            )

        (
            manager,
            _placed,
            _placement_context,
            _partitioned,
            _partition_context,
            manager_planned,
            _planning_context,
        ) = _compile_through_n4()
        context = ProjectToIR2Context.create(
            producer_pass="stage2_ce_rollback",
            state_transfers=(),
        )
        before = manager.snapshot
        with self.assertRaisesRegex(
            UnsupportedFeatureError,
            "carrier support is not implemented.*ce_forward",
        ):
            manager.run_pass(
                "project_to_ir2",
                manager_planned,
                reject_ce,
                context=context,
            )
        self.assertEqual(manager.snapshot, before)
        self.assertIs(manager.snapshot.phase, PipelinePhase.INTERDIE_PLANNED)

    def test_tp1_tp2_tp4_projection_counts_roles_and_fusion_exclusion(self) -> None:
        expected = {
            1: ((44, 0), 25),
            2: ((80, 80), 58),
            4: ((136, 136, 136, 136), 148),
        }
        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                planned, projected = _project_tiny(tp)
                projection = projected.entries[0].projection
                self.assertEqual(
                    planned.schema_version,
            "wafer_frontend.inter_die_plan_bundle/v1alpha4",
                )
                self.assertEqual(
                    projected.schema_version,
                    "wafer_frontend.projected_ir2_bundle/v1alpha4",
                )
                self.assertEqual(
                    projection.schema_version,
                    "wafer_frontend.ir2_projection_result/v1alpha10",
                )
                self.assertTrue(
                    all(
                        dag.schema_version
                        == "wafer_frontend.intra_die_dag/v1alpha11"
                        for dag in projection.dags
                    )
                )
                self.assertTrue(
                    all(
                        plan.schema_version
                        == "wafer_frontend.fusion_plan/v1alpha10"
                        for plan in planned.entries[0].fusion_plans
                    )
                )
                self.assertEqual(
                    tuple(len(dag.tasks) for dag in projection.dags),
                    expected[tp][0],
                )
                self.assertEqual(
                    sum(
                        task.compute is not None
                        for dag in projection.dags
                        for task in dag.tasks
                    ),
                    expected[tp][1],
                )
                ordinary = {
                    task.member_id: task
                    for dag in projection.dags
                    for task in dag.tasks
                    if isinstance(task.origin_ref, OrdinaryNodeOrigin)
                    and task.compute is not None
                }
                expected_roles = {
                    "P0.embedding": (
                        ("indices", "table"),
                        ("activation",),
                    ),
                    "P0.layer0.norm1": (
                        ("activation", "weight"),
                        ("normalized",),
                    ),
                    "P0.layer0.qkv": (("lhs", "rhs"), ("output",)),
                    "P0.layer0.rope": (
                        ("packed_qkv",),
                        ("packed_qkv",),
                    ),
                    "P0.layer0.attention": (
                        ("packed_qkv",),
                        ("attention_output",),
                    ),
                    "P0.layer0.swiglu": (("gate_up",), ("swiglu",)),
                    "P0.layer0.residual1": (
                        ("residual", "branch"),
                        ("output",),
                    ),
                    "P0.lm_head": (("lhs", "rhs"), ("output",)),
                }
                for node_id, roles in expected_roles.items():
                    task = ordinary[node_id]
                    assert task.compute is not None
                    self.assertEqual(
                        tuple(item.role for item in task.compute.inputs),
                        roles[0],
                    )
                    self.assertEqual(
                        tuple(item.role for item in task.compute.outputs),
                        roles[1],
                    )
                fused_members = {
                    member_id
                    for skeleton in planned.entries[0].graph.fused_op_skeletons
                    for member_id in skeleton.member_node_ids
                }
                self.assertNotIn("P0.lm_head", fused_members)

        _planned, projected = _project_tiny(1)
        dag = projected.entries[0].projection.dags[0]
        embedding = next(
            task for task in dag.tasks if task.member_id == "P0.embedding"
        )
        assert embedding.compute is not None
        for label, bad_compute, message in (
            (
                "input_role",
                replace(
                    embedding.compute,
                    inputs=(
                        replace(embedding.compute.inputs[0], role="table"),
                        embedding.compute.inputs[1],
                    ),
                ),
                "roles/arity",
            ),
            (
                "missing_input",
                replace(
                    embedding.compute,
                    inputs=embedding.compute.inputs[:1],
                ),
                "roles/arity",
            ),
            (
                "output_role",
                replace(
                    embedding.compute,
                    outputs=(
                        replace(
                            embedding.compute.outputs[0],
                            role="output",
                        ),
                    ),
                ),
                "roles/arity",
            ),
            (
                "workload_kind",
                replace(embedding.compute, op_kind=OpKind.ROPE),
                "workload type",
            ),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                SchemaError, message
            ):
                replace(embedding, compute=bad_compute).validate("embedding")

    def test_tp1_greedy_projection_is_exact_and_typed(self) -> None:
        planned, projected = _project_tiny(1, output="greedy_sample")
        projection = projected.entries[0].projection
        self.assertEqual(
            tuple(len(dag.tasks) for dag in projection.dags),
            (45, 0),
        )
        greedy = next(
            task
            for dag in projection.dags
            for task in dag.tasks
            if task.member_id == "P0.greedy_sample"
        )
        assert greedy.compute is not None
        self.assertEqual(
            tuple(item.role for item in greedy.compute.inputs),
            ("logits",),
        )
        self.assertEqual(
            tuple(item.role for item in greedy.compute.outputs),
            ("sample_ids",),
        )
        self.assertEqual(
            greedy.write_values,
            ("P0.sampled_ids",),
        )
        fused_members = {
            member_id
            for skeleton in planned.entries[0].graph.fused_op_skeletons
            for member_id in skeleton.member_node_ids
        }
        self.assertNotIn(greedy.member_id, fused_members)

    def test_multi_profile_oracles_validate_each_profile_independently(self) -> None:
        raw = valid_spec()
        raw["model"].update(  # type: ignore[index]
            V=32,
            H=16,
            I=32,
            NH=4,
            KVH=4,
            DH=4,
            rotary_dim=4,
            L=2,
            max_position_embeddings=128,
        )
        raw["parallel"]["instances"][0].update(tp=2, sp=True)  # type: ignore[index]
        raw["workload"]["infer"] = {  # type: ignore[index]
            "source": "shape_dist",
            "output": "logits",
            "shape_dist": {
                "profiles": [
                    {
                        "key": {
                            "prefill_tokens": tokens,
                            "decode_tokens": 0,
                            "num_seqs": 1,
                            "context_sum": tokens,
                            "context_max": tokens,
                            "kv_pages": 1,
                            "expert_load": None,
                        },
                        "weight": weight,
                    }
                    for tokens, weight in ((8, 0.4), (16, 0.6))
                ]
            },
        }
        spec = from_data(ExperimentSpec, raw, path="spec")
        template = build_ir0(spec)
        bundle = logical_expand(template)
        self.assertEqual(len(bundle.entries), 2)

        fabric = load_physical_fabric(_HARDWARE, _MAPPING)
        context = PlacementContext.create(
            producer_pass="load_physical_fabric",
            fabric=fabric,
            placement=spec.placement,
            hbm_address_spaces=valid_hbm_address_spaces(fabric),
        )
        placed = place_bundle(bundle, context)
        validate_placement_against(placed, bundle, context)

        for source_entry, placed_entry in zip(bundle.entries, placed.entries):
            graph = source_entry.graph
            oracle = build_stage2_dense_forward_oracle(
                template,
                graph.profile,
                tp_degree=2,
            )
            oracle.validate_against_ir0(template, graph)
            oracle.validate_against_ir1(template, graph, placed_entry.graph)

    def test_tp1_logits_full_graph_is_exact(self) -> None:
        template = _template()
        bundle = logical_expand(template)
        graph = bundle.entries[0].graph
        DenseIR0Validator.validate(graph)
        DenseLogicalBundleValidator.validate(template, bundle)
        oracle = build_stage2_dense_forward_oracle(
            template, graph.profile, tp_degree=1
        )
        oracle.validate_against_ir0(template, graph)

        self.assertEqual((len(graph.nodes), len(graph.values), len(graph.edges)), (25, 41, 28))
        self.assertEqual(graph.nodes[0].id, "P0.embedding")
        self.assertEqual(graph.nodes[-2].id, "P0.final_norm")
        self.assertEqual(graph.nodes[-1].id, "P0.lm_head")
        self.assertIs(type(graph.nodes[0].workload), EmbeddingWorkload)
        self.assertIs(type(graph.nodes[-2].workload), RmsNormWorkload)
        self.assertIs(type(graph.nodes[-1].workload), GemmWorkload)
        self.assertIs(graph.nodes[-1].workload.partition, GemmPartition.REPLICATED)
        self.assertEqual(graph.values[0].dtype, DType.INT32)
        self.assertEqual(graph.values[-1].id, "P0.logits")
        self.assertEqual(
            tuple(node.id for node in graph.nodes if node.kind is OpKind.ROPE),
            ("P0.layer0.rope", "P0.layer1.rope"),
        )
        self.assertTrue(
            all(type(node.workload) is RopeQkWorkload for node in graph.nodes if node.kind is OpKind.ROPE)
        )

        parameters = tuple(
            state
            for state in graph.persistent_states
            if state.identity.kind is StateKind.PARAMETER
        )
        kv = tuple(
            state
            for state in graph.persistent_states
            if state.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        self.assertEqual((len(parameters), len(kv), len(graph.state_accesses)), (15, 4, 19))
        self.assertEqual(sum(2 * prod(state.shape) for state in parameters), 12448)
        self.assertEqual(
            {
                state.identity.tensor_ref
                for state in parameters
                if state.identity.tensor_ref in {
                    "P0.tok_embeddings.weight",
                    "P0.final_norm.weight",
                    "P0.lm_head.weight",
                }
            },
            {
                "P0.tok_embeddings.weight",
                "P0.final_norm.weight",
                "P0.lm_head.weight",
            },
        )

    def test_tp1_greedy_adds_one_typed_terminal(self) -> None:
        template = _template(output="greedy_sample")
        bundle = logical_expand(template)
        graph = bundle.entries[0].graph
        self.assertEqual((len(graph.nodes), len(graph.values), len(graph.edges)), (26, 42, 29))
        sample = graph.nodes[-1]
        self.assertEqual(sample.id, "P0.greedy_sample")
        self.assertIs(type(sample.workload), GreedySampleWorkload)
        self.assertEqual(graph.values[-1].id, "P0.sampled_ids")
        self.assertIs(graph.values[-1].dtype, DType.INT32)

    def test_tp2_tp4_shapes_states_and_ir1_placement_are_exact(self) -> None:
        expected_bytes = {2: 14656, 4: 19072}
        for tp in (2, 4):
            with self.subTest(tp=tp):
                fabric = load_physical_fabric(
                    _HARDWARE_TP4 if tp == 4 else _HARDWARE,
                    _MAPPING,
                )
                spec = _spec(tp=tp)
                template = build_ir0(spec)
                bundle = logical_expand(template)
                graph = bundle.entries[0].graph
                oracle = build_stage2_dense_forward_oracle(
                    template, graph.profile, tp_degree=tp
                )
                oracle.validate_against_ir0(template, graph)
                self.assertEqual(
                    (len(graph.nodes), len(graph.values), len(graph.edges)),
                    (33, 49, 36),
                )
                embedding = graph.nodes[0].workload
                self.assertIs(type(embedding), EmbeddingWorkload)
                self.assertEqual(embedding.rank_index_shape, (8 // tp,))
                self.assertEqual(embedding.rank_table_shape, (32, 16))
                self.assertEqual(embedding.rank_output_shape, (8 // tp, 16))
                ropes = tuple(
                    node.workload for node in graph.nodes if node.kind is OpKind.ROPE
                )
                self.assertEqual(len(ropes), 2)
                self.assertTrue(
                    all(work.rank_input_shape == (8, 48 // tp) for work in ropes)
                )
                lm_head = graph.nodes[-1].workload
                self.assertIs(type(lm_head), GemmWorkload)
                self.assertIs(
                    lm_head.partition,
                    GemmPartition.SEQUENCE_PARALLEL_REPLICATED_WEIGHT,
                )
                self.assertEqual(lm_head.rank_shape, (8 // tp, 32, 16))

                parameters = tuple(
                    state
                    for state in graph.persistent_states
                    if state.identity.kind is StateKind.PARAMETER
                )
                self.assertEqual(len(parameters), 15 * tp)
                self.assertEqual(
                    sum(2 * prod(state.shape) for state in parameters),
                    expected_bytes[tp],
                )
                self.assertEqual(len(graph.persistent_states), 19 * tp)

                context = PlacementContext.create(
                    producer_pass="load_physical_fabric",
                    fabric=fabric,
                    placement=spec.placement,
                    hbm_address_spaces=valid_hbm_address_spaces(fabric),
                )
                placed = place_bundle(bundle, context)
                validate_placement_against(placed, bundle, context)
                ir1 = placed.entries[0].graph
                oracle.validate_against_ir1(template, graph, ir1)
                self.assertEqual(ir1.schema_version, "wafer_frontend.ir1/v1alpha14")
                self.assertEqual(
                    placed.schema_version,
                    "wafer_frontend.placed_ir1_bundle/v1alpha4",
                )
                self.assertEqual(len(ir1.nodes), 33)
                self.assertIsNotNone(ir1.persistent_state_manifest)
                assert ir1.persistent_state_manifest is not None
                self.assertEqual(
                    sum(
                        binding.size_bytes
                        for binding in ir1.persistent_state_manifest.bindings
                        if next(
                            state
                            for state in ir1.persistent_state_manifest.declarations
                            if state.id == binding.state_ref
                        ).identity.kind
                        is StateKind.PARAMETER
                    ),
                    expected_bytes[tp],
                )

    def test_bundle_rejects_self_consistent_alternate_rope_heads(self) -> None:
        template = _template()
        graph = logical_expand(template).entries[0].graph
        rope = next(node for node in graph.nodes if node.id == "P0.layer0.rope")
        bad_work = replace(
            rope.workload,
            num_heads=6,
            num_kv_heads=3,
            rank_num_heads=6,
            rank_num_kv_heads=3,
        )
        bad = _rebuild_graph(
            graph,
            nodes=tuple(
                replace(node, workload=bad_work) if node.id == rope.id else node
                for node in graph.nodes
            ),
        )
        DenseIR0Validator.validate(bad)
        oracle = build_stage2_dense_forward_oracle(
            template, graph.profile, tp_degree=1
        )
        with self.assertRaisesRegex(
            SchemaError, "ROPE|graph-derived|template|attention model/profile"
        ):
            oracle.validate_against_ir0(template, bad)
        with self.assertRaisesRegex(SchemaError, "attention model/profile|provenance"):
            DenseLogicalBundleValidator.validate(template, _bundle(template, bad))

    def test_missing_global_parameter_state_fails_closed(self) -> None:
        template = _template()
        graph = logical_expand(template).entries[0].graph
        removed = next(
            state
            for state in graph.persistent_states
            if state.identity.tensor_ref == "P0.final_norm.weight"
        )
        bad = _rebuild_graph(
            graph,
            persistent_states=tuple(state for state in graph.persistent_states if state.id != removed.id),
            state_accesses=tuple(access for access in graph.state_accesses if access.state_ref != removed.id),
        )
        with self.assertRaisesRegex(SchemaError, "state declarations are not exact"):
            build_stage2_dense_forward_oracle(
                template, graph.profile, tp_degree=1
            ).validate_against_ir0(template, bad)

    def test_oracle_cross_rejects_collective_kv_and_hbm_binding_tampers(self) -> None:
        spec = _spec(tp=2)
        template = build_ir0(spec)
        bundle = logical_expand(template)
        graph = bundle.entries[0].graph
        oracle = build_stage2_dense_forward_oracle(
            template, graph.profile, tp_degree=2
        )

        collective = next(
            node for node in graph.nodes if type(node.workload) is CollectiveWorkload
        )
        collective_work = replace(
            collective.workload,
            rank_logical_payload_bytes=(
                collective.workload.rank_logical_payload_bytes + 1
            ),
        )
        bad_collective = _rebuild_graph(
            graph,
            nodes=tuple(
                replace(node, workload=collective_work)
                if node.id == collective.id
                else node
                for node in graph.nodes
            ),
        )
        with self.assertRaisesRegex(SchemaError, "payload|collective"):
            oracle.validate_against_ir0(template, bad_collective)

        attention = next(
            node for node in graph.nodes if type(node.workload) is AttentionWorkload
        )
        bad_attention_work = replace(
            attention.workload,
            logical_kv_write_bytes=attention.workload.logical_kv_write_bytes + 4,
        )
        bad_kv = _rebuild_graph(
            graph,
            nodes=tuple(
                replace(node, workload=bad_attention_work)
                if node.id == attention.id
                else node
                for node in graph.nodes
            ),
        )
        with self.assertRaisesRegex(SchemaError, "logical_kv_write_bytes|KV"):
            oracle.validate_against_ir0(template, bad_kv)

        fabric = load_physical_fabric(_HARDWARE, _MAPPING)
        context = PlacementContext.create(
            producer_pass="load_physical_fabric",
            fabric=fabric,
            placement=spec.placement,
            hbm_address_spaces=valid_hbm_address_spaces(fabric),
        )
        ir1 = place_bundle(bundle, context).entries[0].graph
        manifest = ir1.persistent_state_manifest
        assert manifest is not None

        forged_node = replace(
            ir1.nodes[0],
            execution_group_ref="forged.execution.group",
        )
        bad_execution_group = _rebuild_ir1(
            ir1,
            nodes=(forged_node, *ir1.nodes[1:]),
        )
        with self.assertRaisesRegex(SchemaError, "execution group"):
            oracle.validate_against_ir1(template, graph, bad_execution_group)

        original = manifest.bindings[0]
        oversized = HbmBinding.create(
            state_ref=original.state_ref,
            die_id=original.die_id,
            address=original.address,
            size_bytes=original.size_bytes + 64,
        )
        with self.assertRaisesRegex(SchemaError, "size.*state tensor byte size"):
            PersistentStateManifest.create(
                address_spaces=manifest.address_spaces,
                declarations=manifest.declarations,
                bindings=tuple(
                    oversized if binding.id == original.id else binding
                    for binding in manifest.bindings
                ),
            )

        other_die = 1 - original.die_id
        other_space = next(
            space for space in manifest.address_spaces if space.die_id == other_die
        )
        forged_binding = HbmBinding.create(
            state_ref=original.state_ref,
            die_id=other_die,
            address=other_space.base_address + other_space.size_bytes - 65536,
            size_bytes=original.size_bytes,
        )
        forged_manifest = PersistentStateManifest.create(
            address_spaces=manifest.address_spaces,
            declarations=manifest.declarations,
            bindings=tuple(
                forged_binding if binding.id == original.id else binding
                for binding in manifest.bindings
            ),
        )
        bad_ir1 = _rebuild_ir1(ir1, persistent_state_manifest=forged_manifest)
        with self.assertRaisesRegex(SchemaError, "home die|size/home"):
            oracle.validate_against_ir1(template, graph, bad_ir1)


if __name__ == "__main__":
    unittest.main()
