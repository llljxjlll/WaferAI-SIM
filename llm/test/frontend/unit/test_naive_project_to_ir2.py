from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.schema.action import (
    FusionPlan,
    StandaloneCollectivePlan,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind, OpKind
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir2 import (
    FusedNodeOrigin,
    IR2ProjectionResult,
    IntraDieDAG,
    OrdinaryNodeOrigin,
    RegionLowering,
    StateIoOrigin,
    SemanticTaskKind,
    StandaloneNodeOrigin,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from test_naive_inter_die import _partitioned_graph


def _ordinary_tp1_graph() -> IR1:
    source = _partitioned_graph(tp=1)
    node = next(item for item in source.nodes if item.kind is not OpKind.COLLECTIVE)
    value_ids = set(node.inputs + node.outputs)
    values = tuple(
        replace(
            value,
            producer=node.id if value.id in node.outputs else None,
            consumers=(node.id,) if value.id in node.inputs else (),
        )
        for value in source.values
        if value.id in value_ids
    )
    fields = source._semantic_key()
    fields.update(
        instances=tuple(
            replace(instance, node_ids=(node.id,))
            if instance.id == node.instance_id
            else replace(instance, node_ids=())
            for instance in source.instances
        ),
        nodes=(node,),
        values=values,
        edges=(),
        state_accesses=(),
        persistent_state_manifest=None,
        fusion_candidates=(),
        fused_op_skeletons=(),
    )
    graph = IR1.create(producer_pass="fusion_partition", **fields)
    graph.validate()
    return graph


def _fused_graph(tp: int) -> IR1:
    source = _partitioned_graph(tp=tp)
    skeleton = source.fused_op_skeletons[0]
    member_ids = set(skeleton.member_node_ids)
    nodes = tuple(node for node in source.nodes if node.id in member_ids)
    value_ids = {
        value_id
        for node in nodes
        for value_id in node.inputs + node.outputs
    }
    values = tuple(
        replace(
            value,
            producer=value.producer if value.producer in member_ids else None,
            consumers=tuple(
                consumer for consumer in value.consumers if consumer in member_ids
            ),
        )
        for value in source.values
        if value.id in value_ids
    )
    fields = source._semantic_key()
    fields.update(
        instances=tuple(
            replace(
                instance,
                node_ids=tuple(
                    node_id for node_id in instance.node_ids if node_id in member_ids
                ),
            )
            for instance in source.instances
        ),
        nodes=nodes,
        values=values,
        edges=tuple(
            edge
            for edge in source.edges
            if edge.source_node in member_ids and edge.destination_node in member_ids
        ),
        fusion_candidates=tuple(
            candidate
            for candidate in source.fusion_candidates
            if candidate.id == skeleton.fusion_ref
        ),
        state_accesses=(),
        persistent_state_manifest=None,
        fused_op_skeletons=(skeleton,),
    )
    graph = IR1.create(producer_pass="fusion_partition", **fields)
    graph.validate()
    return graph


def _standalone_tp2_graph() -> IR1:
    source = _partitioned_graph(tp=2)
    fused_members = {
        member_id
        for skeleton in source.fused_op_skeletons
        for member_id in skeleton.member_node_ids
    }
    node = next(
        item
        for item in source.nodes
        if item.id not in fused_members
        and getattr(item.workload, "collective", None)
        is CollectiveKind.ALL_GATHER
    )
    value_ids = set(node.inputs + node.outputs)
    values = tuple(
        replace(
            value,
            producer=node.id if value.id in node.outputs else None,
            consumers=(node.id,) if value.id in node.inputs else (),
        )
        for value in source.values
        if value.id in value_ids
    )
    fields = source._semantic_key()
    fields.update(
        instances=tuple(
            replace(
                instance,
                node_ids=(node.id,) if instance.id == node.instance_id else (),
            )
            for instance in source.instances
        ),
        nodes=(node,),
        values=values,
        edges=(),
        state_accesses=(),
        persistent_state_manifest=None,
        fusion_candidates=(),
        fused_op_skeletons=(),
    )
    graph = IR1.create(producer_pass="fusion_partition", **fields)
    graph.validate()
    return graph


def _stateful_tp2_projection() -> tuple[
    IR1,
    tuple[FusionPlan, ...],
    tuple[StandaloneCollectivePlan, ...],
    IR2ProjectionResult,
]:
    graph = _partitioned_graph(tp=2)
    fusion_plans = tuple(
        NaiveInterDiePolicy().plan(graph, skeleton, graph.profile)
        for skeleton in graph.fused_op_skeletons
    )
    fused_members = {
        member_id
        for skeleton in graph.fused_op_skeletons
        for member_id in skeleton.member_node_ids
    }
    standalone_plans = tuple(
        DirectAllGatherPolicy().plan(graph, node, graph.profile)
        for node in graph.nodes
        if node.id not in fused_members
        and node.kind is OpKind.COLLECTIVE
        and getattr(node.workload, "collective", None)
        is CollectiveKind.ALL_GATHER
    )
    projection = NaiveProjectToIR2().run(
        graph, fusion_plans, standalone_plans, state_transfers=()
    )
    return graph, fusion_plans, standalone_plans, projection


def _replace_dag(
    dag: IntraDieDAG,
    **updates: object,
) -> IntraDieDAG:
    semantic_key = dag._semantic_key()
    semantic_key.update(updates)
    return IntraDieDAG.create(
        producer_pass=dag.producer_pass,
        **semantic_key,
    )


def _replace_projection_dag(
    projection: IR2ProjectionResult,
    dag: IntraDieDAG,
    **updates: object,
) -> IR2ProjectionResult:
    replacement = _replace_dag(dag, **updates)
    semantic_key = projection._semantic_key()
    semantic_key["dags"] = tuple(
        replacement if item.die_id == dag.die_id else item
        for item in projection.dags
    )
    return IR2ProjectionResult.create(
        producer_pass=projection.producer_pass,
        **semantic_key,
    )



class NaiveProjectToIR2Test(unittest.TestCase):
    def test_ordinary_tp1_is_exact_deterministic_and_valid(self) -> None:
        graph = _ordinary_tp1_graph()
        source_digest = canonical_digest(graph)
        policy = NaiveProjectToIR2()
        projection = policy.run(graph, (), (), state_transfers=())
        self.assertEqual(
            policy.run(graph, (), (), state_transfers=()), projection
        )
        projection.validate_against(graph, (), ())
        self.assertEqual(projection.producer_pass, "project_to_ir2")
        self.assertEqual(
            tuple(dag.die_id for dag in projection.dags),
            tuple(die.id for die in graph.fabric.dies),
        )
        tasks = tuple(task for dag in projection.dags for task in dag.tasks)
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertIs(task.kind, SemanticTaskKind.COMP)
        self.assertIsInstance(task.origin_ref, OrdinaryNodeOrigin)
        self.assertEqual(task.deps, ())
        owning_dag = next(dag for dag in projection.dags if dag.tasks)
        self.assertEqual(len(owning_dag.regions), 1)
        self.assertIs(
            owning_dag.regions[0].lowering, RegionLowering.JSON_COARSE
        )
        self.assertEqual(owning_dag.regions[0].task_ids, (task.id,))
        self.assertEqual(canonical_digest(graph), source_digest)

    def test_unplanned_collective_fails_closed(self) -> None:
        graph = _partitioned_graph(tp=2)
        with self.assertRaisesRegex(SchemaError, "collective nodes require"):
            NaiveProjectToIR2().run(graph, (), (), state_transfers=())

    def test_fused_tp2_projects_each_action_once_with_exact_flows(self) -> None:
        graph = _fused_graph(2)
        plan = NaiveInterDiePolicy().plan(
            graph, graph.fused_op_skeletons[0], graph.profile
        )
        projection = NaiveProjectToIR2().run(graph, (plan,), (), state_transfers=())
        projection.validate_against(graph, (plan,), ())
        expected_actions = tuple(
            action
            for program in plan.rank_programs
            for action in program.actions
        )
        tasks = tuple(task for dag in projection.dags for task in dag.tasks)
        self.assertEqual(len(tasks), len(expected_actions))
        self.assertEqual(tuple(len(dag.tasks) for dag in projection.dags), (6, 6))
        self.assertTrue(
            all(isinstance(task.origin_ref, FusedNodeOrigin) for task in tasks)
        )
        self.assertEqual(
            Counter(task.kind for task in tasks),
            Counter(SemanticTaskKind(action.kind.value) for action in expected_actions),
        )
        self.assertEqual(sum(len(dag.flows) for dag in projection.dags), 4)
        self.assertTrue(
            all(
                region.lowering is RegionLowering.ISA_REGION
                for dag in projection.dags
                for region in dag.regions
            )
        )

    def test_standalone_tp2_all_gather_is_exact(self) -> None:
        graph = _standalone_tp2_graph()
        node = graph.nodes[0]
        plan = DirectAllGatherPolicy().plan(graph, node, graph.profile)
        projection = NaiveProjectToIR2().run(graph, (), (plan,), state_transfers=())
        projection.validate_against(graph, (), (plan,))
        expected_actions = tuple(
            action
            for program in plan.rank_programs
            for action in program.actions
        )
        tasks = tuple(task for dag in projection.dags for task in dag.tasks)
        self.assertEqual(len(tasks), len(expected_actions))
        self.assertEqual(tuple(len(dag.tasks) for dag in projection.dags), (4, 4))
        self.assertTrue(
            all(
                isinstance(task.origin_ref, StandaloneNodeOrigin)
                for task in tasks
            )
        )
        self.assertEqual(
            Counter(task.kind for task in tasks),
            Counter(SemanticTaskKind(action.kind.value) for action in expected_actions),
        )
        self.assertEqual(sum(len(dag.flows) for dag in projection.dags), 4)
        self.assertTrue(
            all(
                region.lowering is RegionLowering.STRICT_ACTIONS
                for dag in projection.dags
                for region in dag.regions
            )
        )

    def test_fused_tp4_2x2_projects_exact_transit_hops(self) -> None:
        graph = _fused_graph(4)
        plan = NaiveInterDiePolicy().plan(
            graph, graph.fused_op_skeletons[0], graph.profile
        )
        projection = NaiveProjectToIR2().run(graph, (plan,), (), state_transfers=())
        projection.validate_against(graph, (plan,), ())
        planned_action_count = sum(
            len(program.actions) for program in plan.rank_programs
        )
        tasks = tuple(task for dag in projection.dags for task in dag.tasks)
        transits = tuple(
            task for task in tasks if task.kind is SemanticTaskKind.TRANSIT
        )
        self.assertEqual(planned_action_count, 56)
        self.assertEqual(len(transits), 4)
        self.assertEqual(len(tasks), 60)
        self.assertEqual(tuple(len(dag.tasks) for dag in projection.dags), (15,) * 4)
        self.assertEqual(sum(len(dag.flows) for dag in projection.dags), 28)
        self.assertTrue(
            all(
                task.deps == ()
                and task.read_values == ()
                and task.write_values == ()
                for task in transits
            )
        )

    def test_no_state_tp2_tp4_shared_rhs_has_no_writer_and_all_consumers(self) -> None:
        for tp in (2, 4):
            with self.subTest(tp=tp):
                graph = _fused_graph(tp)
                skeleton = graph.fused_op_skeletons[0]
                gemm = next(
                    node
                    for node in graph.nodes
                    if node.id == skeleton.member_node_ids[0]
                )
                plan = NaiveInterDiePolicy().plan(
                    graph, skeleton, graph.profile
                )
                projection = NaiveProjectToIR2().run(graph, (plan,), (), state_transfers=())
                projection.validate_against(graph, (plan,), ())
                group = next(item for item in graph.groups if item.id == plan.group_ref)
                rank_to_die = {
                    placement.rank: placement.die_id
                    for placement in group.placements
                }
                for program in plan.rank_programs:
                    comp_actions = tuple(
                        action
                        for action in program.actions
                        if action.kind.value == "comp"
                    )
                    rhs_id = f"temp.{skeleton.id}.rank.{program.rank}.b"
                    self.assertEqual(
                        {action.reads[1] for action in comp_actions},
                        {rhs_id},
                    )
                    dag = next(
                        item
                        for item in projection.dags
                        if item.die_id == rank_to_die[program.rank]
                    )
                    rhs_value = next(value for value in dag.values if value.id == rhs_id)
                    self.assertEqual(rhs_value.origin_value_id, gemm.inputs[1])
                    self.assertEqual(rhs_value.producer_tasks, ())
                    self.assertEqual(
                        rhs_value.consumer_tasks,
                        tuple(f"task.{action.id}" for action in comp_actions),
                    )

    def test_complete_tp2_l1_has_exact_state_projection_per_die(self) -> None:
        graph = _partitioned_graph(tp=2)
        fusion_plans = tuple(
            NaiveInterDiePolicy().plan(graph, skeleton, graph.profile)
            for skeleton in graph.fused_op_skeletons
        )
        fused_members = {
            member_id
            for skeleton in graph.fused_op_skeletons
            for member_id in skeleton.member_node_ids
        }
        standalone_nodes = tuple(
            node
            for node in graph.nodes
            if node.id not in fused_members
            and node.kind is OpKind.COLLECTIVE
            and getattr(node.workload, "collective", None)
            is CollectiveKind.ALL_GATHER
        )
        standalone_plans = tuple(
            DirectAllGatherPolicy().plan(graph, node, graph.profile)
            for node in standalone_nodes
        )
        policy = NaiveProjectToIR2()
        projection = policy.run(
            graph, fusion_plans, standalone_plans, state_transfers=()
        )
        self.assertEqual(
            policy.run(
                graph, fusion_plans, standalone_plans, state_transfers=()
            ),
            projection,
        )
        # D2-3A adds seven ordinary typed full-forward computes per die while
        # leaving the frozen state-DMA and collective projection unchanged.
        self.assertEqual(tuple(len(dag.tasks) for dag in projection.dags), (43, 43))
        node_index = {node.id: node for node in graph.nodes}
        for dag in projection.dags:
            state_tasks = tuple(
                task
                for task in dag.tasks
                if isinstance(task.origin_ref, StateIoOrigin)
            )
            self.assertEqual(len(dag.state_staging_values), 11)
            self.assertEqual(
                Counter(task.kind for task in state_tasks),
                Counter(
                    {
                        SemanticTaskKind.DMA_IN: 9,
                        SemanticTaskKind.DMA_OUT: 2,
                    }
                ),
            )
            self.assertEqual(
                sum(
                    task.bytes
                    for task in state_tasks
                    if task.kind is SemanticTaskKind.DMA_IN
                ),
                1_115_648,
            )
            self.assertEqual(
                sum(
                    task.bytes
                    for task in state_tasks
                    if task.kind is SemanticTaskKind.DMA_OUT
                ),
                8_192,
            )

            task_index = {task.id: task for task in dag.tasks}
            fused_parameter_dma = tuple(
                task
                for task in state_tasks
                if task.kind is SemanticTaskKind.DMA_IN
                and task.dma is not None
                and len(task.dma.access_task_refs) > 1
            )
            self.assertEqual(len(fused_parameter_dma), 2)
            for dma_task in fused_parameter_dma:
                assert dma_task.dma is not None
                expected_targets = tuple(
                    task.id
                    for task in dag.tasks
                    if task.kind is SemanticTaskKind.COMP
                    and task.member_id == dma_task.origin_ref.node_ref
                    and getattr(task.origin_ref, "rank", None)
                    == dma_task.origin_ref.rank
                )
                self.assertEqual(
                    dma_task.dma.access_task_refs, expected_targets
                )
                self.assertTrue(
                    all(
                        dma_task.dma.local_value_ref
                        in task_index[target_id].read_values
                        for target_id in expected_targets
                    )
                )

            for task in dag.tasks:
                if (
                    task.kind is SemanticTaskKind.COMP
                    and task.member_id is not None
                    and node_index[task.member_id].kind is OpKind.ATTENTION
                ):
                    assert task.compute is not None
                    self.assertEqual(
                        tuple(
                            operand.value_id
                            for operand in task.compute.inputs
                        ),
                        node_index[task.member_id].inputs,
                    )
                    self.assertEqual(
                        tuple(operand.role for operand in task.compute.inputs),
                        ("packed_qkv",),
                    )
                    self.assertEqual(
                        tuple(operand.role for operand in task.compute.outputs),
                        ("attention_output",),
                    )
        self.assertEqual(sum(len(dag.flows) for dag in projection.dags), 16)
        self.assertFalse(
            any(
                task.kind is SemanticTaskKind.TRANSIT
                for dag in projection.dags
                for task in dag.tasks
            )
        )
        with self.assertRaisesRegex(SchemaError, "skeleton source order"):
            policy.run(
                graph,
                tuple(reversed(fusion_plans)),
                standalone_plans,
                state_transfers=(),
            )

    def test_state_multi_target_refs_are_exact(self) -> None:
        graph, fusion_plans, standalone_plans, projection = (
            _stateful_tp2_projection()
        )
        dag = projection.dags[0]
        dma_task = next(
            task
            for task in dag.tasks
            if task.kind is SemanticTaskKind.DMA_IN
            and task.dma is not None
            and len(task.dma.access_task_refs) > 1
        )
        assert dma_task.dma is not None
        refs = dma_task.dma.access_task_refs
        extra_ref = next(
            task.id
            for task in dag.tasks
            if task.kind is SemanticTaskKind.COMP and task.id not in refs
        )
        cases = (
            ("missing", refs[:-1], "exactly cover IR-1 node/rank"),
            ("extra", refs + (extra_ref,), None),
            ("reordered", tuple(reversed(refs)), None),
        )
        for label, tampered_refs, message in cases:
            with self.subTest(label=label):
                tampered_dma = replace(
                    dma_task,
                    dma=replace(
                        dma_task.dma,
                        access_task_refs=tampered_refs,
                    ),
                )
                tampered = _replace_projection_dag(
                    projection,
                    dag,
                    tasks=tuple(
                        tampered_dma if task.id == dma_task.id else task
                        for task in dag.tasks
                    ),
                )
                if message is None:
                    with self.assertRaises(SchemaError):
                        tampered.validate_against(
                            graph, fusion_plans, standalone_plans
                        )
                else:
                    with self.assertRaisesRegex(SchemaError, message):
                        tampered.validate_against(
                            graph, fusion_plans, standalone_plans
                        )

    def test_state_task_and_region_identity_partition_are_exact(self) -> None:
        graph, fusion_plans, standalone_plans, projection = (
            _stateful_tp2_projection()
        )
        dag = projection.dags[0]
        state_tasks = tuple(
            task
            for task in dag.tasks
            if isinstance(task.origin_ref, StateIoOrigin)
        )
        state_task = state_tasks[0]
        assert state_task.region_id is not None

        with self.subTest(label="synchronized_task_id_tamper"):
            replacement_id = f"{state_task.id}.tampered"
            tasks = []
            for task in dag.tasks:
                updated = (
                    replace(task, id=replacement_id)
                    if task.id == state_task.id
                    else task
                )
                tasks.append(
                    replace(
                        updated,
                        deps=tuple(
                            replacement_id if dependency == state_task.id
                            else dependency
                            for dependency in updated.deps
                        ),
                    )
                )
            regions = tuple(
                replace(
                    region,
                    task_ids=tuple(
                        replacement_id if task_id == state_task.id else task_id
                        for task_id in region.task_ids
                    ),
                )
                for region in dag.regions
            )
            staging_values = tuple(
                replace(
                    value,
                    producer_tasks=tuple(
                        replacement_id if task_id == state_task.id else task_id
                        for task_id in value.producer_tasks
                    ),
                    consumer_tasks=tuple(
                        replacement_id if task_id == state_task.id else task_id
                        for task_id in value.consumer_tasks
                    ),
                )
                for value in dag.state_staging_values
            )
            tampered = _replace_projection_dag(
                projection,
                dag,
                tasks=tuple(tasks),
                regions=regions,
                state_staging_values=staging_values,
            )
            with self.assertRaisesRegex(SchemaError, "identity is not canonical"):
                tampered.validate_against(
                    graph, fusion_plans, standalone_plans
                )

        with self.subTest(label="synchronized_region_id_tamper"):
            replacement_region_id = f"{state_task.region_id}.tampered"
            tampered = _replace_projection_dag(
                projection,
                dag,
                tasks=tuple(
                    replace(task, region_id=replacement_region_id)
                    if task.id == state_task.id
                    else task
                    for task in dag.tasks
                ),
                regions=tuple(
                    replace(region, id=replacement_region_id)
                    if region.id == state_task.region_id
                    else region
                    for region in dag.regions
                ),
            )
            with self.assertRaisesRegex(SchemaError, "identity is not canonical"):
                tampered.validate_against(
                    graph, fusion_plans, standalone_plans
                )

        with self.subTest(label="merged_state_regions"):
            second_task = state_tasks[1]
            assert second_task.region_id is not None
            merged_task_ids = tuple(
                task.id
                for task in dag.tasks
                if task.id in (state_task.id, second_task.id)
            )
            tampered = _replace_projection_dag(
                projection,
                dag,
                tasks=tuple(
                    replace(task, region_id=state_task.region_id)
                    if task.id == second_task.id
                    else task
                    for task in dag.tasks
                ),
                regions=tuple(
                    replace(region, task_ids=merged_task_ids)
                    if region.id == state_task.region_id
                    else region
                    for region in dag.regions
                    if region.id != second_task.region_id
                ),
            )
            with self.assertRaises(SchemaError):
                tampered.validate_against(
                    graph, fusion_plans, standalone_plans
                )


if __name__ == "__main__":
    unittest.main()
