from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.policies.identity_intra_die_refine import (
    IdentityIntraDieRefinePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
from llm.frontend.wafer_frontend.policies.optimized_intra_die import OptimizedIntraDiePolicy
from llm.frontend.wafer_frontend.policies.split_k_intra_die_refine import (
    _refine_dag,
    refine_split_k_projection,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.schema.intra_die_refine import (
    IntraDieRefineContext,
    IntraDieRefineContract,
    SplitKRefineOptions,
)
from llm.frontend.wafer_frontend.schema.ir0 import GemmWorkload, OpKind
from llm.frontend.wafer_frontend.schema.ir1 import IR1, MemoryInitiator
from llm.frontend.wafer_frontend.schema.ir2 import (
    SemanticTaskKind, _validate_split_k_row_pack,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.split_k_refine import (
    SplitKRefinedProjection,
    split_k_accumulator_value_id,
    split_k_compute_event_id,
    split_k_part_task_id,
    split_k_pack_row_task_id,
    split_k_pack_value_id,
    split_k_partial_value_id,
    split_k_ready_event_id,
    split_k_reduce_task_id,
    split_k_reduce_step_task_id,
    split_k_stage_task_id,
    split_k_staged_value_id,
)

from test_naive_inter_die import _partitioned_graph


def _ordinary_gemm_tp1_graph() -> IR1:
    source = _partitioned_graph(tp=1)
    node = next(item for item in source.nodes if item.kind is OpKind.GEMM)
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


def _m1_gemm_graph() -> IR1:
    graph = _ordinary_gemm_tp1_graph()
    node = graph.nodes[0]
    _logical_m, logical_n, logical_k = node.workload.logical_shape
    _rank_m, rank_n, rank_k = node.workload.rank_shape
    node = replace(
        node,
        workload=replace(
            node.workload,
            logical_shape=(1, logical_n, logical_k),
            rank_shape=(1, rank_n, rank_k),
        ),
    )
    values = tuple(
        replace(value, shape=(1, value.shape[1]))
        if value.id in (node.inputs[0], node.outputs[0])
        else value
        for value in graph.values
    )
    fields = graph._semantic_key()
    fields.update(nodes=(node,), values=values)
    result = IR1.create(producer_pass=graph.producer_pass, **fields)
    result.validate("m1_gemm_graph")
    return result


def _source_projection():
    graph = _ordinary_gemm_tp1_graph()
    projection = NaiveProjectToIR2().run(
        graph, (), (), state_transfers=()
    )
    projection.validate_against(graph, (), ())
    return graph, projection


def _context(
    parts: int,
    *,
    reduce: bool = False,
    double_buffer: bool = False,
) -> IntraDieRefineContext:
    selection = production_registry().instantiate(
        RegistryKind.INTRA_DIE, "naive"
    ).selection
    return IntraDieRefineContext.create(
        producer_pass="test_split_k_intra_die_refine",
        policy=selection,
        contract=(
            IntraDieRefineContract.SPLIT_K_REDUCE_DOUBLE_BUFFER_V2
        ),
        options=SplitKRefineOptions(
            split_k_parts=parts,
            enable_reduce=reduce,
            enable_double_buffer=double_buffer,
        ),
    )


class SplitKIntraDieRefineTest(unittest.TestCase):
    def test_16_core_tree_reduce_and_target_dma_are_exact(self) -> None:
        graph = _partitioned_graph(tp=1)
        projection = NaiveProjectToIR2().run(
            graph, (), (), state_transfers=()
        )
        refined = refine_split_k_projection(
            projection,
            SplitKRefineOptions(
                split_k_parts=16, enable_reduce=True,
                compute_groups_per_die=16, enable_tree_reduce=True,
                enable_direct_dma=True,
            ),
            graph,
        )
        rewrite = refined.rewrites[0]
        self.assertTrue(rewrite.enable_tree_reduce)
        self.assertTrue(rewrite.enable_direct_dma)
        self.assertEqual(len(rewrite.direct_dma_task_ids), 16)
        self.assertEqual(len(rewrite.local_handoffs), 15)
        self.assertEqual(
            tuple(
                (item.source_compute_group, item.destination_compute_group)
                for item in rewrite.local_handoffs
            ),
            (
                (1, 0), (3, 2), (5, 4), (7, 6),
                (9, 8), (11, 10), (13, 12), (15, 14),
                (2, 0), (6, 4), (10, 8), (14, 12),
                (4, 0), (12, 8), (8, 0),
            ),
        )
        tasks = {
            task.id: task for dag in refined.projection.dags
            for task in dag.tasks
        }
        original_dma = next(
            task for dag in projection.dags for task in dag.tasks
            if task.kind is SemanticTaskKind.DMA_IN
            and rewrite.source_task_id in task.dma.access_task_refs
        )
        self.assertEqual(
            sum(tasks[task_id].bytes for task_id in rewrite.direct_dma_task_ids),
            original_dma.bytes,
        )
        schedule = NaiveIntraDiePolicy().schedule(
            refined.projection, graph
        ).schedules[0]
        placements = {item.task_id: item.core_id for item in schedule.placements}
        self.assertEqual(
            tuple(placements[task_id] for task_id in rewrite.reduction_task_ids),
            (0, 2, 4, 6, 8, 10, 12, 14, 0, 4, 8, 12, 0, 8, 0),
        )
        for task_id in rewrite.direct_dma_task_ids:
            target = tasks[task_id].dma.access_task_refs[0]
            self.assertEqual(placements[task_id], placements[target])

    def test_16_core_barrier_and_streaming_schedules_are_exact(self) -> None:
        graph, projection = _source_projection()
        core_counts = []
        first_reduce_dependency_counts = []
        for streaming in (False, True):
            refined = refine_split_k_projection(
                projection,
                SplitKRefineOptions(
                    split_k_parts=16,
                    enable_reduce=True,
                    compute_groups_per_die=16,
                    enable_streaming_reduce=streaming,
                ),
                graph,
            )
            rewrite = refined.rewrites[0]
            schedule = NaiveIntraDiePolicy().schedule(
                refined.projection, graph
            ).schedules[0]
            tasks = {
                task.id: task for task in refined.projection.dags[0].tasks
            }
            core_counts.append(len(schedule.core_orders))
            first_reduce_dependency_counts.append(
                len(tasks[rewrite.reduction_task_ids[0]].deps)
            )
            self.assertEqual(rewrite.compute_group_count, 16)
            self.assertEqual(
                tuple(order.core_id for order in schedule.core_orders),
                tuple(range(16)),
            )
        self.assertEqual(core_counts, [16, 16])
        self.assertEqual(first_reduce_dependency_counts, [16, 2])

    def test_disabled_v2_preserves_exact_identity(self) -> None:
        graph, projection = _source_projection()
        policy = IdentityIntraDieRefinePolicy()
        context = _context(1)

        self.assertIs(policy.refine(projection, graph), projection)
        self.assertIsNone(policy.refine_graph(projection, graph, context))
        projection.validate_against(graph, (), ())

    def test_parts_reduce_and_double_buffer_matrix_is_deterministic(self) -> None:
        graph, projection = _source_projection()
        policy = IdentityIntraDieRefinePolicy()
        source_digest = canonical_digest(projection)
        cases = (
            (2, False, False),
            (2, True, False),
            (4, False, True),
            (4, True, True),
        )

        for parts, enable_reduce, enable_double_buffer in cases:
            with self.subTest(
                parts=parts,
                reduce=enable_reduce,
                double_buffer=enable_double_buffer,
            ):
                context = _context(
                    parts,
                    reduce=enable_reduce,
                    double_buffer=enable_double_buffer,
                )
                first = policy.refine_graph(projection, graph, context)
                second = policy.refine_graph(projection, graph, context)
                self.assertIsInstance(first, SplitKRefinedProjection)
                self.assertEqual(first, second)
                assert first is not None
                first.validate_against(
                    projection,
                    graph,
                    split_k_parts=parts,
                    enable_reduce=enable_reduce,
                    enable_double_buffer=enable_double_buffer,
                )
                first.projection.validate_against(graph, (), ())
                projection.validate_against(graph, (), ())
                self.assertEqual(canonical_digest(projection), source_digest)
                self._assert_rewrite_shape(
                    first,
                    parts=parts,
                    enable_reduce=enable_reduce,
                    enable_double_buffer=enable_double_buffer,
                )

    def _assert_rewrite_shape(
        self,
        refined: SplitKRefinedProjection,
        *,
        parts: int,
        enable_reduce: bool,
        enable_double_buffer: bool,
    ) -> None:
        self.assertEqual(len(refined.rewrites), 1)
        rewrite = refined.rewrites[0]
        dag = next(
            item
            for item in refined.projection.dags
            if item.id != rewrite.source_dag_id and item.tasks
        )
        tasks = {task.id: task for task in dag.tasks}
        values = {value.id: value for value in dag.values}

        expected_parts = tuple(
            split_k_part_task_id(rewrite.source_task_id, part)
            for part in range(parts)
        )
        expected_partials = tuple(
            split_k_partial_value_id(
                rewrite.source_task_id,
                rewrite.source_output_value_id,
                part,
            )
            for part in range(parts)
        )
        self.assertEqual(rewrite.part_task_ids, expected_parts)
        self.assertEqual(rewrite.partial_value_ids, expected_partials)

        part_tasks = tuple(tasks[task_id] for task_id in expected_parts)
        self.assertTrue(
            all(task.kind is SemanticTaskKind.COMP for task in part_tasks)
        )
        source_workload = next(
            task.compute.workload
            for source_dag in refined.projection.dags
            for task in source_dag.tasks
            if task.id in expected_parts and task.compute is not None
        )
        self.assertIsInstance(source_workload, GemmWorkload)
        for part, (task, value_id) in enumerate(
            zip(part_tasks, expected_partials, strict=True)
        ):
            self.assertEqual(task.chunk_id, part)
            self.assertEqual(task.write_values, (value_id,))
            self.assertEqual(values[value_id].producer_tasks, (task.id,))
            self.assertEqual(
                task.sync.completion_event if task.sync is not None else None,
                split_k_compute_event_id(rewrite.source_task_id, part),
            )
        if enable_reduce:
            reduce_id = split_k_reduce_task_id(rewrite.source_task_id)
            self.assertEqual(rewrite.reduce_task_id, reduce_id)
            self.assertIsNone(rewrite.semantic_commit_task_id)
            self.assertEqual(len(rewrite.reduction_task_ids), parts - 1)
            self.assertEqual(len(rewrite.reduction_accumulator_value_ids), parts - 2)
            handoff_by_part = {item.part_index: item for item in rewrite.local_handoffs}
            ready_by_part = tuple(
                handoff_by_part[index].wait_task_id
                if index in handoff_by_part else task_id
                for index, task_id in enumerate(expected_parts)
            )
            previous_value = expected_partials[0]
            previous_task_id = None
            for step in range(1, parts):
                task_id = split_k_reduce_step_task_id(
                    rewrite.source_task_id, step, parts
                )
                reduce_task = tasks[task_id]
                self.assertIs(reduce_task.kind, SemanticTaskKind.REDUCE)
                self.assertEqual(
                    reduce_task.read_values, (previous_value, expected_partials[step])
                )
                self.assertEqual(
                    reduce_task.deps,
                    tuple(dict.fromkeys(
                        ((ready_by_part[0],) if step == 1 else (previous_task_id,))
                        + (ready_by_part[step],)
                    )),
                )
                output_id = (
                    rewrite.source_output_value_id if step == parts - 1
                    else split_k_accumulator_value_id(
                        rewrite.source_task_id, rewrite.source_output_value_id, step
                    )
                )
                self.assertEqual(reduce_task.write_values, (output_id,))
                previous_value, previous_task_id = output_id, task_id
            for step, value_id in enumerate(expected_partials):
                self.assertEqual(
                    values[value_id].consumer_tasks,
                    (rewrite.reduction_task_ids[max(0, step - 1)],),
                )
        else:
            self.assertIsNone(rewrite.reduce_task_id)
            self.assertEqual(
                rewrite.semantic_commit_task_id, rewrite.source_task_id
            )
            commit = tasks[rewrite.source_task_id]
            self.assertIs(commit.kind, SemanticTaskKind.COMP)
            self.assertTrue(set(expected_parts).issubset(commit.deps))
            for value_id in expected_partials:
                self.assertEqual(values[value_id].consumer_tasks, ())

        copies = tuple(
            task
            for task in dag.tasks
            if task.kind is SemanticTaskKind.LOCAL_COPY
        )
        if not enable_double_buffer:
            self.assertEqual(copies, ())
            self.assertEqual(rewrite.double_buffer_versions, ())
            return

        self.assertEqual(len(copies), parts * 2)
        self.assertEqual(
            tuple(version.slot for version in rewrite.double_buffer_versions),
            tuple((part // rewrite.compute_group_count) % 2 for part in range(parts)),
        )
        for version in rewrite.double_buffer_versions:
            part = version.part_index
            part_task = tasks[expected_parts[part]]
            self.assertEqual(part_task.read_values, version.staged_value_ids)
            self.assertTrue(set(version.stage_task_ids).issubset(part_task.deps))
            for operand, stage_id in enumerate(version.stage_task_ids):
                expected_stage_id = split_k_stage_task_id(
                    rewrite.source_task_id, part, operand
                )
                expected_staged_id = split_k_staged_value_id(
                    rewrite.source_task_id,
                    tasks[rewrite.source_task_id].read_values[operand]
                    if not enable_reduce
                    else next(
                        task
                        for task in part_tasks
                        if task.chunk_id == part
                    ).compute.tile.input_slices[operand].source_value_id,
                    part,
                    operand,
                    rewrite.compute_group_count,
                )
                self.assertEqual(stage_id, expected_stage_id)
                self.assertEqual(
                    version.staged_value_ids[operand], expected_staged_id
                )
                stage = tasks[stage_id]
                staged = values[expected_staged_id]
                self.assertEqual(stage.write_values, (expected_staged_id,))
                self.assertEqual(staged.producer_tasks, (stage_id,))
                self.assertEqual(staged.consumer_tasks, (part_task.id,))
                self.assertIsNone(staged.alias_set)
                self.assertIn(
                    f".slot.{(part // rewrite.compute_group_count) % 2}.version.{part}", staged.id
                )
                self.assertEqual(
                    stage.sync.completion_event
                    if stage.sync is not None
                    else None,
                    split_k_ready_event_id(
                        rewrite.source_task_id, part, operand,
                        rewrite.compute_group_count,
                    ),
                )
                if part >= 2:
                    self.assertIn(expected_parts[part - 2], stage.deps)

    def test_remote_inputs_receive_directly_into_stage_slots_exactly(self) -> None:
        graph = _partitioned_graph(tp=1)
        projection = NaiveProjectToIR2().run(
            graph, (), (), state_transfers=()
        )
        rewrites = []
        refined_dags = []
        for dag in projection.dags:
            refined, dag_rewrites = _refine_dag(
                dag, SplitKRefineOptions(4, True, True), graph
            )
            refined_dags.append(refined)
            rewrites.extend(dag_rewrites)
        self.assertEqual(len(rewrites), 1)
        rewrite = rewrites[0]
        refined = next(dag for dag in refined_dags if any(
            task.id == rewrite.reduce_task_id for task in dag.tasks
        ))
        tasks = {task.id: task for task in refined.tasks}
        values = {value.id: value for value in refined.values}
        self.assertEqual(len(rewrite.input_handoffs), 4)
        self.assertEqual(
            sum(len(version.direct_receive_operand_indices)
                for version in rewrite.double_buffer_versions),
            4,
        )
        split_tasks = tuple(
            task for task in refined.tasks if ".split_k." in task.id
        )
        self.assertEqual(
            sum(task.kind is SemanticTaskKind.LOCAL_COPY for task in split_tasks),
            4,
        )
        self.assertEqual(
            sum(task.kind is SemanticTaskKind.LOCAL_SEND for task in split_tasks),
            6,
        )
        self.assertEqual(
            sum(task.kind is SemanticTaskKind.LOCAL_RECV for task in split_tasks),
            6,
        )
        self.assertEqual(
            sum(task.kind is SemanticTaskKind.LOCAL_WAIT for task in split_tasks),
            6,
        )
        for handoff in rewrite.input_handoffs:
            send = tasks[handoff.send_task_id]
            recv = tasks[handoff.recv_task_id]
            wait = tasks[handoff.wait_task_id]
            destination = tasks[handoff.destination_task_id]
            self.assertEqual(send.tensor_slice.value_id, handoff.value_id)
            self.assertEqual(
                recv.tensor_slice.value_id, handoff.destination_value_id
            )
            self.assertEqual(
                (recv.tensor_slice.offset, recv.tensor_slice.shape),
                (send.tensor_slice.offset, send.tensor_slice.shape),
            )
            self.assertEqual(wait.deps, (recv.id,))
            self.assertIn(wait.id, destination.deps)
            self.assertEqual(
                values[handoff.destination_value_id].producer_tasks, ()
            )
            self.assertNotIn(
                split_k_stage_task_id(
                    rewrite.source_task_id, handoff.part_index,
                    handoff.operand_index,
                ),
                tasks,
            )


    def test_parts8_double_buffer_reuses_two_physical_slots_exactly(self) -> None:
        graph = _m1_gemm_graph()
        projection = NaiveProjectToIR2().run(
            graph, (), (), state_transfers=()
        )
        refined = IdentityIntraDieRefinePolicy().refine_graph(
            projection,
            graph,
            _context(8, reduce=True, double_buffer=True),
        )
        assert refined is not None
        rewrite = refined.rewrites[0]
        self.assertEqual(rewrite.compute_group_count, 2)
        schedule_set = OptimizedIntraDiePolicy().schedule(
            refined.projection, graph
        )
        schedule = next(
            item for item in schedule_set.schedules
            if any(
                placement.task_id == rewrite.reduce_task_id
                for placement in item.placements
            )
        )
        bindings = {item.value_id: item for item in schedule.buffer_bindings}
        tasks = {
            task.id: task
            for dag in refined.projection.dags for task in dag.tasks
        }
        for left_index, right_index in ((0, 4), (1, 5), (2, 6), (3, 7)):
            left = rewrite.double_buffer_versions[left_index]
            right = rewrite.double_buffer_versions[right_index]
            for operand_index, (left_value, right_value) in enumerate(
                zip(left.staged_value_ids, right.staged_value_ids, strict=True)
            ):
                left_binding = bindings[left_value]
                right_binding = bindings[right_value]
                self.assertEqual(left_binding.core_id, right_binding.core_id)
                self.assertEqual(
                    left_binding.region_offset_bytes,
                    right_binding.region_offset_bytes,
                )
                self.assertNotEqual(
                    left_binding.storage_id, right_binding.storage_id
                )
                self.assertLessEqual(
                    left_binding.lifetime_end_exclusive,
                    right_binding.lifetime_start,
                )
                stage = tasks[left.stage_task_ids[operand_index]]
                self.assertEqual(
                    left_binding.size_bytes,
                    2 * __import__("math").prod(stage.shape),
                )
                source_shape = next(
                    value.shape for value in graph.values
                    if value.id == stage.read_values[0]
                )
                self.assertLess(
                    left_binding.size_bytes,
                    2 * __import__("math").prod(source_shape),
                )

    def test_m_gt_1_lhs_row_pack_has_exact_action_count_and_coverage(self) -> None:
        graph = _partitioned_graph(tp=1)
        projection = NaiveProjectToIR2().run(
            graph, (), (), state_transfers=()
        )
        refined_dags = []
        rewrites = []
        for dag in projection.dags:
            refined, dag_rewrites = _refine_dag(
                dag, SplitKRefineOptions(2, True, False), graph
            )
            refined_dags.append(refined)
            rewrites.extend(dag_rewrites)
        self.assertEqual(len(rewrites), 1)
        rewrite = rewrites[0]
        refined = next(
            dag for dag in refined_dags
            if rewrite.reduce_task_id in {task.id for task in dag.tasks}
        )
        tasks = {task.id: task for task in refined.tasks}
        values = {value.id: value for value in refined.values}
        source = next(
            task for dag in projection.dags for task in dag.tasks
            if task.id == rewrite.source_task_id
        )
        assert source.compute is not None
        rank_m = source.compute.workload.rank_shape[0]
        lhs_id = source.read_values[0]
        expected_values = tuple(
            split_k_pack_value_id(rewrite.source_task_id, lhs_id, part, 0)
            for part in range(2)
        )
        self.assertEqual(
            tuple(value_id for value_id in values if value_id.endswith(".packed")),
            expected_values,
        )
        pack_tasks = tuple(
            task for task in refined.tasks if ".pack.row." in task.id
        )
        self.assertEqual(len(pack_tasks), 2 * rank_m)
        for part, value_id in enumerate(expected_values):
            value = values[value_id]
            writers = tuple(tasks[task_id] for task_id in value.producer_tasks)
            self.assertEqual(len(writers), rank_m)
            part_task = tasks[rewrite.part_task_ids[part]]
            binding = next(
                item for item in part_task.compute.tile.input_slices
                if item.operand_id == value_id
            )
            for row, writer in enumerate(writers):
                self.assertEqual(
                    writer.id,
                    split_k_pack_row_task_id(
                        rewrite.source_task_id, part, 0, row
                    ),
                )
                self.assertEqual(writer.tensor_slice.shape, (1, binding.logical_shape[1]))
                self.assertEqual(
                    writer.tensor_slice.offset,
                    (binding.logical_offset[0] + row, binding.logical_offset[1]),
                )
                if row:
                    self.assertEqual(writer.deps, (writers[row - 1].id,))
            self.assertTrue(_validate_split_k_row_pack(
                value, writers, tasks, path="test.pack"
            ))
            with self.assertRaisesRegex(
                Exception, "writer count must equal the target row count"
            ):
                _validate_split_k_row_pack(
                    value, writers[:-1], tasks, path="test.pack"
                )

    def test_single_core_degrades_to_same_core_reduce_without_local_flow(self) -> None:
        graph = _m1_gemm_graph()
        fields = graph._semantic_key()
        default_profile = graph.fabric.sram_profiles[0]
        compute_only_profile = replace(
            default_profile,
            id=f"{default_profile.id}.compute_only",
            regions=tuple(
                replace(
                    region,
                    access=(
                        MemoryInitiator.COMPUTE, MemoryInitiator.LSU,
                    ),
                )
                for region in default_profile.regions
            ),
        )
        fields["fabric"] = replace(
            graph.fabric,
            sram_profiles=(default_profile, compute_only_profile),
            dies=tuple(
                replace(
                    die,
                    cores=(die.cores[0],) + tuple(
                        replace(core, sram_profile_ref=compute_only_profile.id)
                        for core in die.cores[1:]
                    ),
                )
                for die in graph.fabric.dies
            ),
        )
        graph = IR1.create(producer_pass=graph.producer_pass, **fields)
        graph.validate("single_core_graph")
        projection = NaiveProjectToIR2().run(
            graph, (), (), state_transfers=()
        )
        refined = IdentityIntraDieRefinePolicy().refine_graph(
            projection,
            graph,
            _context(2, reduce=True, double_buffer=False),
        )
        assert refined is not None
        rewrite = refined.rewrites[0]
        self.assertEqual(rewrite.compute_group_count, 1)
        self.assertEqual(rewrite.part_compute_groups, (0, 0))
        self.assertEqual(rewrite.local_handoffs, ())
        self.assertFalse(
            any(
                task.kind in (
                    SemanticTaskKind.LOCAL_SEND,
                    SemanticTaskKind.LOCAL_RECV,
                    SemanticTaskKind.LOCAL_WAIT,
                )
                for dag in refined.projection.dags for task in dag.tasks
            )
        )
        schedule_set = NaiveIntraDiePolicy().schedule(
            refined.projection, graph
        )
        schedule = next(
            item for item in schedule_set.schedules
            if any(
                placement.task_id == rewrite.reduce_task_id
                for placement in item.placements
            )
        )
        placements = {item.task_id: item.core_id for item in schedule.placements}
        self.assertEqual(
            {placements[item] for item in (*rewrite.part_task_ids, rewrite.reduce_task_id)},
            {placements[rewrite.reduce_task_id]},
        )

    def test_carrier_rejects_missing_rewrite_closure(self) -> None:
        graph, projection = _source_projection()
        context = _context(2, reduce=True, double_buffer=True)
        refined = IdentityIntraDieRefinePolicy().refine_graph(
            projection, graph, context
        )
        assert refined is not None
        forged = SplitKRefinedProjection.create(
            source_projection_id=projection.id,
            projection=refined.projection,
            rewrites=(),
            search_decision=refined.search_decision,
        )
        with self.assertRaisesRegex(
            Exception, "exactly cover every eligible ordinary GEMM"
        ):
            forged.validate_against(
                projection,
                graph,
                split_k_parts=2,
                enable_reduce=True,
                enable_double_buffer=True,
            )


if __name__ == "__main__":
    unittest.main()
