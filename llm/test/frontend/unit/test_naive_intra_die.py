from __future__ import annotations

import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferAccess,
    BufferOwnership,
    dense_row_major_view_byte_addend,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind, OpKind
from llm.frontend.wafer_frontend.schema.ir2 import (
    IR2ProjectionResult,
    IntraDieDAG,
    IntraDieRegion,
    IntraDieSchedule,
    IntraDieScheduleSet,
    IntraDieValue,
    BufferUseRole,
    FlowRouteRole,
    RegionLowering,
    SemanticTaskKind,
    StandaloneNodeOrigin,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from test_ir2_schema import ordinary_projection, valid_reduce_case
from test_ir2_route_schedule import _two_by_two_case, _two_die_case
from test_n4_pipeline import _compile_through_n4
from test_naive_project_to_ir2 import _partitioned_graph
from test_naive_project_state import (
    _ordinary_kv_tp1,
    _ordinary_parameter_tp1,
)


def _complete_projection(*, tp: int, large_sram: bool):
    graph = _partitioned_graph(tp=tp)
    if large_sram:
        profiles = tuple(
            replace(
                profile,
                capacity_bytes=64 * 1024 * 1024,
                regions=tuple(
                    replace(region, size_bytes=64 * 1024 * 1024)
                    for region in profile.regions
                ),
            )
            for profile in graph.fabric.sram_profiles
        )
        graph = IR1.create(
            producer_pass=graph.producer_pass,
            **{
                **graph._semantic_key(),
                "fabric": replace(graph.fabric, sram_profiles=profiles),
            },
        )
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
        and node.workload.collective is CollectiveKind.ALL_GATHER
    )
    standalone_plans = tuple(
        DirectAllGatherPolicy().plan(graph, node, graph.profile)
        for node in standalone_nodes
    )
    return graph, NaiveProjectToIR2().run(
        graph,
        fusion_plans,
        standalone_plans,
        state_transfers=(),
    )


def _complete_tp2_projection(*, large_sram: bool):
    return _complete_projection(tp=2, large_sram=large_sram)


def _recreate_dag(dag: IntraDieDAG, **changes: object) -> IntraDieDAG:
    fields = dag._semantic_key()
    fields.update(changes)
    return IntraDieDAG.create(producer_pass=dag.producer_pass, **fields)


def _recreate_schedule(
    schedule: IntraDieSchedule, **changes: object
) -> IntraDieSchedule:
    fields = schedule._semantic_key()
    fields.update(changes)
    return IntraDieSchedule.create(
        producer_pass=schedule.producer_pass, **fields
    )


def _component_projection():
    ir1, projection = ordinary_projection()
    template_dag = projection.dags[0]
    template_task = template_dag.tasks[0]
    input_value, output_value = template_dag.values

    def task(
        task_id: str,
        region_id: str,
        input_id: str,
        output_id: str,
        deps: tuple[str, ...],
    ):
        compute = replace(
            template_task.compute,
            inputs=(
                replace(
                    template_task.compute.inputs[0],
                    value_id=input_id,
                ),
            ),
            outputs=(
                replace(
                    template_task.compute.outputs[0],
                    value_id=output_id,
                ),
            ),
        )
        return replace(
            template_task,
            id=task_id,
            region_id=region_id,
            read_values=(input_id,),
            write_values=(output_id,),
            compute=compute,
            deps=deps,
        )

    task_a = task("z_task_a", "region_a", "v_input", "v_mid", ())
    task_c = task("a_task_c", "region_c", "v_input", "v_out_c", ())
    task_b = task(
        "b_task_b",
        "region_b",
        "v_mid",
        "v_out_b",
        (task_a.id,),
    )

    def value(
        template,
        value_id: str,
        producers: tuple[str, ...],
        consumers: tuple[str, ...],
    ):
        return IntraDieValue(
            value_id,
            template.origin_value_id,
            template.shape,
            template.dtype,
            template.logical_layout,
            template.sharding,
            None,
            producers,
            consumers,
        )

    dag = IntraDieDAG.create(
        producer_pass="component_projection_fixture",
        source_ir1_id=ir1.id,
        die_id=template_dag.die_id,
        fusion_plan_ids=(),
        standalone_collective_plan_ids=(),
        ordinary_node_ids=template_dag.ordinary_node_ids,
        tasks=(task_a, task_c, task_b),
        values=(
            value(input_value, "v_input", (), (task_a.id, task_c.id)),
            value(output_value, "v_mid", (task_a.id,), (task_b.id,)),
            value(output_value, "v_out_c", (task_c.id,), ()),
            value(output_value, "v_out_b", (task_b.id,), ()),
        ),
        flows=(),
        regions=(
            IntraDieRegion(
                "region_a", None, None, RegionLowering.JSON_COARSE, (task_a.id,)
            ),
            IntraDieRegion(
                "region_c", None, None, RegionLowering.JSON_COARSE, (task_c.id,)
            ),
            IntraDieRegion(
                "region_b", None, None, RegionLowering.JSON_COARSE, (task_b.id,)
            ),
        ),
    )
    result = IR2ProjectionResult.create(
        producer_pass="component_projection_fixture",
        source_ir1_id=ir1.id,
        fusion_plan_ids=(),
        standalone_collective_plan_ids=(),
        dags=(dag,),
    )
    result.validate()
    return ir1, result


class NaiveIntraDieOrdinaryTest(unittest.TestCase):
    def test_canonical_kahn_component_rr_and_binding_sharing(self) -> None:
        ir1, projection = _component_projection()
        schedule = NaiveIntraDiePolicy().schedule(
            projection,
            ir1,
        ).schedules[0]
        placement = {
            item.task_id: item.core_id for item in schedule.placements
        }
        runtime_cores = tuple(
            sorted(core.runtime_core_id for core in ir1.fabric.dies[0].cores)
        )
        self.assertEqual(placement["z_task_a"], runtime_cores[0])
        self.assertEqual(placement["b_task_b"], runtime_cores[0])
        self.assertEqual(placement["a_task_c"], runtime_cores[1])
        orders = {item.core_id: item.task_ids for item in schedule.core_orders}
        self.assertEqual(orders[runtime_cores[0]], ("z_task_a", "b_task_b"))
        self.assertEqual(orders[runtime_cores[1]], ("a_task_c",))
        mid_bindings = tuple(
            binding
            for binding in schedule.buffer_bindings
            if binding.value_id == "v_mid"
        )
        self.assertEqual(len(mid_bindings), 1)
        self.assertEqual(mid_bindings[0].lifetime_start, 0)
        self.assertEqual(mid_bindings[0].lifetime_end_exclusive, 2)
        self.assertEqual(len(schedule.buffer_bindings), 5)

    def test_ordinary_projection_is_exact_deterministic_and_non_mutating(self) -> None:
        ir1, projection = ordinary_projection()
        before = canonical_digest((ir1, projection))
        first = NaiveIntraDiePolicy().schedule(projection, ir1)
        second = NaiveIntraDiePolicy().schedule(projection, ir1)
        first.validate_against(projection, ir1)
        self.assertEqual(first, second)
        self.assertEqual(canonical_digest((ir1, projection)), before)
        self.assertEqual(
            tuple(schedule.dag_id for schedule in first.schedules),
            tuple(dag.id for dag in projection.dags),
        )
        for dag, schedule in zip(projection.dags, first.schedules):
            minimum_runtime_core = min(
                core.runtime_core_id
                for die in ir1.fabric.dies
                if die.id == dag.die_id
                for core in die.cores
            )
            self.assertEqual(schedule.placements[0].core_id, minimum_runtime_core)
            self.assertEqual(len(schedule.task_buffer_uses), 2)
            self.assertEqual(
                tuple(binding.ownership for binding in schedule.buffer_bindings),
                (BufferOwnership.BORROWED, BufferOwnership.OWNED),
            )
        self.assertEqual(
            tuple(
                (
                    schedule.task_buffer_uses[0].tensor_slice.offset,
                    schedule.task_buffer_uses[0].tensor_slice.shape,
                )
                for schedule in first.schedules
            ),
            (((0, 0), (32, 128)), ((0, 128), (32, 128))),
        )
        for schedule in first.schedules:
            bindings = {
                binding.id: binding for binding in schedule.buffer_bindings
            }
            self.assertTrue(
                all(
                    bindings[use.binding_id].tensor_slice == use.tensor_slice
                    for use in schedule.task_buffer_uses
                )
            )

    def test_core_order_is_runtime_id_not_tuple_or_local_id(self) -> None:
        ir1, projection = ordinary_projection()
        die = ir1.fabric.dies[0]
        remapped = tuple(
            replace(
                core,
                id=(
                    f"z_core_{core.local_core_id}"
                    if core.local_core_id == 0
                    else f"a_core_{core.local_core_id}"
                ),
            )
            for core in die.cores
        )
        remapped = remapped[5:] + remapped[:5]
        fabric = replace(
            ir1.fabric,
            dies=(replace(die, cores=remapped),) + ir1.fabric.dies[1:],
        )
        changed_ir1 = IR1.create(
            producer_pass=ir1.producer_pass,
            **{**ir1._semantic_key(), "fabric": fabric},
        )
        dags = tuple(
            IntraDieDAG.create(
                producer_pass=dag.producer_pass,
                **{
                    **dag._semantic_key(),
                    "source_ir1_id": changed_ir1.id,
                },
            )
            for dag in projection.dags
        )
        changed_projection = IR2ProjectionResult.create(
            producer_pass=projection.producer_pass,
            **{
                **projection._semantic_key(),
                "source_ir1_id": changed_ir1.id,
                "dags": dags,
            },
        )
        result = NaiveIntraDiePolicy().schedule(
            changed_projection,
            changed_ir1,
        )
        first_core_id = result.schedules[0].placements[0].core_id
        selected = next(
            core for core in remapped if core.runtime_core_id == first_core_id
        )
        self.assertEqual(
            first_core_id,
            min(core.runtime_core_id for core in remapped),
        )
        self.assertEqual(selected.local_core_id, 0)
        self.assertNotEqual(remapped[0].runtime_core_id, first_core_id)
        self.assertNotEqual(
            min(remapped, key=lambda core: core.id).runtime_core_id,
            first_core_id,
        )


class NaiveIntraDieTransportTest(unittest.TestCase):
    def test_two_by_two_transit_and_empty_die_are_physical_binding_free(self) -> None:
        ir1, projection, _fixture_schedule = _two_by_two_case()
        result = NaiveIntraDiePolicy().schedule(projection, ir1)
        result.validate_against(projection, ir1)
        routed = tuple(
            schedule for schedule in result.schedules if schedule.flow_routes
        )
        self.assertEqual(
            tuple(schedule.flow_routes[0].role for schedule in routed),
            (
                FlowRouteRole.SOURCE,
                FlowRouteRole.TRANSIT,
                FlowRouteRole.DESTINATION,
            ),
        )
        transit = routed[1]
        self.assertFalse(transit.placements)
        self.assertFalse(transit.buffer_bindings)
        self.assertFalse(transit.task_buffer_uses)
        self.assertFalse(transit.runtime_bindings)
        empty = result.schedules[2]
        self.assertFalse(empty.placements)
        self.assertFalse(empty.flow_routes)
        self.assertFalse(empty.core_orders)

    def test_two_die_send_recv_routes_buffers_and_runtime_are_exact(self) -> None:
        ir1, projection, _fixture_schedule = _two_die_case()
        result = NaiveIntraDiePolicy().schedule(projection, ir1)
        result.validate_against(projection, ir1)
        source, destination = result.schedules
        self.assertEqual(source.flow_routes[0].role, FlowRouteRole.SOURCE)
        self.assertEqual(
            destination.flow_routes[0].role,
            FlowRouteRole.DESTINATION,
        )
        self.assertEqual(
            source.task_buffer_uses[0].role,
            BufferUseRole.SEND_SOURCE,
        )
        self.assertEqual(
            destination.task_buffer_uses[0].role,
            BufferUseRole.RECV_DESTINATION,
        )
        self.assertEqual(
            source.runtime_bindings[0].channel_symbol,
            projection.dags[0].flows[0].logical_channel,
        )
        self.assertEqual(
            destination.runtime_bindings[0].channel_symbol,
            projection.dags[1].flows[0].logical_channel,
        )


class NaiveIntraDieReduceTest(unittest.TestCase):
    def test_local_reduce_inputs_are_rank_major_tight_stride(self) -> None:
        ir1, dag, _fixture_schedule = valid_reduce_case()
        projection = IR2ProjectionResult.create(
            producer_pass="project_to_ir2",
            source_ir1_id=ir1.id,
            fusion_plan_ids=dag.fusion_plan_ids,
            standalone_collective_plan_ids=(),
            dags=(dag,),
        )
        result = NaiveIntraDiePolicy().schedule(projection, ir1)
        result.validate_against(projection, ir1)
        schedule = result.schedules[0]
        bindings = {binding.id: binding for binding in schedule.buffer_bindings}
        inputs = tuple(
            (use.contribution_rank, bindings[use.binding_id])
            for use in schedule.task_buffer_uses
            if use.role is BufferUseRole.REDUCE_INPUT
        )
        output = next(
            bindings[use.binding_id]
            for use in schedule.task_buffer_uses
            if use.role is BufferUseRole.REDUCE_OUTPUT
        )
        self.assertEqual(tuple(rank for rank, _binding in inputs), (0, 1))
        self.assertEqual(
            tuple(binding.region_offset_bytes for _rank, binding in inputs),
            (0, 32),
        )
        self.assertEqual(output.region_offset_bytes, 64)
        self.assertEqual(
            {binding.core_id for _rank, binding in inputs} | {output.core_id},
            {0},
        )
        self.assertEqual(
            {binding.region_ref for _rank, binding in inputs}
            | {output.region_ref},
            {"sram_main"},
        )
        self.assertTrue(
            all(binding.alignment_bytes == 2 for _rank, binding in inputs)
        )
        self.assertEqual(output.alignment_bytes, 2)

    def test_complete_tp2_l1_preserves_tiles_and_reduce_staging(self) -> None:
        graph, projection = _complete_tp2_projection(large_sram=True)
        result = NaiveIntraDiePolicy().schedule(projection, graph)
        result.validate_against(projection, graph)
        self.assertEqual(tuple(len(dag.tasks) for dag in projection.dags), (43, 43))
        self.assertEqual(
            tuple(len(schedule.buffer_bindings) for schedule in result.schedules),
            (38, 38),
        )
        self.assertEqual(
            tuple(
                max(
                    binding.region_offset_bytes + binding.size_bytes
                    for binding in schedule.buffer_bindings
                )
                for schedule in result.schedules
            ),
            (1427008, 1427008),
        )
        for dag, schedule in zip(projection.dags, result.schedules):
            staging_ids = {
                value.id for value in dag.state_staging_values
            }
            state_bindings = tuple(
                binding
                for binding in schedule.buffer_bindings
                if binding.value_id in staging_ids
            )
            dma_in = tuple(
                task for task in dag.tasks
                if task.kind is SemanticTaskKind.DMA_IN
            )
            dma_out = tuple(
                task for task in dag.tasks
                if task.kind is SemanticTaskKind.DMA_OUT
            )
            self.assertEqual(
                (
                    len(staging_ids),
                    len(dma_in),
                    sum(task.bytes for task in dma_in),
                    len(dma_out),
                    sum(task.bytes for task in dma_out),
                    len(schedule.task_state_uses),
                    len(state_bindings),
                    sum(binding.size_bytes for binding in state_bindings),
                ),
                (11, 9, 1115648, 2, 8192, 11, 11, 1123840),
            )
            if dag.die_id == 1:
                self.assertEqual(
                    tuple(
                        binding.tensor_slice.offset
                        for binding in state_bindings
                        if len(binding.tensor_slice.shape) == 2
                    ),
                    (
                        (0, 0),
                        (0, 256),
                        (128, 0),
                        (0, 512),
                        (256, 0),
                        (0, 0),
                    ),
                )
            binding_index = {
                binding.id: binding for binding in schedule.buffer_bindings
            }
            runtime_index = {
                binding.task_id: binding for binding in schedule.runtime_bindings
            }
            task_index = {task.id: task for task in dag.tasks}
            uses_by_task = {
                task.id: tuple(
                    use for use in schedule.task_buffer_uses
                    if use.task_id == task.id
                )
                for task in dag.tasks
            }
            for task in dag.tasks:
                if task.kind in (
                    SemanticTaskKind.SEND,
                    SemanticTaskKind.RECV,
                    SemanticTaskKind.WAIT,
                    SemanticTaskKind.BARRIER,
                ):
                    self.assertTrue(runtime_index[task.id].token_symbol)
                if task.kind is SemanticTaskKind.WAIT:
                    self.assertEqual(
                        runtime_index[task.id].event_symbol,
                        task.sync.wait_event,
                    )
                    waited_task = task_index[task.deps[0]]
                    self.assertEqual(
                        runtime_index[task.id].event_symbol,
                        waited_task.sync.completion_event,
                    )
                    self.assertEqual(
                        runtime_index[task.id].token_symbol,
                        runtime_index[waited_task.id].token_symbol,
                    )
                if task.kind is SemanticTaskKind.REDUCE:
                    inputs = tuple(
                        binding_index[use.binding_id]
                        for use in uses_by_task[task.id]
                        if use.role is BufferUseRole.REDUCE_INPUT
                    )
                    output = next(
                        binding_index[use.binding_id]
                        for use in uses_by_task[task.id]
                        if use.role is BufferUseRole.REDUCE_OUTPUT
                    )
                    base = inputs[0].region_offset_bytes
                    self.assertEqual(
                        tuple(item.region_offset_bytes for item in inputs),
                        tuple(base + index * task.bytes for index in range(2)),
                    )
                    self.assertGreaterEqual(
                        output.region_offset_bytes,
                        base + len(inputs) * task.bytes,
                    )
                if (
                    task.kind is SemanticTaskKind.COMP
                    and task.compute is not None
                    and task.compute.tile is not None
                ):
                    uses = uses_by_task[task.id]
                    for role, slices in (
                        (BufferUseRole.COMP_INPUT, task.compute.tile.input_slices),
                        (BufferUseRole.COMP_OUTPUT, task.compute.tile.output_slices),
                    ):
                        role_uses = tuple(use for use in uses if use.role is role)
                        self.assertEqual(len(role_uses), len(slices))
                        for use, expected in zip(role_uses, slices):
                            self.assertEqual(
                                (
                                    use.tensor_slice.offset,
                                    use.tensor_slice.shape,
                                ),
                                (expected.logical_offset, expected.logical_shape),
                            )

            value_index = {value.id: value for value in dag.values}
            for suffix in ("ag1_out", "ag2_out"):
                matches = tuple(
                    binding
                    for binding in schedule.buffer_bindings
                    if binding.value_id.rsplit(".", 1)[-1] == suffix
                )
                self.assertEqual(len(matches), 1)
                root = matches[0]
                value = value_index[root.value_id]
                self.assertEqual(root.tensor_slice.offset, (0, 0))
                self.assertEqual(root.tensor_slice.shape, value.shape)
                views = tuple(
                    use
                    for use in schedule.task_buffer_uses
                    if use.binding_id == root.id
                )
                self.assertEqual(
                    {use.role for use in views},
                    {
                        BufferUseRole.COMP_INPUT,
                        BufferUseRole.LOCAL_COPY_DESTINATION,
                        BufferUseRole.SEND_SOURCE,
                        BufferUseRole.RECV_DESTINATION,
                    },
                )
                self.assertEqual(
                    sorted(
                        {
                            dense_row_major_view_byte_addend(
                                root.tensor_slice,
                                use.tensor_slice,
                                root.dtype,
                            )
                            for use in views
                            if use.role
                            in (
                                BufferUseRole.LOCAL_COPY_DESTINATION,
                                BufferUseRole.RECV_DESTINATION,
                            )
                        }
                    ),
                    [0, root.size_bytes // 2],
                )

            positions = {
                task_id: position
                for order in schedule.core_orders
                for position, task_id in enumerate(order.task_ids)
            }
            for binding in schedule.buffer_bindings:
                uses = tuple(
                    use
                    for use in schedule.task_buffer_uses
                    if use.binding_id == binding.id
                )
                first_position = min(positions[use.task_id] for use in uses)
                first_accesses = {
                    use.access
                    for use in uses
                    if positions[use.task_id] == first_position
                }
                self.assertEqual(
                    first_accesses,
                    {
                        BufferAccess.WRITE
                        if binding.ownership is BufferOwnership.OWNED
                        else BufferAccess.READ
                    },
                )

            standalone_recvs = tuple(
                task
                for task in dag.tasks
                if task.kind is SemanticTaskKind.RECV
                and isinstance(task.origin_ref, StandaloneNodeOrigin)
            )
            self.assertTrue(standalone_recvs)
            waited_tokens = {
                runtime_index[task.id].token_symbol
                for task in dag.tasks
                if task.kind is SemanticTaskKind.WAIT
            }
            self.assertTrue(
                all(
                    runtime_index[task.id].token_symbol
                    and runtime_index[task.id].token_symbol not in waited_tokens
                    for task in standalone_recvs
                )
            )

    def test_wait_runtime_token_lineage_is_exact_and_fail_closed(self) -> None:
        graph, projection = _complete_tp2_projection(large_sram=True)
        result = NaiveIntraDiePolicy().schedule(projection, graph)
        dag = projection.dags[0]
        schedule = result.schedules[0]
        task_index = {task.id: task for task in dag.tasks}
        wait = next(
            task for task in dag.tasks if task.kind is SemanticTaskKind.WAIT
        )
        recv = task_index[wait.deps[0]]

        wrong_token = _recreate_schedule(
            schedule,
            runtime_bindings=tuple(
                replace(binding, token_symbol="forged_wait_token")
                if binding.task_id == wait.id
                else binding
                for binding in schedule.runtime_bindings
            ),
        )
        with self.assertRaisesRegex(SchemaError, "reuse.*RECV.*token"):
            wrong_token.validate_against(dag, graph)

        wait_position = dag.tasks.index(wait)
        other_recv = next(
            task
            for task in dag.tasks[:wait_position]
            if task.kind is SemanticTaskKind.RECV and task.id != recv.id
        )
        ambiguous_tasks = tuple(
            replace(
                task,
                sync=replace(
                    task.sync,
                    completion_event=wait.sync.wait_event,
                ),
            )
            if task.id == other_recv.id
            else replace(task, deps=(other_recv.id, recv.id))
            if task.id == wait.id
            else task
            for task in dag.tasks
        )
        ambiguous_dag = _recreate_dag(dag, tasks=ambiguous_tasks)
        ambiguous_schedule = _recreate_schedule(
            schedule,
            dag_id=ambiguous_dag.id,
            runtime_bindings=tuple(
                replace(binding, event_symbol=wait.sync.wait_event)
                if binding.task_id == other_recv.id
                else binding
                for binding in schedule.runtime_bindings
            ),
        )
        with self.assertRaisesRegex(SchemaError, "exactly one.*RECV"):
            ambiguous_schedule.validate_against(ambiguous_dag, graph)

        placement_index = {
            placement.task_id: placement.core_id
            for placement in schedule.placements
        }
        send = next(
            task
            for task in dag.tasks[:wait_position]
            if task.kind is SemanticTaskKind.SEND
            and task.origin_ref.rank == wait.origin_ref.rank
            and placement_index[task.id] == placement_index[wait.id]
        )
        nonrecv_tasks = tuple(
            replace(
                task,
                sync=replace(
                    task.sync,
                    wait_event=send.sync.completion_event,
                ),
                deps=(send.id, recv.id),
            )
            if task.id == wait.id
            else task
            for task in dag.tasks
        )
        nonrecv_dag = _recreate_dag(dag, tasks=nonrecv_tasks)
        nonrecv_projection = IR2ProjectionResult.create(
            producer_pass=projection.producer_pass,
            **{
                **projection._semantic_key(),
                "dags": (nonrecv_dag,) + projection.dags[1:],
            },
        )
        with self.assertRaisesRegex(SchemaError, "exactly one.*RECV"):
            NaiveIntraDiePolicy().schedule(nonrecv_projection, graph)

    def test_plan_barrier_runtime_event_is_shared_for_tp2_and_tp4(self) -> None:
        for tp in (2, 4):
            with self.subTest(tp=tp):
                graph, projection = _complete_projection(
                    tp=tp, large_sram=True
                )
                result = NaiveIntraDiePolicy().schedule(projection, graph)
                result.validate_against(projection, graph)
                barrier_groups: dict[str, list[tuple[object, object]]] = {}
                for dag, schedule in zip(
                    projection.dags, result.schedules
                ):
                    runtime_index = {
                        binding.task_id: binding
                        for binding in schedule.runtime_bindings
                    }
                    for task in dag.tasks:
                        if task.kind is not SemanticTaskKind.BARRIER:
                            continue
                        barrier_groups.setdefault(
                            task.sync.barrier.id, []
                        ).append((task, runtime_index[task.id]))
                self.assertEqual(len(barrier_groups), 2)
                for barrier_id, entries in barrier_groups.items():
                    self.assertEqual(len(entries), tp)
                    self.assertEqual(
                        {task.origin_ref.rank for task, _binding in entries},
                        set(range(tp)),
                    )
                    self.assertEqual(
                        {binding.event_symbol for _task, binding in entries},
                        {barrier_id},
                    )
                    tokens = {
                        binding.token_symbol for _task, binding in entries
                    }
                    self.assertNotIn(None, tokens)
                    self.assertEqual(len(tokens), tp)

    def test_plan_barrier_runtime_event_is_exact_and_complete(self) -> None:
        graph, projection = _complete_projection(tp=2, large_sram=True)
        result = NaiveIntraDiePolicy().schedule(projection, graph)
        first_dag = projection.dags[0]
        first_schedule = result.schedules[0]
        barrier = next(
            task
            for task in first_dag.tasks
            if task.kind is SemanticTaskKind.BARRIER
        )
        for event_symbol in (
            barrier.sync.completion_event,
            "forged_barrier_id",
        ):
            changed = _recreate_schedule(
                first_schedule,
                runtime_bindings=tuple(
                    replace(binding, event_symbol=event_symbol)
                    if binding.task_id == barrier.id
                    else binding
                    for binding in first_schedule.runtime_bindings
                ),
            )
            with self.subTest(event_symbol=event_symbol), self.assertRaisesRegex(
                SchemaError, "shared barrier id"
            ):
                changed.validate_against(first_dag, graph)

        barrier_id = barrier.sync.barrier.id
        second_dag = projection.dags[1]
        second_schedule = result.schedules[1]
        second_barrier = next(
            task
            for task in second_dag.tasks
            if task.kind is SemanticTaskKind.BARRIER
            and task.sync.barrier.id == barrier_id
        )
        forged_contract = replace(
            second_barrier.sync.barrier,
            id=f"{barrier_id}.rank_different",
        )
        changed_second_dag = _recreate_dag(
            second_dag,
            tasks=tuple(
                replace(
                    task,
                    sync=replace(task.sync, barrier=forged_contract),
                )
                if task.id == second_barrier.id
                else task
                for task in second_dag.tasks
            ),
        )
        changed_second_schedule = _recreate_schedule(
            second_schedule,
            dag_id=changed_second_dag.id,
            runtime_bindings=tuple(
                replace(binding, event_symbol=forged_contract.id)
                if binding.task_id == second_barrier.id
                else binding
                for binding in second_schedule.runtime_bindings
            ),
        )
        changed_projection = IR2ProjectionResult.create(
            producer_pass=projection.producer_pass,
            **{
                **projection._semantic_key(),
                "dags": (first_dag, changed_second_dag),
            },
        )
        changed_set = IntraDieScheduleSet.create(
            producer_pass=result.producer_pass,
            source_projection_id=changed_projection.id,
            source_ir1_id=graph.id,
            schedules=(first_schedule, changed_second_schedule),
        )
        with self.assertRaisesRegex(SchemaError, "one action per participant"):
            changed_set.validate_against(changed_projection, graph)

    def test_real_hardware_comm_capacity_is_fail_closed(self) -> None:
        planned = _compile_through_n4()[5].entries[0]
        projection = NaiveProjectToIR2().run(
            planned.graph,
            planned.fusion_plans,
            planned.standalone_plans,
            state_transfers=(),
        )
        with self.assertRaisesRegex(SchemaError, "SRAM capacity.*comm"):
            NaiveIntraDiePolicy().schedule(projection, planned.graph)


class NaiveIntraDieStateTest(unittest.TestCase):
    def test_parameter_tp1_is_scheduled_deterministically(self) -> None:
        graph, _access, _declaration = _ordinary_parameter_tp1()
        projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
        policy = NaiveIntraDiePolicy()
        first = policy.schedule(projection, graph)
        second = policy.schedule(projection, graph)
        self.assertEqual(first, second)
        first.validate_against(projection, graph)

        dag = next(item for item in projection.dags if item.state_access_ids)
        schedule = next(
            item for item in first.schedules if item.dag_id == dag.id
        )
        dma = next(
            task for task in dag.tasks
            if task.kind is SemanticTaskKind.DMA_IN
        )
        comp = next(
            task for task in dag.tasks
            if task.kind is SemanticTaskKind.COMP
        )
        placement = {
            item.task_id: item.core_id for item in schedule.placements
        }
        self.assertEqual(placement[dma.id], placement[comp.id])
        self.assertEqual(
            tuple(use.task_id for use in schedule.task_state_uses),
            (dma.id,),
        )
        self.assertEqual(
            tuple(
                use.role
                for use in schedule.task_buffer_uses
                if use.task_id == dma.id
            ),
            (BufferUseRole.DMA_DESTINATION,),
        )

    def test_kv_tp1_is_blocking_lsu_and_attention_stays_opaque(self) -> None:
        graph, _accesses, _declarations = _ordinary_kv_tp1()
        projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
        result = NaiveIntraDiePolicy().schedule(projection, graph)
        result.validate_against(projection, graph)

        dag = next(item for item in projection.dags if item.state_access_ids)
        schedule = next(
            item for item in result.schedules if item.dag_id == dag.id
        )
        comp = next(
            task for task in dag.tasks
            if task.kind is SemanticTaskKind.COMP
        )
        self.assertEqual(
            tuple(operand.value_id for operand in comp.compute.inputs),
            graph.nodes[0].inputs,
        )
        staging_ids = {
            value.id for value in dag.state_staging_values
        }
        self.assertFalse(
            staging_ids.intersection(
                use.tensor_slice.value_id
                for use in schedule.task_buffer_uses
                if use.task_id == comp.id
            )
        )
        self.assertEqual(len(schedule.task_state_uses), 2)
        roles_by_kind = {
            kind: tuple(
                use.role
                for use in schedule.task_buffer_uses
                if next(
                    task for task in dag.tasks if task.id == use.task_id
                ).kind is kind
            )
            for kind in (
                SemanticTaskKind.DMA_IN,
                SemanticTaskKind.DMA_OUT,
            )
        }
        self.assertEqual(
            roles_by_kind[SemanticTaskKind.DMA_IN],
            (),
        )
        self.assertEqual(
            roles_by_kind[SemanticTaskKind.DMA_OUT],
            (
                BufferUseRole.DMA_SOURCE,
                BufferUseRole.DMA_SOURCE,
            ),
        )


if __name__ == "__main__":
    unittest.main()
