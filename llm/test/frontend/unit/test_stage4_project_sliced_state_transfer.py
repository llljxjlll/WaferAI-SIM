from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_ir1
from llm.frontend.wafer_frontend.passes.placement import place_stage4_ir0
from llm.frontend.wafer_frontend.passes.stage4_logical_expand import (
    build_stage4_separated_ir0,
)
from llm.frontend.wafer_frontend.passes.stage4_pd import build_stage4_pd_plan
from llm.frontend.wafer_frontend.passes.stage4_state_transfer import (
    build_stage4_state_transfers,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    FlowRouteRole,
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
    canonical_state_transfer_flow_id,
    canonical_state_transfer_task_id,
)
from llm.frontend.wafer_frontend.schema.ir1 import SramAllocator
from llm.frontend.wafer_frontend.schema.persistent_state import (
    canonical_state_staging_value_id,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext

from _fixtures import naive_inter_die_planning_context, valid_hbm_address_spaces
from test_stage4_inter_die_plan import _three_die_fabric
from test_stage4_pd import _profile, _spec
from test_stage4_placement import _tp1_case


def _project_tp1(*, large_sram: bool = False):
    graph, context, pd_plan = _tp1_case()
    if large_sram:
        region_size = 16 * 1024 * 1024
        profiles = tuple(
            replace(
                profile,
                capacity_bytes=len(profile.regions) * region_size,
                regions=tuple(
                    replace(
                        region,
                        base_bytes=index * region_size,
                        size_bytes=region_size,
                        allocator=SramAllocator.BLOCK,
                    )
                    for index, region in enumerate(profile.regions)
                ),
            )
            for profile in context.fabric.sram_profiles
        )
        context = PlacementContext.create(
            producer_pass="stage4_projector_large_sram",
            fabric=replace(context.fabric, sram_profiles=profiles),
            placement=context.placement,
            hbm_address_spaces=context.hbm_address_spaces,
        )
    ir1 = partition_ir1(place_stage4_ir0(graph, context, pd_plan))
    fusion_plans, standalone_plans = plan_ir1(
        ir1,
        naive_inter_die_planning_context("stage4_projector"),
    )
    contracts = build_stage4_state_transfers(ir1, pd_plan)
    projection = NaiveProjectToIR2().run(
        ir1,
        fusion_plans,
        standalone_plans,
        state_transfers=contracts,
    )
    return ir1, fusion_plans, standalone_plans, contracts, projection


class Stage4ProjectSlicedStateTransferTest(unittest.TestCase):
    def test_tp1_schedule_binds_cross_group_routes_exactly(self) -> None:
        ir1, _fusion, _standalone, _contracts, projection = _project_tp1(
            large_sram=True
        )
        schedules = NaiveIntraDiePolicy().schedule(projection, ir1)
        schedules.validate_against(projection, ir1)
        self.assertEqual(
            [
                (
                    schedule.die_id,
                    len(schedule.flow_routes),
                    len(schedule.placements),
                    len(schedule.buffer_bindings),
                )
                for schedule in schedules.schedules
            ],
            [(0, 4, 48, 45), (1, 4, 52, 45)],
        )
        cross_route_ids = {route.id for route in ir1.cross_routes}
        self.assertEqual(len(cross_route_ids), 1)
        for schedule in schedules.schedules:
            expected_role = (
                FlowRouteRole.SOURCE
                if schedule.die_id == 0
                else FlowRouteRole.DESTINATION
            )
            self.assertEqual(
                {binding.pair_route_ref for binding in schedule.flow_routes},
                cross_route_ids,
            )
            self.assertTrue(
                all(binding.role is expected_role for binding in schedule.flow_routes)
            )
            self.assertTrue(
                all(binding.local_noc_path for binding in schedule.flow_routes)
            )
        self.assertEqual(
            NaiveIntraDiePolicy().schedule(projection, ir1),
            schedules,
        )

    def test_tp1_full_head_history_dependencies_bytes_and_order_are_exact(self) -> None:
        ir1, fusion, standalone, contracts, projection = _project_tp1()
        projection.validate_against(ir1, fusion, standalone)
        self.assertEqual(projection.state_transfers, contracts)
        self.assertEqual(len(contracts), 4)
        self.assertEqual([contract.bytes for contract in contracts], [256] * 4)
        self.assertEqual(sum(contract.bytes for contract in contracts), 1024)
        self.assertEqual(
            [(dag.die_id, len(dag.tasks), len(dag.flows)) for dag in projection.dags],
            [(0, 48, 4), (1, 52, 4)],
        )

        access_index = {access.id: access for access in ir1.state_accesses}
        dag_index = {dag.die_id: dag for dag in projection.dags}
        route_index = {route.id: route for route in ir1.cross_routes}
        for contract in contracts:
            route = route_index[contract.cross_group_route_ref]
            source_dag = dag_index[route.die_path[0]]
            destination_dag = dag_index[route.die_path[-1]]
            source_tasks = {task.id: task for task in source_dag.tasks}
            destination_tasks = {task.id: task for task in destination_dag.tasks}
            source_order = {
                task.id: index for index, task in enumerate(source_dag.tasks)
            }
            destination_order = {
                task.id: index for index, task in enumerate(destination_dag.tasks)
            }

            send = source_tasks[
                canonical_state_transfer_task_id(
                    contract.id, SemanticTaskKind.SEND, route.die_path[0]
                )
            ]
            recv = destination_tasks[
                canonical_state_transfer_task_id(
                    contract.id, SemanticTaskKind.RECV, route.die_path[-1]
                )
            ]
            wait = destination_tasks[
                canonical_state_transfer_task_id(
                    contract.id, SemanticTaskKind.WAIT, route.die_path[-1]
                )
            ]
            source_dma = next(
                task
                for task in source_dag.tasks
                if isinstance(task.origin_ref, StateIoOrigin)
                and task.origin_ref.state_access_ref
                == contract.source_state_access_ref
            )
            destination_dma = next(
                task
                for task in destination_dag.tasks
                if isinstance(task.origin_ref, StateIoOrigin)
                and task.origin_ref.state_access_ref
                == contract.destination_state_access_ref
            )
            self.assertEqual(source_dma.kind, SemanticTaskKind.DMA_OUT)
            self.assertEqual(destination_dma.kind, SemanticTaskKind.DMA_OUT)
            assert source_dma.dma is not None
            assert destination_dma.dma is not None
            source_targets = source_dma.dma.access_task_refs
            destination_targets = destination_dma.dma.access_task_refs
            self.assertEqual(send.deps, source_targets)
            self.assertEqual(source_dma.deps, source_targets)
            self.assertEqual(recv.deps, ())
            self.assertEqual(wait.deps, (recv.id,))
            self.assertEqual(destination_dma.deps, destination_targets)
            self.assertTrue(source_targets)
            self.assertTrue(destination_targets)
            self.assertTrue(
                all(wait.id in destination_tasks[target].deps for target in destination_targets)
            )
            self.assertLess(max(source_order[target] for target in source_targets), source_order[send.id])
            self.assertLess(source_order[send.id], source_order[source_dma.id])
            self.assertLess(destination_order[recv.id], destination_order[wait.id])
            self.assertLess(destination_order[wait.id], min(destination_order[target] for target in destination_targets))
            self.assertLess(max(destination_order[target] for target in destination_targets), destination_order[destination_dma.id])

            for task in (send, recv):
                self.assertIsInstance(task.origin_ref, StateTransferOrigin)
                self.assertEqual(task.bytes, contract.bytes)
                self.assertEqual(task.shape, contract.source_local_shape)
            self.assertEqual(send.tensor_slice.offset, contract.source_local_offset)
            self.assertEqual(recv.tensor_slice.offset, contract.destination_local_offset)
            flow_id = canonical_state_transfer_flow_id(contract.id)
            replicas = [
                flow
                for dag in projection.dags
                for flow in dag.flows
                if flow.id == flow_id
            ]
            self.assertEqual(len(replicas), 2)
            self.assertTrue(all(flow.bytes == 256 for flow in replicas))

            source_staging = next(
                value
                for value in source_dag.state_staging_values
                if value.id
                == canonical_state_staging_value_id(
                    contract.source_state_access_ref
                )
            )
            destination_staging = next(
                value
                for value in destination_dag.state_staging_values
                if value.id
                == canonical_state_staging_value_id(
                    contract.destination_state_access_ref
                )
            )
            self.assertNotIn(recv.id, source_staging.producer_tasks)
            self.assertEqual(destination_staging.producer_tasks, (recv.id,))
            self.assertEqual(
                access_index[contract.source_state_access_ref].rank,
                route.source_rank,
            )
            self.assertEqual(
                access_index[contract.destination_state_access_ref].rank,
                route.destination_rank,
            )

        repeated = _project_tp1()[-1]
        self.assertEqual(repeated, projection)

    def test_gather_and_scatter_stop_at_projector_boundary(self) -> None:
        for prefill_tp, decode_tp in ((2, 1), (1, 2)):
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                spec = _spec(prefill_tp, decode_tp)
                spec = replace(
                    spec,
                    parallel=replace(
                        spec.parallel,
                        instances=tuple(
                            replace(instance, sp=instance.tp > 1)
                            for instance in spec.parallel.instances
                        ),
                    ),
                )
                decode_profile = _profile(prefill=False)
                if decode_tp > 1:
                    request = replace(
                        decode_profile.requests[0],
                        decode_tokens=decode_tp,
                        context_tokens=8 + decode_tp,
                    )
                    decode_profile = type(decode_profile).create(
                        key=replace(
                            decode_profile.key,
                            decode_tokens=decode_tp,
                            context_sum=8 + decode_tp,
                            context_max=8 + decode_tp,
                        ),
                        requests=(request,),
                    )
                    spec = replace(
                        spec,
                        workload=replace(
                            spec.workload,
                            infer=replace(
                                spec.workload.infer,
                                pd_static=replace(
                                    spec.workload.infer.pd_static,
                                    decode_profile=decode_profile.key,
                                ),
                            ),
                        ),
                    )
                spec.validate("spec")
                pd_plan = build_stage4_pd_plan(
                    spec,
                    prefill_profile=_profile(prefill=True),
                    decode_profile=decode_profile,
                )
                graph = build_stage4_separated_ir0(spec, pd_plan)
                fabric = _three_die_fabric()
                context = PlacementContext.create(
                    producer_pass="stage4_projector_reshard_negative",
                    fabric=fabric,
                    placement=spec.placement,
                    hbm_address_spaces=valid_hbm_address_spaces(fabric),
                )
                ir1 = place_stage4_ir0(graph, context, pd_plan)
                contracts = build_stage4_state_transfers(ir1, pd_plan)
                self.assertEqual(len(contracts), 8)
                with self.assertRaisesRegex(
                    UnsupportedFeatureError,
                    "equal-TP contiguous full-head THD",
                ):
                    NaiveProjectToIR2().run(
                        ir1,
                        (),
                        (),
                        state_transfers=contracts,
                    )


if __name__ == "__main__":
    unittest.main()
