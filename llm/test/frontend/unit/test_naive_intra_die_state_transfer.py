from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.global_action import (
    build_global_action_dag,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NAIVE_INTRADIE_POLICY_SCHEMA_VERSION,
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.schema.global_action import (
    GLOBAL_ACTION_DAG_SCHEMA_VERSION,
    GLOBAL_ACTION_SCHEMA_VERSION,
)
from llm.frontend.wafer_frontend.schema.ir1 import MemoryInitiator
from llm.frontend.wafer_frontend.schema.ir2 import (
    INTRA_DIE_SCHEDULE_SCHEMA_VERSION,
    INTRA_DIE_SCHEDULE_SET_SCHEMA_VERSION,
    BufferOwnership,
    BufferUseRole,
    FlowRouteRole,
    IntraDieSchedule,
    RegionLowering,
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
    StateUseAccess,
    canonical_state_transfer_flow_id,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
    INTRADIE_SCHEDULING_CONTEXT_SCHEMA_VERSION,
    SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION,
    IntraDieSchedulingContract,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind

from test_naive_project_state_transfer import _project


def _scheduled(kinds: tuple[StateKind, ...]):
    graph, fusion, standalone, contracts, projection = _project(
        kinds,
        kv_shape=(16,),
        sram_capacity_bytes=2 * 1024 * 1024,
    )
    schedules = NaiveIntraDiePolicy().schedule(projection, graph)
    schedules.validate_against(projection, graph)
    return graph, fusion, standalone, contracts, projection, schedules


def _schedule_replacement(
    schedule: IntraDieSchedule,
    **updates: object,
) -> IntraDieSchedule:
    fields = schedule._semantic_key()
    fields.update(updates)
    return IntraDieSchedule.create(
        producer_pass=schedule.producer_pass,
        **fields,
    )


def _is_ancestor(action_by_id: dict[str, object], ancestor: str, target: str) -> bool:
    pending = list(action_by_id[target].deps)
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == ancestor:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(action_by_id[current].deps)
    return False


class NaiveIntraDieStateTransferTest(unittest.TestCase):
    def test_single_k_schedule_closes_endpoint_components_and_staging(self) -> None:
        graph, _fusion, _standalone, contracts, projection, schedules = (
            _scheduled((StateKind.KV_KEY,))
        )
        contract = contracts[0]
        route = next(
            route
            for group in graph.groups
            for route in group.embedding.routes
            if route.id == contract.pair_route_ref
        )
        source_dag = next(
            dag for dag in projection.dags if dag.die_id == route.die_path[0]
        )
        destination_dag = next(
            dag for dag in projection.dags if dag.die_id == route.die_path[-1]
        )
        schedule_by_dag = {
            schedule.dag_id: schedule for schedule in schedules.schedules
        }
        source_schedule = schedule_by_dag[source_dag.id]
        destination_schedule = schedule_by_dag[destination_dag.id]
        source_placement = {
            placement.task_id: placement.core_id
            for placement in source_schedule.placements
        }
        destination_placement = {
            placement.task_id: placement.core_id
            for placement in destination_schedule.placements
        }

        send = next(
            task
            for task in source_dag.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
            and task.origin_ref.state_transfer_ref == contract.id
            and task.kind is SemanticTaskKind.SEND
        )
        source_dma = next(
            task
            for task in source_dag.tasks
            if task.kind is SemanticTaskKind.DMA_IN
            and task.dma is not None
            and task.dma.local_value_ref == send.read_values[0]
        )
        assert source_dma.dma is not None
        source_targets = tuple(
            next(task for task in source_dag.tasks if task.id == target_ref)
            for target_ref in source_dma.dma.access_task_refs
        )
        self.assertEqual(send.deps, source_dma.dma.access_task_refs)
        self.assertTrue(
            all(
                source_dma.id in target.deps
                and target.kind is SemanticTaskKind.COMP
                for target in source_targets
            )
        )
        self.assertEqual(
            {
                source_placement[task.id]
                for task in (source_dma, *source_targets, send)
            },
            {source_placement[send.id]},
        )
        source_use_by_task = {
            task_id: tuple(
                use
                for use in source_schedule.task_buffer_uses
                if use.task_id == task_id
            )
            for task_id in (source_dma.id, send.id)
        }
        self.assertEqual(
            source_use_by_task[source_dma.id][0].role,
            BufferUseRole.DMA_DESTINATION,
        )
        self.assertEqual(
            source_use_by_task[send.id][0].role,
            BufferUseRole.SEND_SOURCE,
        )
        self.assertEqual(
            source_use_by_task[source_dma.id][0].binding_id,
            source_use_by_task[send.id][0].binding_id,
        )
        self.assertEqual(
            tuple(
                use.access
                for use in source_schedule.task_state_uses
                if use.task_id == source_dma.id
            ),
            (StateUseAccess.READ,),
        )
        self.assertFalse(
            any(
                use.task_id == send.id
                for use in source_schedule.task_state_uses
            )
        )

        recv, wait = (
            next(
                task
                for task in destination_dag.tasks
                if isinstance(task.origin_ref, StateTransferOrigin)
                and task.origin_ref.state_transfer_ref == contract.id
                and task.kind is kind
            )
            for kind in (SemanticTaskKind.RECV, SemanticTaskKind.WAIT)
        )
        destination_dma = next(
            task
            for task in destination_dag.tasks
            if task.kind is SemanticTaskKind.DMA_OUT
            and task.dma is not None
            and task.dma.local_value_ref == recv.write_values[0]
        )
        assert destination_dma.dma is not None
        destination_targets = tuple(
            next(
                task
                for task in destination_dag.tasks
                if task.id == target_ref
            )
            for target_ref in destination_dma.dma.access_task_refs
        )
        self.assertEqual(wait.deps, (recv.id,))
        self.assertTrue(
            all(
                wait.id in target.deps
                and target.kind is SemanticTaskKind.COMP
                for target in destination_targets
            )
        )
        self.assertEqual(
            destination_dma.deps,
            destination_dma.dma.access_task_refs,
        )
        self.assertEqual(
            {
                destination_placement[task.id]
                for task in (recv, wait, *destination_targets, destination_dma)
            },
            {destination_placement[recv.id]},
        )
        destination_use_by_task = {
            task_id: tuple(
                use
                for use in destination_schedule.task_buffer_uses
                if use.task_id == task_id
            )
            for task_id in (recv.id, destination_dma.id)
        }
        self.assertEqual(
            destination_use_by_task[recv.id][0].binding_id,
            destination_use_by_task[destination_dma.id][0].binding_id,
        )
        recv_runtime = next(
            binding
            for binding in destination_schedule.runtime_bindings
            if binding.task_id == recv.id
        )
        wait_runtime = next(
            binding
            for binding in destination_schedule.runtime_bindings
            if binding.task_id == wait.id
        )
        self.assertEqual(recv_runtime.token_symbol, wait_runtime.token_symbol)
        self.assertEqual(recv_runtime.event_symbol, wait_runtime.event_symbol)
        self.assertEqual(
            tuple(
                use.access
                for use in destination_schedule.task_state_uses
                if use.task_id == destination_dma.id
            ),
            (StateUseAccess.WRITE,),
        )
        self.assertFalse(
            any(
                use.task_id in (recv.id, wait.id)
                for use in destination_schedule.task_state_uses
            )
        )

        flow_id = canonical_state_transfer_flow_id(contract.id)
        source_route = next(
            binding
            for binding in source_schedule.flow_routes
            if binding.flow_id == flow_id
        )
        destination_route = next(
            binding
            for binding in destination_schedule.flow_routes
            if binding.flow_id == flow_id
        )
        self.assertEqual(source_route.role, FlowRouteRole.SOURCE)
        self.assertEqual(destination_route.role, FlowRouteRole.DESTINATION)

        source_die = next(
            die for die in graph.fabric.dies if die.id == source_dag.die_id
        )
        destination_die = next(
            die
            for die in graph.fabric.dies
            if die.id == destination_dag.die_id
        )
        profile_by_id = {
            profile.id: profile for profile in graph.fabric.sram_profiles
        }
        for die, core_id, required in (
            (
                source_die,
                source_placement[send.id],
                {
                    MemoryInitiator.LSU,
                    MemoryInitiator.COMPUTE,
                    MemoryInitiator.DTE,
                },
            ),
            (
                destination_die,
                destination_placement[recv.id],
                {
                    MemoryInitiator.NOC_RX,
                    MemoryInitiator.COMPUTE,
                    MemoryInitiator.LSU,
                },
            ),
        ):
            core = next(
                core for core in die.cores if core.runtime_core_id == core_id
            )
            self.assertTrue(
                any(
                    required.issubset(region.access)
                    for region in profile_by_id[
                        core.sram_profile_ref
                    ].regions
                )
            )

    def test_schedule_rejects_split_staging_and_wait_token(self) -> None:
        graph, _fusion, _standalone, contracts, projection, schedules = (
            _scheduled((StateKind.KV_KEY,))
        )
        contract = contracts[0]
        destination_dag = next(
            dag
            for dag in projection.dags
            if any(
                isinstance(task.origin_ref, StateTransferOrigin)
                and task.origin_ref.state_transfer_ref == contract.id
                and task.kind is SemanticTaskKind.RECV
                for task in dag.tasks
            )
        )
        destination_schedule = next(
            schedule
            for schedule in schedules.schedules
            if schedule.dag_id == destination_dag.id
        )
        recv = next(
            task
            for task in destination_dag.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
            and task.origin_ref.state_transfer_ref == contract.id
            and task.kind is SemanticTaskKind.RECV
        )
        wait = next(
            task
            for task in destination_dag.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
            and task.origin_ref.state_transfer_ref == contract.id
            and task.kind is SemanticTaskKind.WAIT
        )
        recv_use = next(
            use
            for use in destination_schedule.task_buffer_uses
            if use.task_id == recv.id
        )
        root = next(
            binding
            for binding in destination_schedule.buffer_bindings
            if binding.id == recv_use.binding_id
        )
        alias = replace(
            root,
            id="forged.transfer.alias",
            ownership=BufferOwnership.ALIASED,
            alias_of=root.id,
        )
        split_uses = tuple(
            replace(use, binding_id=alias.id)
            if use.task_id == recv.id
            else use
            for use in destination_schedule.task_buffer_uses
        )
        split_schedule = _schedule_replacement(
            destination_schedule,
            buffer_bindings=destination_schedule.buffer_bindings + (alias,),
            task_buffer_uses=split_uses,
        )
        with self.assertRaisesRegex(SchemaError, "reuse one staging root"):
            split_schedule.validate_against(destination_dag, graph)

        wait_binding = next(
            binding
            for binding in destination_schedule.runtime_bindings
            if binding.task_id == wait.id
        )
        forged_runtime = tuple(
            replace(binding, token_symbol="forged.transfer.token")
            if binding.task_id == wait.id
            else binding
            for binding in destination_schedule.runtime_bindings
        )
        token_schedule = _schedule_replacement(
            destination_schedule,
            runtime_bindings=forged_runtime,
        )
        with self.assertRaisesRegex(SchemaError, "reuse its RECV event and token"):
            token_schedule.validate_against(destination_dag, graph)
        self.assertIsNotNone(wait_binding.token_symbol)

    def test_global_action_preserves_transfer_lowering_and_wait_ancestry(self) -> None:
        graph, _fusion, _standalone, contracts, projection, schedules = (
            _scheduled((StateKind.KV_KEY,))
        )
        actions = build_global_action_dag(graph, projection, schedules)
        actions.validate_against(graph, projection, schedules)
        transfer_actions = tuple(
            action
            for action in actions.actions
            if isinstance(action.origin_ref, StateTransferOrigin)
        )
        self.assertEqual(
            tuple(action.task_kind for action in transfer_actions).count(
                SemanticTaskKind.SEND
            ),
            1,
        )
        self.assertEqual(
            tuple(action.task_kind for action in transfer_actions).count(
                SemanticTaskKind.RECV
            ),
            1,
        )
        self.assertEqual(
            tuple(action.task_kind for action in transfer_actions).count(
                SemanticTaskKind.WAIT
            ),
            1,
        )
        self.assertTrue(
            all(
                action.lowering is RegionLowering.STRICT_STATE_TRANSFER
                and not action.state_uses
                for action in transfer_actions
            )
        )
        contract = contracts[0]
        endpoint_accesses = {
            contract.source_state_access_ref,
            contract.destination_state_access_ref,
        }
        endpoint_actions = tuple(
            action
            for action in actions.actions
            if isinstance(action.origin_ref, StateIoOrigin)
            and action.origin_ref.state_access_ref in endpoint_accesses
        )
        self.assertEqual(len(endpoint_actions), 2)
        self.assertTrue(
            all(
                action.lowering is RegionLowering.STRICT_STATE_IO
                and len(action.state_uses) == 1
                for action in endpoint_actions
            )
        )
        wait_action = next(
            action
            for action in transfer_actions
            if action.task_kind is SemanticTaskKind.WAIT
        )
        store_action = next(
            action
            for action in endpoint_actions
            if action.task_kind is SemanticTaskKind.DMA_OUT
        )
        action_by_id = {action.id: action for action in actions.actions}
        self.assertTrue(
            _is_ancestor(action_by_id, wait_action.id, store_action.id)
        )

        with self.assertRaisesRegex(SchemaError, "exactly accompany"):
            replace(
                transfer_actions[0],
                lowering=RegionLowering.STRICT_ACTIONS,
            ).validate("action")
        ordinary_comp = next(
            action
            for action in actions.actions
            if action.task_kind is SemanticTaskKind.COMP
            and not isinstance(action.origin_ref, StateTransferOrigin)
        )
        with self.assertRaisesRegex(SchemaError, "exactly accompany"):
            replace(
                ordinary_comp,
                lowering=RegionLowering.STRICT_STATE_TRANSFER,
            ).validate("action")

    def test_kv_pair_freezes_numeric_transport_and_hbm_endpoints(self) -> None:
        graph, _fusion, _standalone, contracts, projection, schedules = (
            _scheduled((StateKind.KV_KEY, StateKind.KV_VALUE))
        )
        actions = build_global_action_dag(graph, projection, schedules)
        transfer_actions = tuple(
            action
            for action in actions.actions
            if isinstance(action.origin_ref, StateTransferOrigin)
        )
        sends = tuple(
            action
            for action in transfer_actions
            if action.task_kind is SemanticTaskKind.SEND
        )
        recvs = tuple(
            action
            for action in transfer_actions
            if action.task_kind is SemanticTaskKind.RECV
        )
        waits = tuple(
            action
            for action in transfer_actions
            if action.task_kind is SemanticTaskKind.WAIT
        )
        self.assertEqual((len(sends), len(recvs), len(waits)), (2, 2, 2))
        self.assertEqual({action.bytes for action in sends + recvs}, {32})
        self.assertEqual(sum(action.bytes for action in sends), 64)
        self.assertEqual(sum(action.bytes for action in recvs), 64)
        self.assertTrue(all(not action.state_uses for action in transfer_actions))

        source_accesses = {
            contract.source_state_access_ref for contract in contracts
        }
        destination_accesses = {
            contract.destination_state_access_ref for contract in contracts
        }
        loads = tuple(
            action
            for action in actions.actions
            if isinstance(action.origin_ref, StateIoOrigin)
            and action.origin_ref.state_access_ref in source_accesses
        )
        stores = tuple(
            action
            for action in actions.actions
            if isinstance(action.origin_ref, StateIoOrigin)
            and action.origin_ref.state_access_ref in destination_accesses
        )
        self.assertEqual((len(loads), len(stores)), (2, 2))
        self.assertEqual(sum(action.bytes for action in loads), 64)
        self.assertEqual(sum(action.bytes for action in stores), 64)
        self.assertTrue(
            all(
                action.task_kind is SemanticTaskKind.DMA_IN
                and action.state_uses[0].access is StateUseAccess.READ
                for action in loads
            )
        )
        self.assertTrue(
            all(
                action.task_kind is SemanticTaskKind.DMA_OUT
                and action.state_uses[0].access is StateUseAccess.WRITE
                for action in stores
            )
        )

    def test_p2_versions_and_contract_are_frozen(self) -> None:
        self.assertEqual(
            NAIVE_INTRADIE_POLICY_SCHEMA_VERSION,
            "wafer_frontend.naive_intra_die_policy/v8",
        )
        self.assertEqual(
            INTRA_DIE_SCHEDULE_SCHEMA_VERSION,
            "wafer_frontend.intra_die_schedule/v1alpha14",
        )
        self.assertEqual(
            INTRA_DIE_SCHEDULE_SET_SCHEMA_VERSION,
            "wafer_frontend.intra_die_schedule_set/v1alpha9",
        )
        self.assertEqual(
            INTRADIE_SCHEDULING_CONTEXT_SCHEMA_VERSION,
            "wafer_frontend.intra_die_scheduling_context/v1alpha8",
        )
        self.assertEqual(
            IntraDieSchedulingContract
            .NAIVE_COMPONENT_RR_XY_SEQUENTIAL_STATE_TRANSFER_V5.value,
            "naive_component_rr_xy_sequential_state_transfer/v5",
        )
        self.assertEqual(
            SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION,
            "wafer_frontend.scheduled_ir2_bundle/v1alpha6",
        )
        self.assertEqual(
            GLOBAL_ACTION_SCHEMA_VERSION,
            "wafer_frontend.global_action/v1alpha8",
        )
        self.assertEqual(
            GLOBAL_ACTION_DAG_SCHEMA_VERSION,
            "wafer_frontend.global_action_dag/v1alpha11",
        )
        self.assertEqual(
            GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
            "wafer_frontend.global_action_bundle/v1alpha6",
        )


if __name__ == "__main__":
    unittest.main()
