from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.stage4_segmented_state_transfer import (
    build_stage4_segmented_state_transfers,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    INTRA_DIE_DAG_SCHEMA_VERSION,
    IR2_PROJECTION_RESULT_SCHEMA_VERSION,
    IR2ProjectionResult,
    IntraDieDAG,
    RegionLowering,
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
    canonical_state_transfer_flow_id,
    canonical_state_transfer_task_id,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)

from test_stage4_carriers import _chain


def _case():
    planned = _chain(2, 1)[-1]
    contracts = build_stage4_segmented_state_transfers(planned)
    projection = NaiveProjectToIR2().run(
        planned.graph,
        planned.fusion_plans,
        planned.standalone_plans,
        state_transfers=contracts,
    )
    return planned, contracts, projection


def _rebuild_dag(dag: IntraDieDAG, **changes: object) -> IntraDieDAG:
    fields = dag._semantic_key()
    fields.update(changes)
    return IntraDieDAG.create(
        producer_pass=dag.producer_pass,
        **fields,
    )


def _rebuild_projection(
    projection: IR2ProjectionResult,
    replacement: IntraDieDAG,
) -> IR2ProjectionResult:
    fields = projection._semantic_key()
    fields["dags"] = tuple(
        replacement if dag.die_id == replacement.die_id else dag
        for dag in projection.dags
    )
    return IR2ProjectionResult.create(
        producer_pass=projection.producer_pass,
        **fields,
    )


class Stage4ProjectSegmentedStateTransferTest(unittest.TestCase):
    def test_tp2_to_tp1_segments_have_exact_tasks_flows_and_dependencies(self) -> None:
        planned, contracts, projection = _case()
        projection.validate_against(
            planned.graph,
            planned.fusion_plans,
            planned.standalone_plans,
        )
        self.assertEqual(
            (INTRA_DIE_DAG_SCHEMA_VERSION, IR2_PROJECTION_RESULT_SCHEMA_VERSION),
            (
                "wafer_frontend.intra_die_dag/v1alpha14",
                "wafer_frontend.ir2_projection_result/v1alpha13",
            ),
        )
        self.assertEqual(
            loads_dataclass(
                IR2ProjectionResult, canonical_json(projection)
            ),
            projection,
        )
        self.assertEqual(len(contracts), 8)
        self.assertEqual({len(contract.segments) for contract in contracts}, {8})
        self.assertEqual(sum(contract.bytes for contract in contracts), 1024)
        self.assertEqual(
            [
                (
                    dag.die_id,
                    len(dag.tasks),
                    len(dag.flows),
                    len(dag.regions),
                    len(dag.state_transfer_ids),
                )
                for dag in projection.dags
            ],
            [(0, 112, 48, 52, 4), (1, 144, 80, 56, 8), (2, 172, 64, 52, 8)],
        )
        transfer_tasks = tuple(
            task
            for dag in projection.dags
            for task in dag.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
        )
        self.assertEqual(
            Counter(task.kind for task in transfer_tasks),
            Counter(
                {
                    SemanticTaskKind.SEND: 64,
                    SemanticTaskKind.RECV: 64,
                    SemanticTaskKind.WAIT: 64,
                    SemanticTaskKind.TRANSIT: 32,
                }
            ),
        )
        self.assertEqual(
            sum(flow.bytes for dag in projection.dags for flow in dag.flows),
            6656,
        )

        dag_index = {dag.die_id: dag for dag in projection.dags}
        route_index = {route.id: route for route in planned.graph.cross_routes}
        for contract in contracts:
            route = route_index[contract.cross_group_route_ref]
            source_dag = dag_index[route.die_path[0]]
            destination_dag = dag_index[route.die_path[-1]]
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
            self.assertIs(source_dma.kind, SemanticTaskKind.DMA_OUT)
            self.assertIs(destination_dma.kind, SemanticTaskKind.DMA_OUT)
            assert source_dma.dma is not None
            assert destination_dma.dma is not None
            source_targets = source_dma.dma.access_task_refs
            destination_targets = destination_dma.dma.access_task_refs
            source_task_index = {task.id: task for task in source_dag.tasks}
            destination_task_index = {
                task.id: task for task in destination_dag.tasks
            }
            waits: list[str] = []
            for segment_index, segment in enumerate(contract.segments):
                send = source_task_index[
                    canonical_state_transfer_task_id(
                        contract.id,
                        SemanticTaskKind.SEND,
                        route.die_path[0],
                        segment_index,
                    )
                ]
                recv = destination_task_index[
                    canonical_state_transfer_task_id(
                        contract.id,
                        SemanticTaskKind.RECV,
                        route.die_path[-1],
                        segment_index,
                    )
                ]
                wait = destination_task_index[
                    canonical_state_transfer_task_id(
                        contract.id,
                        SemanticTaskKind.WAIT,
                        route.die_path[-1],
                        segment_index,
                    )
                ]
                waits.append(wait.id)
                self.assertEqual(send.origin_ref.segment_index, segment_index)
                self.assertEqual(recv.origin_ref.segment_index, segment_index)
                self.assertEqual(wait.origin_ref.segment_index, segment_index)
                self.assertEqual(send.deps, source_targets)
                self.assertEqual(recv.deps, ())
                self.assertEqual(wait.deps, (recv.id,))
                self.assertEqual(
                    (send.tensor_slice.offset, send.tensor_slice.shape),
                    (segment.source_local_offset, segment.source_local_shape),
                )
                self.assertEqual(
                    (recv.tensor_slice.offset, recv.tensor_slice.shape),
                    (
                        segment.destination_local_offset,
                        segment.destination_local_shape,
                    ),
                )
                self.assertEqual((send.bytes, recv.bytes), (16, 16))
                flow_id = canonical_state_transfer_flow_id(
                    contract.id, segment_index
                )
                replicas = tuple(
                    flow
                    for die_id in route.die_path
                    for flow in dag_index[die_id].flows
                    if flow.id == flow_id
                )
                self.assertEqual(len(replicas), len(route.die_path))
                self.assertTrue(
                    all(
                        flow.bytes == segment.bytes
                        and flow.tensor_slice.shape
                        == segment.source_local_shape
                        for flow in replicas
                    )
                )
            for target_id in destination_targets:
                self.assertTrue(
                    set(waits).issubset(destination_task_index[target_id].deps)
                )
            for die_id in route.die_path:
                region = next(
                    region
                    for region in dag_index[die_id].regions
                    if region.state_transfer_ref == contract.id
                )
                self.assertIs(
                    region.lowering, RegionLowering.STRICT_STATE_TRANSFER
                )
                expected_task_count = (
                    16 if die_id == route.die_path[-1] else 8
                )
                self.assertEqual(len(region.task_ids), expected_task_count)

        self.assertEqual(_case()[2], projection)

    def test_segment_identity_region_flow_and_wait_tampering_fail_closed(self) -> None:
        planned, contracts, projection = _case()
        first = contracts[0]
        route = next(
            route for route in planned.graph.cross_routes
            if route.id == first.cross_group_route_ref
        )
        source_dag = next(
            dag for dag in projection.dags if dag.die_id == route.die_path[0]
        )
        destination_dag = next(
            dag for dag in projection.dags if dag.die_id == route.die_path[-1]
        )
        send_id = canonical_state_transfer_task_id(
            first.id, SemanticTaskKind.SEND, route.die_path[0], 0
        )
        send = next(task for task in source_dag.tasks if task.id == send_id)
        assert isinstance(send.origin_ref, StateTransferOrigin)
        forged_send = replace(
            send,
            origin_ref=replace(send.origin_ref, segment_index=1),
        )
        bad_source = _rebuild_dag(
            source_dag,
            tasks=tuple(
                forged_send if task.id == send.id else task
                for task in source_dag.tasks
            ),
        )
        with self.assertRaisesRegex(SchemaError, "canonical"):
            bad_source.validate()

        source_region = next(
            region for region in source_dag.regions
            if region.state_transfer_ref == first.id
        )
        bad_region_dag = _rebuild_dag(
            source_dag,
            regions=tuple(
                replace(region, task_ids=tuple(reversed(region.task_ids)))
                if region.id == source_region.id
                else region
                for region in source_dag.regions
            ),
        )
        with self.assertRaises(SchemaError):
            bad_region_dag.validate()

        flow_id = canonical_state_transfer_flow_id(first.id, 0)
        bad_flow_dag = _rebuild_dag(
            source_dag,
            flows=tuple(
                replace(
                    flow,
                    tensor_slice=replace(
                        flow.tensor_slice, offset=(1, 0, 0)
                    ),
                )
                if flow.id == flow_id
                else flow
                for flow in source_dag.flows
            ),
        )
        with self.assertRaisesRegex(SchemaError, "flow replica"):
            _rebuild_projection(
                projection, bad_flow_dag
            ).validate_against(
                planned.graph,
                planned.fusion_plans,
                planned.standalone_plans,
            )

        destination_dma = next(
            task
            for task in destination_dag.tasks
            if isinstance(task.origin_ref, StateIoOrigin)
            and task.origin_ref.state_access_ref
            == first.destination_state_access_ref
        )
        assert destination_dma.dma is not None
        target_id = destination_dma.dma.access_task_refs[0]
        wait_id = canonical_state_transfer_task_id(
            first.id, SemanticTaskKind.WAIT, route.die_path[-1], 0
        )
        bad_wait_dag = _rebuild_dag(
            destination_dag,
            tasks=tuple(
                replace(
                    task,
                    deps=tuple(dep for dep in task.deps if dep != wait_id),
                )
                if task.id == target_id
                else task
                for task in destination_dag.tasks
            ),
        )
        with self.assertRaisesRegex(SchemaError, "destination WAIT"):
            _rebuild_projection(
                projection, bad_wait_dag
            ).validate_against(
                planned.graph,
                planned.fusion_plans,
                planned.standalone_plans,
            )

        with self.assertRaisesRegex(SchemaError, "cannot mix"):
            NaiveProjectToIR2().run(
                planned.graph,
                planned.fusion_plans,
                planned.standalone_plans,
                state_transfers=(first, first.logical_slice),
            )


if __name__ == "__main__":
    unittest.main()
