from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    OpKind,
    StateAccess,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir2 import (
    IR2ProjectionResult,
    IntraDieDAG,
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
    TensorSlice,
    canonical_state_staging_value_id,
    canonical_state_transfer_flow_id,
    canonical_state_transfer_payload_id,
)
from llm.frontend.wafer_frontend.schema.persistent_state import (
    HbmBinding,
    PersistentStateDecl,
    PersistentStateManifest,
    StateKind,
)

from test_state_transfer_schema import _contract, _transfer_graph


def _with_sram_capacity(graph: IR1, capacity_bytes: int) -> IR1:
    profiles = []
    for profile in graph.fabric.sram_profiles:
        if len(profile.regions) != 1:
            raise AssertionError("large-SRAM fixture requires one named region")
        profiles.append(
            replace(
                profile,
                capacity_bytes=capacity_bytes,
                regions=(
                    replace(
                        profile.regions[0],
                        size_bytes=capacity_bytes,
                    ),
                ),
            )
        )
    fields = graph._semantic_key()
    fields["fabric"] = replace(
        graph.fabric,
        sram_profiles=tuple(profiles),
    )
    result = IR1.create(producer_pass=graph.producer_pass, **fields)
    result.validate()
    return result


def _with_kv_shape(graph: IR1, shape: tuple[int, ...]) -> IR1:
    manifest = graph.persistent_state_manifest
    assert manifest is not None
    replacement_by_state: dict[str, PersistentStateDecl] = {}
    declarations = []
    for declaration in manifest.declarations:
        if declaration.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
            replacement = PersistentStateDecl.create(
                identity=declaration.identity,
                shape=shape,
                dtype=declaration.dtype,
                layout=declaration.layout,
                lifetime=declaration.lifetime,
                access=declaration.access,
            )
            replacement_by_state[declaration.id] = replacement
            declarations.append(replacement)
        else:
            declarations.append(declaration)
    bindings = tuple(
        HbmBinding.create(
            state_ref=replacement_by_state.get(
                binding.state_ref,
                next(
                    declaration
                    for declaration in manifest.declarations
                    if declaration.id == binding.state_ref
                ),
            ).id,
            die_id=binding.die_id,
            address=binding.address,
            size_bytes=replacement_by_state.get(
                binding.state_ref,
                next(
                    declaration
                    for declaration in manifest.declarations
                    if declaration.id == binding.state_ref
                ),
            ).tensor_bytes,
        )
        for binding in manifest.bindings
    )
    rewritten_accesses = tuple(
        StateAccess.create(
            node_ref=access.node_ref,
            state_ref=replacement_by_state.get(
                access.state_ref,
                next(
                    declaration
                    for declaration in manifest.declarations
                    if declaration.id == access.state_ref
                ),
            ).id,
            mode=access.mode,
            rank=access.rank,
        )
        for access in graph.state_accesses
    )
    fields = graph._semantic_key()
    fields["persistent_state_manifest"] = PersistentStateManifest.create(
        address_spaces=manifest.address_spaces,
        declarations=tuple(declarations),
        bindings=bindings,
    )
    fields["state_accesses"] = tuple(
        sorted(
            rewritten_accesses,
            key=lambda item: (
                item.node_ref,
                item.state_ref,
                item.rank,
                item.id,
            ),
        )
    )
    result = IR1.create(producer_pass=graph.producer_pass, **fields)
    result.validate()
    return result


def _project(
    kinds: tuple[StateKind, ...],
    *,
    kv_shape: tuple[int, ...] | None = None,
    sram_capacity_bytes: int | None = None,
):
    graph, accesses, forward, _reverse = _transfer_graph()
    if sram_capacity_bytes is not None:
        graph = _with_sram_capacity(graph, sram_capacity_bytes)
    if kv_shape is not None:
        graph = _with_kv_shape(graph, kv_shape)
        manifest = graph.persistent_state_manifest
        assert manifest is not None
        declaration_index = {
            declaration.id: declaration for declaration in manifest.declarations
        }
        attention = next(
            node for node in graph.nodes if node.kind is OpKind.ATTENTION
        )
        accesses = {
            (access.rank, declaration_index[access.state_ref].identity.kind): access
            for access in graph.state_accesses
            if access.node_ref == attention.id
            and declaration_index[access.state_ref].identity.kind
            in (StateKind.KV_KEY, StateKind.KV_VALUE)
        }
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
    contracts = tuple(
        sorted(
            (
                _contract(
                    graph,
                    accesses,
                    forward,
                    source_kind=kind,
                    destination_kind=kind,
                )
                for kind in kinds
            ),
            key=lambda item: (
                item.source_ir1_id,
                item.source_state_access_ref,
                item.destination_state_access_ref,
                item.pair_route_ref,
                item.id,
            ),
        )
    )
    projection = NaiveProjectToIR2().run(
        graph,
        fusion_plans,
        standalone_plans,
        state_transfers=contracts,
    )
    projection.validate_against(graph, fusion_plans, standalone_plans)
    return graph, fusion_plans, standalone_plans, contracts, projection


def _replace_projection_dag(
    projection: IR2ProjectionResult,
    source_dag: IntraDieDAG,
    **updates: object,
) -> IR2ProjectionResult:
    dag_fields = source_dag._semantic_key()
    dag_fields.update(updates)
    replacement_dag = IntraDieDAG.create(
        producer_pass=source_dag.producer_pass,
        **dag_fields,
    )
    projection_fields = projection._semantic_key()
    projection_fields["dags"] = tuple(
        replacement_dag if dag.id == source_dag.id else dag
        for dag in projection.dags
    )
    return IR2ProjectionResult.create(
        producer_pass=projection.producer_pass,
        **projection_fields,
    )


class NaiveProjectStateTransferTest(unittest.TestCase):
    def test_single_k_transfer_closes_exact_endpoint_chain(self) -> None:
        graph, _fusion, _standalone, contracts, projection = _project(
            (StateKind.KV_KEY,)
        )
        contract = contracts[0]
        access_index = {access.id: access for access in graph.state_accesses}
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
        source_access = access_index[contract.source_state_access_ref]
        destination_access = access_index[
            contract.destination_state_access_ref
        ]
        source_staging = canonical_state_staging_value_id(source_access.id)
        destination_staging = canonical_state_staging_value_id(
            destination_access.id
        )

        source_dma = next(
            task
            for task in source_dag.tasks
            if isinstance(task.origin_ref, StateIoOrigin)
            and task.origin_ref.state_access_ref == source_access.id
            and task.kind is SemanticTaskKind.DMA_IN
        )
        send = next(
            task
            for task in source_dag.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
        )
        destination_tasks = tuple(
            task
            for task in destination_dag.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
        )
        recv = next(
            task
            for task in destination_tasks
            if task.kind is SemanticTaskKind.RECV
        )
        wait = next(
            task
            for task in destination_tasks
            if task.kind is SemanticTaskKind.WAIT
        )
        destination_dma = next(
            task
            for task in destination_dag.tasks
            if isinstance(task.origin_ref, StateIoOrigin)
            and task.origin_ref.state_access_ref == destination_access.id
            and task.kind is SemanticTaskKind.DMA_OUT
        )
        assert source_dma.dma is not None
        assert destination_dma.dma is not None
        self.assertEqual(send.deps, source_dma.dma.access_task_refs)
        self.assertEqual(wait.deps, (recv.id,))
        self.assertTrue(
            all(
                wait.id
                in next(
                    task.deps
                    for task in destination_dag.tasks
                    if task.id == target_id
                )
                for target_id in destination_dma.dma.access_task_refs
            )
        )
        self.assertEqual(
            destination_dma.deps, destination_dma.dma.access_task_refs
        )
        self.assertEqual(send.read_values, (source_staging,))
        self.assertEqual(recv.write_values, (destination_staging,))
        self.assertNotEqual(source_staging, destination_staging)

        flow_id = canonical_state_transfer_flow_id(contract.id)
        logical_payload = canonical_state_transfer_payload_id(contract.id)
        replicas = tuple(
            flow
            for dag in projection.dags
            for flow in dag.flows
            if flow.id == flow_id
        )
        self.assertEqual(len(replicas), len(route.die_path))
        self.assertTrue(
            all(flow.tensor_slice.value_id == logical_payload for flow in replicas)
        )
        self.assertEqual(send.flow_id, flow_id)
        self.assertEqual(recv.flow_id, flow_id)
        self.assertEqual(send.tensor_slice.value_id, source_staging)
        self.assertEqual(recv.tensor_slice.value_id, destination_staging)


    def test_kv_pair_freezes_32_byte_payloads_and_four_replicas(self) -> None:
        graph, _fusion, _standalone, contracts, projection = _project(
            (StateKind.KV_KEY, StateKind.KV_VALUE),
            kv_shape=(16,),
        )
        self.assertEqual(len(contracts), 2)
        manifest = graph.persistent_state_manifest
        assert manifest is not None
        access_index = {access.id: access for access in graph.state_accesses}
        declaration_index = {
            declaration.id: declaration for declaration in manifest.declarations
        }
        transfer_flow_ids = {
            canonical_state_transfer_flow_id(contract.id)
            for contract in contracts
        }
        replicas = tuple(
            flow
            for dag in projection.dags
            for flow in dag.flows
            if flow.id in transfer_flow_ids
        )
        self.assertEqual(len(replicas), 4)
        self.assertEqual({flow.bytes for flow in replicas}, {32})
        # Replica bytes are structural local views, not the D2D accounting sum.
        self.assertEqual(sum(flow.bytes for flow in replicas), 128)
        self.assertEqual(
            sum(
                declaration_index[
                    access_index[contract.source_state_access_ref].state_ref
                ].tensor_bytes
                for contract in contracts
            ),
            64,
        )

        transferred_access_ids = {
            access_ref
            for contract in contracts
            for access_ref in (
                contract.source_state_access_ref,
                contract.destination_state_access_ref,
            )
        }
        transferred_staging = tuple(
            value
            for dag in projection.dags
            for value in dag.state_staging_values
            if value.state_access_ref in transferred_access_ids
        )
        self.assertEqual(len(transferred_staging), 4)
        transfer_tasks = tuple(
            task
            for dag in projection.dags
            for task in dag.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
        )
        self.assertEqual(
            tuple(task.kind for task in transfer_tasks).count(
                SemanticTaskKind.SEND
            ),
            2,
        )
        self.assertEqual(
            tuple(task.kind for task in transfer_tasks).count(
                SemanticTaskKind.RECV
            ),
            2,
        )
        self.assertEqual(
            tuple(task.kind for task in transfer_tasks).count(
                SemanticTaskKind.WAIT
            ),
            2,
        )

        attention = next(
            node for node in graph.nodes if node.kind is OpKind.ATTENTION
        )
        attention_tasks = tuple(
            task
            for dag in projection.dags
            for task in dag.tasks
            if task.member_id == attention.id
            and task.kind is SemanticTaskKind.COMP
        )
        self.assertEqual(len(attention_tasks), 2)
        for task in attention_tasks:
            assert task.compute is not None
            self.assertEqual(task.read_values, attention.inputs)
            self.assertEqual(
                tuple(operand.value_id for operand in task.compute.inputs),
                attention.inputs,
            )



    def test_tampered_logical_payload_id_fails_closed(self) -> None:
        graph, fusion, standalone, contracts, projection = _project(
            (StateKind.KV_KEY,)
        )
        flow_id = canonical_state_transfer_flow_id(contracts[0].id)
        dag = next(
            dag
            for dag in projection.dags
            if any(flow.id == flow_id for flow in dag.flows)
        )
        flows = tuple(
            replace(
                flow,
                tensor_slice=TensorSlice(
                    "forged.logical.payload",
                    flow.tensor_slice.offset,
                    flow.tensor_slice.shape,
                ),
            )
            if flow.id == flow_id
            else flow
            for flow in dag.flows
        )
        tampered = _replace_projection_dag(projection, dag, flows=flows)
        with self.assertRaisesRegex(
            SchemaError,
            "logical/local payload geometry disagrees",
        ):
            tampered.validate_against(graph, fusion, standalone)

    def test_missing_destination_wait_edge_fails_closed(self) -> None:
        graph, fusion, standalone, contracts, projection = _project(
            (StateKind.KV_KEY,)
        )
        contract = contracts[0]
        destination_dag = next(
            dag
            for dag in projection.dags
            if sum(
                1
                for task in dag.tasks
                if isinstance(task.origin_ref, StateTransferOrigin)
                and task.origin_ref.state_transfer_ref == contract.id
            )
            == 2
        )
        wait = next(
            task
            for task in destination_dag.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
            and task.kind is SemanticTaskKind.WAIT
        )
        target = next(
            task
            for task in destination_dag.tasks
            if wait.id in task.deps
        )
        tasks = tuple(
            replace(
                task,
                deps=tuple(dep for dep in task.deps if dep != wait.id),
            )
            if task.id == target.id
            else task
            for task in destination_dag.tasks
        )
        tampered = _replace_projection_dag(
            projection, destination_dag, tasks=tasks
        )
        with self.assertRaisesRegex(
            SchemaError,
            "consumer dependency closure omits a relevant slice producer",
        ):
            tampered.validate_against(graph, fusion, standalone)

    def test_wrong_endpoint_local_staging_fails_closed(self) -> None:
        graph, fusion, standalone, contracts, projection = _project(
            (StateKind.KV_KEY, StateKind.KV_VALUE),
            kv_shape=(16,),
        )
        contract = contracts[0]
        other_contract = contracts[1]
        source_dag = next(
            dag
            for dag in projection.dags
            if any(
                isinstance(task.origin_ref, StateTransferOrigin)
                and task.origin_ref.state_transfer_ref == contract.id
                and task.kind is SemanticTaskKind.SEND
                for task in dag.tasks
            )
        )
        send = next(
            task
            for task in source_dag.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
            and task.origin_ref.state_transfer_ref == contract.id
            and task.kind is SemanticTaskKind.SEND
        )
        other_source_staging = canonical_state_staging_value_id(
            other_contract.source_state_access_ref
        )
        assert send.tensor_slice is not None
        tasks = tuple(
            replace(
                task,
                tensor_slice=TensorSlice(
                    other_source_staging,
                    task.tensor_slice.offset,
                    task.tensor_slice.shape,
                ),
                read_values=(other_source_staging,),
            )
            if task.id == send.id
            else task
            for task in source_dag.tasks
        )
        staging_values = tuple(
            replace(
                value,
                consumer_tasks=tuple(
                    task.id for task in tasks if value.id in task.read_values
                ),
            )
            for value in source_dag.state_staging_values
        )
        tampered = _replace_projection_dag(
            projection,
            source_dag,
            tasks=tasks,
            state_staging_values=staging_values,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "state transfer task fields are not contract-exact",
        ):
            tampered.validate_against(graph, fusion, standalone)

    def test_reordered_destination_transfer_region_fails_closed(self) -> None:
        graph, fusion, standalone, contracts, projection = _project(
            (StateKind.KV_KEY,)
        )
        contract = contracts[0]
        destination_dag = next(
            dag
            for dag in projection.dags
            if any(
                region.state_transfer_ref == contract.id
                and len(region.task_ids) == 2
                for region in dag.regions
            )
        )
        regions = tuple(
            replace(region, task_ids=tuple(reversed(region.task_ids)))
            if region.state_transfer_ref == contract.id
            else region
            for region in destination_dag.regions
        )
        tampered = _replace_projection_dag(
            projection, destination_dag, regions=regions
        )
        with self.assertRaisesRegex(
            SchemaError,
            "region task_ids must exactly follow local task order",
        ):
            tampered.validate_against(graph, fusion, standalone)



if __name__ == "__main__":
    unittest.main()
