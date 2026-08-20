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
    StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.n5 import ProjectToIR2Context
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    to_primitive,
)
from llm.frontend.wafer_frontend.schema.state_transfer import (
    StateTransferContract,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind

from test_naive_inter_die import _partitioned_graph


def _transfer_graph() -> tuple[
    IR1,
    dict[tuple[int, StateKind], StateAccess],
    str,
    str,
]:
    source = _partitioned_graph(tp=2)
    manifest = source.persistent_state_manifest
    assert manifest is not None
    declaration_index = {
        declaration.id: declaration for declaration in manifest.declarations
    }
    attention = next(
        node for node in source.nodes if node.kind is OpKind.ATTENTION
    )
    rewritten: list[StateAccess] = []
    for access in source.state_accesses:
        declaration = declaration_index[access.state_ref]
        if access.node_ref != attention.id or declaration.identity.kind not in (
            StateKind.KV_KEY,
            StateKind.KV_VALUE,
        ):
            rewritten.append(access)
            continue
        rewritten.append(
            StateAccess.create(
                node_ref=access.node_ref,
                state_ref=access.state_ref,
                mode=(
                    StateAccessMode.READ
                    if access.rank == 0
                    else StateAccessMode.WRITE
                ),
                rank=access.rank,
            )
        )
    fields = source._semantic_key()
    fields["state_accesses"] = tuple(
        sorted(
            rewritten,
            key=lambda item: (
                item.node_ref,
                item.state_ref,
                item.rank,
                item.id,
            ),
        )
    )
    graph = IR1.create(producer_pass=source.producer_pass, **fields)
    graph.validate()
    accesses = {
        (access.rank, declaration_index[access.state_ref].identity.kind): access
        for access in graph.state_accesses
        if access.node_ref == attention.id
        and declaration_index[access.state_ref].identity.kind
        in (StateKind.KV_KEY, StateKind.KV_VALUE)
    }
    group = next(
        item for item in graph.groups if item.id == attention.execution_group_ref
    )
    forward = next(
        route.id
        for route in group.embedding.routes
        if (route.source_rank, route.destination_rank) == (0, 1)
    )
    reverse = next(
        route.id
        for route in group.embedding.routes
        if (route.source_rank, route.destination_rank) == (1, 0)
    )
    return graph, accesses, forward, reverse


def _contract(
    graph: IR1,
    accesses: dict[tuple[int, StateKind], StateAccess],
    route_ref: str,
    *,
    source_kind: StateKind = StateKind.KV_KEY,
    destination_kind: StateKind = StateKind.KV_KEY,
) -> StateTransferContract:
    return StateTransferContract.create(
        producer_pass="unit",
        source_ir1_id=graph.id,
        source_state_access_ref=accesses[(0, source_kind)].id,
        destination_state_access_ref=accesses[(1, destination_kind)].id,
        pair_route_ref=route_ref,
    )


class StateTransferContractTest(unittest.TestCase):
    def test_exact_contract_round_trips_and_validates_against_ir1(self) -> None:
        graph, accesses, forward, _reverse = _transfer_graph()
        contract = _contract(graph, accesses, forward)
        contract.validate_against(graph)
        decoded = from_data(
            StateTransferContract,
            to_primitive(contract),
            path="state_transfer_contract",
        )
        self.assertEqual(decoded, contract)
        self.assertEqual(canonical_json(decoded), canonical_json(contract))

    def test_strict_serde_rejects_missing_unknown_and_stale_identity(self) -> None:
        graph, accesses, forward, _reverse = _transfer_graph()
        contract = _contract(graph, accesses, forward)
        primitive = to_primitive(contract)
        assert isinstance(primitive, dict)
        for mutation in ("missing", "unknown", "stale_version", "stale_id"):
            with self.subTest(mutation=mutation), self.assertRaises(SchemaError):
                payload = dict(primitive)
                if mutation == "missing":
                    del payload["pair_route_ref"]
                elif mutation == "unknown":
                    payload["extra"] = True
                elif mutation == "stale_version":
                    payload["schema_version"] = "state_transfer/v0"
                else:
                    payload["id"] = "state_transfer_contract_stale"
                from_data(
                    StateTransferContract,
                    payload,
                    path="state_transfer_contract",
                )

    def test_unknown_access_same_state_lineage_and_route_fail_closed(self) -> None:
        graph, accesses, forward, reverse = _transfer_graph()
        valid = _contract(graph, accesses, forward)
        cases = (
            StateTransferContract.create(
                producer_pass="unit",
                source_ir1_id=graph.id,
                source_state_access_ref="missing.access",
                destination_state_access_ref=valid.destination_state_access_ref,
                pair_route_ref=forward,
            ),
            StateTransferContract.create(
                producer_pass="unit",
                source_ir1_id=graph.id,
                source_state_access_ref=valid.source_state_access_ref,
                destination_state_access_ref=valid.source_state_access_ref,
                pair_route_ref=forward,
            ),
            _contract(
                graph,
                accesses,
                forward,
                destination_kind=StateKind.KV_VALUE,
            ),
            _contract(graph, accesses, reverse),
            replace(valid, source_ir1_id="different.ir1"),
        )
        for contract in cases:
            with self.subTest(contract=contract), self.assertRaises(SchemaError):
                contract.validate_against(graph)

    def test_projection_context_is_canonical_strict_and_round_trips(self) -> None:
        graph, accesses, forward, _reverse = _transfer_graph()
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
                    for kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
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
        context = ProjectToIR2Context.create(
            producer_pass="unit",
            state_transfers=contracts,
        )
        context.validate()
        decoded = from_data(
            ProjectToIR2Context,
            to_primitive(context),
            path="project_to_ir2_context",
        )
        self.assertEqual(decoded, context)
        self.assertEqual(canonical_json(decoded), canonical_json(context))

        for state_transfers in (tuple(reversed(contracts)), (contracts[0],) * 2):
            with self.subTest(state_transfers=state_transfers), self.assertRaises(
                SchemaError
            ):
                ProjectToIR2Context.create(
                    producer_pass="unit",
                    state_transfers=state_transfers,
                ).validate()

    def test_nonempty_projection_is_published_after_exact_validation(self) -> None:
        graph, accesses, forward, _reverse = _transfer_graph()
        contract = _contract(graph, accesses, forward)
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
            graph,
            fusion_plans,
            standalone_plans,
            state_transfers=(contract,),
        )
        projection.validate_against(graph, fusion_plans, standalone_plans)
        self.assertEqual(projection.state_transfers, (contract,))


if __name__ == "__main__":
    unittest.main()
