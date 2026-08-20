from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.placement import place_stage4_ir0
from llm.frontend.wafer_frontend.passes.stage4_logical_expand import (
    build_stage4_separated_ir0,
)
from llm.frontend.wafer_frontend.passes.stage4_pd import (
    build_stage4_pd_plan,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    StateAccess,
    StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.ir1 import (
    CrossGroupRoute,
    IR1,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    to_primitive,
)
from llm.frontend.wafer_frontend.schema.state_transfer import (
    SlicedKvStateTransferContract,
    StateTransferLike,
)

from _fixtures import valid_hbm_address_spaces, valid_ir1
from test_stage4_pd import _profile, _spec


def _stage4_tp1_ir1() -> tuple[IR1, CrossGroupRoute]:
    spec = _spec(1, 1)
    plan = build_stage4_pd_plan(
        spec,
        prefill_profile=_profile(prefill=True),
        decode_profile=_profile(prefill=False),
    )
    logical = build_stage4_separated_ir0(spec, plan)
    fabric = valid_ir1().fabric
    context = PlacementContext.create(
        producer_pass="test_stage4_sliced_state_transfer",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    graph = place_stage4_ir0(logical, context, plan)
    self_route = graph.cross_routes
    if len(self_route) != 1:
        raise AssertionError("TP1 Stage4 placement must produce one route")
    return graph, self_route[0]


def _access(
    graph: IR1,
    *,
    instance_ref: str,
    layer: int,
    kind: StateKind,
) -> StateAccess:
    assert graph.persistent_state_manifest is not None
    declarations = {
        declaration.id: declaration
        for declaration in graph.persistent_state_manifest.declarations
    }
    return next(
        access
        for access in graph.state_accesses
        if (
            declarations[access.state_ref].identity.instance_ref,
            declarations[access.state_ref].identity.layer_index,
            declarations[access.state_ref].identity.kind,
            access.rank,
        )
        == (instance_ref, layer, kind, 0)
    )


def _contract(
    graph: IR1,
    route: CrossGroupRoute,
    *,
    source_kind: StateKind = StateKind.KV_KEY,
    destination_kind: StateKind = StateKind.KV_KEY,
    source_layer: int = 0,
    destination_layer: int = 0,
    source_access: StateAccess | None = None,
    **updates: object,
) -> SlicedKvStateTransferContract:
    source = source_access or _access(
        graph,
        instance_ref="P0",
        layer=source_layer,
        kind=source_kind,
    )
    destination = _access(
        graph,
        instance_ref="D0",
        layer=destination_layer,
        kind=destination_kind,
    )
    fields: dict[str, object] = {
        "producer_pass": "test_stage4_sliced_state_transfer",
        "source_ir1_id": graph.id,
        "source_state_access_ref": source.id,
        "destination_state_access_ref": destination.id,
        "cross_group_route_ref": route.id,
        "source_local_offset": (0, 0, 0),
        "source_local_shape": (8, 4, 4),
        "destination_local_offset": (0, 0, 0),
        "destination_local_shape": (8, 4, 4),
        "bytes": 256,
    }
    fields.update(updates)
    return SlicedKvStateTransferContract.create(**fields)  # type: ignore[arg-type]


def _replace_access(
    graph: IR1,
    old: StateAccess,
    new: StateAccess,
) -> IR1:
    fields = graph._semantic_key()
    fields["state_accesses"] = tuple(
        sorted(
            tuple(
                new if access.id == old.id else access
                for access in graph.state_accesses
            ),
            key=lambda access: (
                access.node_ref,
                access.state_ref,
                access.rank,
                access.id,
            ),
        )
    )
    result = IR1.create(producer_pass=graph.producer_pass, **fields)
    result.validate("stage4_ir1")
    return result


class Stage4SlicedStateTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph, cls.forward = _stage4_tp1_ir1()

    def test_tp1_k_and_v_are_exact_stable_and_round_trip(self) -> None:
        contracts = tuple(
            _contract(
                self.graph,
                self.forward,
                source_kind=kind,
                destination_kind=kind,
            )
            for kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        for contract in contracts:
            contract.validate_against(self.graph)
            self.assertEqual(
                (
                    contract.source_local_offset,
                    contract.source_local_shape,
                    contract.destination_local_offset,
                    contract.destination_local_shape,
                    contract.bytes,
                ),
                ((0, 0, 0), (8, 4, 4), (0, 0, 0), (8, 4, 4), 256),
            )
            primitive = to_primitive(contract)
            decoded = from_data(
                SlicedKvStateTransferContract,
                primitive,
                path="contract",
            )
            decoded_union = from_data(
                StateTransferLike,
                primitive,
                path="contract",
            )
            self.assertEqual(decoded, contract)
            self.assertEqual(decoded_union, contract)
            self.assertEqual(canonical_json(decoded), canonical_json(contract))
        self.assertNotEqual(contracts[0].id, contracts[1].id)
        self.assertEqual(
            _contract(self.graph, self.forward), contracts[0]
        )

    def test_strict_serde_rejects_missing_unknown_and_stale_identity(self) -> None:
        contract = _contract(self.graph, self.forward)
        primitive = to_primitive(contract)
        assert isinstance(primitive, dict)
        for mutation in ("missing", "unknown", "version", "id"):
            with self.subTest(mutation=mutation), self.assertRaises(SchemaError):
                payload = dict(primitive)
                if mutation == "missing":
                    del payload["source_local_shape"]
                elif mutation == "unknown":
                    payload["extra"] = True
                elif mutation == "version":
                    payload["schema_version"] = (
                        "wafer_frontend.state_transfer_contract/v1alpha1"
                    )
                else:
                    payload["id"] = "stale"
                from_data(
                    SlicedKvStateTransferContract,
                    payload,
                    path="contract",
                )

    def test_mode_kind_and_lineage_fail_closed(self) -> None:
        source = _access(
            self.graph,
            instance_ref="P0",
            layer=0,
            kind=StateKind.KV_KEY,
        )
        read_source = StateAccess.create(
            node_ref=source.node_ref,
            state_ref=source.state_ref,
            mode=StateAccessMode.READ,
            rank=source.rank,
            read_offset=source.write_offset,
            read_shape=source.write_shape,
        )
        read_graph = _replace_access(self.graph, source, read_source)
        cases = (
            (
                _contract(
                    read_graph,
                    self.forward,
                    source_access=read_source,
                ),
                read_graph,
            ),
            (
                _contract(
                    self.graph,
                    self.forward,
                    destination_kind=StateKind.KV_VALUE,
                ),
                self.graph,
            ),
            (
                _contract(
                    self.graph,
                    self.forward,
                    destination_layer=1,
                ),
                self.graph,
            ),
        )
        for contract, graph in cases:
            with self.subTest(contract=contract), self.assertRaises(SchemaError):
                contract.validate_against(graph)

    def test_slice_offset_bytes_and_route_tamper_fail_closed(self) -> None:
        cases = (
            _contract(
                self.graph,
                self.forward,
                source_local_offset=(0, 1, 0),
            ),
            _contract(
                self.graph,
                self.forward,
                source_local_shape=(8, 3, 4),
                destination_local_shape=(8, 3, 4),
            ),
            _contract(
                self.graph,
                self.forward,
                destination_local_offset=(1, 0, 0),
            ),
            _contract(self.graph, self.forward, bytes=255),
            _contract(
                self.graph,
                self.forward,
                cross_group_route_ref="missing.route",
            ),
        )
        for contract in cases:
            with self.subTest(contract=contract), self.assertRaises(SchemaError):
                contract.validate_against(self.graph)


if __name__ == "__main__":
    unittest.main()
