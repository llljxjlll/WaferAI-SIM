from __future__ import annotations

import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.ir0 import (
    InstanceProfileBinding,
    LogicalRole,
)
from llm.frontend.wafer_frontend.schema.ir1 import (
    CrossGroupRoute,
    GroupEmbedding,
    IR1,
    KvRoute,
    PhysicalGroup,
    PhysicalInstance,
    RankPlacement,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)

from _fixtures import valid_ir1


def _rebuild(source: IR1, **updates: object) -> IR1:
    fields = {
        "producer_pass": source.producer_pass,
        "source_ir0_id": source.source_ir0_id,
        "profile": source.profile,
        "fabric": source.fabric,
        "instances": source.instances,
        "groups": source.groups,
        "nodes": source.nodes,
        "values": source.values,
        "edges": source.edges,
        "fusion_candidates": source.fusion_candidates,
        "fused_op_skeletons": source.fused_op_skeletons,
        "cross_routes": source.cross_routes,
        "state_accesses": source.state_accesses,
        "persistent_state_manifest": source.persistent_state_manifest,
        "instance_profiles": source.instance_profiles,
        "pd_plan_id": source.pd_plan_id,
    }
    fields.update(updates)
    return IR1.create(**fields)  # type: ignore[arg-type]


def _two_group_ir1() -> tuple[IR1, CrossGroupRoute, CrossGroupRoute]:
    base = valid_ir1()
    source_group = base.groups[0]
    route_0_1, route_1_0 = source_group.embedding.routes
    destination_group = PhysicalGroup(
        id="group_decode_tp",
        instance_id="D0",
        mesh_ref="D0.mesh.tp",
        axis=source_group.axis,
        logical_shape=(2,),
        placements=(
            RankPlacement(rank=0, die_id=1, logical_coord=(0,)),
            RankPlacement(rank=1, die_id=0, logical_coord=(1,)),
        ),
        embedding=GroupEmbedding((), (), ()),
    )
    source_instance = replace(
        base.instances[0], group_ids=(source_group.id,)
    )
    destination_instance = PhysicalInstance(
        id="D0",
        origin_instance_id="D0",
        role=LogicalRole.DECODE,
        die_region=(1, 0),
        group_ids=(destination_group.id,),
        node_ids=(),
    )
    forward = CrossGroupRoute.create(
        source_group_ref=source_group.id,
        source_rank=0,
        destination_group_ref=destination_group.id,
        destination_rank=0,
        die_path=route_0_1.die_path,
        hops=route_0_1.hops,
        resource_ids=route_0_1.resource_ids,
    )
    reverse = CrossGroupRoute.create(
        source_group_ref=source_group.id,
        source_rank=1,
        destination_group_ref=destination_group.id,
        destination_rank=1,
        die_path=route_1_0.die_path,
        hops=route_1_0.hops,
        resource_ids=route_1_0.resource_ids,
    )
    bindings = tuple(
        sorted(
            (
                InstanceProfileBinding("D0", base.profile),
                InstanceProfileBinding(source_instance.id, base.profile),
            ),
            key=lambda item: (
                item.instance_ref,
                item.profile.stable_id(),
            ),
        )
    )
    result = _rebuild(
        base,
        instances=(source_instance, destination_instance),
        groups=(source_group, destination_group),
        cross_routes=(forward, reverse),
        instance_profiles=bindings,
        pd_plan_id="stage4_plan",
    )
    result.validate("stage4_ir1")
    return result, forward, reverse


class Stage4CrossGroupRouteTest(unittest.TestCase):
    def test_equal_local_rank_route_is_exact_and_round_trips(self) -> None:
        graph, forward, _reverse = _two_group_ir1()
        self.assertEqual((forward.source_rank, forward.destination_rank), (0, 0))
        self.assertNotEqual(
            forward.source_group_ref, forward.destination_group_ref
        )
        self.assertEqual(forward.die_path, (0, 1))
        decoded = loads_dataclass(IR1, canonical_json(graph), path="ir1")
        self.assertEqual(decoded, graph)
        decoded.validate("ir1")

    def test_same_group_and_legacy_kv_route_fail_closed(self) -> None:
        graph, forward, _reverse = _two_group_ir1()
        pair = graph.groups[0].embedding.routes[0]
        with self.assertRaisesRegex(SchemaError, "distinct groups"):
            CrossGroupRoute.create(
                source_group_ref=graph.groups[0].id,
                source_rank=0,
                destination_group_ref=graph.groups[0].id,
                destination_rank=1,
                die_path=pair.die_path,
                hops=pair.hops,
                resource_ids=pair.resource_ids,
            )
        legacy = KvRoute(
            id="legacy",
            source_instance_id=graph.instances[0].id,
            destination_instance_id=graph.instances[1].id,
            value_id=graph.values[0].id,
            bytes=16,
            die_path=(0, 1),
        )
        with self.assertRaisesRegex(SchemaError, "CrossGroupRoute"):
            _rebuild(graph, cross_routes=(legacy,)).validate()  # type: ignore[arg-type]
        self.assertIs(type(forward), CrossGroupRoute)

    def test_rank_path_hop_resource_and_order_are_exact(self) -> None:
        graph, forward, reverse = _two_group_ir1()
        bad_rank = CrossGroupRoute.create(
            source_group_ref=forward.source_group_ref,
            source_rank=forward.source_rank,
            destination_group_ref=forward.destination_group_ref,
            destination_rank=2,
            die_path=forward.die_path,
            hops=forward.hops,
            resource_ids=forward.resource_ids,
        )
        with self.assertRaisesRegex(SchemaError, "group-local rank"):
            _rebuild(graph, cross_routes=(bad_rank,)).validate()

        wrong_path = CrossGroupRoute.create(
            source_group_ref=forward.source_group_ref,
            source_rank=forward.source_rank,
            destination_group_ref=forward.destination_group_ref,
            destination_rank=forward.destination_rank,
            die_path=reverse.die_path,
            hops=reverse.hops,
            resource_ids=reverse.resource_ids,
        )
        with self.assertRaisesRegex(SchemaError, "rank placement"):
            _rebuild(graph, cross_routes=(wrong_path,)).validate()

        bad_hop = CrossGroupRoute.create(
            source_group_ref=forward.source_group_ref,
            source_rank=forward.source_rank,
            destination_group_ref=forward.destination_group_ref,
            destination_rank=forward.destination_rank,
            die_path=forward.die_path,
            hops=(replace(forward.hops[0], link_ref="missing"),),
            resource_ids=forward.resource_ids,
        )
        with self.assertRaisesRegex(SchemaError, "unknown directed link"):
            _rebuild(graph, cross_routes=(bad_hop,)).validate()

        forged_hop = replace(forward.hops[0], resource_ids=("forged",))
        bad_resource = CrossGroupRoute.create(
            source_group_ref=forward.source_group_ref,
            source_rank=forward.source_rank,
            destination_group_ref=forward.destination_group_ref,
            destination_rank=forward.destination_rank,
            die_path=forward.die_path,
            hops=(forged_hop,),
            resource_ids=("forged",),
        )
        with self.assertRaisesRegex(SchemaError, "source egress port"):
            _rebuild(graph, cross_routes=(bad_resource,)).validate()

        with self.assertRaisesRegex(SchemaError, "canonical endpoint/id order"):
            _rebuild(graph, cross_routes=(reverse, forward)).validate()

    def test_route_ids_versions_and_profile_provenance_fail_closed(self) -> None:
        graph, forward, _reverse = _two_group_ir1()
        colliding = replace(
            forward, id=graph.groups[0].embedding.routes[0].id
        )
        with self.assertRaisesRegex(SchemaError, "globally unique"):
            _rebuild(graph, cross_routes=(colliding,)).validate()
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                graph, schema_version="wafer_frontend.ir1/v1alpha11"
            ).validate()
        with self.assertRaisesRegex(SchemaError, "requires instance_profiles"):
            _rebuild(
                graph,
                instance_profiles=(),
                pd_plan_id=None,
            ).validate()
        with self.assertRaisesRegex(SchemaError, "required with instance_profiles"):
            _rebuild(graph, pd_plan_id=None).validate()
        with self.assertRaisesRegex(SchemaError, "bind every instance"):
            _rebuild(
                graph,
                instance_profiles=(graph.instance_profiles[0],),
            ).validate()


if __name__ == "__main__":
    unittest.main()
