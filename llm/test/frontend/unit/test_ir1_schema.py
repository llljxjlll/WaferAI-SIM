from __future__ import annotations

import dataclasses
import json
import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import UINT64_MAX
from llm.frontend.wafer_frontend.schema.ir1 import (
    C2CPort,
    Direction,
    IR1,
    IR1_SCHEMA_VERSION,
    RouteHop,
    RoutingMode,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, from_data, loads_dataclass

from _fixtures import valid_ir1


def rebuild(ir1: IR1, **updates: object) -> IR1:
    fields = {
        "producer_pass": ir1.producer_pass,
        "source_ir0_id": ir1.source_ir0_id,
        "profile": ir1.profile,
        "fabric": ir1.fabric,
        "instances": ir1.instances,
        "groups": ir1.groups,
        "nodes": ir1.nodes,
        "values": ir1.values,
        "edges": ir1.edges,
        "fusion_candidates": ir1.fusion_candidates,
        "fused_op_skeletons": ir1.fused_op_skeletons,
        "cross_routes": ir1.cross_routes,
    }
    fields.update(updates)
    return IR1.create(**fields)  # type: ignore[arg-type]


class IR1SchemaTest(unittest.TestCase):
    def test_canonical_round_trip_is_self_contained(self) -> None:
        ir1 = valid_ir1()
        self.assertEqual(IR1_SCHEMA_VERSION, "wafer_frontend.ir1/v1alpha14")
        decoded = loads_dataclass(IR1, canonical_json(ir1), path="ir1")
        self.assertEqual(decoded, ir1)
        self.assertIsInstance(decoded.values, tuple)
        self.assertEqual(decoded.values[1].producer, "p_gemm_0")
        self.assertEqual(decoded.fusion_candidates, ir1.fusion_candidates)
        self.assertFalse(hasattr(decoded, "ir0"))
        self.assertIsInstance(decoded.source_ir0_id, str)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decoded.values = ()  # type: ignore[misc]

    def test_value_table_missing_consumer_and_wrong_producer_fail(self) -> None:
        ir1 = valid_ir1()
        missing_consumer = replace(ir1.values[1], consumers=())
        with self.assertRaisesRegex(SchemaError, "missing this node"):
            rebuild(
                ir1,
                values=(ir1.values[0], missing_consumer, ir1.values[2]),
            ).validate()
        wrong_producer = replace(ir1.values[1], producer="p_rs_0")
        with self.assertRaises(SchemaError):
            rebuild(
                ir1,
                values=(ir1.values[0], wrong_producer, ir1.values[2]),
            ).validate()

    def test_duplicate_ids_and_dangling_instance_refs_fail(self) -> None:
        ir1 = valid_ir1()
        duplicate = replace(ir1.nodes[1], id=ir1.nodes[0].id)
        with self.assertRaisesRegex(SchemaError, "duplicate id"):
            rebuild(ir1, nodes=(ir1.nodes[0], duplicate)).validate()
        instance = replace(ir1.instances[0], group_ids=("missing_group",))
        with self.assertRaisesRegex(SchemaError, "dangling group"):
            rebuild(ir1, instances=(instance,)).validate()

    def test_rank_placement_and_route_refs_are_checked(self) -> None:
        ir1 = valid_ir1()
        placements = (
            ir1.groups[0].placements[0],
            replace(ir1.groups[0].placements[1], die_id=9),
        )
        group = replace(ir1.groups[0], placements=placements)
        with self.assertRaises(SchemaError):
            rebuild(ir1, groups=(group,)).validate()
        embedding = replace(
            ir1.groups[0].embedding,
            resource_capacities=tuple(
                capacity
                for capacity in ir1.groups[0].embedding.resource_capacities
                if capacity.id != "d2d_0_1"
            ),
        )
        group = replace(ir1.groups[0], embedding=embedding)
        with self.assertRaisesRegex(SchemaError, "unknown resource"):
            rebuild(ir1, groups=(group,)).validate()

    def test_execution_group_is_explicit_and_matches_instance_mesh(self) -> None:
        ir1 = valid_ir1()
        self.assertEqual(ir1.nodes[0].execution_group_ref, "group_tp")
        with self.assertRaisesRegex(SchemaError, "dangling execution group"):
            rebuild(
                ir1,
                nodes=(
                    replace(ir1.nodes[0], execution_group_ref="missing"),
                    ir1.nodes[1],
                ),
            ).validate()

        other_embedding = replace(
            ir1.groups[0].embedding,
            routes=tuple(
                replace(route, id=f"{route.id}_other")
                for route in ir1.groups[0].embedding.routes
            ),
        )
        other_group = replace(
            ir1.groups[0],
            id="group_other",
            mesh_ref="mesh_other",
            embedding=other_embedding,
        )
        instance = replace(
            ir1.instances[0],
            group_ids=(ir1.groups[0].id, other_group.id),
        )
        mismatched_node = replace(
            ir1.nodes[0], execution_group_ref=other_group.id
        )
        with self.assertRaisesRegex(SchemaError, "must match node instance and mesh"):
            rebuild(
                ir1,
                instances=(instance,),
                groups=(ir1.groups[0], other_group),
                nodes=(mismatched_node, ir1.nodes[1]),
            ).validate()

    def test_missing_values_unknown_fields_enum_and_uint64_fail_decode(self) -> None:
        raw = json.loads(canonical_json(valid_ir1()))
        del raw["values"]
        with self.assertRaisesRegex(SchemaError, "values"):
            from_data(IR1, raw, path="ir1")
        raw = json.loads(canonical_json(valid_ir1()))
        raw["global_value_table"] = []
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            from_data(IR1, raw, path="ir1")
        raw = json.loads(canonical_json(valid_ir1()))
        raw["fabric"]["dies"][0]["ports"][0]["direction"] = "EAST"
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            from_data(IR1, raw, path="ir1")
        raw = json.loads(canonical_json(valid_ir1()))
        port = raw["fabric"]["dies"][0]["ports"][0]
        port["resource_id"] = port.pop("egress_resource_id")
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            from_data(IR1, raw, path="ir1")
        raw = json.loads(canonical_json(valid_ir1()))
        raw["fabric"]["dies"][0]["ports"][0]["link_group_ref"] = "cut_0_1"
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            from_data(IR1, raw, path="ir1")
        raw = json.loads(canonical_json(valid_ir1()))
        raw["fabric"]["sram_profiles"][0]["capacity_bytes"] = UINT64_MAX + 1
        with self.assertRaisesRegex(SchemaError, "unsigned 64-bit"):
            from_data(IR1, raw, path="ir1")
        raw = json.loads(canonical_json(valid_ir1()))
        del raw["nodes"][0]["execution_group_ref"]
        with self.assertRaisesRegex(SchemaError, "execution_group_ref"):
            from_data(IR1, raw, path="ir1")

    def test_stable_id_detects_unversioned_semantic_change(self) -> None:
        raw = json.loads(canonical_json(valid_ir1()))
        raw["source_ir0_id"] = "different_ir0"
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            from_data(IR1, raw, path="ir1")

    def test_fusion_candidates_are_required_and_content_addressed(self) -> None:
        ir1 = valid_ir1()
        raw = json.loads(canonical_json(ir1))
        del raw["fusion_candidates"]
        with self.assertRaisesRegex(SchemaError, "fusion_candidates"):
            from_data(IR1, raw, path="ir1")

        raw = json.loads(canonical_json(ir1))
        raw["fusion_candidates"][0]["origin"] = "discovered"
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            from_data(IR1, raw, path="ir1")

    def test_fusion_candidate_refs_and_uniqueness_are_checked(self) -> None:
        ir1 = valid_ir1()
        candidate = ir1.fusion_candidates[0]
        self.assertEqual(candidate.impl.value, "none")

        with self.assertRaisesRegex(SchemaError, "dangling member"):
            rebuild(
                ir1,
                fusion_candidates=(replace(candidate, members=("missing",)),),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "dangling boundary input"):
            rebuild(
                ir1,
                fusion_candidates=(
                    replace(candidate, boundary_inputs=("missing",)),
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "dangling boundary output"):
            rebuild(
                ir1,
                fusion_candidates=(
                    replace(candidate, boundary_outputs=("missing",)),
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "duplicate id"):
            rebuild(ir1, fusion_candidates=(candidate, candidate)).validate()

    def test_fused_skeleton_is_semantically_self_contained(self) -> None:
        ir1 = valid_ir1()
        skeleton = ir1.fused_op_skeletons[0]
        self.assertEqual(skeleton.semantic_contract.output_layout, "MN_shard_tp")
        raw = json.loads(canonical_json(ir1))
        del raw["fused_op_skeletons"][0]["semantic_contract"]
        with self.assertRaisesRegex(SchemaError, "semantic_contract"):
            from_data(IR1, raw, path="ir1")
        invalid_contract = replace(skeleton.semantic_contract, output_layout="")
        with self.assertRaisesRegex(SchemaError, "output_layout"):
            rebuild(
                ir1,
                fused_op_skeletons=(replace(skeleton, semantic_contract=invalid_contract),),
            ).validate()

    def test_fabric_contract_is_explicit_and_backend_addressable(self) -> None:
        ir1 = valid_ir1()
        fabric = ir1.fabric
        self.assertEqual(fabric.routing_mode, RoutingMode.BACKEND_XY_V1)
        self.assertEqual(fabric.dies[1].cores[0].runtime_core_id, 16)
        self.assertEqual(fabric.dies[1].cores[-1].runtime_core_id, 31)
        self.assertEqual(fabric.dies[0].noc_bytes_per_cycle, 128)
        profile = fabric.sram_profiles[0]
        self.assertTrue(profile.real_data_path)
        self.assertTrue(profile.manual_regions)
        self.assertTrue(profile.manual_memory_schedule)
        self.assertEqual(profile.regions[0].name, "sram")
        route = ir1.groups[0].embedding.routes[0]
        self.assertEqual(route.hops[0].link_ref, "link_0_1")
        self.assertEqual(
            route.hops[0].resource_ids,
            ("port_0_east", "d2d_0_1", "cut_0_1"),
        )
        self.assertEqual(fabric.dies[0].ports[0].egress_resource_id, "port_0_east")

    def test_core_and_die_backend_numbering_fail_closed(self) -> None:
        fabric = valid_ir1().fabric
        die = fabric.dies[1]
        wrong_runtime = replace(die.cores[0], runtime_core_id=63)
        with self.assertRaisesRegex(SchemaError, "runtime_core_id must be 16"):
            replace(fabric, dies=(fabric.dies[0], replace(die, cores=(wrong_runtime,) + die.cores[1:]))).validate("fabric")

        overflow = replace(die.cores[0], runtime_core_id=1 << 16)
        with self.assertRaisesRegex(SchemaError, "uint16"):
            replace(fabric, dies=(fabric.dies[0], replace(die, cores=(overflow,) + die.cores[1:]))).validate("fabric")

        wrong_local = replace(fabric.dies[0].cores[0], local_core_id=16)
        with self.assertRaisesRegex(SchemaError, "row-major local_core_id"):
            replace(
                fabric,
                dies=(replace(fabric.dies[0], cores=(wrong_local,) + fabric.dies[0].cores[1:]), fabric.dies[1]),
            ).validate("fabric")

        wrong_die_id = replace(fabric.dies[1], id=2)
        with self.assertRaisesRegex(SchemaError, "row-major die id"):
            replace(fabric, dies=(fabric.dies[0], wrong_die_id)).validate("fabric")

        nonuniform_cores = tuple(
            replace(core, noc_coord=(core.local_core_id % 8, core.local_core_id // 8))
            for core in die.cores
        )
        with self.assertRaisesRegex(SchemaError, "uniform noc_grid"):
            replace(
                fabric,
                dies=(fabric.dies[0], replace(die, noc_grid=(8, 2), cores=nonuniform_cores)),
            ).validate("fabric")

    def test_sram_profile_alignment_capacity_access_and_capabilities(self) -> None:
        fabric = valid_ir1().fabric
        profile = fabric.sram_profiles[0]
        region = profile.regions[0]
        invalid_profiles = (
            (replace(profile, real_data_path=False), "real_data_path"),
            (replace(profile, manual_regions=False), "manual_regions"),
            (replace(profile, manual_memory_schedule=False), "manual_memory_schedule"),
            (replace(profile, regions=(replace(region, base_bytes=1),)), "alignment"),
            (replace(profile, regions=(replace(region, size_bytes=profile.capacity_bytes + 64),)), "capacity"),
            (replace(profile, regions=(replace(region, access=()),)), "initiator"),
            (replace(profile, regions=(replace(region, name="x" * 65),)), "64 bytes"),
            (
                replace(
                    profile,
                    regions=(
                        replace(region, size_bytes=128),
                        replace(region, id="overlap", name="overlap", base_bytes=64, size_bytes=64),
                    ),
                ),
                "overlaps",
            ),
        )
        for invalid_profile, message in invalid_profiles:
            with self.subTest(message=message), self.assertRaisesRegex(SchemaError, message):
                replace(fabric, sram_profiles=(invalid_profile,)).validate("fabric")

    def test_port_and_link_constraints_fail_closed(self) -> None:
        fabric = valid_ir1().fabric
        die0 = fabric.dies[0]
        duplicate_direction = C2CPort(
            id="east_0_second",
            runtime_port_id=1,
            side=Direction.EAST,
            direction=Direction.EAST,
            noc_coord=(3, 2),
            egress_resource_id="port_0_east_second",
            bytes_per_cycle=64,
            buffer_packets=8,
        )
        with self.assertRaisesRegex(SchemaError, "one C2C port per direction"):
            replace(
                fabric,
                dies=(replace(die0, ports=die0.ports + (duplicate_direction,)), fabric.dies[1]),
            ).validate("fabric")

        off_edge = replace(die0.ports[0], noc_coord=(2, 1))
        with self.assertRaisesRegex(SchemaError, "declared side"):
            replace(
                fabric,
                dies=(replace(die0, ports=(off_edge,)), fabric.dies[1]),
            ).validate("fabric")

        independent_link_bandwidth = replace(
            fabric,
            links=tuple(replace(link, bytes_per_cycle=32) for link in fabric.links),
        )
        independent_link_bandwidth.validate("fabric")
        self.assertEqual(independent_link_bandwidth.dies[0].ports[0].bytes_per_cycle, 64)
        self.assertEqual(independent_link_bandwidth.links[0].bytes_per_cycle, 32)

        with self.assertRaisesRegex(SchemaError, "missing reciprocal"):
            replace(fabric, links=(fabric.links[0],)).validate("fabric")

        directional_group = replace(fabric.links[0], link_group_ref="other_cut")
        asymmetric_groups = replace(
            fabric, links=(directional_group, fabric.links[1])
        )
        asymmetric_groups.validate("fabric")
        self.assertNotEqual(
            asymmetric_groups.links[0].link_group_ref,
            asymmetric_groups.links[1].link_group_ref,
        )

        aliased_cut = replace(
            fabric.links[0], link_group_ref=fabric.links[0].resource_id
        )
        with self.assertRaisesRegex(SchemaError, "distinct directed cut"):
            replace(fabric, links=(aliased_cut, fabric.links[1])).validate("fabric")

    def test_route_is_exact_xy_and_hop_resources_are_lossless(self) -> None:
        ir1 = valid_ir1()
        forward, reverse = ir1.groups[0].embedding.routes
        missing_cut_hop = replace(
            forward.hops[0],
            resource_ids=("port_0_east", "d2d_0_1"),
        )
        missing_cut_route = replace(
            forward,
            hops=(missing_cut_hop,),
            resource_ids=missing_cut_hop.resource_ids,
        )
        embedding = replace(ir1.groups[0].embedding, routes=(missing_cut_route, reverse))
        with self.assertRaisesRegex(SchemaError, "directed shared-cut resource"):
            rebuild(ir1, groups=(replace(ir1.groups[0], embedding=embedding),)).validate()

        wrong_link_hop = replace(forward.hops[0], link_ref="link_1_0")
        wrong_link_route = replace(forward, hops=(wrong_link_hop,))
        embedding = replace(ir1.groups[0].embedding, routes=(wrong_link_route, reverse))
        with self.assertRaisesRegex(SchemaError, "disagree with directed link"):
            rebuild(ir1, groups=(replace(ir1.groups[0], embedding=embedding),)).validate()

        detour_hops = (
            forward.hops[0],
            replace(reverse.hops[0], index=1),
            replace(forward.hops[0], index=2),
        )
        detour = replace(
            forward,
            die_path=(0, 1, 0, 1),
            hops=detour_hops,
            resource_ids=(
                "port_0_east",
                "d2d_0_1",
                "cut_0_1",
                "port_1_west",
                "d2d_1_0",
                "cut_1_0",
            ),
        )
        embedding = replace(ir1.groups[0].embedding, routes=(detour, reverse))
        with self.assertRaisesRegex(SchemaError, "X-then-Y"):
            rebuild(ir1, groups=(replace(ir1.groups[0], embedding=embedding),)).validate()

    def test_pair_route_ids_are_globally_unambiguous(self) -> None:
        ir1 = valid_ir1()
        duplicate_route_group = replace(ir1.groups[0], id="group_duplicate_routes")
        instance = replace(
            ir1.instances[0],
            group_ids=ir1.instances[0].group_ids + (duplicate_route_group.id,),
        )
        with self.assertRaisesRegex(SchemaError, "globally unique"):
            rebuild(
                ir1,
                instances=(instance,),
                groups=ir1.groups + (duplicate_route_group,),
            ).validate()


if __name__ == "__main__":
    unittest.main()
