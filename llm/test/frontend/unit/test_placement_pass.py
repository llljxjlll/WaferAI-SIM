from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_ir0,
    logical_expand,
    place_bundle as public_place_bundle,
    place_ir0 as public_place_ir0,
    validate_placement_against as public_validate_placement_against,
)
from llm.frontend.wafer_frontend.passes.placement import (
    place_bundle,
    place_ir0,
    validate_placement_against,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.persistent_state import (
    HbmAddressSpace,
    HbmBinding,
    PersistentStateManifest,
    StateKind,
)
from llm.frontend.wafer_frontend.schema.placed_ir1 import (
    PlacedIR1Bundle,
    PlacedProfileIR1,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, from_data

from _fixtures import valid_hbm_address_spaces, valid_ir1, valid_spec


def source_and_context():
    spec = from_data(ExperimentSpec, valid_spec(), path="spec")
    source = logical_expand(build_ir0(spec))
    fabric = valid_ir1().fabric
    context = PlacementContext.create(
        producer_pass="test_placement_pass",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    return source, context


def rebuild_graph(graph: IR1, **updates: object) -> IR1:
    fields = {
        "producer_pass": graph.producer_pass,
        "source_ir0_id": graph.source_ir0_id,
        "profile": graph.profile,
        "fabric": graph.fabric,
        "instances": graph.instances,
        "groups": graph.groups,
        "nodes": graph.nodes,
        "values": graph.values,
        "edges": graph.edges,
        "fusion_candidates": graph.fusion_candidates,
        "fused_op_skeletons": graph.fused_op_skeletons,
        "cross_routes": graph.cross_routes,
        "state_accesses": graph.state_accesses,
        "persistent_state_manifest": graph.persistent_state_manifest,
    }
    fields.update(updates)
    result = IR1.create(**fields)  # type: ignore[arg-type]
    result.validate()
    return result


def rebuild_bundle(source, context, original, graph):
    entry = PlacedProfileIR1.create(
        source_expanded_entry_id=original.entries[0].source_expanded_entry_id,
        weight=original.entries[0].weight,
        graph=graph,
    )
    return PlacedIR1Bundle.create(
        source_expanded_bundle=source,
        placement_context=context,
        entries=(entry,),
    )


class PlacementPassTest(unittest.TestCase):
    def test_public_exports_and_exact_source_preservation(self) -> None:
        self.assertIs(public_place_bundle, place_bundle)
        self.assertIs(public_place_ir0, place_ir0)
        self.assertIs(
            public_validate_placement_against,
            validate_placement_against,
        )
        source, context = source_and_context()
        result = place_bundle(source, context)
        validate_placement_against(result, source, context)

        logical = source.entries[0].graph
        physical = result.entries[0].graph
        self.assertEqual(physical.source_ir0_id, logical.id)
        self.assertEqual(physical.fabric, context.fabric)
        self.assertEqual(physical.values, logical.values)
        self.assertEqual(physical.edges, logical.edges)
        self.assertEqual(physical.fusion_candidates, logical.fusion_candidates)
        self.assertEqual(physical.fused_op_skeletons, ())
        self.assertEqual(physical.cross_routes, ())
        self.assertEqual(
            tuple(node.id for node in physical.nodes),
            tuple(node.id for node in logical.nodes),
        )
        self.assertTrue(
            all(
                node.origin_node_id == node.id
                and node.execution_group_ref == physical.groups[0].id
                for node in physical.nodes
            )
        )
        self.assertEqual(
            tuple(item.die_id for item in physical.groups[0].placements),
            (0, 1),
        )
        self.assertEqual(
            physical.groups[0].embedding.canonical_profiles[0].lane_eq_bandwidth,
            64.0,
        )

    def test_persistent_state_is_preserved_and_uses_canonical_first_fit(self) -> None:
        source, context = source_and_context()
        result = place_bundle(source, context)
        logical = source.entries[0].graph
        physical = result.entries[0].graph
        manifest = physical.persistent_state_manifest
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(manifest.declarations, logical.persistent_states)
        self.assertEqual(physical.state_accesses, logical.state_accesses)
        self.assertEqual(manifest.address_spaces, context.hbm_address_spaces)
        self.assertEqual(
            {binding.state_ref for binding in manifest.bindings},
            {declaration.id for declaration in logical.persistent_states},
        )

        group_by_owner = {
            (group.instance_id, group.mesh_ref): group
            for group in physical.groups
        }
        binding_by_state = {
            binding.state_ref: binding for binding in manifest.bindings
        }
        kind_order = {kind: index for index, kind in enumerate(StateKind)}
        golden_offsets = {
            ("parameter", "P0.final_norm.weight"): 0,
            ("parameter", "P0.layer0.w_down"): 512,
            ("parameter", "P0.layer0.w_gate_up"): 131584,
            ("parameter", "P0.layer0.w_norm1"): 393728,
            ("parameter", "P0.layer0.w_norm2"): 394240,
            ("parameter", "P0.layer0.w_o"): 394752,
            ("parameter", "P0.layer0.w_qkv"): 460288,
            ("parameter", "P0.lm_head.weight"): 591360,
            ("parameter", "P0.tok_embeddings.weight"): 853504,
            ("kv_key", None): 1115648,
            ("kv_value", None): 1119744,
        }
        self.assertEqual(
            {
                (
                    declaration.identity.shard_index,
                    declaration.identity.kind.value,
                    declaration.identity.tensor_ref,
                ): (
                    binding_by_state[declaration.id].die_id,
                    binding_by_state[declaration.id].address,
                )
                for declaration in manifest.declarations
            },
            {
                (rank, kind, tensor_ref): (
                    rank,
                    context.hbm_address_spaces[rank].base_address + offset,
                )
                for rank in (0, 1)
                for (kind, tensor_ref), offset in golden_offsets.items()
            },
        )

        def placement_key(declaration):
            identity = declaration.identity
            group = group_by_owner[
                (identity.instance_ref, identity.mesh_ref)
            ]
            home_die = next(
                placement.die_id
                for placement in group.placements
                if placement.rank == identity.shard_index
            )
            return (
                identity.instance_ref,
                identity.mesh_ref,
                identity.shard_index,
                home_die,
                kind_order[identity.kind],
                identity.request_ref or "",
                -1 if identity.layer_index is None else identity.layer_index,
                identity.tensor_ref or "",
                identity.generation,
                declaration.id,
            )

        spaces = {space.die_id: space for space in manifest.address_spaces}
        for die_id, space in spaces.items():
            expected_declarations = tuple(
                sorted(
                    (
                        declaration
                        for declaration in manifest.declarations
                        if binding_by_state[declaration.id].die_id == die_id
                    ),
                    key=placement_key,
                )
            )
            actual_bindings = tuple(
                sorted(
                    (
                        binding
                        for binding in manifest.bindings
                        if binding.die_id == die_id
                    ),
                    key=lambda binding: binding.address,
                )
            )
            self.assertEqual(
                tuple(binding.state_ref for binding in actual_bindings),
                tuple(declaration.id for declaration in expected_declarations),
            )
            cursor = space.base_address
            for binding in actual_bindings:
                expected_address = ((cursor + 63) // 64) * 64
                self.assertEqual(binding.address, expected_address)
                self.assertEqual(binding.address % 64, 0)
                cursor = binding.address + binding.size_bytes

    def test_self_consistent_non_first_fit_binding_fails_recomputation(self) -> None:
        source, context = source_and_context()
        result = place_bundle(source, context)
        graph = result.entries[0].graph
        manifest = graph.persistent_state_manifest
        assert manifest is not None
        target = max(
            (
                binding
                for binding in manifest.bindings
                if binding.die_id == manifest.address_spaces[0].die_id
            ),
            key=lambda binding: binding.address,
        )
        moved = HbmBinding.create(
            state_ref=target.state_ref,
            die_id=target.die_id,
            address=target.address + 64,
            size_bytes=target.size_bytes,
        )
        forged_manifest = PersistentStateManifest.create(
            address_spaces=manifest.address_spaces,
            declarations=manifest.declarations,
            bindings=tuple(
                moved if binding.id == target.id else binding
                for binding in manifest.bindings
            ),
        )
        forged_graph = rebuild_graph(
            graph,
            persistent_state_manifest=forged_manifest,
        )
        forged = rebuild_bundle(source, context, result, forged_graph)
        forged.validate_against(source, context)
        with self.assertRaisesRegex(SchemaError, "canonical 64-byte first-fit"):
            validate_placement_against(forged, source, context)

    def test_binding_on_non_home_die_fails_self_validation(self) -> None:
        source, context = source_and_context()
        result = place_bundle(source, context)
        graph = result.entries[0].graph
        manifest = graph.persistent_state_manifest
        assert manifest is not None
        target = next(
            binding for binding in manifest.bindings if binding.die_id == 0
        )
        wrong_space = next(
            space for space in manifest.address_spaces if space.die_id == 1
        )
        wrong_address = (
            (
                wrong_space.base_address
                + wrong_space.size_bytes
                - target.size_bytes
            )
            // 64
        ) * 64
        wrong_home = HbmBinding.create(
            state_ref=target.state_ref,
            die_id=1,
            address=wrong_address,
            size_bytes=target.size_bytes,
        )
        forged_manifest = PersistentStateManifest.create(
            address_spaces=manifest.address_spaces,
            declarations=manifest.declarations,
            bindings=tuple(
                wrong_home if binding.id == target.id else binding
                for binding in manifest.bindings
            ),
        )
        with self.assertRaisesRegex(SchemaError, "home die"):
            rebuild_graph(
                graph,
                persistent_state_manifest=forged_manifest,
            )

    def test_missing_or_insufficient_hbm_capacity_fails_closed(self) -> None:
        source, context = source_and_context()
        empty_context = PlacementContext.create(
            producer_pass=context.producer_pass,
            fabric=context.fabric,
            placement=context.placement,
        )
        with self.assertRaisesRegex(SchemaError, "explicit HBM address spaces"):
            place_bundle(source, empty_context)

        tiny_spaces = tuple(
            HbmAddressSpace.create(
                die_id=space.die_id,
                base_address=space.base_address,
                size_bytes=64,
                alignment_bytes=64,
            )
            for space in context.hbm_address_spaces
        )
        tiny_context = PlacementContext.create(
            producer_pass=context.producer_pass,
            fabric=context.fabric,
            placement=context.placement,
            hbm_address_spaces=tiny_spaces,
        )
        with self.assertRaisesRegex(SchemaError, "HBM capacity"):
            place_bundle(source, tiny_context)

    def test_repeat_is_stable_and_inputs_are_unchanged(self) -> None:
        source, context = source_and_context()
        source_digest = canonical_digest(source)
        context_digest = canonical_digest(context)
        first = place_bundle(source, context)
        second = place_bundle(source, context)
        self.assertEqual(first, second)
        self.assertEqual(canonical_digest(source), source_digest)
        self.assertEqual(canonical_digest(context), context_digest)

    def test_recomputed_embedding_rejects_self_consistent_false_bandwidth(self) -> None:
        source, context = source_and_context()
        result = place_bundle(source, context)
        graph = result.entries[0].graph
        group = graph.groups[0]
        profile = group.embedding.canonical_profiles[0]
        forged_group = replace(
            group,
            embedding=replace(
                group.embedding,
                canonical_profiles=(
                    replace(profile, lane_eq_bandwidth=profile.lane_eq_bandwidth / 2),
                ),
            ),
        )
        forged_graph = rebuild_graph(graph, groups=(forged_group,))
        forged = rebuild_bundle(source, context, result, forged_graph)
        forged.validate_against(source, context)
        with self.assertRaisesRegex(SchemaError, "canonical profile"):
            validate_placement_against(forged, source, context)

    def test_recomputed_embedding_rejects_missing_ordered_pair_route(self) -> None:
        source, context = source_and_context()
        result = place_bundle(source, context)
        graph = result.entries[0].graph
        group = graph.groups[0]
        forged_group = replace(
            group,
            embedding=replace(
                group.embedding,
                routes=group.embedding.routes[:1],
            ),
        )
        forged_graph = rebuild_graph(graph, groups=(forged_group,))
        forged = rebuild_bundle(source, context, result, forged_graph)
        forged.validate_against(source, context)
        with self.assertRaisesRegex(SchemaError, "every ordered pair route"):
            validate_placement_against(forged, source, context)


if __name__ == "__main__":
    unittest.main()
