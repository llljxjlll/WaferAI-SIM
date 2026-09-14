from __future__ import annotations

import dataclasses
import json
import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema import (
    PLACEMENT_CONTEXT_SCHEMA_VERSION,
    ExplicitGroupPlacement,
    PlacementContext,
    PlacementSpec,
    PlacementStrategy,
    TrafficTemplate,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.placement import (
    PersistentStateReservationPolicy,
    PersistentStateSlotReservation,
)

from _fixtures import valid_hbm_address_spaces, valid_ir1


def compact_context() -> PlacementContext:
    fabric = valid_ir1().fabric
    result = PlacementContext.create(
        producer_pass="load_fabric",
        fabric=fabric,
        placement=PlacementSpec(PlacementStrategy.COMPACT, ()),
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    result.validate()
    return result


class PlacementContextSchemaTest(unittest.TestCase):
    def test_compact_context_is_frozen_stable_and_round_trips(self) -> None:
        context = compact_context()
        self.assertEqual(
            context.schema_version,
            "wafer_frontend.placement_context/v1alpha3",
        )
        self.assertEqual(context.schema_version, PLACEMENT_CONTEXT_SCHEMA_VERSION)
        self.assertEqual(
            context.traffic_templates,
            (TrafficTemplate.DIRECT_A2A_UNIT_CHUNK_V1,),
        )
        decoded = loads_dataclass(
            PlacementContext,
            canonical_json(context),
            path="placement_context",
        )
        self.assertEqual(decoded, context)
        self.assertEqual(canonical_digest(decoded), canonical_digest(context))
        self.assertEqual(compact_context().id, context.id)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decoded.id = "changed"  # type: ignore[misc]

    def test_explicit_context_embeds_exact_placement_and_checks_dies(self) -> None:
        placement = PlacementSpec(
            PlacementStrategy.EXPLICIT,
            (ExplicitGroupPlacement("P0", "mesh_tp", (0, 1)),),
        )
        context = PlacementContext.create(
            producer_pass="load_fabric",
            fabric=valid_ir1().fabric,
            placement=placement,
        )
        context.validate()
        self.assertEqual(context.placement, placement)

        bad_placement = PlacementSpec(
            PlacementStrategy.EXPLICIT,
            (ExplicitGroupPlacement("P0", "mesh_tp", (0, 2)),),
        )
        with self.assertRaisesRegex(SchemaError, "outside the physical fabric"):
            PlacementContext.create(
                producer_pass="load_fabric",
                fabric=context.fabric,
                placement=bad_placement,
            ).validate()

    def test_missing_duplicate_and_wrong_traffic_template_fail(self) -> None:
        context = compact_context()
        for templates, message in (
            ((), "exactly"),
            (
                (
                    TrafficTemplate.DIRECT_A2A_UNIT_CHUNK_V1,
                    TrafficTemplate.DIRECT_A2A_UNIT_CHUNK_V1,
                ),
                "exactly",
            ),
            (("direct_a2a_unit_chunk/v1",), "TrafficTemplate"),
        ):
            with self.subTest(templates=templates), self.assertRaisesRegex(
                SchemaError, message
            ):
                replace(context, traffic_templates=templates).validate()  # type: ignore[arg-type]

        raw = json.loads(canonical_json(context))
        raw["traffic_templates"] = ["ring/v1"]
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            from_data(PlacementContext, raw, path="placement_context")

    def test_semantic_mutations_change_id_and_stale_ids_fail(self) -> None:
        context = compact_context()
        explicit = PlacementSpec(
            PlacementStrategy.EXPLICIT,
            (ExplicitGroupPlacement("P0", "mesh_tp", (1, 0)),),
        )
        changed = PlacementContext.create(
            producer_pass=context.producer_pass,
            fabric=context.fabric,
            placement=explicit,
        )
        changed.validate()
        self.assertNotEqual(changed.id, context.id)
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(context, placement=explicit).validate()

        # Producer metadata does not alter the semantic identity, matching the
        # other versioned frontend artifacts.
        same_semantics = PlacementContext.create(
            producer_pass="unit_fixture",
            fabric=context.fabric,
            placement=context.placement,
            hbm_address_spaces=context.hbm_address_spaces,
        )
        self.assertEqual(same_semantics.id, context.id)

    def test_unknown_fields_and_invalid_producer_fail_closed(self) -> None:
        context = compact_context()
        raw = json.loads(canonical_json(context))
        raw["group_registry"] = []
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            from_data(PlacementContext, raw, path="placement_context")
        with self.assertRaisesRegex(SchemaError, "producer_pass"):
            replace(context, producer_pass="").validate()

    def test_persistent_state_reservation_policy_round_trip_and_validation(self) -> None:
        slots = tuple(
            PersistentStateSlotReservation.create(
                producer_pass="sequence_test",
                kind=kind,
                slot_bytes=192,
            )
            for kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        policy = PersistentStateReservationPolicy.create(
            producer_pass="sequence_test",
            slots=slots,
        )
        context = PlacementContext.create(
            producer_pass="sequence_test",
            fabric=valid_ir1().fabric,
            placement=PlacementSpec(PlacementStrategy.COMPACT, ()),
            hbm_address_spaces=valid_hbm_address_spaces(valid_ir1().fabric),
            persistent_state_reservation_policy=policy,
        )
        decoded = loads_dataclass(
            PlacementContext,
            canonical_json(context),
            path="placement_context",
        )
        self.assertEqual(decoded, context)
        self.assertEqual(decoded.persistent_state_reservation_policy, policy)
        self.assertNotEqual(decoded.id, compact_context().id)

        with self.assertRaisesRegex(SchemaError, "positive multiple of 64"):
            PersistentStateSlotReservation.create(
                producer_pass="sequence_test",
                kind=StateKind.KV_KEY,
                slot_bytes=160,
            )
        with self.assertRaisesRegex(SchemaError, "duplicate state kind"):
            PersistentStateReservationPolicy.create(
                producer_pass="sequence_test",
                slots=(slots[0], slots[0]),
            )


if __name__ == "__main__":
    unittest.main()
