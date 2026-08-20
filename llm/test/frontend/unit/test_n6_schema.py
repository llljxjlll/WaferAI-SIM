from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import (
    LoweringContext,
    NaiveManifestLinker,
    add_fixed_sram_lifecycle,
)
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.logical import ProfileEntry
from llm.frontend.wafer_frontend.schema.n5 import (
    GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
    GlobalActionBundle,
    GlobalActionProfile,
)
from llm.frontend.wafer_frontend.schema.n6 import (
    LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    LinkedProgramBundle,
    LinkedProgramProfile,
    LoweredProgramBundle,
    LoweredProgramProfile,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.ir2 import IntraDieScheduleSet
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)

from test_linked_program_manifest_schema import (
    _recreate_manifest,
    valid_exact_linked_manifest,
)
from test_global_action_schema import _create_global


def _restable_profile(
    entry: LinkedProgramProfile,
    **updates: object,
) -> LinkedProgramProfile:
    changed = replace(entry, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "linked_program_profile",
            changed._semantic_key(),
            schema_version=LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _restable_lowered_profile(
    entry: LoweredProgramProfile,
    **updates: object,
) -> LoweredProgramProfile:
    changed = replace(entry, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "lowered_program_profile",
            changed._semantic_key(),
            schema_version=LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _restable_lowered_bundle(
    bundle: LoweredProgramBundle,
    **updates: object,
) -> LoweredProgramBundle:
    changed = replace(bundle, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "lowered_program_bundle",
            changed._semantic_key(),
            schema_version=LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _restable_bundle(
    bundle: LinkedProgramBundle,
    **updates: object,
) -> LinkedProgramBundle:
    changed = replace(bundle, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "linked_program_bundle",
            changed._semantic_key(),
            schema_version=LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _source_profile(
    context: LoweringContext,
    *,
    suffix: str,
    weight: float,
) -> GlobalActionProfile:
    entry = GlobalActionProfile(
        id="pending",
        source_scheduled_entry_id=f"scheduled_entry_{suffix}",
        source_projected_entry_id=f"projected_entry_{suffix}",
        source_planned_entry_id=f"planned_entry_{suffix}",
        source_partitioned_entry_id=f"partitioned_entry_{suffix}",
        source_ir1_id=context.ir1.id,
        projection_context_id="projection_context_fixture",
        scheduling_context_id="scheduling_context_fixture",
        profile_id=context.ir1.profile.stable_id(),
        weight=weight,
        graph=context.ir1,
        fusion_plans=context.fusion_plans,
        standalone_plans=context.standalone_plans,
        projection=context.projection,
        schedule_set=context.schedule_set,
        global_dag=context.global_dag,
    )
    return replace(
        entry,
        id=stable_artifact_id(
            "global_action_profile",
            entry._semantic_key(),
            schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _source_bundle(
    entries: tuple[GlobalActionProfile, ...],
) -> GlobalActionBundle:
    profiles = tuple(
        ProfileEntry.create(key=entry.graph.profile, weight=entry.weight)
        for entry in entries
    )
    bundle = GlobalActionBundle(
        schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
        producer_pass="global_action_dag",
        id="pending",
        source_scheduled_bundle_id="scheduled_bundle_fixture",
        source_projected_bundle_id="projected_bundle_fixture",
        source_inter_die_bundle_id="planned_bundle_fixture",
        source_partitioned_bundle_id="partitioned_bundle_fixture",
        placement_context_id="placement_context_fixture",
        partition_context_id="partition_context_fixture",
        planning_context_id="planning_context_fixture",
        projection_context_id="projection_context_fixture",
        scheduling_context_id="scheduling_context_fixture",
        source_profiles=profiles,
        entries=entries,
    )
    return replace(
        bundle,
        id=stable_artifact_id(
            "global_action_bundle",
            bundle._semantic_key(),
            schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _single_profile_fixture() -> tuple[
    GlobalActionBundle,
    LoweredProgramBundle,
    LinkedProgramBundle,
]:
    context, manifest = valid_exact_linked_manifest()
    ir1 = replace(context.ir1, producer_pass="fusion_partition")
    projection = replace(context.projection, producer_pass="project_to_ir2")
    schedules = tuple(
        replace(schedule, producer_pass="intra_die_schedule")
        for schedule in context.schedule_set.schedules
    )
    schedule_set = IntraDieScheduleSet.create(
        producer_pass="intra_die_schedule",
        source_projection_id=projection.id,
        source_ir1_id=ir1.id,
        schedules=schedules,
    )
    global_dag = _create_global(ir1, projection, schedule_set)
    global_dag = replace(global_dag, producer_pass="global_action_dag")

    fragment_map: dict[str, CommandFragment] = {}
    for fragment in manifest.fragments:
        if type(fragment) is not CommandFragment:
            raise AssertionError("exact N6 fixture requires leaf command fragments")
        fragment_map[fragment.id] = CommandFragment.create(
            producer_pass=fragment.producer_pass,
            **{
                **fragment._semantic_key(),
                "source_global_dag_id": global_dag.id,
            },
        )
    fragments = tuple(
        sorted(fragment_map.values(), key=lambda fragment: fragment.id)
    )
    context = LoweringContext(
        ir1=ir1,
        fusion_plans=context.fusion_plans,
        standalone_plans=context.standalone_plans,
        projection=projection,
        schedule_set=schedule_set,
        global_dag=global_dag,
    )
    fragments = tuple(
        add_fixed_sram_lifecycle(fragment, context)
        for fragment in fragments
    )
    manifest = NaiveManifestLinker().link(context, fragments)
    source_entry = _source_profile(context, suffix="p0", weight=1.0)
    source = _source_bundle((source_entry,))
    lowered_entry = LoweredProgramProfile.create(
        source=source_entry,
        lowering_context=context,
        fragments=manifest.fragments,
    )
    lowered = LoweredProgramBundle.create(
        source=source,
        entries=(lowered_entry,),
    )
    lowered.validate_against(source)
    linked_entry = LinkedProgramProfile.create(
        source=lowered_entry,
        manifest=manifest,
    )
    bundle = LinkedProgramBundle.create(source=lowered, entries=(linked_entry,))
    bundle.validate_against(lowered)
    return source, lowered, bundle


class LinkedProgramBundleSchemaTest(unittest.TestCase):
    def test_strict_round_trip_stable_id_and_exact_source(self) -> None:
        source, lowered, bundle = _single_profile_fixture()
        decoded_lowered = loads_dataclass(
            LoweredProgramBundle,
            canonical_json(lowered),
            path="lowered_program_bundle",
        )
        decoded_lowered.validate_against(source)
        self.assertEqual(decoded_lowered, lowered)
        self.assertEqual(
            canonical_digest(decoded_lowered),
            canonical_digest(lowered),
        )
        self.assertEqual(
            decoded_lowered.schema_version,
            LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
        )
        decoded = loads_dataclass(
            LinkedProgramBundle,
            canonical_json(bundle),
            path="linked_program_bundle",
        )
        decoded.validate_against(lowered)
        self.assertEqual(decoded, bundle)
        self.assertEqual(canonical_digest(decoded), canonical_digest(bundle))
        self.assertEqual(
            decoded.schema_version,
            LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
        )

    def test_profile_rejects_context_manifest_and_leaf_impostors(self) -> None:
        source, lowered, bundle = _single_profile_fixture()
        source_entry = lowered.entries[0]
        entry = bundle.entries[0]
        with self.assertRaisesRegex(SchemaError, "lowered source provenance"):
            _restable_profile(
                entry,
                source_scheduled_entry_id="scheduled_entry_impostor",
            ).validate_against(source_entry)

        forged_manifest = _recreate_manifest(
            entry.manifest,
            source_global_dag_id="global_dag_impostor",
        )
        with self.assertRaises(SchemaError):
            _restable_profile(entry, manifest=forged_manifest).validate()

        with self.assertRaisesRegex(SchemaError, "leaf fragments"):
            _restable_profile(entry, leaf_fragments=()).validate()
        with self.assertRaisesRegex(SchemaError, "canonical id order"):
            _restable_profile(
                entry,
                leaf_fragments=tuple(reversed(entry.leaf_fragments)),
            ).validate()

    def test_bundle_rejects_missing_duplicate_weight_and_upstream_impostors(self) -> None:
        _source, lowered, bundle = _single_profile_fixture()
        entry = bundle.entries[0]
        with self.assertRaisesRegex(SchemaError, "one immutable entry"):
            LinkedProgramBundle.create(source=lowered, entries=()).validate()
        with self.assertRaisesRegex(SchemaError, "source profile weight"):
            _restable_bundle(
                bundle,
                entries=(_restable_profile(entry, weight=0.5),),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "lowered bundle provenance"):
            _restable_bundle(
                bundle,
                source_global_action_bundle_id="bundle_impostor",
            ).validate_against(lowered)
        with self.assertRaisesRegex(SchemaError, "lowered bundle provenance"):
            _restable_bundle(
                bundle,
                planning_context_id="planning_context_impostor",
            ).validate_against(lowered)

    def test_stale_ids_and_strict_serde_fail_closed(self) -> None:
        _source, lowered, bundle = _single_profile_fixture()
        with self.assertRaisesRegex(SchemaError, "unstable entry id"):
            replace(lowered.entries[0], id="stale").validate()
        with self.assertRaisesRegex(SchemaError, "schema_version"):
            replace(
                bundle,
                schema_version="wafer_frontend.linked_program_bundle/v1alpha2",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unstable bundle id"):
            replace(lowered, id="stale").validate()
        with self.assertRaisesRegex(SchemaError, "unstable entry id"):
            replace(bundle.entries[0], id="stale").validate()
        with self.assertRaisesRegex(SchemaError, "unstable bundle id"):
            replace(bundle, id="stale").validate()

        primitive = __import__(
            "llm.frontend.wafer_frontend.schema.serde",
            fromlist=["to_primitive"],
        ).to_primitive(bundle)
        assert isinstance(primitive, dict)
        missing = dict(primitive)
        missing.pop("entries")
        with self.assertRaises(SchemaError):
            loads_dataclass(LinkedProgramBundle, canonical_json(missing))
        unknown = dict(primitive)
        unknown["unexpected"] = True
        with self.assertRaises(SchemaError):
            loads_dataclass(LinkedProgramBundle, canonical_json(unknown))

        lowered_primitive = __import__(
            "llm.frontend.wafer_frontend.schema.serde",
            fromlist=["to_primitive"],
        ).to_primitive(lowered)
        assert isinstance(lowered_primitive, dict)
        missing_lowered = dict(lowered_primitive)
        missing_lowered.pop("entries")
        with self.assertRaises(SchemaError):
            loads_dataclass(
                LoweredProgramBundle,
                canonical_json(missing_lowered),
            )
        unknown_lowered = dict(lowered_primitive)
        unknown_lowered["unexpected"] = True
        with self.assertRaises(SchemaError):
            loads_dataclass(
                LoweredProgramBundle,
                canonical_json(unknown_lowered),
            )

    def test_lowered_rejects_missing_duplicate_reordered_and_impostor_fragments(self) -> None:
        source, lowered, _bundle = _single_profile_fixture()
        entry = lowered.entries[0]
        self.assertGreaterEqual(len(entry.fragments), 2)

        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            _restable_lowered_profile(
                entry,
                fragments=entry.fragments[:-1],
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unique leaves"):
            _restable_lowered_profile(
                entry,
                fragments=(entry.fragments[0], entry.fragments[0]),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "canonical leaf-id order"):
            _restable_lowered_profile(
                entry,
                fragments=tuple(reversed(entry.fragments)),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "global-action entry"):
            _restable_lowered_profile(
                entry,
                source_global_action_entry_id="global_entry_impostor",
            ).validate_against(source.entries[0])
        with self.assertRaisesRegex(SchemaError, "global-action bundle"):
            _restable_lowered_bundle(
                lowered,
                source_global_action_bundle_id="global_bundle_impostor",
            ).validate_against(source)

    def test_linked_mixed_fragments_match_by_exact_leaf_outer_bijection(self) -> None:
        _source, lowered, linked_bundle = _single_profile_fixture()
        base = lowered.entries[0]
        self.assertEqual(len(base.fragments), 2)
        command_fragments = tuple(base.fragments)
        self.assertTrue(
            all(type(fragment) is CommandFragment for fragment in command_fragments)
        )

        selected = None
        for wrapped_index in range(2):
            leaf = command_fragments[wrapped_index]
            assert type(leaf) is CommandFragment
            region = RegionManifest.create(
                producer_pass="isa_region_lowering",
                region_id=f"region.fixture.{wrapped_index}",
                fusion_plan_id=f"fusion.fixture.{wrapped_index}",
                target_dies=tuple(
                    sorted(
                        {
                            stream.logical_core.die_id
                            for stream in leaf.core_streams
                        }
                    )
                ),
                fragment=leaf,
            )
            mixed = list(command_fragments)
            mixed[wrapped_index] = region
            lowered_order = tuple(mixed)
            manifest_order = tuple(sorted(mixed, key=lambda fragment: fragment.id))
            if lowered_order != manifest_order:
                selected = (lowered_order, manifest_order, wrapped_index)
                break
        self.assertIsNotNone(selected)
        assert selected is not None
        lowered_order, manifest_order, wrapped_index = selected

        mixed_source = _restable_lowered_profile(
            base,
            fragments=lowered_order,
        )
        mixed_manifest = replace(
            linked_bundle.entries[0].manifest,
            fragments=manifest_order,
        )
        mixed_linked = LinkedProgramProfile.create(
            source=mixed_source,
            manifest=mixed_manifest,
        )
        with (
            patch.object(
                LoweredProgramProfile,
                "validate",
                autospec=True,
                return_value=None,
            ),
            patch.object(
                LinkedProgramProfile,
                "validate",
                autospec=True,
                return_value=None,
            ),
        ):
            mixed_linked.validate_against(mixed_source)

            original_region = lowered_order[wrapped_index]
            assert type(original_region) is RegionManifest
            impostor_region = RegionManifest.create(
                producer_pass=original_region.producer_pass,
                region_id=f"{original_region.region_id}.impostor",
                fusion_plan_id=original_region.fusion_plan_id,
                target_dies=original_region.target_dies,
                fragment=original_region.fragment,
            )
            impostor_manifest_fragments = tuple(
                impostor_region if fragment is original_region else fragment
                for fragment in manifest_order
            )
            impostor_manifest = replace(
                mixed_manifest,
                fragments=impostor_manifest_fragments,
            )
            impostor_linked = LinkedProgramProfile.create(
                source=mixed_source,
                manifest=impostor_manifest,
            )
            with self.assertRaisesRegex(SchemaError, "bijectively consume"):
                impostor_linked.validate_against(mixed_source)


if __name__ == "__main__":
    unittest.main()
