from __future__ import annotations

import dataclasses
import json
import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir2 import (
    IR2ProjectionResult,
    IntraDieDAG,
    IntraDieSchedule,
    IntraDieScheduleSet,
)
from llm.frontend.wafer_frontend.schema.n4 import (
    FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
    FusionPartitionedIR1Bundle,
    InterDiePlanBundle,
    InterDiePlannedProfile,
    InterDiePlanningContext,
)
from llm.frontend.wafer_frontend.schema.logical import ProfileEntry
from llm.frontend.wafer_frontend.schema.n5 import (
    GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
    INTRADIE_SCHEDULING_CONTEXT_SCHEMA_VERSION,
    PROJECTED_IR2_BUNDLE_SCHEMA_VERSION,
    PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION,
    SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION,
    IntraDieSchedulingContext,
    GlobalActionBundle,
    GlobalActionProfile,
    ProjectToIR2Context,
    ProjectedIR2Bundle,
    ProjectedProfileIR2,
    ScheduledIR2Bundle,
    ScheduledProfileIR2,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)

from _fixtures import (
    naive_inter_die_planning_context,
    naive_intra_die_scheduling_context,
)
from test_lowering_context import valid_lowering_context
from test_global_action_schema import _create_global
from test_n4_schema import (
    _restable_partition_entry,
    _single_profile_partition,
)


def _projected_fixture():
    lowering = valid_lowering_context()
    _partition_context, partitioned = _single_profile_partition(lowering.ir1)
    planning_context = naive_inter_die_planning_context("n5_schema_fixture")
    planned_entry = InterDiePlannedProfile.create(
        source=partitioned.entries[0],
        context=planning_context,
        fusion_plans=(),
        standalone_plans=(),
    )
    planned = InterDiePlanBundle.create(
        source=partitioned,
        context=planning_context,
        entries=(planned_entry,),
    )
    planned.validate_against(partitioned, planning_context)

    projection_context = ProjectToIR2Context.create(
        producer_pass="n5_schema_fixture",
        state_transfers=(),
    )
    projection = replace(
        lowering.projection,
        producer_pass="project_to_ir2",
    )
    projected_entry = ProjectedProfileIR2.create(
        source=planned_entry,
        context=projection_context,
        projection=projection,
    )
    projected = ProjectedIR2Bundle.create(
        source=planned,
        context=projection_context,
        entries=(projected_entry,),
    )
    projected.validate_against(planned, projection_context)
    return planned, projection_context, projected


def _restable_projected_entry(
    entry: ProjectedProfileIR2,
    **updates: object,
) -> ProjectedProfileIR2:
    changed = replace(entry, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "projected_profile_ir2",
            changed._semantic_key(),
            schema_version=PROJECTED_IR2_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _restable_projected_bundle(
    bundle: ProjectedIR2Bundle,
    **updates: object,
) -> ProjectedIR2Bundle:
    changed = replace(bundle, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "projected_ir2_bundle",
            changed._semantic_key(),
            schema_version=PROJECTED_IR2_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _scheduled_fixture():
    _planned, _projection_context, projected = _projected_fixture()
    lowering = valid_lowering_context()
    schedules = tuple(
        IntraDieSchedule.create(
            producer_pass="intra_die_schedule",
            **schedule._semantic_key(),
        )
        for schedule in lowering.schedule_set.schedules
    )
    schedule_set = IntraDieScheduleSet.create(
        producer_pass="intra_die_schedule",
        source_projection_id=projected.entries[0].projection.id,
        source_ir1_id=projected.entries[0].graph.id,
        schedules=schedules,
    )
    scheduling_context = naive_intra_die_scheduling_context("n5_schema_fixture")
    scheduled_entry = ScheduledProfileIR2.create(
        source=projected.entries[0],
        context=scheduling_context,
        schedule_set=schedule_set,
    )
    scheduled = ScheduledIR2Bundle.create(
        source=projected,
        context=scheduling_context,
        entries=(scheduled_entry,),
    )
    scheduled.validate_against(projected, scheduling_context)
    return projected, scheduling_context, scheduled


def _restable_scheduled_entry(
    entry: ScheduledProfileIR2,
    **updates: object,
) -> ScheduledProfileIR2:
    changed = replace(entry, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "scheduled_profile_ir2",
            changed._semantic_key(),
            schema_version=SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _restable_scheduled_bundle(
    bundle: ScheduledIR2Bundle,
    **updates: object,
) -> ScheduledIR2Bundle:
    changed = replace(bundle, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "scheduled_ir2_bundle",
            changed._semantic_key(),
            schema_version=SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _global_fixture():
    _projected, _scheduling_context, scheduled = _scheduled_fixture()
    source = scheduled.entries[0]
    global_dag = replace(
        _create_global(
            source.graph,
            source.projection,
            source.schedule_set,
        ),
        producer_pass="global_action_dag",
    )
    entry = GlobalActionProfile.create(
        source=source,
        global_dag=global_dag,
    )
    bundle = GlobalActionBundle.create(
        source=scheduled,
        entries=(entry,),
    )
    bundle.validate_against(scheduled)
    return scheduled, bundle


def _restable_global_entry(
    entry: GlobalActionProfile,
    **updates: object,
) -> GlobalActionProfile:
    changed = replace(entry, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "global_action_profile",
            changed._semantic_key(),
            schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _restable_global_bundle(
    bundle: GlobalActionBundle,
    **updates: object,
) -> GlobalActionBundle:
    changed = replace(bundle, **updates)
    return replace(
        changed,
        id=stable_artifact_id(
            "global_action_bundle",
            changed._semantic_key(),
            schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
        ),
    )


def _projection_for_graph(graph: IR1) -> IR2ProjectionResult:
    template = valid_lowering_context().projection
    dags = tuple(
        IntraDieDAG.create(
            producer_pass="project_to_ir2",
            **{
                **dag._semantic_key(),
                "source_ir1_id": graph.id,
            },
        )
        for dag in template.dags
    )
    return IR2ProjectionResult.create(
        producer_pass="project_to_ir2",
        source_ir1_id=graph.id,
        fusion_plan_ids=(),
        standalone_collective_plan_ids=(),
        dags=dags,
    )


def _schedule_for_projection(
    graph: IR1,
    projection: IR2ProjectionResult,
) -> IntraDieScheduleSet:
    template = valid_lowering_context().schedule_set
    schedules = tuple(
        IntraDieSchedule.create(
            producer_pass="intra_die_schedule",
            **{
                **schedule._semantic_key(),
                "dag_id": dag.id,
            },
        )
        for schedule, dag in zip(template.schedules, projection.dags)
    )
    return IntraDieScheduleSet.create(
        producer_pass="intra_die_schedule",
        source_projection_id=projection.id,
        source_ir1_id=graph.id,
        schedules=schedules,
    )


def _multi_profile_fixture():
    base = valid_lowering_context().ir1
    profiles = (
        base.profile,
        replace(base.profile, kv_pages=base.profile.kv_pages + 1),
    )
    partition_parts = []
    for profile in profiles:
        graph = IR1.create(
            producer_pass=base.producer_pass,
            **{**base._semantic_key(), "profile": profile},
        )
        partition_context, partitioned = _single_profile_partition(graph)
        partition_parts.append((partition_context, partitioned.entries[0]))
    ordered = sorted(partition_parts, key=lambda item: item[1].profile_id)
    weights = (0.25, 0.75)
    partition_context = ordered[0][0]
    partition_entries = tuple(
        _restable_partition_entry(entry, weight=weight)
        for (_context, entry), weight in zip(ordered, weights)
    )
    source_profiles = tuple(
        ProfileEntry.create(key=entry.graph.profile, weight=entry.weight)
        for entry in partition_entries
    )
    partition_key = {
        "source_placed_bundle_id": "placed_bundle_fixture",
        "placement_context_id": "placement_context_fixture",
        "partition_context_id": partition_context.id,
        "source_profiles": source_profiles,
        "entries": partition_entries,
    }
    partitioned = FusionPartitionedIR1Bundle(
        schema_version=FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
        producer_pass="fusion_partition",
        id=stable_artifact_id(
            "fusion_partitioned_ir1_bundle",
            partition_key,
            schema_version=FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
        ),
        **partition_key,
    )
    partitioned.validate()

    planning_context = naive_inter_die_planning_context("n5_schema_fixture")
    planned_entries = tuple(
        InterDiePlannedProfile.create(
            source=entry,
            context=planning_context,
            fusion_plans=(),
            standalone_plans=(),
        )
        for entry in partition_entries
    )
    planned = InterDiePlanBundle.create(
        source=partitioned,
        context=planning_context,
        entries=planned_entries,
    )
    planned.validate_against(partitioned, planning_context)

    projection_context = ProjectToIR2Context.create(
        producer_pass="n5_schema_fixture",
        state_transfers=(),
    )
    projected_entries = tuple(
        ProjectedProfileIR2.create(
            source=entry,
            context=projection_context,
            projection=_projection_for_graph(entry.graph),
        )
        for entry in planned.entries
    )
    projected = ProjectedIR2Bundle.create(
        source=planned,
        context=projection_context,
        entries=projected_entries,
    )
    projected.validate_against(planned, projection_context)

    scheduling_context = naive_intra_die_scheduling_context("n5_schema_fixture")
    scheduled_entries = tuple(
        ScheduledProfileIR2.create(
            source=entry,
            context=scheduling_context,
            schedule_set=_schedule_for_projection(entry.graph, entry.projection),
        )
        for entry in projected.entries
    )
    scheduled = ScheduledIR2Bundle.create(
        source=projected,
        context=scheduling_context,
        entries=scheduled_entries,
    )
    scheduled.validate_against(projected, scheduling_context)

    global_entries = tuple(
        GlobalActionProfile.create(
            source=entry,
            global_dag=replace(
                _create_global(
                    entry.graph,
                    entry.projection,
                    entry.schedule_set,
                ),
                producer_pass="global_action_dag",
            ),
        )
        for entry in scheduled.entries
    )
    global_bundle = GlobalActionBundle.create(
        source=scheduled,
        entries=global_entries,
    )
    global_bundle.validate_against(scheduled)
    return (
        planned,
        projection_context,
        projected,
        scheduling_context,
        scheduled,
        global_bundle,
    )


class N5ContextSchemaTest(unittest.TestCase):
    def test_contexts_are_typed_frozen_stable_and_strict_round_trip(self) -> None:
        projection = ProjectToIR2Context.create(
            producer_pass="unit", state_transfers=()
        )
        scheduling = naive_intra_die_scheduling_context("unit")
        projection.validate()
        scheduling.validate()
        self.assertEqual(
            projection.schema_version,
            PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION,
        )
        self.assertEqual(
            scheduling.schema_version,
            INTRADIE_SCHEDULING_CONTEXT_SCHEMA_VERSION,
        )
        self.assertEqual(
            projection.contract.value,
            "exact_naive_projection_state_transfer/v6",
        )
        self.assertEqual(
            scheduling.contract.value,
            "naive_component_rr_xy_sequential_state_transfer/v5",
        )
        self.assertEqual(
            loads_dataclass(
                ProjectToIR2Context,
                canonical_json(projection),
                path="projection_context",
            ),
            projection,
        )
        self.assertEqual(
            loads_dataclass(
                IntraDieSchedulingContext,
                canonical_json(scheduling),
                path="scheduling_context",
            ),
            scheduling,
        )
        self.assertEqual(
            ProjectToIR2Context.create(
                producer_pass="other", state_transfers=()
            ).id,
            projection.id,
        )
        self.assertEqual(
            naive_intra_die_scheduling_context("other").id,
            scheduling.id,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            projection.id = "changed"  # type: ignore[misc]
        with self.assertRaisesRegex(SchemaError, "ProjectToIR2Contract"):
            replace(
                projection,
                contract="exact_naive_projection/v1",  # type: ignore[arg-type]
            ).validate()
        with self.assertRaisesRegex(SchemaError, "naive intra-die policy"):
            replace(
                scheduling,
                policy=naive_inter_die_planning_context(
                    "wrong_policy"
                ).fused_policy,
            ).validate()
        raw = json.loads(canonical_json(scheduling))
        raw["contract"] = "unsupported/v1"
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            from_data(
                IntraDieSchedulingContext,
                raw,
                path="scheduling_context",
            )
        retired = json.loads(canonical_json(projection))
        retired["contract"] = "exact_naive_projection_state_transfer/v2"
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            from_data(
                ProjectToIR2Context,
                retired,
                path="projection_context",
            )
        retired = json.loads(canonical_json(projection))
        retired["schema_version"] = (
            "wafer_frontend.project_to_ir2_context/v1alpha2"
        )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            from_data(
                ProjectToIR2Context,
                retired,
                path="projection_context",
            )


class ProjectedIR2BundleSchemaTest(unittest.TestCase):
    def test_round_trip_stable_digest_and_exact_provenance(self) -> None:
        planned, context, bundle = _projected_fixture()
        decoded = loads_dataclass(
            ProjectedIR2Bundle,
            canonical_json(bundle),
            path="projected_ir2_bundle",
        )
        decoded.validate_against(planned, context)
        self.assertEqual(decoded, bundle)
        self.assertEqual(canonical_digest(decoded), canonical_digest(bundle))
        self.assertEqual(decoded.entries[0].graph, planned.entries[0].graph)
        self.assertEqual(
            decoded.entries[0].fusion_plans,
            planned.entries[0].fusion_plans,
        )
        self.assertEqual(
            decoded.entries[0].standalone_plans,
            planned.entries[0].standalone_plans,
        )

    def test_entry_source_context_weight_and_projection_are_exact(self) -> None:
        planned, context, bundle = _projected_fixture()
        source = planned.entries[0]
        entry = bundle.entries[0]
        cases = (
            (
                _restable_projected_entry(
                    entry,
                    source_planned_entry_id="planned_entry_impostor",
                ),
                "source planned entry",
            ),
            (
                _restable_projected_entry(
                    entry,
                    source_partitioned_entry_id="partitioned_entry_impostor",
                ),
                "source partitioned entry",
            ),
            (
                _restable_projected_entry(
                    entry,
                    projection_context_id="projection_context_impostor",
                ),
                "projection context",
            ),
            (
                _restable_projected_entry(entry, weight=entry.weight + 0.125),
                "profile and weight",
            ),
        )
        for candidate, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError,
                message,
            ):
                candidate.validate_against(source, context)

        fields = entry.projection._semantic_key()
        fields["source_ir1_id"] = "ir1_impostor"
        wrong_projection = IR2ProjectionResult.create(
            producer_pass="project_to_ir2",
            **fields,
        )
        candidate = _restable_projected_entry(
            entry,
            projection=wrong_projection,
        )
        with self.assertRaisesRegex(SchemaError, "different IR-1"):
            candidate.validate()

    def test_bundle_source_context_manifest_and_entry_coverage_are_exact(self) -> None:
        planned, context, bundle = _projected_fixture()
        cases = (
            (
                _restable_projected_bundle(
                    bundle,
                    source_inter_die_bundle_id="planned_bundle_impostor",
                ),
                "source inter-die bundle",
            ),
            (
                _restable_projected_bundle(
                    bundle,
                    source_partitioned_bundle_id="partitioned_bundle_impostor",
                ),
                "source partitioned bundle",
            ),
            (
                _restable_projected_bundle(
                    bundle,
                    projection_context_id="projection_context_impostor",
                ),
                "projection context",
            ),
        )
        for candidate, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError,
                message,
            ):
                candidate.validate_against(planned, context)

        for entries in ((), (bundle.entries[0], bundle.entries[0])):
            candidate = ProjectedIR2Bundle.create(
                source=planned,
                context=context,
                entries=entries,
            )
            with self.subTest(entries=len(entries)), self.assertRaises(SchemaError):
                candidate.validate()


class ScheduledIR2BundleSchemaTest(unittest.TestCase):
    def test_round_trip_stable_digest_and_exact_provenance(self) -> None:
        projected, context, bundle = _scheduled_fixture()
        decoded = loads_dataclass(
            ScheduledIR2Bundle,
            canonical_json(bundle),
            path="scheduled_ir2_bundle",
        )
        decoded.validate_against(projected, context)
        self.assertEqual(decoded, bundle)
        self.assertEqual(canonical_digest(decoded), canonical_digest(bundle))
        entry = decoded.entries[0]
        self.assertEqual(entry.graph, projected.entries[0].graph)
        self.assertEqual(entry.projection, projected.entries[0].projection)
        self.assertEqual(
            tuple(schedule.dag_id for schedule in entry.schedule_set.schedules),
            tuple(dag.id for dag in entry.projection.dags),
        )

    def test_entry_source_context_payload_and_schedule_are_exact(self) -> None:
        projected, context, bundle = _scheduled_fixture()
        source = projected.entries[0]
        entry = bundle.entries[0]
        cases = (
            (
                _restable_scheduled_entry(
                    entry,
                    source_projected_entry_id="projected_entry_impostor",
                ),
                "source projected entry",
            ),
            (
                _restable_scheduled_entry(
                    entry,
                    source_planned_entry_id="planned_entry_impostor",
                ),
                "upstream provenance",
            ),
            (
                _restable_scheduled_entry(
                    entry,
                    scheduling_context_id="scheduling_context_impostor",
                ),
                "scheduling context",
            ),
            (
                _restable_scheduled_entry(entry, weight=entry.weight + 0.125),
                "profile and weight",
            ),
        )
        for candidate, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError,
                message,
            ):
                candidate.validate_against(source, context)

        reversed_set = IntraDieScheduleSet.create(
            producer_pass="intra_die_schedule",
            source_projection_id=entry.projection.id,
            source_ir1_id=entry.graph.id,
            schedules=tuple(reversed(entry.schedule_set.schedules)),
        )
        with self.assertRaisesRegex(SchemaError, "DAG order"):
            _restable_scheduled_entry(
                entry,
                schedule_set=reversed_set,
            ).validate()

        first = entry.schedule_set.schedules[0]
        wrong_schedule = replace(first, producer_pass="fixture")
        wrong_set = IntraDieScheduleSet.create(
            producer_pass="intra_die_schedule",
            source_projection_id=entry.projection.id,
            source_ir1_id=entry.graph.id,
            schedules=(wrong_schedule,) + entry.schedule_set.schedules[1:],
        )
        with self.assertRaisesRegex(SchemaError, "intra_die_schedule"):
            _restable_scheduled_entry(
                entry,
                schedule_set=wrong_set,
            ).validate()

    def test_bundle_source_context_and_entry_coverage_are_exact(self) -> None:
        projected, context, bundle = _scheduled_fixture()
        cases = (
            (
                _restable_scheduled_bundle(
                    bundle,
                    source_projected_bundle_id="projected_bundle_impostor",
                ),
                "source projected bundle",
            ),
            (
                _restable_scheduled_bundle(
                    bundle,
                    source_inter_die_bundle_id="planned_bundle_impostor",
                ),
                "upstream provenance",
            ),
            (
                _restable_scheduled_bundle(
                    bundle,
                    scheduling_context_id="scheduling_context_impostor",
                ),
                "scheduling context",
            ),
        )
        for candidate, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError,
                message,
            ):
                candidate.validate_against(projected, context)

        for entries in ((), (bundle.entries[0], bundle.entries[0])):
            candidate = ScheduledIR2Bundle.create(
                source=projected,
                context=context,
                entries=entries,
            )
            with self.subTest(entries=len(entries)), self.assertRaises(SchemaError):
                candidate.validate()


class GlobalActionBundleSchemaTest(unittest.TestCase):
    def test_round_trip_exact_provenance_and_validated_lowering_context(self) -> None:
        scheduled, bundle = _global_fixture()
        self.assertEqual(
            GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
            "wafer_frontend.global_action_bundle/v1alpha6",
        )
        decoded = loads_dataclass(
            GlobalActionBundle,
            canonical_json(bundle),
            path="global_action_bundle",
        )
        decoded.validate_against(scheduled)
        self.assertEqual(decoded, bundle)
        self.assertEqual(canonical_digest(decoded), canonical_digest(bundle))
        lowering = decoded.entries[0].lowering_context()
        lowering.validate()
        self.assertEqual(lowering.ir1, scheduled.entries[0].graph)
        self.assertEqual(lowering.global_dag, decoded.entries[0].global_dag)

    def test_entry_source_payload_global_and_lowering_gate_are_exact(self) -> None:
        scheduled, bundle = _global_fixture()
        source = scheduled.entries[0]
        entry = bundle.entries[0]
        cases = (
            (
                _restable_global_entry(
                    entry,
                    source_scheduled_entry_id="scheduled_entry_impostor",
                ),
                "source scheduled entry",
            ),
            (
                _restable_global_entry(
                    entry,
                    source_projected_entry_id="projected_entry_impostor",
                ),
                "upstream provenance",
            ),
            (
                _restable_global_entry(entry, weight=entry.weight + 0.125),
                "profile and weight",
            ),
        )
        for candidate, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError,
                message,
            ):
                candidate.validate_against(source)

        wrong_producer = replace(
            entry.global_dag,
            producer_pass="fixture",
        )
        with self.assertRaisesRegex(SchemaError, "global_action_dag"):
            _restable_global_entry(
                entry,
                global_dag=wrong_producer,
            ).validate()

        with self.assertRaisesRegex(SchemaError, "unstable entry id"):
            replace(entry, id="arbitrary").lowering_context()

    def test_bundle_source_upstream_and_entry_coverage_are_exact(self) -> None:
        scheduled, bundle = _global_fixture()
        cases = (
            (
                _restable_global_bundle(
                    bundle,
                    source_scheduled_bundle_id="scheduled_bundle_impostor",
                ),
                "source scheduled bundle",
            ),
            (
                _restable_global_bundle(
                    bundle,
                    source_projected_bundle_id="projected_bundle_impostor",
                ),
                "upstream provenance",
            ),
            (
                _restable_global_bundle(
                    bundle,
                    scheduling_context_id="scheduling_context_impostor",
                ),
                "scheduling context",
            ),
        )
        for candidate, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                SchemaError,
                message,
            ):
                candidate.validate_against(scheduled)

        for entries in ((), (bundle.entries[0], bundle.entries[0])):
            candidate = GlobalActionBundle.create(
                source=scheduled,
                entries=entries,
            )
            with self.subTest(entries=len(entries)), self.assertRaises(SchemaError):
                candidate.validate()


class N5MultiProfileSchemaTest(unittest.TestCase):
    def test_all_layers_preserve_profile_order_weight_and_exact_coverage(self) -> None:
        (
            planned,
            projection_context,
            projected,
            scheduling_context,
            scheduled,
            global_bundle,
        ) = _multi_profile_fixture()
        self.assertEqual(len(global_bundle.entries), 2)
        self.assertEqual(
            loads_dataclass(
                GlobalActionBundle,
                canonical_json(global_bundle),
                path="global_action_bundle",
            ),
            global_bundle,
        )

        projected_factories = (
            lambda entries: ProjectedIR2Bundle.create(
                source=planned,
                context=projection_context,
                entries=entries,
            ),
            lambda entries: ScheduledIR2Bundle.create(
                source=projected,
                context=scheduling_context,
                entries=entries,
            ),
            lambda entries: GlobalActionBundle.create(
                source=scheduled,
                entries=entries,
            ),
        )
        bundles = (projected, scheduled, global_bundle)
        restabilizers = (
            _restable_projected_entry,
            _restable_scheduled_entry,
            _restable_global_entry,
        )
        for factory, bundle, restabilize in zip(
            projected_factories,
            bundles,
            restabilizers,
        ):
            invalid_entries = (
                bundle.entries[:1],
                (bundle.entries[0], bundle.entries[0]),
                tuple(reversed(bundle.entries)),
                (
                    restabilize(
                        bundle.entries[0],
                        weight=bundle.entries[0].weight + 0.125,
                    ),
                )
                + bundle.entries[1:],
            )
            for entries in invalid_entries:
                with self.subTest(
                    bundle=type(bundle).__name__,
                    entries=len(entries),
                ), self.assertRaises(SchemaError):
                    factory(entries).validate()

            reversed_manifest = replace(
                bundle,
                source_profiles=tuple(reversed(bundle.source_profiles)),
            )
            with self.subTest(
                bundle=type(bundle).__name__,
                case="profile_order",
            ), self.assertRaises(SchemaError):
                reversed_manifest.validate()


if __name__ == "__main__":
    unittest.main()
