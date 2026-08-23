from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import (
    NaiveManifestLinker,
    NaiveStateTransferLowering,
)
from llm.frontend.wafer_frontend.passes.lower_program import lower_profile
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    CommandFragment,
    FragmentKind,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.ir2 import (
    SemanticTaskKind,
    StateTransferOrigin,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
    GlobalActionProfile,
)
from llm.frontend.wafer_frontend.schema.n6 import (
    LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    LoweredProgramProfile,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind

from test_state_transfer_lowering import _case


def _profile(kinds: tuple[StateKind, ...]):
    context, contracts, groups = _case(kinds)
    fields = {
        "source_scheduled_entry_id": "scheduled.pd1",
        "source_projected_entry_id": "projected.pd1",
        "source_planned_entry_id": "planned.pd1",
        "source_partitioned_entry_id": "partitioned.pd1",
        "source_ir1_id": context.ir1.id,
        "projection_context_id": "projection_context.pd1",
        "scheduling_context_id": "scheduling_context.pd1",
        "profile_id": context.ir1.profile.stable_id(),
        "weight": 1.0,
        "graph": context.ir1,
        "fusion_plans": context.fusion_plans,
        "standalone_plans": context.standalone_plans,
        "projection": context.projection,
        "schedule_set": context.schedule_set,
        "global_dag": context.global_dag,
    }
    semantic_key = {
        **{
            name: fields[name]
            for name in (
                "source_scheduled_entry_id",
                "source_projected_entry_id",
                "source_planned_entry_id",
                "source_partitioned_entry_id",
                "source_ir1_id",
                "projection_context_id",
                "scheduling_context_id",
                "profile_id",
                "weight",
            )
        },
        "graph_id": context.ir1.id,
        "fusion_plan_ids": tuple(
            plan.id for plan in context.fusion_plans
        ),
        "standalone_plan_ids": tuple(
            plan.id for plan in context.standalone_plans
        ),
        "projection_id": context.projection.id,
        "schedule_set_id": context.schedule_set.id,
        "global_dag_id": context.global_dag.id,
    }
    profile = GlobalActionProfile(
        id=stable_artifact_id(
            "global_action_profile",
            semantic_key,
            schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
        ),
        **fields,
    )
    profile.validate()
    return profile, context, contracts, groups


class _RecordingStateTransfer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, tuple[SemanticTaskKind, ...]]] = []
        self.delegate = NaiveStateTransferLowering()

    def lower(self, actions, context):
        origin = actions[0].origin_ref
        assert isinstance(origin, StateTransferOrigin)
        core = actions[0].logical_core
        assert core is not None
        self.calls.append(
            (
                origin.state_transfer_ref,
                core.die_id,
                tuple(action.task_kind for action in actions),
            )
        )
        return self.delegate.lower(actions, context)


class LowerProgramStateTransferTest(unittest.TestCase):
    def test_recording_dispatches_once_per_contract_endpoint(self) -> None:
        profile, context, contracts, _groups = _profile(
            (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        recording = _RecordingStateTransfer()
        result = lower_profile(
            profile, state_transfer_lowerer=recording
        )
        expected = []
        route_index = {
            route.id: route
            for group in context.ir1.groups
            for route in group.embedding.routes
        }
        for contract in contracts:
            route = route_index[contract.pair_route_ref]
            expected.extend(
                (
                    (
                        contract.id,
                        route.die_path[0],
                        (SemanticTaskKind.SEND,),
                    ),
                    (
                        contract.id,
                        route.die_path[-1],
                        (SemanticTaskKind.RECV, SemanticTaskKind.WAIT),
                    ),
                )
            )
        self.assertEqual(recording.calls, expected)
        transfer_fragments = tuple(
            fragment
            for fragment in result.fragments
            if type(fragment) is CommandFragment
            and fragment.kind is FragmentKind.STATE_TRANSFER
        )
        self.assertEqual(len(transfer_fragments), 4)
        self.assertTrue(
            all(
                action.task_kind is not SemanticTaskKind.TRANSIT
                for fragment in transfer_fragments
                for action in context.global_dag.actions
                if action.id in fragment.claimed_action_ids
            )
        )

    def test_single_k_default_lifecycle_and_exact_coverage(self) -> None:
        profile, context, _contracts, _groups = _profile(
            (StateKind.KV_KEY,)
        )
        result = lower_profile(profile)
        result.validate_against(profile)
        transfer_fragments = tuple(
            fragment
            for fragment in result.fragments
            if type(fragment) is CommandFragment
            and fragment.kind is FragmentKind.STATE_TRANSFER
        )
        self.assertEqual((len(result.fragments), len(transfer_fragments)), (54, 2))
        action_index = {action.id: action for action in context.global_dag.actions}
        source = next(
            fragment
            for fragment in transfer_fragments
            if len(fragment.claimed_action_ids) == 1
        )
        destination = next(
            fragment
            for fragment in transfer_fragments
            if len(fragment.claimed_action_ids) == 2
        )
        self.assertEqual(
            tuple(
                action_index[action_id].task_kind
                for action_id in source.claimed_action_ids
            ),
            (SemanticTaskKind.SEND,),
        )
        self.assertEqual(
            {
                action_index[action_id].task_kind
                for action_id in destination.claimed_action_ids
            },
            {SemanticTaskKind.RECV, SemanticTaskKind.WAIT},
        )
        self.assertEqual(
            tuple(record.opcode for record in source.core_streams[0].records),
            (RecordOpcode.DTE_SEND, RecordOpcode.SRAM_FREE),
        )
        self.assertEqual(
            tuple(
                record.opcode
                for record in destination.core_streams[0].records
            ),
            (
                RecordOpcode.SRAM_ALLOC_AT,
                RecordOpcode.DTE_RECV,
                RecordOpcode.DTE_WAIT,
            ),
        )
        self.assertTrue(
            all(
                len(fragment.buffer_abi) == 1 and not fragment.state_abi
                for fragment in transfer_fragments
            )
        )

        missing = LoweredProgramProfile.create(
            source=profile,
            lowering_context=context,
            fragments=tuple(
                fragment
                for fragment in result.fragments
                if fragment.id != source.id
            ),
        )
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            missing.validate_against(profile)
        duplicate = LoweredProgramProfile.create(
            source=profile,
            lowering_context=context,
            fragments=result.fragments + (source,),
        )
        with self.assertRaisesRegex(SchemaError, "unique leaves"):
            duplicate.validate_against(profile)

    def test_kv_default_links_generic_buffer_closure(self) -> None:
        profile, context, _contracts, _groups = _profile(
            (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        lowered = lower_profile(profile)
        transfer_fragments = tuple(
            fragment
            for fragment in lowered.fragments
            if type(fragment) is CommandFragment
            and fragment.kind is FragmentKind.STATE_TRANSFER
        )
        self.assertEqual((len(lowered.fragments), len(transfer_fragments)), (56, 4))
        self.assertEqual(
            sum(
                len(stream.records)
                for fragment in transfer_fragments
                for stream in fragment.core_streams
            ),
            10,
        )
        manifest = NaiveManifestLinker().link(context, lowered.fragments)
        manifest.validate_against(
            context.ir1,
            context.fusion_plans,
            context.standalone_plans,
            context.projection,
            context.schedule_set,
            context.global_dag,
            manifest.fragments,
        )
        transfer_ids = {fragment.id for fragment in transfer_fragments}
        self.assertEqual(len(manifest.fragments), 56)
        self.assertEqual(len(manifest.address_operand_bindings), 446)
        self.assertEqual(
            sum(
                binding.fragment_id in transfer_ids
                for binding in manifest.address_operand_bindings
            ),
            10,
        )
        self.assertEqual(len(manifest.state_operand_bindings), 22)
        self.assertTrue(
            all(
                binding.fragment_id not in transfer_ids
                for binding in manifest.state_operand_bindings
            )
        )
        with self.assertRaisesRegex(SchemaError, "cover every executable"):
            NaiveManifestLinker().link(
                context,
                tuple(
                    fragment
                    for fragment in lowered.fragments
                    if fragment.id != transfer_fragments[0].id
                ),
            )

    def test_breaking_versions_and_stale_artifacts_fail_closed(self) -> None:
        profile, context, _contracts, _groups = _profile(
            (StateKind.KV_KEY,)
        )
        lowered = lower_profile(profile)
        fragment = next(
            fragment
            for fragment in lowered.fragments
            if type(fragment) is CommandFragment
            and fragment.kind is FragmentKind.STATE_TRANSFER
        )
        manifest = NaiveManifestLinker().link(context, lowered.fragments)
        self.assertEqual(
            COMMAND_FRAGMENT_SCHEMA_VERSION,
            "wafer_frontend.command_fragment/v1alpha13",
        )
        self.assertEqual(
            LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
            "wafer_frontend.lowered_program_bundle/v1alpha9",
        )
        self.assertEqual(
            LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
            "wafer_frontend.linked_program_manifest/v1alpha14",
        )
        self.assertEqual(
            LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
            "wafer_frontend.linked_program_bundle/v1alpha10",
        )
        with self.assertRaisesRegex(SchemaError, "schema version"):
            replace(
                fragment,
                schema_version="wafer_frontend.command_fragment/v1alpha10",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "schema version"):
            replace(
                manifest,
                schema_version=(
                    "wafer_frontend.linked_program_manifest/v1alpha11"
                ),
            ).validate()


if __name__ == "__main__":
    unittest.main()
