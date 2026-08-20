from __future__ import annotations

import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import (
    NaiveIsaRegionLowering,
    NaiveStateDmaLowering,
    NaiveStandaloneCollectiveLowering,
)
from llm.frontend.wafer_frontend.passes import lower_bundle, lower_profile
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    FragmentKind,
    RecordOpcode,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    FusedNodeOrigin,
    OrdinaryNodeOrigin,
    SemanticTaskKind,
    StateIoOrigin,
    StandaloneNodeOrigin,
)
from llm.frontend.wafer_frontend.schema.n6 import (
    LoweredProgramBundle,
    LoweredProgramProfile,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from test_n5_pipeline import _compile_through_n5
from test_n5_schema import _multi_profile_fixture
from test_n6_schema import _single_profile_fixture


class _RecordingCoarse:
    def __init__(
        self,
        trace: list[tuple[object, ...]],
        fragment: CommandFragment,
    ) -> None:
        self.trace = trace
        self.fragment = fragment

    def lower(self, action, context):
        self.trace.append(("ordinary", action.id))
        return self.fragment


class _RecordingState:
    def __init__(self, trace: list[tuple[object, ...]]) -> None:
        self.trace = trace
        self.delegate = NaiveStateDmaLowering()
        self.outputs: list[CommandFragment] = []

    def lower(self, action, context):
        self.trace.append(("state", action.id))
        result = self.delegate.lower(action, context)
        self.outputs.append(result)
        return result


class _RecordingIsa:
    def __init__(self, trace: list[tuple[object, ...]]) -> None:
        self.trace = trace
        self.delegate = NaiveIsaRegionLowering()
        self.outputs: list[tuple[RegionManifest, ...]] = []

    def lower(self, plan, actions, context):
        self.trace.append(
            ("fusion", plan.id, tuple(action.id for action in actions))
        )
        result = self.delegate.lower(plan, actions, context)
        self.outputs.append(result)
        return result


class _RecordingStandalone:
    def __init__(self, trace: list[tuple[object, ...]]) -> None:
        self.trace = trace
        self.delegate = NaiveStandaloneCollectiveLowering()
        self.outputs: list[CommandFragment] = []

    def lower(self, actions, context):
        plan_id = actions[0].origin_ref.collective_plan_id
        self.trace.append(
            ("standalone", plan_id, tuple(action.id for action in actions))
        )
        result = self.delegate.lower(actions, context)
        self.outputs.append(result)
        return result


class LowerProgramPassTest(unittest.TestCase):
    def test_real_ordinary_profile_and_bundle_are_exact_and_deterministic(self) -> None:
        source, _fixture_lowered, _fixture_linked = _single_profile_fixture()
        before = canonical_digest(source)

        first = lower_profile(source.entries[0])
        second = lower_profile(source.entries[0])
        first.validate_against(source.entries[0])
        self.assertEqual(first, second)
        self.assertEqual(first.lowering_context, source.entries[0].lowering_context())
        self.assertFalse(hasattr(first, "manifest"))

        bundle = lower_bundle(source)
        bundle.validate_against(source)
        self.assertEqual(bundle.entries, (first,))
        self.assertEqual(bundle.producer_pass, "lowering")
        self.assertEqual(canonical_digest(source), before)

    def test_default_tp2_lowers_exact_parameter_and_kv_state_leaf_counts(self) -> None:
        entry = _compile_through_n5()[-1].entries[0]
        result = lower_profile(entry)
        state_fragments = tuple(
            fragment
            for fragment in result.fragments
            if type(fragment) is CommandFragment
            and fragment.kind is FragmentKind.STATE_IO
        )
        state_actions = {
            action.id: action
            for action in entry.global_dag.actions
            if isinstance(action.origin_ref, StateIoOrigin)
        }
        self.assertEqual(len(state_fragments), 22)
        self.assertEqual(
            set(state_actions),
            {fragment.claimed_action_ids[0] for fragment in state_fragments},
        )
        opcodes = tuple(
            record.opcode
            for fragment in state_fragments
            for stream in fragment.core_streams
            for record in stream.records
            if record.opcode in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE)
        )
        self.assertEqual(opcodes.count(RecordOpcode.LSU_LOAD), 18)
        self.assertEqual(opcodes.count(RecordOpcode.LSU_STORE), 4)
        self.assertTrue(
            all(len(fragment.state_abi) == 1 for fragment in state_fragments)
        )

        missing = LoweredProgramProfile.create(
            source=entry,
            lowering_context=entry.lowering_context(),
            fragments=tuple(
                fragment
                for fragment in result.fragments
                if fragment != state_fragments[0]
            ),
        )
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            missing.validate_against(entry)

        duplicate = LoweredProgramProfile.create(
            source=entry,
            lowering_context=entry.lowering_context(),
            fragments=result.fragments + (state_fragments[0],),
        )
        with self.assertRaisesRegex(SchemaError, "unique leaves"):
            duplicate.validate_against(entry)

    def test_mixed_groups_call_protocols_in_frozen_order_and_decorate_each_leaf(self) -> None:
        source = _compile_through_n5()[-1]
        entry = source.entries[0]
        context = entry.lowering_context()
        before = canonical_digest(source)
        _ordinary_source, dummy_lowered, _dummy_linked = _single_profile_fixture()
        dummy_fragment = dummy_lowered.entries[0].fragments[0]
        assert type(dummy_fragment) is CommandFragment

        trace: list[tuple[object, ...]] = []
        lifecycle_calls: list[tuple[CommandFragment, object]] = []
        coarse = _RecordingCoarse(trace, dummy_fragment)
        state = _RecordingState(trace)
        isa = _RecordingIsa(trace)
        standalone = _RecordingStandalone(trace)

        with (
            patch(
                "llm.frontend.wafer_frontend.passes.lower_program."
                "add_fixed_sram_lifecycle",
                side_effect=lambda fragment, lifecycle_context: (
                    lifecycle_calls.append((fragment, lifecycle_context))
                    or fragment
                ),
            ),
            patch.object(
                CommandFragment,
                "validate_against",
                autospec=True,
                return_value=None,
            ),
            patch.object(
                LoweredProgramProfile,
                "validate_against",
                autospec=True,
                return_value=None,
            ),
        ):
            result = lower_profile(
                entry,
                coarse_lowerer=coarse,
                isa_lowerer=isa,
                standalone_lowerer=standalone,
                state_dma_lowerer=state,
            )

        expected: list[tuple[object, ...]] = [
            (
                (
                    "state"
                    if isinstance(action.origin_ref, StateIoOrigin)
                    else "ordinary"
                ),
                action.id,
            )
            for action in context.global_dag.actions
            if isinstance(action.origin_ref, (StateIoOrigin, OrdinaryNodeOrigin))
        ]
        expected.extend(
            (
                "fusion",
                plan.id,
                tuple(
                    action.id
                    for action in context.global_dag.actions
                    if isinstance(action.origin_ref, FusedNodeOrigin)
                    and action.origin_ref.plan_id == plan.id
                    and action.task_kind is not SemanticTaskKind.TRANSIT
                ),
            )
            for plan in context.fusion_plans
        )
        expected.extend(
            (
                "standalone",
                plan.id,
                tuple(
                    action.id
                    for action in context.global_dag.actions
                    if isinstance(action.origin_ref, StandaloneNodeOrigin)
                    and action.origin_ref.collective_plan_id == plan.id
                ),
            )
            for plan in context.standalone_plans
        )
        self.assertEqual(trace, expected)

        ordinary_count = sum(
            isinstance(action.origin_ref, OrdinaryNodeOrigin)
            for action in context.global_dag.actions
        )
        region_count = sum(len(regions) for regions in isa.outputs)
        expected_leaf_count = (
            ordinary_count
            + len(state.outputs)
            + region_count
            + len(standalone.outputs)
        )
        self.assertEqual(len(result.fragments), expected_leaf_count)
        self.assertEqual(len(lifecycle_calls), expected_leaf_count)
        self.assertTrue(
            all(call_context == context for _fragment, call_context in lifecycle_calls)
        )

        standalone_call_ids = {
            action_id
            for call in expected
            if call[0] == "standalone"
            for action_id in call[2]
        }
        transit_ids = {
            action.id
            for action in context.global_dag.actions
            if action.task_kind is SemanticTaskKind.TRANSIT
        }
        self.assertTrue(transit_ids.issubset(standalone_call_ids))
        self.assertEqual(canonical_digest(source), before)

    def test_bundle_uses_shared_fakes_once_per_entry_in_source_order(self) -> None:
        source = _multi_profile_fixture()[-1]
        _ordinary_source, dummy_lowered, _dummy_linked = _single_profile_fixture()
        fragment = dummy_lowered.entries[0].fragments[0]
        assert type(fragment) is CommandFragment
        calls: list[str] = []

        class Coarse:
            def lower(self, action, context):
                calls.append(context.ir1.profile.stable_id())
                return fragment

        class Unexpected:
            def lower(self, *args):
                raise AssertionError("plan lowerer must not be called")

        expected_calls = tuple(
            entry.profile_id
            for entry in source.entries
            for _action in entry.global_dag.actions
        )
        before = canonical_digest(source)
        with (
            patch(
                "llm.frontend.wafer_frontend.passes.lower_program."
                "add_fixed_sram_lifecycle",
                side_effect=lambda value, _context: value,
            ),
            patch.object(
                CommandFragment,
                "validate_against",
                autospec=True,
                return_value=None,
            ),
            patch.object(
                LoweredProgramProfile,
                "validate_against",
                autospec=True,
                return_value=None,
            ),
            patch.object(
                LoweredProgramBundle,
                "validate_against",
                autospec=True,
                return_value=None,
            ),
        ):
            result = lower_bundle(
                source,
                coarse_lowerer=Coarse(),
                isa_lowerer=Unexpected(),
                standalone_lowerer=Unexpected(),
            )

        self.assertEqual(tuple(calls), expected_calls)
        self.assertEqual(
            tuple(entry.source_global_action_entry_id for entry in result.entries),
            tuple(entry.id for entry in source.entries),
        )
        self.assertEqual(canonical_digest(source), before)

    def test_wrong_source_and_lowerer_output_fail_closed(self) -> None:
        source, _lowered, _linked = _single_profile_fixture()

        with self.assertRaisesRegex(SchemaError, "GlobalActionProfile"):
            lower_profile(source)  # type: ignore[arg-type]
        with self.assertRaisesRegex(SchemaError, "GlobalActionBundle"):
            lower_bundle(source.entries[0])  # type: ignore[arg-type]

        class WrongCoarse:
            def lower(self, action, context):
                return object()

        with self.assertRaisesRegex(SchemaError, "CommandFragment"):
            lower_profile(source.entries[0], coarse_lowerer=WrongCoarse())


if __name__ == "__main__":
    unittest.main()
