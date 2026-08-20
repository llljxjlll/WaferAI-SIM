from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import link_bundle, link_profile
from llm.frontend.wafer_frontend.schema.n6 import (
    LinkedProgramBundle,
    LinkedProgramProfile,
    LoweredProgramBundle,
    LoweredProgramProfile,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from test_n5_schema import _multi_profile_fixture
from test_n6_schema import _single_profile_fixture


class LinkProgramPassTest(unittest.TestCase):
    def test_real_ordinary_profile_and_bundle_link_once_and_deterministically(self) -> None:
        _global_source, lowered, _fixture_linked = _single_profile_fixture()
        before = canonical_digest(lowered)

        first = link_profile(lowered.entries[0])
        second = link_profile(lowered.entries[0])
        first.validate_against(lowered.entries[0])
        self.assertEqual(first, second)
        self.assertEqual(first.source_lowered_entry_id, lowered.entries[0].id)

        bundle = link_bundle(lowered)
        bundle.validate_against(lowered)
        self.assertEqual(bundle.entries, (first,))
        self.assertEqual(bundle.source_lowered_bundle_id, lowered.id)
        self.assertEqual(canonical_digest(lowered), before)

    def test_injected_linker_is_called_once_per_profile_in_source_order(self) -> None:
        global_source = _multi_profile_fixture()[-1]
        _ordinary_source, ordinary_lowered, ordinary_linked = (
            _single_profile_fixture()
        )
        dummy_fragments = ordinary_lowered.entries[0].fragments
        dummy_manifest = ordinary_linked.entries[0].manifest
        entries = tuple(
            LoweredProgramProfile.create(
                source=entry,
                lowering_context=entry.lowering_context(),
                fragments=dummy_fragments,
            )
            for entry in global_source.entries
        )
        source = LoweredProgramBundle.create(
            source=global_source,
            entries=entries,
        )
        calls: list[tuple[str, tuple[str, ...]]] = []

        class Linker:
            def link(self, context, fragments):
                calls.append(
                    (
                        context.ir1.profile.stable_id(),
                        tuple(fragment.id for fragment in fragments),
                    )
                )
                return dummy_manifest

        before = canonical_digest(source)
        with (
            patch.object(
                LoweredProgramProfile,
                "validate",
                autospec=True,
                return_value=None,
            ),
            patch.object(
                LoweredProgramBundle,
                "validate",
                autospec=True,
                return_value=None,
            ),
            patch.object(
                LinkedProgramProfile,
                "validate_against",
                autospec=True,
                return_value=None,
            ),
            patch.object(
                LinkedProgramBundle,
                "validate_against",
                autospec=True,
                return_value=None,
            ),
        ):
            result = link_bundle(source, linker=Linker())

        self.assertEqual(
            tuple(profile_id for profile_id, _fragment_ids in calls),
            tuple(entry.profile_id for entry in global_source.entries),
        )
        self.assertEqual(
            tuple(fragment_ids for _profile_id, fragment_ids in calls),
            tuple(
                tuple(fragment.id for fragment in entry.fragments)
                for entry in source.entries
            ),
        )
        self.assertEqual(
            tuple(entry.source_lowered_entry_id for entry in result.entries),
            tuple(entry.id for entry in source.entries),
        )
        self.assertEqual(canonical_digest(source), before)

    def test_wrong_source_output_and_producer_fail_closed(self) -> None:
        global_source, lowered, linked = _single_profile_fixture()

        with self.assertRaisesRegex(SchemaError, "LoweredProgramProfile"):
            link_profile(global_source.entries[0])  # type: ignore[arg-type]
        with self.assertRaisesRegex(SchemaError, "LoweredProgramBundle"):
            link_bundle(global_source)  # type: ignore[arg-type]

        class WrongOutput:
            def link(self, context, fragments):
                return object()

        with self.assertRaisesRegex(SchemaError, "LinkedProgramManifest"):
            link_profile(lowered.entries[0], linker=WrongOutput())

        forged_manifest = replace(
            linked.entries[0].manifest,
            producer_pass="impostor",
        )

        class WrongProducer:
            def link(self, context, fragments):
                return forged_manifest

        with self.assertRaisesRegex(SchemaError, "manifest_linker"):
            link_profile(lowered.entries[0], linker=WrongProducer())


if __name__ == "__main__":
    unittest.main()
