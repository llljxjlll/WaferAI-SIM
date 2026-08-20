from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
import unittest

from test_train_forward_global_action import _global_action

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import link_train, lower_train
from llm.frontend.wafer_frontend.schema import TrainLinkedProgram
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    EmptyCoreAckPolicy,
    LinkedProgramManifest,
    ManifestInputKind,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.train_n6 import (
    TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
)


def _recreate_manifest(
    manifest: LinkedProgramManifest,
    **changes: object,
) -> LinkedProgramManifest:
    semantic_key = manifest._semantic_key()
    semantic_key.update(changes)
    return LinkedProgramManifest.create(
        producer_pass=manifest.producer_pass,
        **semantic_key,
    )


def _recreate_train_linked(
    result: TrainLinkedProgram,
    manifest: LinkedProgramManifest,
) -> TrainLinkedProgram:
    semantic_key = {"source": result.source, "manifest": manifest}
    return replace(
        result,
        manifest=manifest,
        id=stable_artifact_id(
            "train_linked_program",
            semantic_key,
            schema_version=TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
        ),
    )


class TrainForwardLinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, train_global = _global_action()
        cls.source = lower_train(train_global)
        cls.result = link_train(cls.source)

    def test_dp2_is_one_exact_canonical_manifest(self) -> None:
        result = self.result
        manifest = result.manifest
        result.validate()
        self.assertEqual(
            result.schema_version,
            TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
        )
        self.assertEqual(
            (
                manifest.source_ir1_id,
                manifest.source_projection_id,
                manifest.source_schedule_set_id,
                manifest.source_global_dag_id,
            ),
            (
                self.source.source_planned_carrier_id,
                self.source.source_projected_carrier_id,
                self.source.source_scheduled_carrier_id,
                self.source.source_global_action_carrier_id,
            ),
        )
        self.assertEqual(
            (
                len(manifest.fragments),
                len(manifest.fragment_interfaces),
                len(manifest.core_bindings),
                len(manifest.core_streams),
                sum(
                    len(stream.records)
                    for linked in manifest.fragments
                    for stream in (
                        linked.fragment
                        if isinstance(linked, RegionManifest)
                        else linked
                    ).core_streams
                ),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
                len(manifest.input_digests),
                len(manifest.envelope.start_events),
            ),
            (172, 172, 4, 4, 996, 164, 613, 1592, 60, 213, 4),
        )
        self.assertEqual(
            Counter(item.kind for item in manifest.input_digests),
            Counter(
                {
                    ManifestInputKind.TRAIN_LOWERED_PROGRAM: 1,
                    ManifestInputKind.IR1: 2,
                    ManifestInputKind.IR2_PROJECTION: 2,
                    ManifestInputKind.SCHEDULE_SET: 2,
                    ManifestInputKind.GLOBAL_ACTION_DAG: 2,
                    ManifestInputKind.FUSION_PLAN: 8,
                    ManifestInputKind.STANDALONE_PLAN: 8,
                    ManifestInputKind.COMMAND_FRAGMENT: 172,
                    ManifestInputKind.REGION_MANIFEST: 16,
                }
            ),
        )

    def test_shared_region_symbol_has_one_global_exporter(self) -> None:
        manifest = self.result.manifest
        fragments_by_symbol: dict[str, set[str]] = defaultdict(set)
        for linked in manifest.fragments:
            leaf = (
                linked.fragment if isinstance(linked, RegionManifest) else linked
            )
            for symbol in leaf.program_symbols:
                fragments_by_symbol[symbol.id].add(leaf.id)
        shared = {
            symbol_id: fragment_ids
            for symbol_id, fragment_ids in fragments_by_symbol.items()
            if len(fragment_ids) > 1
        }
        self.assertTrue(shared)
        exporters = Counter(
            symbol_id
            for interface in manifest.fragment_interfaces
            for symbol_id in interface.program_exports
        )
        for symbol_id in shared:
            self.assertEqual(exporters[symbol_id], 1)
        definitions = {
            definition.symbol.id: definition
            for definition in manifest.program_symbol_definitions
        }
        self.assertTrue(
            any(len(definitions[symbol_id].logical_cores) > 1 for symbol_id in shared)
        )

    def test_strict_serde_stable_id_and_version(self) -> None:
        encoded = canonical_json(self.result)
        decoded = loads_dataclass(
            TrainLinkedProgram,
            encoded,
            path="train_linked",
        )
        self.assertEqual(decoded, self.result)
        decoded.validate()

        raw = json.loads(encoded)
        raw["unexpected"] = True
        with self.assertRaises(SchemaError):
            loads_dataclass(
                TrainLinkedProgram,
                canonical_json(raw),
                path="train_linked",
            )
        del raw["unexpected"]
        del raw["manifest"]
        with self.assertRaises(SchemaError):
            loads_dataclass(
                TrainLinkedProgram,
                canonical_json(raw),
                path="train_linked",
            )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                self.result,
                schema_version="wafer_frontend.train_linked_program/v1alpha0",
            ).validate()

    def test_train_digest_lineage_and_top_ids_fail_closed(self) -> None:
        manifest = self.result.manifest
        without_train = tuple(
            digest
            for digest in manifest.input_digests
            if digest.kind is not ManifestInputKind.TRAIN_LOWERED_PROGRAM
        )
        with self.assertRaisesRegex(SchemaError, "different global DAG"):
            replace(manifest, input_digests=without_train).validate()

        missing_ir1 = tuple(
            digest
            for digest in manifest.input_digests
            if not (
                digest.kind is ManifestInputKind.IR1
                and digest.artifact_id
                == self.source.replicas[0].lowering_context.ir1.id
            )
        )
        with self.assertRaisesRegex(SchemaError, "equal non-zero"):
            replace(manifest, input_digests=missing_ir1).validate()

        global_dag_id = (
            self.source.replicas[0].lowering_context.global_dag.id
        )
        forged_global = tuple(
            sorted(
                (
                    replace(digest, artifact_id="forged_global_dag")
                    if digest.kind is ManifestInputKind.GLOBAL_ACTION_DAG
                    and digest.artifact_id == global_dag_id
                    else digest
                    for digest in manifest.input_digests
                ),
                key=lambda item: (item.kind.value, item.artifact_id),
            )
        )
        with self.assertRaisesRegex(SchemaError, "lacks an exact input digest"):
            replace(manifest, input_digests=forged_global).validate()

        forged_top = _recreate_manifest(
            manifest,
            source_ir1_id="forged_planned_carrier",
        )
        forged_result = _recreate_train_linked(self.result, forged_top)
        with self.assertRaisesRegex(SchemaError, "top ids"):
            forged_result.validate()

    def test_restable_digest_and_envelope_tamper_fail_closed(self) -> None:
        manifest = self.result.manifest
        train_digest = next(
            item
            for item in manifest.input_digests
            if item.kind is ManifestInputKind.TRAIN_LOWERED_PROGRAM
        )
        forged_digests = tuple(
            replace(item, digest="0" * 64)
            if item is train_digest
            else item
            for item in manifest.input_digests
        )
        forged_manifest = _recreate_manifest(
            manifest,
            input_digests=forged_digests,
        )
        forged = self.result
        with self.assertRaisesRegex(SchemaError, "strict unified"):
            _recreate_train_linked(forged, forged_manifest).validate()

        forged_envelope = replace(
            manifest.envelope,
            empty_core_ack_policy=EmptyCoreAckPolicy.EXCLUDE_EMPTY,
        )
        envelope_manifest = _recreate_manifest(
            manifest,
            envelope=forged_envelope,
        )
        with self.assertRaisesRegex(SchemaError, "strict unified"):
            _recreate_train_linked(forged, envelope_manifest).validate()

    def test_public_pass_rejects_non_train_lowering(self) -> None:
        with self.assertRaisesRegex(SchemaError, "TrainLoweredProgram"):
            link_train(self.source.replicas[0])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
