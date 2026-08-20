from __future__ import annotations

import dataclasses
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import (
    ArtifactMetadata,
    ProfileKey,
    UINT64_MAX,
    stable_artifact_id,
    validate_dependency_dag,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)


@dataclasses.dataclass(frozen=True, slots=True)
class DependencyNode:
    id: str
    deps: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class NestedValue:
    value: int


@dataclasses.dataclass(frozen=True, slots=True)
class NestedDocument:
    nested: NestedValue


class SchemaSerdeTest(unittest.TestCase):
    def test_profile_round_trip_and_canonical_order(self) -> None:
        raw = {
            "num_seqs": 1,
            "decode_tokens": 0,
            "prefill_tokens": 32,
            "context_max": 32,
            "context_sum": 32,
            "kv_pages": 2,
            "expert_load": None,
        }
        profile = from_data(ProfileKey, raw, path="profile")
        self.assertEqual(profile.prefill_tokens, 32)
        self.assertEqual(
            canonical_json(profile),
            canonical_json(from_data(ProfileKey, raw, path="profile")),
        )

    def test_profile_missing_and_unknown_fields_are_rejected(self) -> None:
        raw = {
            "prefill_tokens": 1,
            "decode_tokens": 0,
            "num_seqs": 1,
            "context_sum": 1,
            "context_max": 1,
        }
        with self.assertRaisesRegex(SchemaError, "kv_pages"):
            from_data(ProfileKey, raw, path="profile")
        raw["kv_pages"] = 1
        with self.assertRaisesRegex(SchemaError, "expert_load"):
            from_data(ProfileKey, raw, path="profile")
        raw["expert_load"] = None
        raw["M"] = 1
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            from_data(ProfileKey, raw, path="profile")

    def test_profile_range_and_cross_field_checks(self) -> None:
        base = {
            "prefill_tokens": 1,
            "decode_tokens": 0,
            "num_seqs": 1,
            "context_sum": 1,
            "context_max": 1,
            "kv_pages": 1,
            "expert_load": None,
        }
        for bad in (-1, UINT64_MAX + 1, True):
            raw = dict(base, kv_pages=bad)
            with self.assertRaises(SchemaError):
                from_data(ProfileKey, raw, path="profile")
        with self.assertRaisesRegex(SchemaError, "context_sum"):
            from_data(ProfileKey, dict(base, context_max=2), path="profile")

    def test_metadata_is_frozen(self) -> None:
        metadata = ArtifactMetadata("v1", "unit", "artifact_0")
        metadata.validate()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            metadata.id = "changed"  # type: ignore[misc]

    def test_stable_id_and_digest_ignore_mapping_insertion_order(self) -> None:
        left = {"a": 1, "b": (2, 3)}
        right = {"b": (2, 3), "a": 1}
        self.assertEqual(canonical_digest(left), canonical_digest(right))
        self.assertEqual(
            stable_artifact_id("x", left, schema_version="v1"),
            stable_artifact_id("x", right, schema_version="v1"),
        )

    def test_dependency_dag_preserves_small_graph_error_semantics(self) -> None:
        graph = (
            DependencyNode("a", ()),
            DependencyNode("b", ("a",)),
            DependencyNode("c", ("a", "b")),
        )
        self.assertEqual(
            tuple(validate_dependency_dag(graph, "graph")),
            ("a", "b", "c"),
        )
        for nodes, message in (
            ((DependencyNode("a", ()), DependencyNode("a", ())), "duplicate id"),
            ((DependencyNode("a", ("missing",)),), "dangling dependency"),
            (
                (
                    DependencyNode("a", ("b",)),
                    DependencyNode("b", ("a",)),
                ),
                "dependency graph contains a cycle",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    validate_dependency_dag(nodes, "graph")

    def test_dependency_dag_accepts_a_5000_node_chain(self) -> None:
        nodes = tuple(
            DependencyNode(
                f"node_{index}",
                () if index == 0 else (f"node_{index - 1}",),
            )
            for index in range(5000)
        )
        self.assertEqual(len(validate_dependency_dag(nodes, "graph")), 5000)

    def test_dependency_dag_reports_a_cycle_after_a_deep_prefix(self) -> None:
        prefix = tuple(
            DependencyNode(
                f"node_{index}",
                () if index == 0 else (f"node_{index - 1}",),
            )
            for index in range(4998)
        )
        nodes = prefix + (
            DependencyNode("node_4998", ("node_4997", "node_4999")),
            DependencyNode("node_4999", ("node_4998",)),
        )
        with self.assertRaisesRegex(
            SchemaError, "dependency graph contains a cycle"
        ):
            validate_dependency_dag(nodes, "graph")

    def test_strict_json_rejects_top_level_and_nested_duplicate_keys(self) -> None:
        valid = loads_dataclass(
            NestedDocument,
            '{"nested":{"value":1}}',
            path="document",
        )
        self.assertEqual(valid, NestedDocument(NestedValue(1)))
        for text, error_path in (
            (
                '{"nested":{"value":1},"nested":{"value":2}}',
                "document.nested",
            ),
            ('{"nested":{"value":1,"value":2}}', "document.nested.value"),
        ):
            with self.subTest(error_path=error_path):
                with self.assertRaisesRegex(
                    SchemaError, f"{error_path}.*duplicate object key"
                ):
                    loads_dataclass(NestedDocument, text, path="document")

    def test_strict_json_rejects_nonfinite_decode_and_encode(self) -> None:
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                with self.assertRaisesRegex(
                    SchemaError, "document.nested.value.*non-finite JSON"
                ):
                    loads_dataclass(
                        NestedDocument,
                        f'{{"nested":{{"value":{token}}}}}',
                        path="document",
                    )
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    SchemaError, r"\$\.payload\[0\].*non-finite floats"
                ):
                    canonical_json({"payload": (value,)})


if __name__ == "__main__":
    unittest.main()
