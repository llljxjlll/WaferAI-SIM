"""Reject incomplete or overlapping Dense 100-shape release shards."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from llm.test.frontend.integration.audit_dense_native_mesh_release import (
    _SHAPES, audit_fresh_source_tool, check_release_partition,
)


class DenseReleasePartitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bindings = tuple({
            "shard_index": index,
            "shard_count": 4,
            "shapes": list(_SHAPES[index::4]),
        } for index in range(4))

    def test_complete_disjoint_canonical_partition(self) -> None:
        check_release_partition(self.bindings)
        self.assertEqual(sum(len(item["shapes"]) for item in self.bindings), 100)

    def test_missing_duplicate_reordered_and_extra_are_rejected(self) -> None:
        for mutate in (
            lambda b: b[0]["shapes"].pop(),
            lambda b: b[1]["shapes"].__setitem__(0, b[0]["shapes"][0]),
            lambda b: b[2]["shapes"].reverse(),
            lambda b: b[3]["shapes"].append("11x11"),
            lambda b: b[3].__setitem__("shard_index", 2),
            lambda b: b[0].__setitem__("shard_count", 5),
        ):
            with self.subTest(mutate=mutate):
                binding = [dict(item, shapes=list(item["shapes"])) for item in self.bindings]
                mutate(binding)
                with self.assertRaises(ValueError):
                    check_release_partition(tuple(binding))


class DenseReleaseSourceToolTest(unittest.TestCase):
    def test_fresh_must_bind_exact_frozen_source_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            fresh = root / "fresh0"
            (source / "pkg").mkdir(parents=True)
            fresh.mkdir()
            module = source / "pkg" / "driver.py"
            module.write_text("value = 1\n", encoding="utf-8")
            digest = hashlib.sha256(module.read_bytes()).hexdigest()
            binding = {
                "source_tool_at_entry": {
                    "tool_sha256": {"npusim": "native-digest"},
                    "imported_python_sha256": {"pkg/driver.py": digest},
                },
                "additional_imported_python_sha256": {},
            }
            sidecar = fresh / "source_tool_binding.json"
            sidecar.write_text(json.dumps(binding), encoding="utf-8")
            audit_fresh_source_tool(fresh, source, {"npusim": "native-digest"}, {})
            module.write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source bytes differ"):
                audit_fresh_source_tool(fresh, source, {"npusim": "native-digest"}, {})
            module.write_text("value = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tool bytes differ"):
                audit_fresh_source_tool(fresh, source, {"npusim": "other"}, {})
            binding["source_tool_at_entry"]["imported_python_sha256"] = {
                "../outside.py": digest,
            }
            sidecar.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "path or digest is invalid"):
                audit_fresh_source_tool(fresh, source, {"npusim": "native-digest"}, {})


if __name__ == "__main__":
    unittest.main()
