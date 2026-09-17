"""Reject incomplete or overlapping Dense 100-shape release shards."""
from __future__ import annotations

import unittest

from llm.test.frontend.integration.audit_dense_native_mesh_release import (
    _SHAPES, check_release_partition,
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


if __name__ == "__main__":
    unittest.main()
