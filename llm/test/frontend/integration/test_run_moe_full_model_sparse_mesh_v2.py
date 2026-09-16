"""Fail-closed route checks for the versioned sparse MoE inference adapter."""

import unittest

from llm.test.frontend.integration import run_moe_full_model_sparse_mesh_v2 as sparse


def _binding(source: int, destination: int, columns: int) -> dict:
    flows = [{
        "id": "test_flow",
        "source_rank": source,
        "destination_rank": destination,
        "logical_bytes": 8,
    }]
    links, _ = sparse.canonical._recompute_flow_evidence(
        flows, rows=10, columns=columns,
    )
    return {"expected_remote_flows": flows, "expected_d2d_links": links}


class SparseMeshRouteTest(unittest.TestCase):
    def test_same_row_route_is_identical(self) -> None:
        binding = _binding(0, 3, 10)
        self.assertEqual(
            sparse._flow_route_binding(binding, "10x10", "11x11"),
            binding["expected_d2d_links"],
        )

    def test_row_wrap_change_fails_closed(self) -> None:
        binding = _binding(0, 99, 10)
        with self.assertRaisesRegex(ValueError, "not route-equivalent"):
            sparse._flow_route_binding(binding, "10x10", "11x11")

    def test_signed_logical_links_cannot_be_replaced(self) -> None:
        binding = _binding(0, 3, 10)
        binding["expected_d2d_links"] = []
        with self.assertRaisesRegex(ValueError, "not route-equivalent"):
            sparse._flow_route_binding(binding, "10x10", "11x11")


if __name__ == "__main__":
    unittest.main()
