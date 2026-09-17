from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
    compile_dense_e2e_sequence,
)
from llm.frontend.wafer_frontend.passes.dense_inference_rect_physical_source import (
    build_dense_inference_rect_physical_source,
)
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryObjectKind,
    MemoryTier,
)
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_dense_compile_sequence import _two_by_two_case


class DenseInferenceRectPhysicalSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        resident, template, fabric = _two_by_two_case()
        with builder_validation_session():
            sequence = compile_dense_e2e_sequence(
                resident,
                template,
                fabric,
                hbm_address_spaces=valid_hbm_address_spaces(fabric),
            )
        cls.source = build_dense_inference_rect_physical_source(sequence)

    def test_reconciles_all_true_tp4_physical_state_abis(self) -> None:
        source = self.source
        self.assertEqual(len(source.declarations), 76)
        self.assertEqual(
            sum(item.physical_bytes for item in source.declarations),
            113920,
        )
        self.assertEqual(
            tuple(
                (
                    die,
                    sum(item.role == "parameter" for item in source.declarations
                        if item.die_id == die),
                    sum(item.role != "parameter" for item in source.declarations
                        if item.die_id == die),
                    sum(item.physical_bytes for item in source.declarations
                        if item.die_id == die and item.role == "parameter"),
                    sum(item.physical_bytes for item in source.declarations
                        if item.die_id == die and item.role != "parameter"),
                )
                for die in range(4)
            ),
            (
                (0, 15, 4, 26944, 1536),
                (1, 15, 4, 26944, 1536),
                (2, 15, 4, 26944, 1536),
                (3, 15, 4, 26944, 1536),
            ),
        )
        self.assertEqual(
            len({item.external_allocation_ref for item in source.declarations}),
            76,
        )
        self.assertEqual(
            len({item.linked_state_abi_ids for item in source.declarations}),
            76,
        )

    def test_same_model_resident_is_rejected_at_12288_bytes_per_die(self) -> None:
        source = self.source
        self.assertIn("memory_capacity_exceeded", source.resident_rejection)
        self.assertEqual(
            tuple(item.capacity_bytes for item in source.hbm_capacities),
            (12288, 12288, 12288, 12288),
        )
        self.assertEqual(source.p3_parameter_bytes_per_die, 14416)
        self.assertEqual(source.physical_parameter_bytes_per_die, 26944)
        self.assertEqual(source.physical_kv_bytes_per_die, 1536)

    def test_external_manifest_uses_physical_abi_sized_state(self) -> None:
        source = self.source
        physical_ids = {item.inventory_ref for item in source.declarations}
        inventory = {
            item.id: item for item in source.external_manifest.state_inventory
        }
        self.assertEqual(set(inventory).intersection(physical_ids), physical_ids)
        self.assertEqual(
            sum(inventory[item].size_bytes for item in physical_ids),
            113920,
        )
        self.assertTrue(all(
            inventory[item].object_kind in (
                MemoryObjectKind.PARAMETER, MemoryObjectKind.KV,
            )
            for item in physical_ids
        ))
        requests = {
            item.id: item
            for item in source.external_manifest.memory_plan.requests
        }
        allocations = {
            item.id: item
            for item in source.external_manifest.memory_plan.allocations
        }
        self.assertTrue(all(
            requests[allocations[item.external_allocation_ref].request_ref].tier
            is MemoryTier.EXTERNAL
            for item in source.declarations
        ))


if __name__ == "__main__":
    unittest.main()
