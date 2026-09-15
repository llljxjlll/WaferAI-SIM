"""Resident-only HBM must reject actual linked Dense parameter StateABIs."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_training_physical_offload_source import (
    build_dense_training_physical_offload_source,
)
from llm.frontend.wafer_frontend.passes.dense_training_physical_resident_capacity import (
    reject_physically_resident_dense_training,
)

from .run_dense_training_sequence_runtime_canary import _offload_sequence


class PhysicalDenseResidentCapacityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sequence = _offload_sequence()
        cls.sequence.validate()
        cls.source = build_dense_training_physical_offload_source(
            cls.sequence, hbm_capacity_bytes=34880,
            external_capacity_bytes=32768,
        )

    def test_native_resident_planner_rejects_all_real_linked_states(self) -> None:
        rejection = reject_physically_resident_dense_training(
            self.sequence, self.source,
        )
        self.assertEqual(rejection.linked_state_abi_count, 15)
        self.assertEqual(rejection.linked_state_logical_bytes, 17344)
        self.assertEqual(rejection.linked_state_padding_bytes, 0)
        self.assertEqual(rejection.hbm_capacity_bytes_per_die, 34880)
        self.assertEqual(rejection.rejection_code, "memory_capacity_exceeded")
        self.assertEqual(rejection.source_digest, self.source.digest)
        self.assertEqual(rejection.workload_request_digest,
                         self.sequence.materialization.request.digest)
        self.assertTrue(rejection.state_inventory_digest)
        self.assertTrue(rejection.hbm_requests_digest)

    def test_forged_linked_address_and_inventory_fail_before_capacity(self) -> None:
        declaration = self.source.declarations[0]
        for forged in (
            replace(declaration, hbm_address=declaration.hbm_address + 16),
            replace(declaration, logical_bytes=declaration.logical_bytes + 1),
        ):
            source = replace(self.source, declarations=(
                forged, *self.source.declarations[1:],
            ))
            with self.subTest(forged=forged), self.assertRaises(SchemaError) as caught:
                reject_physically_resident_dense_training(self.sequence, source)
            self.assertIn("declared parameter does not match linked StateABI",
                          str(caught.exception))


if __name__ == "__main__":
    unittest.main()
