from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership

from test_naive_intra_die import _component_projection, _complete_projection


class NaiveWireSafeSramTest(unittest.TestCase):
    def test_optional_wire_budget_reuses_a_seed_only_after_its_reader(self) -> None:
        ir1, projection = _component_projection()
        default = NaiveIntraDiePolicy().schedule(projection, ir1).schedules[0]
        compact = NaiveIntraDiePolicy(
            wire_address_limit_bytes=65536,
        ).schedule(projection, ir1).schedules[0]
        self.assertEqual(default.core_orders, compact.core_orders)
        self.assertEqual(default.placements, compact.placements)
        compact.validate_against(projection.dags[0], ir1)
        ordinary = {binding.value_id: binding for binding in default.buffer_bindings
                    if binding.core_id == 0}
        wire = {binding.value_id: binding for binding in compact.buffer_bindings
                if binding.core_id == 0}
        seed, after = wire["v_input"], wire["v_out_b"]
        self.assertIs(seed.ownership, BufferOwnership.BORROWED)
        self.assertIs(after.ownership, BufferOwnership.OWNED)
        self.assertEqual((seed.region_offset_bytes, after.region_offset_bytes),
                         (0, 0))
        self.assertLessEqual(seed.lifetime_end_exclusive, after.lifetime_start)
        self.assertEqual(ordinary["v_out_b"].region_offset_bytes, 16384)
        self.assertTrue(all(binding.region_offset_bytes + binding.size_bytes
                            <= 65536 for binding in compact.buffer_bindings))

    def test_real_oversized_staging_fails_before_record_or_finalizer(self) -> None:
        ir1, projection = _complete_projection(tp=2, large_sram=True)
        with self.assertRaisesRegex(SchemaError,
                                    "wire-addressable SRAM capacity exceeded"):
            NaiveIntraDiePolicy(
                wire_address_limit_bytes=65536,
            ).schedule(projection, ir1)

    def test_wire_budget_cannot_exceed_uint16_or_accept_unaligned_value(self) -> None:
        for invalid in (True, 0, 65537, 65535):
            with self.subTest(budget=invalid):
                with self.assertRaises(SchemaError):
                    NaiveIntraDiePolicy(wire_address_limit_bytes=invalid)


if __name__ == "__main__":
    unittest.main()
