"""Real Dense terminal timing outputs need unshared physical SRAM bytes."""

from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
    compile_dense_e2e_sequence_runtime_profiles,
)
from llm.frontend.wafer_frontend.schema._validation_session import (
    builder_validation_session,
)
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.test.frontend.integration.run_dense_sequence_runtime_canary import (
    _all_die_scaled_model_case,
)
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces


class DenseTerminalSramLifetimeTest(unittest.TestCase):
    def test_all_die_single_die_terminal_never_reuses_earlier_bytes(self) -> None:
        manifest, template, fabric = _all_die_scaled_model_case(1, 1)
        with builder_validation_session():
            sequence, profiles = compile_dense_e2e_sequence_runtime_profiles(
                manifest, template, fabric,
                hbm_address_spaces=valid_hbm_address_spaces(fabric),
                intra_die_wire_address_limit_bytes=65536,
            )
        self.assertEqual(len(sequence.segments), 3)
        for index, profile in enumerate(profiles):
            terminals = {value.id for value in profile.lowering_context.ir1.values
                         if not value.consumers}
            roots = [binding for schedule in
                     profile.lowering_context.schedule_set.schedules
                     for binding in schedule.buffer_bindings
                     if binding.ownership is not BufferOwnership.ALIASED]
            outputs = [item for item in roots if item.value_id in terminals
                       and item.ownership is BufferOwnership.OWNED]
            self.assertTrue(outputs, f'segment {index} lacks terminal output')
            for output in outputs:
                overlaps = [other.value_id for other in roots
                            if other.id != output.id
                            and other.core_id == output.core_id
                            and other.region_ref == output.region_ref
                            and other.region_offset_bytes
                            < output.region_offset_bytes + output.size_bytes
                            and output.region_offset_bytes
                            < other.region_offset_bytes + other.size_bytes]
                self.assertEqual(overlaps, [],
                                 f'segment {index} {output.value_id} reused SRAM')


if __name__ == '__main__':
    unittest.main()
