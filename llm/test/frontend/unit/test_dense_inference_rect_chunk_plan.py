"""A paged DMA range cannot conceal an untiled full-span native LSU read."""
from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.dense_inference_rect_chunk_plan import (
    plan_bounded_parameter_chunks,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import StateABI, StateKind
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess, PersistentStateLifetime,
)


def _state() -> StateABI:
    return StateABI.create(
        state_ref="real_tp6_parameter_0", hbm_binding_ref="real_hbm_binding_0",
        kind=StateKind.PARAMETER, lifetime=PersistentStateLifetime.PERSISTENT,
        access=PersistentStateAccess.READ_ONLY,
        shape=(48, 128), dtype=DType.FP16, layout="HV_replicated",
        die_id=0, address=65536, size_bytes=12288, alignment_bytes=64,
    )


class DenseRectChunkPlanTest(unittest.TestCase):
    def test_real_source_byte_ranges_cover_state_without_overlap(self) -> None:
        state = _state()
        plan = plan_bounded_parameter_chunks(
            (state,), {state.id: (8192, 4096)},
            hbm_capacity_bytes_per_die=12288, reserved_by_die={0: ((8192, 12288),)},
        )
        self.assertFalse(plan.native_execution_admitted)
        self.assertEqual(
            [(part.source_offset_bytes, part.size_bytes, part.hbm_address)
             for part in plan.chunks], [(0, 8192, 0), (8192, 4096, 0)],
        )

    def test_full_lsu_read_cannot_be_claimed_as_executable_chunks(self) -> None:
        state = _state()
        with self.assertRaisesRegex(UnsupportedFeatureError, "requires_tiled_consumer"):
            plan_bounded_parameter_chunks(
                (state,), {state.id: (12288,)},
                hbm_capacity_bytes_per_die=12288,
                reserved_by_die={0: ((8192, 12288),)},
            )

    def test_missing_consumer_and_invalid_reserved_home_fail_closed(self) -> None:
        state = _state()
        with self.assertRaisesRegex(SchemaError, "bound LSU consumers"):
            plan_bounded_parameter_chunks(
                (state,), {}, hbm_capacity_bytes_per_die=12288,
                reserved_by_die={0: ((8192, 12288),)},
            )
        with self.assertRaisesRegex(SchemaError, "reservations overlap"):
            plan_bounded_parameter_chunks(
                (state,), {state.id: (4096,)}, hbm_capacity_bytes_per_die=12288,
                reserved_by_die={0: ((0, 4096), (2048, 8192))},
            )

    def test_stale_duplicate_state_abi_cannot_erase_parameter(self) -> None:
        state = _state()
        with self.assertRaisesRegex(SchemaError, "duplicated"):
            plan_bounded_parameter_chunks(
                (state, state), {state.id: (4096,)}, hbm_capacity_bytes_per_die=12288,
                reserved_by_die={0: ((8192, 12288),)},
            )


if __name__ == "__main__":
    unittest.main()
