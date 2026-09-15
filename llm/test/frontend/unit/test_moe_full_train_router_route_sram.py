"""One real P2 route trace → 80B borrowed INT32 route SRAM, old leaf FAIL."""

from dataclasses import replace
import hashlib
import struct
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_router_native_protocol import (
    build_moe_router_native_protocol,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_return_protocol import (
    build_moe_router_signed_return_protocol,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_route_sram import (
    build_moe_router_route_sram,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_score_source import (
    build_moe_trainable_signed_router_requirements,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeFullTrainRouterRouteSramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.score = build_moe_trainable_signed_router_requirements(
            Fixture.sequence)
        cls.returned = build_moe_router_signed_return_protocol(
            cls.score, Fixture.sequence)
        cls.native = build_moe_router_native_protocol(
            cls.score, cls.returned, Fixture.sequence)
        cls.route = build_moe_router_route_sram(
            cls.score, cls.returned, cls.native, Fixture.sequence)

    def test_four_step_layer_route_blobs_are_one_nonzero_little_endian_trace(self):
        self.route.validate_against(self.score, self.returned,
                                    self.native, Fixture.sequence)
        self.assertEqual(len(self.route.routes), 4)
        for seed in self.route.routes:
            self.assertEqual((seed.source_rank, seed.route_rows,
                              seed.route_bytes), (0, 4, 80))
            actual = [struct.unpack_from("<IIIII", seed.payload, index * 20)
                      for index in range(4)]
            self.assertEqual(actual, [(0, 0, 0, 0, 0),
                                      (1, 0, 1, 1, 0),
                                      (2, 0, 0, 0, 1),
                                      (3, 0, 1, 1, 1)])
            self.assertEqual(hashlib.sha256(seed.payload).hexdigest(),
                             seed.sha256)
            self.assertEqual(seed.payload[-4:], b"\x01\x00\x00\x00")

    def test_missing_native_route_reader_remains_fail_closed(self):
        with self.assertRaisesRegex(SchemaError,
                                    "public native route address roles are not implemented"):
            self.route.require_physical_route_reads(
                self.score, self.returned, self.native,
                Fixture.sequence, {})

    def test_last_token_or_selected_expert_source_rewrite_rejected(self):
        seed = self.route.routes[0]
        for byte_index in (60, 68, 76):
            altered = bytearray(seed.payload)
            altered[byte_index] ^= 1
            fake = replace(seed, payload_hex=bytes(altered).hex(),
                           sha256=hashlib.sha256(altered).hexdigest())
            with self.subTest(byte=byte_index), self.assertRaisesRegex(
                    SchemaError, "one frozen P2 source trace"):
                replace(self.route, routes=(fake,
                    *self.route.routes[1:])).validate_against(
                        self.score, self.returned, self.native,
                        Fixture.sequence)

    def test_old_case_and_short_int32_buffer_fail_source_gate(self):
        fake_score = replace(self.score,
            dynamic_score_case_ref=self.score.original_static_case_ref)
        with self.assertRaisesRegex(SchemaError,
                                    "source/hardware/route/score contract"):
            build_moe_router_route_sram(fake_score,
                self.returned, self.native, Fixture.sequence)
        seed = replace(self.route.routes[0], route_bytes=40)
        with self.assertRaisesRegex(SchemaError,
                                    "one frozen P2 source trace"):
            replace(self.route, routes=(seed,
                *self.route.routes[1:])).validate_against(
                    self.score, self.returned, self.native,
                    Fixture.sequence)


if __name__ == "__main__":
    unittest.main()
