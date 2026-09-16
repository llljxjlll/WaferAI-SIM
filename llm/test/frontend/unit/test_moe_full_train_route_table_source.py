"""Full-model route bytes must be exactly the frozen P2 physical trace."""

from dataclasses import replace
import hashlib
import struct
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_compile_sequence import compile_moe_sequence
from llm.frontend.wafer_frontend.passes.moe_full_train_forward_ir0 import (
    build_moe_full_train_forward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_route_table_source import (
    build_moe_full_train_route_table_source,
)
from llm.frontend.wafer_frontend.schema.workload_run import WorkloadFamily
from llm.test.frontend.unit.test_moe_compile_sequence import _manifest
from llm.test.frontend.unit.test_moe_full_train_forward_ir0 import (
    MoeFullTrainForwardIr0Test as Fixture,
)


class MoeFullTrainRouteTableSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.dense = Fixture.forward
        cls.ep2 = Fixture.sequence
        cls.ep1 = compile_moe_sequence(
            _manifest(WorkloadFamily.MOE_TRAINING, rows=1, columns=1),
            source_rank_policy="rank0_shared_spine",
        )

    def test_both_layers_and_ep_degrees_serialize_one_full_native_route(self):
        for sequence in (self.ep1, self.ep2):
            phase = build_moe_full_train_forward_ir0(self.dense, sequence)
            source = build_moe_full_train_route_table_source(phase, sequence)
            source.validate_against(phase, sequence)
            self.assertEqual(len(source.seeds), 2)
            for seed in source.seeds:
                self.assertEqual(seed.route_rows, 4)
                self.assertEqual(seed.route_bytes, 80)
                self.assertEqual(seed.payload_sha256,
                                 hashlib.sha256(seed.payload).hexdigest())
                unit = next(unit for unit in sequence.units
                            if unit.step == seed.step and unit.layer == seed.layer)
                expected = tuple((a.token_index, a.source_rank, a.expert_index,
                                  a.expert_home_rank, a.slot_index)
                                 for a in sorted(unit.spec.trace.assignments,
                                                 key=lambda a: a.token_index))
                self.assertEqual(tuple(struct.unpack_from("<IIIII", seed.payload,
                                                          row * 20)
                                       for row in range(seed.route_rows)), expected)
                self.assertEqual(seed.route_value_ref,
                                 f"T0.layer{seed.layer}.moe.route_ids")

    def test_single_route_field_forgery_fails_source_reopen(self):
        phase = build_moe_full_train_forward_ir0(self.dense, self.ep2)
        source = build_moe_full_train_route_table_source(phase, self.ep2)
        forged = bytearray(source.seeds[0].payload)
        forged[8] ^= 1  # selected expert in token 0, not another trace.
        first = replace(source.seeds[0], payload_hex=forged.hex())
        with self.assertRaisesRegex(SchemaError, "frozen trace"):
            replace(source, seeds=(first, source.seeds[1])).validate_against(
                phase, self.ep2)

    def test_route_source_rejects_a_different_sequence(self):
        phase = build_moe_full_train_forward_ir0(self.dense, self.ep1)
        with self.assertRaisesRegex(SchemaError, "sequence differs"):
            build_moe_full_train_route_table_source(phase, self.ep2)


if __name__ == "__main__":
    unittest.main()
