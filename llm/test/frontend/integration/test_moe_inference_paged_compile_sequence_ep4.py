"""True four-core MoE inference source relinks without changing its records."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.moe_full_model_compile_sequence import (
    compile_moe_full_model_inference_sequence,
)
from llm.frontend.wafer_frontend.passes.moe_inference_paged_compile_sequence_ep4 import (
    relink_moe_inference_paged_segment_ep4,
    unique_state_abis,
)
from llm.frontend.wafer_frontend.schema.workload_run import WorkloadFamily
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_moe_compile_sequence import _manifest
from llm.test.frontend.unit.test_moe_full_model_compile_sequence import _legacy_template


class MoePagedEp4RelinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fabric = physical_fabric_from_data(
            minimal_hardware(4, 1, sram_bytes=65536)
        )
        cls.sequence = compile_moe_full_model_inference_sequence(
            _manifest(WorkloadFamily.MOE_INFERENCE, rows=1, columns=4),
            _legacy_template(), cls.fabric,
            hbm_address_spaces=valid_hbm_address_spaces(cls.fabric),
        )

    def test_all_three_true_ep4_sources_relink(self) -> None:
        for step, segment in enumerate(self.sequence.segments):
            source = segment.executable_manifest
            paged = relink_moe_inference_paged_segment_ep4(source, step)
            self.assertEqual(len(unique_state_abis(paged)), 31)
            self.assertEqual(
                [stream.runtime_core_id for stream in paged.core_streams],
                [0, 4, 8, 12],
            )
            self.assertEqual(
                [len(stream.records) for stream in paged.core_streams],
                [len(stream.records) for stream in source.core_streams],
            )
            for abi in unique_state_abis(paged):
                relative = abi.address - (abi.die_id << 30)
                self.assertGreaterEqual(relative, 512)
                self.assertLessEqual(relative + abi.size_bytes, 960)
            paged.validate("test.ep4")

    def test_ep4_source_rejects_wrong_step(self) -> None:
        source = self.sequence.segments[0].executable_manifest
        with self.assertRaises(SchemaError):
            relink_moe_inference_paged_segment_ep4(source, 3)


if __name__ == "__main__":
    unittest.main()
