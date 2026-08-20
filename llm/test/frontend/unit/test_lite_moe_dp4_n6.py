from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.lowering.lite_moe_dp4 import (
    lower_lite_moe_dp4_infer,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4 import (
    build_lite_moe_dp4_ir0_adapter,
    build_lite_moe_dp4_oracle,
    build_lite_moe_dp4_spec,
    build_lite_moe_dp4_topology,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_execution import (
    build_lite_moe_dp4_execution_case,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_n6 import (
    build_lite_moe_dp4_infer_n6_intent,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_link_program import (
    link_lite_moe_dp4_backward_program,
    link_lite_moe_dp4_infer_program,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_lower_program import (
    lower_lite_moe_dp4_backward_program,
    lower_lite_moe_dp4_train_forward_program,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_backward import (
    build_lite_moe_dp4_backward,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_train_forward import (
    build_lite_moe_dp4_train_forward,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.policies.registry import production_registry
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    FragmentKind,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.lowering.lite_moe_dp4_linker import (
    link_lite_moe_dp4_train_forward_manifest,
)
from llm.frontend.wafer_frontend.schema.lite_moe_dp4_n6 import (
    LiteMoeDp4InferLoweredProgram,
)
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanningContext,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.policy import RegistryKind
from llm.test.frontend.integration.lite_moe_cases import build_lite_moe_source_case


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x2.json"


def _case():
    source = build_lite_moe_source_case()
    spec = build_lite_moe_dp4_spec(source.moe_spec.trace)
    topology = build_lite_moe_dp4_topology(spec)
    oracle = build_lite_moe_dp4_oracle(spec, topology)
    adapter = build_lite_moe_dp4_ir0_adapter(
        source.spec, spec, topology, oracle
    )
    hardware = json.loads(_HARDWARE.read_text(encoding="utf-8"))
    placement = PlacementContext.create(
        producer_pass="lite_moe_dp4_n6_test",
        fabric=physical_fabric_from_data(hardware),
        placement=source.spec.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(hardware),
    )
    registry = production_registry()
    partition = FusionPartitionContext.create(
        producer_pass="lite_moe_dp4_n6_test"
    )
    planning = InterDiePlanningContext.create(
        producer_pass="lite_moe_dp4_n6_test",
        fused_policy=registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"
        ).selection,
    )
    return build_lite_moe_dp4_execution_case(
        source.spec, adapter, placement, partition, planning
    )


class LiteMoeDp4N6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = _case()
        cls.intent = build_lite_moe_dp4_infer_n6_intent(cls.source)
        cls.fragments = lower_lite_moe_dp4_infer(cls.intent, cls.source)

    def test_production_infer_intent_and_lowering(self) -> None:
        self.assertEqual(len(self.intent.buffer_abis), 68)
        self.assertEqual(
            (len(self.intent.state_loads), len(self.intent.compute_units), len(self.intent.dte_units)),
            (24, 32, 12),
        )
        self.assertEqual(len(self.fragments), 80)
        self.assertEqual(
            Counter(fragment.kind for fragment in self.fragments),
            Counter({FragmentKind.STATE_IO: 24, FragmentKind.COARSE: 32, FragmentKind.MOE_TRANSFER: 24}),
        )
        lowered = LiteMoeDp4InferLoweredProgram.create(
            source=self.source,
            intent=self.intent,
            fragments=self.fragments,
        )
        lowered.validate()
        linked = link_lite_moe_dp4_infer_program(lowered)
        linked.validate()
        self.assertEqual(len(linked.manifest.fragments), 80)
        self.assertEqual(len(linked.manifest.core_streams), 4)

    def test_production_train_forward_tape_lowering(self) -> None:
        source = build_lite_moe_dp4_train_forward(self.source)
        lowered = lower_lite_moe_dp4_train_forward_program(source)
        self.assertEqual(len(lowered.fragments), 88)
        self.assertEqual(len(lowered.tape_buffer_abis), 8)
        self.assertEqual(len(lowered.tape_fragments), 8)
        self.assertEqual(
            Counter(
                record.opcode
                for fragment in lowered.tape_fragments
                for record in fragment.core_streams[0].records
            ),
            Counter({
                # One terminal allocation and one local-copy issue/wait per tape.
                # No FREE: ProgramIo must be able to probe the produced tape.
                RecordOpcode.SRAM_ALLOC_AT: 8,
                RecordOpcode.DTE_ISSUE: 8,
                RecordOpcode.DTE_WAIT: 8,
            }),
        )
        manifest = link_lite_moe_dp4_train_forward_manifest(lowered)
        self.assertEqual(len(manifest.fragments), 88)
        self.assertEqual(len(manifest.core_streams), 4)

    def test_production_backward_lower_and_single_manifest(self) -> None:
        source = build_lite_moe_dp4_backward(
            build_lite_moe_dp4_train_forward(self.source)
        )
        lowered = lower_lite_moe_dp4_backward_program(source)
        self.assertEqual(len(lowered.fragments), 32)
        self.assertEqual(
            Counter(fragment.kind for fragment in lowered.fragments),
            Counter({
                FragmentKind.STATE_IO: 8,
                FragmentKind.COARSE: 12,
                FragmentKind.MOE_TRANSFER: 12,
            }),
        )
        self.assertEqual(
            sum(
                len(stream.records)
                for fragment in lowered.fragments
                for stream in fragment.core_streams
            ),
            114,
        )
        linked = link_lite_moe_dp4_backward_program(lowered)
        manifest = linked.manifest
        self.assertEqual(len(manifest.fragments), 32)
        self.assertEqual(len(manifest.core_streams), 4)
        self.assertEqual(len(manifest.address_operand_bindings), 190)
        self.assertEqual(len(manifest.state_operand_bindings), 8)
        self.assertEqual(len(manifest.runtime_symbol_definitions), 28)
        self.assertEqual(len(manifest.program_symbol_definitions), 73)
        self.assertEqual(len(manifest.input_digests), 37)


if __name__ == "__main__":
    unittest.main()
