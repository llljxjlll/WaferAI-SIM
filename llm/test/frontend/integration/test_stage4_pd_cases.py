from __future__ import annotations

from collections import Counter
import json
import unittest

from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    FragmentKind,
    ProgramSymbolKind,
)
from llm.frontend.wafer_frontend.schema.ir2 import StateTransferOrigin
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoTargetKind
from llm.frontend.wafer_frontend.schema.stage4_pd import (
    Stage4KvReshardKind,
    Stage4PdMode,
)

from stage4_pd_cases import Stage4PdCaseKind, build_stage4_pd_case


class Stage4PdCasesTest(unittest.TestCase):
    def test_pds_tp1_is_self_contained_exact_and_deterministic(self) -> None:
        case = build_stage4_pd_case(Stage4PdCaseKind.PDS)
        self.assertEqual(case.pd_plan.mode, Stage4PdMode.SEPARATED)
        self.assertEqual(len(case.graph.cross_routes), 1)
        self.assertEqual(
            len(case.profile.lowering_context.projection.state_transfers),
            4,
        )
        self.assertEqual(len(case.global_carrier.global_dag.actions), 100)
        self.assertEqual(len(case.lowered.fragments), 96)
        self.assertEqual(
            (
                len(case.manifest.fragments),
                len(case.manifest.address_operand_bindings),
                len(case.manifest.state_operand_bindings),
                len(case.manifest.runtime_symbol_definitions),
                len(case.manifest.program_symbol_definitions),
                len(case.manifest.fragment_interfaces),
                len(case.manifest.core_streams),
            ),
            (96, 564, 38, 18, 220, 96, 2),
        )
        self.assertEqual(
            Counter(
                definition.symbol.kind
                for definition in case.manifest.program_symbol_definitions
            ),
            {
                ProgramSymbolKind.ABSOLUTE_ADDRESS: 128,
                ProgramSymbolKind.SRAM_LABEL: 90,
                ProgramSymbolKind.SRAM_REGION: 2,
            },
        )
        self.assertEqual(
            {
                definition.name
                for definition in case.manifest.program_symbol_definitions
                if definition.symbol.kind is ProgramSymbolKind.SRAM_REGION
            },
            {"comm", "double_a"},
        )
        region_names = {
            region.id: region.name
            for profile in case.graph.fabric.sram_profiles
            for region in profile.regions
        }
        self.assertEqual(
            Counter(
                region_names[binding.region_ref]
                for schedule in case.global_carrier.schedule_set.schedules
                for binding in schedule.buffer_bindings
            ),
            {"double_a": 82, "comm": 8},
        )
        self.assertEqual(
            sum(
                isinstance(action.origin_ref, StateTransferOrigin)
                for action in case.global_carrier.global_dag.actions
            ),
            12,
        )
        self.assertEqual(
            (len(case.state_seed_refs), len(case.state_expected_refs)),
            (30, 0),
        )
        self.assertEqual(
            (
                len(case.program_io.initializations),
                len(case.program_io.output_probes),
            ),
            (120, 2),
        )
        self.assertEqual(
            sum(
                item.target.kind is ProgramIoTargetKind.HBM
                for item in case.program_io.initializations
            ),
            30,
        )
        self.assertTrue(
            all(
                item.target.kind is ProgramIoTargetKind.SRAM
                for item in case.program_io.output_probes
            )
        )
        self.assertEqual(build_stage4_pd_case(Stage4PdCaseKind.PDS), case)

    def test_fused_tp1_uses_the_same_formal_carrier_path(self) -> None:
        case = build_stage4_pd_case(Stage4PdCaseKind.FUSED)
        self.assertEqual(case.pd_plan.mode, Stage4PdMode.FUSED)
        self.assertEqual(case.graph.cross_routes, ())
        self.assertEqual(
            case.profile.lowering_context.projection.state_transfers,
            (),
        )
        self.assertEqual(len(case.global_carrier.global_dag.actions), 92)
        self.assertEqual(len(case.lowered.fragments), 92)
        self.assertEqual(len(case.manifest.fragments), 92)

    def test_pdr_tp2_to_tp1_uses_production_segmented_chain(self) -> None:
        case = build_stage4_pd_case(Stage4PdCaseKind.PDR)
        transfers = case.profile.lowering_context.projection.state_transfers
        leaves = case.profile.leaf_fragments
        self.assertEqual(
            (
                case.pd_plan.mode,
                case.pd_plan.reshard,
                case.pd_plan.prefill_tp,
                case.pd_plan.decode_tp,
            ),
            (
                Stage4PdMode.SEPARATED,
                Stage4KvReshardKind.GATHER,
                2,
                1,
            ),
        )
        self.assertEqual(
            (
                len(case.graph.fabric.dies),
                len(case.graph.cross_routes),
                len(transfers),
                sum(len(contract.segments) for contract in transfers),
                sum(contract.bytes for contract in transfers),
            ),
            (3, 2, 8, 64, 1024),
        )
        self.assertEqual(
            tuple(route.die_path for route in case.graph.cross_routes),
            ((0, 1, 2), (1, 2)),
        )
        self.assertEqual(
            tuple(
                (
                    schedule.die_id,
                    len(schedule.placements),
                    len(schedule.buffer_bindings),
                    len(schedule.flow_routes),
                )
                for schedule in case.global_carrier.schedule_set.schedules
            ),
            ((0, 112, 69, 48), (1, 112, 69, 80), (2, 172, 45, 64)),
        )
        self.assertEqual(len(case.global_carrier.global_dag.actions), 428)
        self.assertEqual(
            sum(
                isinstance(action.origin_ref, StateTransferOrigin)
                for action in case.global_carrier.global_dag.actions
            ),
            224,
        )
        endpoints = tuple(
            fragment
            for fragment in case.lowered.fragments
            if type(fragment) is CommandFragment
            and fragment.kind is FragmentKind.STATE_TRANSFER
        )
        self.assertEqual((len(case.lowered.fragments), len(endpoints)), (152, 16))
        self.assertEqual(
            (
                len(case.manifest.fragments),
                sum(
                    len(stream.records)
                    for fragment in leaves
                    for stream in fragment.core_streams
                ),
                len(case.manifest.address_operand_bindings),
                len(case.manifest.state_operand_bindings),
                len(case.manifest.runtime_symbol_definitions),
                len(case.manifest.program_symbol_definitions),
                len(case.manifest.fragment_interfaces),
                len(case.manifest.core_streams),
            ),
            (152, 971, 1210, 57, 504, 433, 152, 3),
        )
        self.assertEqual(
            (len(case.state_seed_refs), len(case.state_expected_refs)),
            (45, 0),
        )
        self.assertEqual(
            (
                len(case.program_io.initializations),
                len(case.program_io.output_probes),
                sum(
                    item.target.kind is ProgramIoTargetKind.HBM
                    for item in case.program_io.initializations
                ),
                sum(
                    item.target.kind is ProgramIoTargetKind.SRAM
                    for item in case.program_io.initializations
                ),
            ),
            (228, 3, 45, 183),
        )
        self.assertTrue(
            all(
                item.target.kind is ProgramIoTargetKind.SRAM
                for item in case.program_io.output_probes
            )
        )
        hardware = json.loads(case.runtime_hardware_inputs.hardware_json)
        regions = hardware["memory"]["sram"]["regions"]
        self.assertEqual(
            (
                hardware["die"]["x"],
                len(hardware["memory_system"]["hbm_stacks"]),
                len(
                    hardware["memory_system"]["address_policy"][
                        "home_ranges"
                    ]
                ),
            ),
            (3, 3, 3),
        )
        self.assertEqual(
            tuple(
                (
                    region["name"],
                    region["base_bytes"],
                    region["size_bytes"],
                    region["allocator"],
                )
                for region in regions
            ),
            (
                ("double_a", 0, 14656, "block"),
                ("double_b", 14656, 64, "block"),
                ("input", 14720, 64, "block"),
                ("intermediate", 14784, 64, "block"),
                ("comm", 14848, 3584, "block"),
            ),
        )
        self.assertEqual(
            (
                hardware["memory"]["sram_size"],
                hardware["memory"]["sram"]["capacity_bytes"],
            ),
            (18432, 18432),
        )
        names = {
            region.id: region.name
            for profile in case.graph.fabric.sram_profiles
            for region in profile.regions
        }
        high_water = {}
        for schedule in case.global_carrier.schedule_set.schedules:
            for binding in schedule.buffer_bindings:
                name = names[binding.region_ref]
                high_water[name] = max(
                    high_water.get(name, 0),
                    binding.region_offset_bytes + binding.size_bytes,
                )
        self.assertEqual(high_water, {"comm": 3584, "double_a": 14656})

    def test_non_enum_kind_fails_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "Stage4PdCaseKind"):
            build_stage4_pd_case("pds")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
