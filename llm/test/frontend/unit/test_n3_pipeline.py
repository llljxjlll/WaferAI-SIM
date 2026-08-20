from __future__ import annotations

from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.passes import (
    PassManager,
    PipelinePhase,
    build_ir0,
    load_physical_fabric,
    logical_expand,
    place_bundle,
    validate_placement_against,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.placed_ir1 import PlacedIR1Bundle
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)

from _fixtures import valid_hbm_address_spaces, valid_spec


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "llm/test/sram/hardware_numa.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"


def compile_through_n3():
    spec = from_data(ExperimentSpec, valid_spec(), path="spec")
    manager = PassManager()
    template = manager.run_pass("build_ir0", spec, build_ir0)
    expanded = manager.run_pass("logical_expand", template, logical_expand)
    fabric = load_physical_fabric(_HARDWARE, _MAPPING)
    context = PlacementContext.create(
        producer_pass="load_physical_fabric",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    placed = manager.run_pass(
        "placement",
        expanded,
        place_bundle,
        context=context,
    )
    return spec, manager, expanded, context, placed


class N3PipelineTest(unittest.TestCase):
    def test_real_hardware_pipeline_provenance_roundtrip_and_numeric_gate(self) -> None:
        spec, manager, expanded, context, placed = compile_through_n3()
        self.assertEqual(manager.snapshot.phase, PipelinePhase.IR1_PLACED)
        self.assertEqual(len(manager.snapshot.receipts), 3)
        placement_receipt = manager.snapshot.receipts[-1]
        self.assertEqual(placement_receipt.pass_name, "placement")
        self.assertEqual(
            placement_receipt.input_digest,
            manager.snapshot.receipts[-2].output_digest,
        )
        self.assertEqual(placement_receipt.context_digest, canonical_digest(context))
        self.assertEqual(placement_receipt.output_digest, canonical_digest(placed))
        validate_placement_against(placed, expanded, context)

        decoded = loads_dataclass(
            PlacedIR1Bundle,
            canonical_json(placed),
            path="placed_ir1_bundle",
        )
        self.assertEqual(decoded, placed)
        validate_placement_against(decoded, expanded, context)

        graph = placed.entries[0].graph
        group = graph.groups[0]
        self.assertEqual(
            tuple(item.die_id for item in group.placements),
            (0, 1),
        )
        self.assertEqual(
            tuple(route.die_path for route in group.embedding.routes),
            ((0, 1), (1, 0)),
        )
        self.assertEqual(
            group.embedding.canonical_profiles[0].lane_eq_bandwidth,
            16.0,
        )
        self.assertEqual(graph.fused_op_skeletons, ())
        self.assertEqual(graph.cross_routes, ())
        self.assertEqual(graph.fusion_candidates, expanded.entries[0].graph.fusion_candidates)
        self.assertEqual(spec.placement, context.placement)

    def test_repeat_is_deterministic_and_primary_and_context_inputs_are_immutable(self) -> None:
        spec, first_manager, first_expanded, first_context, first = compile_through_n3()
        expanded_digest = canonical_digest(first_expanded)
        context_digest = canonical_digest(first_context)

        second_spec, second_manager, second_expanded, second_context, second = (
            compile_through_n3()
        )
        self.assertEqual(second_spec, spec)
        self.assertEqual(second_expanded, first_expanded)
        self.assertEqual(second_context, first_context)
        self.assertEqual(second, first)
        self.assertEqual(second_manager.snapshot, first_manager.snapshot)
        self.assertEqual(canonical_digest(first_expanded), expanded_digest)
        self.assertEqual(canonical_digest(first_context), context_digest)


if __name__ == "__main__":
    unittest.main()
