"""Guard the 11x11 fixed-model MoE probe and the 1..10 release boundary."""

from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from .flexible_mesh_release_hardware import specialize_p5_large_release_hardware
from .run_moe_full_model_extended_mesh_probe import probe
from .run_moe_full_model_native_mesh_matrix import RELEASE_SHAPES, _shape


class MoeExtendedMeshProbeTest(unittest.TestCase):
    def test_fixed_model_on_121_physical_dies_exposes_full_mesh_gate(self) -> None:
        evidence = probe()
        self.assertEqual(
            evidence["status"], "blocked_by_full_mesh_compiler_contract"
        )
        self.assertEqual(evidence["schema_version"], "moe-full-model-extended-mesh-probe-v2")
        self.assertEqual(evidence["physical_die_count"], 121)
        self.assertEqual(evidence["fabric_die_count"], 121)
        self.assertEqual(evidence["hbm_space_count"], 121)
        self.assertEqual(evidence["physical_directed_link_budget"], 440)
        self.assertEqual(evidence["fabric_directed_link_count"], 440)
        self.assertEqual(evidence["active_die_count"], 100)
        self.assertEqual(evidence["idle_die_count"], 21)
        self.assertEqual(evidence["active_die_ids"][-1], 108)
        self.assertEqual(evidence["model_expert_count"], 100)
        self.assertEqual(evidence["ep_rank_count"], 100)
        self.assertEqual(
            evidence["baseline_model_digest"], evidence["extended_model_digest"]
        )
        self.assertEqual(
            evidence["baseline_steps_digest"], evidence["extended_steps_digest"]
        )
        self.assertEqual(evidence["route_traffic_status"], "not_compiled")
        self.assertEqual(
            set(evidence["downstream_blocking_reasons"]),
            {
                "request.ep_experts_must_cover_mesh",
                "request.full_row_major_mesh_required",
                "manifest.full_row_major_mesh_required",
            },
        )
        self.assertEqual(
            set(evidence["blocking_reasons"]),
            {
                "manifest.full_mesh_required",
                "request.ep_must_cover_mesh",
                "request.experts_must_cover_mesh",
                "request.full_mesh_required",
            },
        )

    def test_canonical_release_entry_still_rejects_11x11(self) -> None:
        self.assertEqual(len(RELEASE_SHAPES), 100)
        self.assertNotIn("11x11", RELEASE_SHAPES)
        with self.assertRaisesRegex(ValueError, "outside 1..10 release envelope"):
            _shape("11x11")
        with self.assertRaisesRegex(SchemaError, "requires a 1..10 rectangle"):
            specialize_p5_large_release_hardware(11, 11)
        extended = RectMeshSpec(11, 11)
        extended.validate()
        self.assertFalse(extended.within_release_envelope)


if __name__ == "__main__":
    unittest.main()
