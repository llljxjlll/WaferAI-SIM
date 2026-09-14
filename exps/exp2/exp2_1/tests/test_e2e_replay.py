#!/usr/bin/env python3

from __future__ import annotations

import unittest

from e2e_replay import (
    _PairBuilder,
    _add_collective,
    _decode_tensor_spatial_utilization,
    _layer_work,
    _partition_moe_weight_service,
    assert_same_work,
    build_inference_replay,
    build_training_replay,
    estimate_inference_case,
    estimate_training_case,
    replay,
    summarize_replay,
)
from model_manifests import load_model_manifests
from placements import (
    HBM_STACKS,
    INFERENCE_INSTANCES,
    Placement,
    TRAINING_GROUPS,
    coordinate_to_die_id,
    xy_route,
)


DENSE_MANIFEST = {
    "model": "unit-dense",
    "num_layers": 1,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_attention_heads": 8,
    "num_kv_heads": 2,
    "head_dim": 64,
    "attention_type": "GQA",
    "mlp_type": "dense",
    "moe_layer_frequency": 1,
    "routed_expert_count": 0,
    "shared_expert_count": 0,
    "top_k": 1,
    "parameter_count": 10_000_000,
}

MOE_MANIFEST = {
    **DENSE_MANIFEST,
    "model": "unit-moe",
    "mlp_type": "routed_moe",
    "routed_expert_count": 8,
    "top_k": 2,
}


class PlacementTests(unittest.TestCase):
    def test_training_and_inference_each_partition_the_wafer(self) -> None:
        expected = set(range(36))
        for family in (TRAINING_GROUPS, INFERENCE_INSTANCES):
            flattened = [die for placement in family for die in placement.die_ids]
            self.assertEqual(len(flattened), 36)
            self.assertEqual(set(flattened), expected)

    def test_hbm_stacks_are_edge_attached_and_non_overlapping(self) -> None:
        self.assertEqual(
            [stack.home_die for stack in HBM_STACKS],
            [coordinate_to_die_id(1, 0), coordinate_to_die_id(4, 0),
             coordinate_to_die_id(1, 5), coordinate_to_die_id(4, 5)],
        )
        self.assertEqual(sum(stack.capacity_bytes for stack in HBM_STACKS), 64 * 1024**3)
        self.assertEqual(
            [(stack.address_base, stack.address_end) for stack in HBM_STACKS],
            [(index * 16 * 1024**3, (index + 1) * 16 * 1024**3) for index in range(4)],
        )

    def test_xy_route_is_directional_and_x_first(self) -> None:
        source = coordinate_to_die_id(0, 0)
        destination = coordinate_to_die_id(2, 2)
        forward = xy_route(source, destination)
        reverse = xy_route(destination, source)
        self.assertEqual(len(forward), 4)
        self.assertEqual(forward[0].destination_die, coordinate_to_die_id(1, 0))
        self.assertNotEqual({link.resource_id for link in forward}, {link.resource_id for link in reverse})


class ReplayTests(unittest.TestCase):
    def test_full_train_is_same_work_dependency_only_counterfactual(self) -> None:
        pair = build_training_replay({**DENSE_MANIFEST, "num_layers": 2}, 128)
        self.assertIsNotNone(pair.full_train_actions)
        full = pair.full_train_actions or ()
        assert_same_work(pair.base_actions, pair.overlap_actions)
        assert_same_work(pair.base_actions, full)
        base = {action.action_id: action for action in pair.base_actions}
        forward = {action.action_id: action for action in pair.overlap_actions}
        full_by_id = {action.action_id: action for action in full}
        for action_id in base:
            self.assertEqual(base[action_id].invariant_tuple(), full_by_id[action_id].invariant_tuple())
        self.assertTrue(any(
            forward[action_id].deps != full_by_id[action_id].deps
            and forward[action_id].phase in {"backward", "wgrad"}
            for action_id in forward
        ))
        self.assertTrue(all(
            forward[action_id].deps == full_by_id[action_id].deps
            for action_id in forward
            if forward[action_id].phase == "forward"
        ))

    def test_full_train_advances_dx_but_adamw_waits_every_gradient_sync(self) -> None:
        pair = build_training_replay({**DENSE_MANIFEST, "num_layers": 2}, 128)
        forward = {action.action_id: action for action in pair.overlap_actions}
        full = {action.action_id: action for action in pair.full_train_actions or ()}
        next_dx = "backward.l0.DP00.q0.backward.control"
        self.assertTrue(any(dep.startswith("gradient_sync.l1.") for dep in forward[next_dx].deps))
        self.assertFalse(any(dep.startswith("gradient_sync.l1.") for dep in full[next_dx].deps))
        optimizer = full["optimizer.l2.DP00.q0.adamw.control"]
        expected_sync = {
            f"gradient_sync.l{layer}.flow{flow}"
            for layer in range(2)
            for flow in range(4)
        }
        self.assertTrue(expected_sync.issubset(set(optimizer.deps)))

    def test_full_train_summary_fields_and_schedule_ordering(self) -> None:
        result = estimate_training_case({**DENSE_MANIFEST, "num_layers": 2}, 128)
        self.assertLessEqual(
            result["T_full_train_overlap_cycles"], result["T_overlap_cycles"]
        )
        self.assertLessEqual(result["T_overlap_cycles"], result["T_base_cycles"])
        self.assertAlmostEqual(
            result["speedup_full_train"],
            result["T_base_cycles"] / result["T_full_train_overlap_cycles"],
        )
        self.assertEqual(result["overlap_scope"], "forward_only")
        self.assertTrue(result["training_schedule_ordering_passed"])
        self.assertEqual(
            set(result["full_train_phase_cycles"]),
            {"forward", "backward", "wgrad", "optimizer"},
        )
        self.assertTrue(result["full_train_phase_strictly_shorter_than_forward_only"]["backward"])
        self.assertTrue(result["full_train_phase_strictly_shorter_than_forward_only"]["wgrad"])
        self.assertFalse(result["full_train_phase_strictly_shorter_than_forward_only"]["forward"])
        self.assertEqual(
            result["full_train_phase_cycles"]["optimizer"],
            result["overlap_phase_cycles"]["optimizer"],
        )
        self.assertLessEqual(
            result["full_train_theory_lower_cycles"], result["theory_lower_cycles"]
        )
        self.assertAlmostEqual(
            result["T_full_train_overlap_seconds"],
            result["T_full_train_overlap_cycles"] / 500_000_000.0,
        )

    def test_real_deepseek_dense_prefix_uses_18432(self) -> None:
        manifest = load_model_manifests(["deepseek_v3"])["deepseek_v3"]
        for layer in range(3):
            work = _layer_work(manifest, 1, layer)
            self.assertEqual(work["moe"], 0.0)
            self.assertEqual(work["mlp_intermediate_size"], 18432.0)
            self.assertEqual(work["mlp_matrix_count"], 3.0)
        self.assertEqual(_layer_work(manifest, 1, 3)["mlp_intermediate_size"], 2048.0)

    def test_dense_gelu_has_two_matrices_while_swiglu_has_three(self) -> None:
        gelu = _layer_work({**DENSE_MANIFEST, "mlp_type": "dense_gelu"}, 7, 0)
        swiglu = _layer_work({**DENSE_MANIFEST, "mlp_type": "dense_swiglu"}, 7, 0)
        h, intermediate = DENSE_MANIFEST["hidden_size"], DENSE_MANIFEST["intermediate_size"]
        self.assertEqual(gelu["mlp_matrix_count"], 2.0)
        self.assertEqual(swiglu["mlp_matrix_count"], 3.0)
        self.assertEqual(
            swiglu["linear_flops"] - gelu["linear_flops"],
            2.0 * 7 * h * intermediate,
        )
        self.assertEqual(
            swiglu["weight_bytes"] - gelu["weight_bytes"],
            h * intermediate * 2,
        )

    def test_shared_experts_are_opt_in_and_exposed_by_estimate_api(self) -> None:
        manifest = load_model_manifests(["deepseek_v3"])["deepseek_v3"]
        routed_only = _layer_work(manifest, 128, 3)
        with_shared = _layer_work(manifest, 128, 3, include_shared_experts=True)
        self.assertEqual(routed_only["shared_experts_included"], 0.0)
        self.assertEqual(with_shared["shared_experts_included"], 1.0)
        self.assertGreater(with_shared["tensor_flops"], routed_only["tensor_flops"])
        estimate = estimate_inference_case(
            {**MOE_MANIFEST, "shared_expert_count": 1},
            8,
            prefill_seq=16,
            kv_length=32,
            include_shared_experts=True,
        )
        self.assertTrue(estimate["include_shared_experts"])

    def test_moe_hbm_weights_use_touched_experts_and_ep4_partition(self) -> None:
        work = _layer_work(MOE_MANIFEST, 128, 0)
        expected_routed = (
            MOE_MANIFEST["routed_expert_count"]
            * 3
            * MOE_MANIFEST["hidden_size"]
            * MOE_MANIFEST["intermediate_size"]
            * 2
        )
        self.assertEqual(work["touched_routed_experts"], 8.0)
        self.assertEqual(work["routed_weight_bytes"], expected_routed)
        partitioned = _partition_moe_weight_service(work, 4)
        self.assertEqual(
            partitioned["weight_bytes"],
            work["non_routed_weight_bytes"] + expected_routed / 4,
        )

    def test_routing_skew_preserves_work_and_bytes_but_scales_all_moe_phases(self) -> None:
        balanced = build_training_replay(MOE_MANIFEST, 128, routing_skew=1.0)
        skewed = build_training_replay(MOE_MANIFEST, 128, routing_skew=1.5)
        a = {action.action_id: action for action in balanced.overlap_actions}
        b = {action.action_id: action for action in skewed.overlap_actions}
        self.assertEqual(a.keys(), b.keys())
        self.assertEqual(sum(x.logical_work for x in a.values()), sum(x.logical_work for x in b.values()))
        self.assertEqual(sum(x.bytes for x in a.values()), sum(x.bytes for x in b.values()))
        for phase in ("forward", "backward", "wgrad"):
            candidates = [
                action_id for action_id, action in a.items()
                if action.phase == phase and action.operator in {
                    "EXPERT_GEMM", "DISPATCH", "COMBINE", "TOPK_WEIGHTED_REDUCE"
                }
            ]
            self.assertTrue(candidates, phase)
            for action_id in candidates:
                self.assertEqual(b[action_id].logical_work, a[action_id].logical_work)
                self.assertEqual(b[action_id].bytes, a[action_id].bytes)
                self.assertAlmostEqual(b[action_id].runtime_work, 1.5 * a[action_id].runtime_work)
                if a[action_id].operator in {"DISPATCH", "COMBINE"}:
                    # Fixed DTE launch and hop latency do not scale with routing skew.
                    self.assertGreater(b[action_id].duration_cycles, a[action_id].duration_cycles)
                    self.assertLess(b[action_id].duration_cycles, 1.5 * a[action_id].duration_cycles)
                else:
                    self.assertAlmostEqual(
                        b[action_id].duration_cycles,
                        1.5 * a[action_id].duration_cycles,
                    )
        optimizer_ids = [action_id for action_id, action in a.items() if action.phase == "optimizer"]
        self.assertTrue(optimizer_ids)
        self.assertTrue(all(a[action_id].duration_cycles == b[action_id].duration_cycles for action_id in optimizer_ids))
        for balanced_state, skewed_state in (
            (balanced.base_actions, skewed.base_actions),
            (balanced.overlap_actions, skewed.overlap_actions),
            (balanced.full_train_actions or (), skewed.full_train_actions or ()),
        ):
            self.assertEqual(
                sum(action.logical_work for action in balanced_state),
                sum(action.logical_work for action in skewed_state),
            )
            self.assertEqual(
                sum(action.bytes for action in balanced_state),
                sum(action.bytes for action in skewed_state),
            )

    def test_inference_skew_scales_prefill_and_decode_without_changing_work(self) -> None:
        balanced = build_inference_replay(
            MOE_MANIFEST, 8, prefill_seq=16, kv_length=32, routing_skew=1.0
        )
        skewed = build_inference_replay(
            MOE_MANIFEST, 8, prefill_seq=16, kv_length=32, routing_skew=1.25
        )
        a = {action.action_id: action for action in balanced.overlap_actions}
        b = {action.action_id: action for action in skewed.overlap_actions}
        self.assertEqual(sum(x.logical_work for x in a.values()), sum(x.logical_work for x in b.values()))
        self.assertEqual(sum(x.bytes for x in a.values()), sum(x.bytes for x in b.values()))
        for phase in ("prefill", "decode"):
            tensors = [action_id for action_id, action in a.items() if action.phase == phase and action.operator == "EXPERT_GEMM"]
            self.assertTrue(tensors, phase)
            self.assertTrue(all(b[action_id].runtime_work == 1.25 * a[action_id].runtime_work for action_id in tensors))

    def test_training_pair_has_same_work_and_only_forward_dependency_changes(self) -> None:
        pair = build_training_replay(MOE_MANIFEST, 128)
        assert_same_work(pair.base_actions, pair.overlap_actions)
        base = {action.action_id: action for action in pair.base_actions}
        overlap = {action.action_id: action for action in pair.overlap_actions}
        changed_phases = {
            base[action_id].phase
            for action_id in base
            if base[action_id].deps != overlap[action_id].deps
        }
        self.assertEqual(changed_phases, {"forward"})
        for phase in ("backward", "wgrad", "optimizer"):
            self.assertTrue(any(action.phase == phase for action in pair.base_actions))

    def test_all_required_resource_classes_are_explicit(self) -> None:
        pair = build_training_replay(MOE_MANIFEST, 128)
        resources = {resource for action in pair.overlap_actions for resource in action.resource_set}
        for prefix in ("tensor.", "vector.", "hbm.stack", "d2d.", "noc.",
                       "dte.", "control.", "reducer.", "sram."):
            self.assertTrue(any(resource.startswith(prefix) for resource in resources), prefix)
        self.assertEqual(
            {resource for resource in resources if resource.startswith("hbm.stack")},
            {f"hbm.stack{index}" for index in range(4)},
        )

    def test_lower_bound_is_max_of_dependency_and_resource_bounds(self) -> None:
        pair = build_training_replay(DENSE_MANIFEST, 128)
        scheduled = replay(pair.overlap_actions)
        self.assertEqual(
            scheduled.theory_lower_cycles,
            max(scheduled.longest_dependency_path_cycles,
                max(scheduled.resource_service_cycles.values())),
        )
        self.assertLessEqual(scheduled.theory_lower_cycles, scheduled.makespan_cycles + 1e-6)
        for phase, lower in scheduled.phase_theory_lower_cycles.items():
            self.assertGreater(lower, 0, phase)
            self.assertLessEqual(lower, scheduled.phase_cycles[phase] + 1e-6, phase)

    def test_decode_kv_bytes_only_feed_hbm_not_reducer(self) -> None:
        estimate = estimate_inference_case(
            DENSE_MANIFEST, 64, prefill_seq=16, kv_length=1024
        )
        diagnostics = estimate["decode_fidelity_diagnostics"]
        self.assertFalse(diagnostics["kv_bytes_enter_reducer"])
        self.assertEqual(
            diagnostics["activation_reduce_bytes_per_layer"],
            64 * DENSE_MANIFEST["hidden_size"] * 2,
        )
        self.assertEqual(
            diagnostics["kv_hbm_bytes_per_layer_step0"],
            64 * 1024 * (2 * 64) * 2 * 2,
        )
        pair = build_inference_replay(
            DENSE_MANIFEST, 64, prefill_seq=16, kv_length=1024
        )
        reducer_bytes = sum(
            action.bytes
            for action in pair.overlap_actions
            if action.phase == "decode"
            and action.layer == 0
            and ".D0." in action.action_id
            and action.operator == "NORM_RESIDUAL"
        )
        self.assertEqual(reducer_bytes, 64 * DENSE_MANIFEST["hidden_size"] * 2)

    def test_decode_windows_and_tensor_efficiency_are_shape_driven(self) -> None:
        small = build_inference_replay(
            DENSE_MANIFEST, 64, prefill_seq=16, kv_length=32
        )
        large = build_inference_replay(
            DENSE_MANIFEST, 512, prefill_seq=16, kv_length=32
        )
        assert_same_work(small.base_actions, small.overlap_actions)
        assert_same_work(large.base_actions, large.overlap_actions)

        def window_count(pair) -> int:
            return sum(
                action.phase == "decode"
                and action.layer == 0
                and ".D0." in action.action_id
                and action.operator == "CONTROL_ISSUE"
                for action in pair.overlap_actions
            )

        self.assertEqual(window_count(small), 1)
        self.assertEqual(window_count(large), 4)
        dense = _layer_work(DENSE_MANIFEST, 64, 0, decode_kv=32)
        moe = _layer_work(MOE_MANIFEST, 64, 0, decode_kv=32)
        self.assertEqual(_decode_tensor_spatial_utilization(dense), 0.5)
        self.assertLess(_decode_tensor_spatial_utilization(moe), 0.5)
        self.assertEqual(moe["expert_tensor_m"], 16.0)
        large_moe = estimate_inference_case(
            MOE_MANIFEST, 512, prefill_seq=16, kv_length=32
        )["decode_fidelity_diagnostics"]
        self.assertEqual(large_moe["version"], "v4")
        self.assertEqual(set(large_moe["dense_tensor_m_per_window"]), {128.0})
        self.assertEqual(set(large_moe["expert_tensor_m_per_window"]), {32.0})
        self.assertLess(
            large_moe["tensor_spatial_utilization_max"],
            0.5,
        )

    def test_collective_fixed_cost_survives_zero_byte_messages(self) -> None:
        short = Placement("short", "test", (0,), (0, 1))
        long = Placement("long", "test", (0,), (0, 1, 2, 3, 4, 5))

        def first_flow_duration(placement: Placement) -> float:
            builder = _PairBuilder()
            action_ids = _add_collective(
                builder,
                "fixed",
                "decode",
                0,
                0,
                placement,
                0.0,
                (),
                (),
                "ALL_GATHER",
            )
            actions = {action.action_id: action for action in builder.base}
            return actions[action_ids[0]].duration_cycles

        self.assertEqual(first_flow_duration(short), 3.0)
        self.assertEqual(first_flow_duration(long), 7.0)
    def test_collective_waits_for_its_window_hbm_data(self) -> None:
        pair = build_inference_replay(
            DENSE_MANIFEST, 512, prefill_seq=16, kv_length=32
        )
        actions = {action.action_id: action for action in pair.overlap_actions}
        scheduled = replay(pair.overlap_actions)
        prefix = "decode.l0.D0.q0.decode_t1"
        hbm_ids = [
            action_id
            for action_id, action in actions.items()
            if action_id.startswith(prefix)
            and action.operator == "HBM_STREAM"
        ]
        flow_ids = [
            action_id
            for action_id, action in actions.items()
            if action_id.startswith(prefix + ".flow")
            and action.operator == "ALL_GATHER"
        ]
        self.assertEqual(len(hbm_ids), 4)
        self.assertEqual(len(flow_ids), 2)
        for flow_id in flow_ids:
            self.assertEqual(set(actions[flow_id].deps), set(hbm_ids))
            self.assertGreaterEqual(
                scheduled.starts[flow_id],
                max(scheduled.finishes[action_id] for action_id in hbm_ids),
            )

    def test_kv_append_is_explicit_and_gates_next_token(self) -> None:
        pair = build_inference_replay(
            DENSE_MANIFEST, 64, prefill_seq=16, kv_length=32
        )
        actions = {action.action_id: action for action in pair.overlap_actions}
        append_ids = [
            action_id
            for action_id, action in actions.items()
            if action.phase == "decode"
            and action.layer == 0
            and ".D0." in action_id
            and action.operator == "KV_APPEND_HBM_WRITE"
        ]
        self.assertEqual(len(append_ids), 4)
        self.assertEqual(
            sum(actions[action_id].bytes for action_id in append_ids),
            64 * (2 * 64) * 2 * 2,
        )
        commit = actions["decode_t1.D0.token_commit"]
        self.assertTrue(set(append_ids).issubset(set(commit.deps)))
        scheduled = replay(pair.overlap_actions)
        self.assertGreaterEqual(
            scheduled.starts[commit.action_id],
            max(scheduled.finishes[action_id] for action_id in append_ids),
        )


    def test_decode_recurrence_is_per_instance_and_summary_is_commit_interval(self) -> None:
        pair = build_inference_replay(
            DENSE_MANIFEST, 64, prefill_seq=16, kv_length=32
        )
        overlap = {action.action_id: action for action in pair.overlap_actions}
        first_steady_control = overlap[
            "decode.l0.D0.q0.decode_t1.control"
        ]
        self.assertIn("decode_t0.D0.token_commit", first_steady_control.deps)
        self.assertNotIn("decode_t0.D1.token_commit", first_steady_control.deps)

        scheduled = replay(pair.overlap_actions)
        intervals = {
            instance: (
                scheduled.finishes[f"decode_t1.{instance}.token_commit"]
                - scheduled.finishes[f"decode_t0.{instance}.token_commit"]
            )
            for instance in ("D0", "D1")
        }
        summary = summarize_replay(pair, DENSE_MANIFEST)
        self.assertEqual(summary["decode_cycles"], max(intervals.values()))
        self.assertEqual(
            summary["overlap_decode_commit_intervals_cycles"], intervals
        )

    def test_decode_steady_interval_is_independent_of_prefill_length(self) -> None:
        short = estimate_inference_case(
            DENSE_MANIFEST, 64, prefill_seq=16, kv_length=32
        )
        long = estimate_inference_case(
            DENSE_MANIFEST, 64, prefill_seq=128, kv_length=32
        )
        self.assertAlmostEqual(short["decode_cycles"], long["decode_cycles"])

    def test_inference_contains_prefill_four_way_pd_wait_and_decode(self) -> None:
        pair = build_inference_replay(DENSE_MANIFEST, 64, prefill_seq=128, kv_length=1024)
        operators = [action.operator for action in pair.overlap_actions]
        self.assertEqual(operators.count("PD_KV_STATE_TRANSFER"), 4)
        self.assertEqual(operators.count("PD_KV_READY_WAIT"), 2)
        summary = summarize_replay(pair, DENSE_MANIFEST)
        for field in ("prefill_cycles", "handoff_cycles", "handoff_wait_cycles",
                      "decode_cycles", "TTFT_cycles"):
            self.assertGreater(summary[field], 0)
        self.assertGreaterEqual(summary["TTFT_cycles"], summary["prefill_cycles"])
        self.assertEqual(
            summary["decode_theory_lower_cycles"],
            summary["phase_theory_lower_cycles"]["decode"],
        )
        self.assertLessEqual(summary["decode_theory_lower_cycles"], summary["decode_cycles"])

    def test_unvalidated_evidence_cannot_upgrade_source(self) -> None:
        invalid = {
            "unit_closure_passed": True,
            "repeatability_passed": True,
            "validation_passed": True,
            "p95_relative_error": 0.16,
            "evidence_signatures": ["bad-anchor"],
        }
        result = estimate_training_case(DENSE_MANIFEST, 128, evidence=invalid)
        self.assertEqual(result["estimate_source"], "analytical_resource_dag_extrapolation")
        self.assertIn("provided_evidence_failed_release_gate", result["limitation_tags"])
        valid = {**invalid, "p95_relative_error": 0.10}
        result = estimate_training_case(DENSE_MANIFEST, 128, evidence=valid)
        self.assertEqual(result["estimate_source"], "analytical_resource_dag_extrapolation")
        self.assertIn("validated_anchor_not_applied_to_abstract_routes", result["limitation_tags"])

    def test_mla_is_never_silently_treated_as_standard_attention(self) -> None:
        mla = {
            **DENSE_MANIFEST,
            "model_id": "deepseek_v3",
            "attention_type": "MLA",
            "head_dim": None,
            "dense_intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "mlp_type": "deepseek_moe_swiglu",
            "routed_expert_count": 256,
            "top_k": 8,
            "mla_dimensions": {
                "kv_lora_rank": 128,
                "q_lora_rank": 256,
                "qk_nope_head_dim": 64,
                "qk_rope_head_dim": 64,
            },
        }
        result = estimate_inference_case(mla, 64, prefill_seq=128, kv_length=1024)
        self.assertEqual(result["estimate_source"], "analytical_only_mla")
        self.assertIn("analytical_only_mla", result["limitation_tags"])
        dense = _layer_work(mla, 1, 0)
        moe = _layer_work(mla, 1, 3)
        self.assertEqual(dense["moe"], 0.0)
        self.assertEqual(moe["moe"], 1.0)
        self.assertGreater(dense["weight_bytes"], 0)
        self.assertNotEqual(dense["weight_bytes"], moe["weight_bytes"])

    def test_speedup_and_intervals_are_derived_without_positive_clamp(self) -> None:
        result = estimate_training_case(DENSE_MANIFEST, 128)
        self.assertAlmostEqual(result["speedup"], result["T_base_cycles"] / result["T_overlap_cycles"])
        self.assertLess(result["uncertainty_low"], result["speedup"])
        self.assertGreater(result["uncertainty_high"], result["speedup"])


if __name__ == "__main__":
    unittest.main()
