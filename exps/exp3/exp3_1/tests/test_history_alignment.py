from __future__ import annotations

import importlib.util
from pathlib import Path

import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp3_1_legacy_alignment_tested", ROOT / "legacy_alignment.py"
)
assert SPEC is not None and SPEC.loader is not None
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


class HistoryAlignmentTests(unittest.TestCase):
    def test_dense_native_exposes_four_states_and_stage_decomposition(self) -> None:
        result = legacy.dense_native({
            "model": "LLaMA-2-7B",
            "stage": "O-proj",
            "S": 2048,
            "D": 6,
            "Px": 2,
            "Py": 3,
        })
        assert result["state_cycles"] == {
            state: result[state] for state in ("W00", "W10", "W01", "W11")
        }
        assert all(isinstance(result[state], int) and result[state] > 0
                   for state in ("W00", "W10", "W01", "W11"))
        assert result["stages"]["communication"] > 0
        assert result["stages"]["compute_naive"] > 0
        assert result["stages"]["runtime_shape"] == {
            "M": 2048, "N": 4096, "K": 12288,
            "rank_N": 4096, "rank_K": 2048,
        }
        baseline = result["intra_ablation"]["baseline"]
        optimized = result["intra_ablation"]["optimized"]
        assert (baseline["active_cores"], baseline["intra_pm"],
                baseline["intra_pn"], baseline["intra_pk"]) == (16, 4, 4, 1)
        assert optimized["active_cores"] == 16
        assert baseline["local_noc_max_directed_link_bytes"] >= (
            optimized["local_noc_max_directed_link_bytes"]
        )
        assert result["W00"] > result["W11"]
        assert result["state_stages"]["W00"]["intra_schedule"]["active_cores"] == 16
        assert result["state_stages"]["W11"]["intra_schedule"]["active_cores"] == 16
        assert result["evidence"] == (
            "analytical_fixed_16core_core_to_d2d_port_full_and_inter_replay_from_exp1"
        )
        assert result["controlled_evidence"] == (
            "analytical_fixed_16core_core_to_d2d_port_inter_only_replay_from_exp1"
        )
        assert result["controlled_state_cycles"]["C00"] > result["controlled_state_cycles"]["C10"]
        port = result["controlled_state_stages"]["C00"]["inter_port"]
        assert port["dte_channel_count"] == 2
        assert port["gateway_serial_cycles"] > port["shared_port_noc_cycles"] > 0
        assert result["controlled_state_stages"]["C10"]["inter_port_cycles"] > 0
        c00 = result["controlled_state_stages"]["C00"]
        c10 = result["controlled_state_stages"]["C10"]
        assert c00["hbm_cycles"] == c10["hbm_cycles"]
        assert "hbm_intermediate_materialization" not in c00

    
    
    def test_moe_anchor_uses_real_model_profile(self) -> None:
        for model, expected in (
            ("Mixtral-8x7B", (4096, 14336, 8, 2)),
            ("DeepSeek-V3", (7168, 2048, 256, 8)),
        ):
            hidden, intermediate, experts, topk = expected
            anchor = legacy.moe_d4_anchor({
                "model": model,
                "stage": "Dispatch + gate/up",
                "S": 2304,
                "H": hidden,
                "I": intermediate,
                "E": experts,
                "topk": topk,
            })
            record = anchor["legacy_record"]
            assert (record["hidden_size"], record["expert_intermediate_size"],
                    record["expert_count"], record["top_k"]) == expected
            assert anchor["D"] == 4
            assert anchor["profile"] == "H128"
            assert anchor["memory_mode"] == "architecture"
            assert anchor["placement"] == "compact"
            intra = anchor["intra_ablation"]
            assert intra["baseline"]["active_cores"] == 16
            controlled = anchor["controlled_state_stages"]
            assert anchor["controlled_state_cycles"]["C00"] > anchor["controlled_state_cycles"]["C10"]
            assert controlled["C00"]["inter_port"]["dte_channel_count"] == 2
            assert controlled["C00"]["inter_port_cycles"] > 0
            assert (intra["baseline"]["intra_pe"] * intra["baseline"]["intra_pm"]
                    * intra["baseline"]["intra_pn"] * intra["baseline"]["intra_pk"]) == 16
            assert controlled["C00"]["hbm_cycles"] == controlled["C10"]["hbm_cycles"]
            assert "hbm_intermediate_materialization" not in controlled["C00"]
            assert intra["baseline"]["intra_pk"] == 1
    
    
    def test_moe_anchor_rejects_synthetic_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "real-model expert_intermediate_size"):
            legacy.moe_d4_anchor({
                "model": "Mixtral-8x7B", "stage": "dispatch", "S": 2304,
                "H": 4096, "I": 11008, "E": 8, "topk": 2,
            })
    
    
    def test_moe_scaling_is_exact_at_d4_and_labelled_beyond_anchor(self) -> None:
        anchor = legacy.moe_d4_anchor({
            "model": "DeepSeek-V3", "stage": "down + Combine", "S": 36864,
        })
        d4 = legacy.scale_moe_anchor_to_d(anchor, 4)
        assert d4["state_cycles"] == anchor["state_cycles"]
        assert all(d4[state] == anchor[state] for state in ("W00", "W10", "W01", "W11"))
        assert d4["analytical_extrapolation"] is False
    
        d16 = legacy.scale_moe_anchor_to_d(anchor, 16)
        assert d16["compute_scale"] == (0.25)
        assert d16["communication_scale"] == (0.5)
        assert d16["analytical_extrapolation"] is True
        assert "analytical_extrapolation" in d16["evidence"]
        for state in ("W00", "W10", "W01", "W11"):
            assert d16["stages"][state]["compute_cycles"] == (
                anchor["stages"][state]["compute_cycles"] * 0.25
            )
            assert d16["stages"][state]["communication_cycles"] == (
                anchor["stages"][state]["communication_cycles"] * 0.5
            )
    
    
    def test_compatibility_audit_matches_committed_history_cycle_for_cycle(self) -> None:
        audit = legacy.compatibility_audit()
        assert audit["passed"] is True
        assert audit["dense_cases"] == 8
        assert audit["dense_main_matrix_overlap_cases"] == 4
        assert audit["moe_cases"] == 8
        assert audit["comparisons"] == (8 + 8) * 4
        assert len(audit["main_matrix_overlap"]) == 16
        assert all(row["actual"] == row["expected"] for row in audit["main_matrix_overlap"])
        assert audit["mismatches"] == []

if __name__ == "__main__":
    unittest.main()
