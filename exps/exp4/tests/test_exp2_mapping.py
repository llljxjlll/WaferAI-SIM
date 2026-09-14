from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import unittest
import pytest


EXP4 = Path(__file__).resolve().parents[1]
if str(EXP4) not in sys.path:
    sys.path.insert(0, str(EXP4))

from exp2_workload_adapter import (  # noqa: E402
    DEFAULT_EXP2_ROOT,
    load_exp2,
    load_primitive_workloads,
    load_workloads,
    load_sw_opt_only_rows,
    source_inventory,
)
from placement_mapper import (  # noqa: E402
    TopologyCapacityError,
    group_is_connected,
    map_logical_ranks,
    topology_status,
)


def test_exact_36_primitive_inventory_and_field_mapping() -> None:
    rows = load_primitive_workloads()
    assert len(rows) == 36
    assert {kind: sum(row.kind == kind for row in rows) for kind in ("training", "prefill", "decode")} == {
        "training": 12,
        "prefill": 12,
        "decode": 12,
    }
    assert {row.model_id for row in rows} == {
        "llama2_7b", "gpt3_175b", "llama3_8b", "llama3_1_405b", "mixtral_8x7b", "deepseek_v3"
    }
    for row in rows:
        assert row.naive_field == "T_base_cycles"
        assert row.sw_opt_field == (
            "T_full_train_overlap_cycles" if row.kind == "training" else "T_overlap_cycles"
        )
        assert math.isclose(row.sw_opt_speedup, row.naive_cycles / row.sw_opt_cycles, rel_tol=1e-12)
        assert len(row.source_file_digest) == len(row.source_result_digest) == len(row.adapter_digest) == 64


def test_stable_main_pipeline_api() -> None:
    assert load_workloads() == load_primitive_workloads()
    dataset = load_exp2()
    assert len(dataset.workloads) == 36
    assert len(dataset.sw_opt_only_rows) == 48
    assert dataset.provenance["inventory_digest"]


def test_sw_opt_only_is_exact_exp2_row_join() -> None:
    joined = load_sw_opt_only_rows()
    assert len(joined) == 48
    for result in joined:
        filename = {
            "training": "training_e2e.json",
            "prefill": "inference_prefill_pd_breakdown.json",
            "decode": "inference_decode_e2e.json",
            "request": "inference_request_e2e.json",
        }[result.kind]
        source = {
            row["case_id"]: row
            for row in json.loads((DEFAULT_EXP2_ROOT / "results" / filename).read_text())
        }[result.workload_id]
        expected_cycles = source[
            "T_full_train_overlap_cycles" if result.kind == "training" else "T_overlap_cycles"
        ]
        assert result.source_result_digest == source["result_digest"]
        assert result.naive_cycles == source["T_base_cycles"]
        assert result.sw_opt_cycles == expected_cycles
        assert math.isclose(result.speedup, source["T_base_cycles"] / expected_cycles, rel_tol=1e-12)
        assert result.calibration_status == source["calibration_status"]
        assert result.limitation_tags == tuple(source["limitation_tags"])


def test_source_inventory_is_repeatable_and_inherits_publish_status() -> None:
    first = source_inventory()
    second = source_inventory()
    calibration = json.loads((DEFAULT_EXP2_ROOT / "results" / "calibration_summary.json").read_text())
    assert first == second
    assert first["publish_status"] == calibration["publish_status"]
    assert first["calibration_summary_digest"] == calibration["calibration_summary_digest"]
    assert len(first["files"]) == 12
    assert len(first["inventory_digest"]) == 64


@pytest.mark.parametrize("shape", [(6, 6), (8, 5), (22, 16), (9, 4)])
def test_mapping_is_deterministic_unique_and_groups_are_connected(shape: tuple[int, int]) -> None:
    first = map_logical_ranks(*shape)
    second = map_logical_ranks(*shape)
    assert first == second
    assert len(first.rank_to_module) == len(set(first.rank_to_module)) == 36
    assert all(0 <= module < shape[0] * shape[1] for module in first.rank_to_module)
    assert [len(group) for group in first.training_groups] == [9] * 4
    assert [len(group) for group in first.inference_groups] == [6] * 6
    assert all(group_is_connected(group, first) for group in first.training_groups)
    assert all(group_is_connected(group, first) for group in first.inference_groups)
    assert first.replica_count == shape[0] * shape[1] // 36
    assert len(first.mapping_digest) == 64


def test_6x6_mapping_preserves_exp2_row_major_modules() -> None:
    placement = map_logical_ranks(6, 6)
    assert set(placement.rank_to_module) == set(range(36))
    assert placement.selected_boundary_edges == 24


def test_small_topology_has_explicit_infeasible_status() -> None:
    assert topology_status(7, 5) == "topology_capacity_infeasible"
    with pytest.raises(TopologyCapacityError, match="topology_capacity_infeasible"):
        map_logical_ranks(7, 5)
