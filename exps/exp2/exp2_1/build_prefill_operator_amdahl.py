#!/usr/bin/env python3
"""Map exp2 prefill cases to exp1 operators and derive mean speedups."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from build_operator_amdahl import (
    EXP1_1_MODEL,
    EXP1_1_OPERATORS,
    EXP1_1_RESULTS,
    EXP1_2_MODEL,
    EXP1_2_OPERATORS,
    EXP1_2_RESULTS,
    EXPS_ROOT,
    ROOT,
    _canonical_digest,
    _one,
    _read_csv,
    _sha256,
)


PREFILL_RESULTS = ROOT / "results" / "inference_prefill_pd_breakdown.json"
OUTPUT_STEM = ROOT / "results" / "prefill_operator_amdahl"
TARGET_SEQUENCE_LENGTHS = (2304, 36864)
EXP1_1_SOURCE_SEQ_BY_TARGET = {
    2304: 2048,
    36864: 32768,
}
TARGET_MESH = "2x3"


def _exp1_1_item(
    rows: list[dict[str, str]],
    model_id: str,
    layer: str,
    operator: str,
    source_digest: str,
    target_seq_len: int,
) -> dict[str, Any]:
    source_model = EXP1_1_MODEL[model_id]
    source_seq_len = EXP1_1_SOURCE_SEQ_BY_TARGET[target_seq_len]
    source = _one(
        (
            item
            for item in rows
            if item["mesh"] == TARGET_MESH
            and item["operator"] == operator
            and item["model"] == source_model
            and item["layer"] == layer
            and int(item["seq_len"]) == source_seq_len
        ),
        f"prefill/exp1-1/{model_id}/{layer}/{operator}",
    )
    return {
        "source_experiment": "exp1-1",
        "source_path": str(EXP1_1_RESULTS.relative_to(EXPS_ROOT)),
        "source_file_digest": source_digest,
        "source_case_id": source["case_id"],
        "operator": operator,
        "operator_scope": f"prefill_tp_{layer}_suboperator",
        "layer": layer,
        "source_model": source_model,
        "source_mesh": TARGET_MESH,
        "target_mesh": TARGET_MESH,
        "mesh_match": "exact",
        "source_seq_len": source_seq_len,
        "target_seq_len": target_seq_len,
        "seq_match": "nearest_same_mesh_proxy",
        "target_over_source_seq_ratio": target_seq_len / source_seq_len,
        "speedup_metric": "total_speedup=T00/T11",
        "speedup": float(source["total_speedup"]),
    }


def _exp1_2_item(
    rows: list[dict[str, str]],
    model_id: str,
    operator: str,
    source_digest: str,
    target_seq_len: int,
) -> dict[str, Any]:
    source_model = EXP1_2_MODEL[model_id]
    source = _one(
        (
            item
            for item in rows
            if item["profile"] == "H128"
            and item["placement"] == "compact"
            and item["operator"] == operator
            and item["model"] == source_model
            and int(item["seq_len"]) == target_seq_len
        ),
        f"prefill/exp1-2/{model_id}/{operator}",
    )
    return {
        "source_experiment": "exp1-2",
        "source_path": str(EXP1_2_RESULTS.relative_to(EXPS_ROOT)),
        "source_file_digest": source_digest,
        "source_case_id": source["case_id"],
        "operator": operator,
        "operator_scope": "prefill_moe_dispatch_or_combine_composite",
        "layer": "moe_mlp",
        "source_model": source_model,
        "source_mesh": source["physical_mesh"],
        "target_mesh": TARGET_MESH,
        "source_placement": "compact",
        "target_placement": "pd_4p_2d_each_2x3",
        "placement_match": "compact_four_rank_proxy_for_contiguous_2x3_instance",
        "source_profile": "H128",
        "source_seq_len": target_seq_len,
        "target_seq_len": target_seq_len,
        "seq_match": "exact",
        "target_over_source_seq_ratio": 1.0,
        "speedup_metric": "actual_speedup=T00/T11",
        "speedup": float(source["actual_speedup"]),
    }


def build_records() -> list[dict[str, Any]]:
    prefill = json.loads(PREFILL_RESULTS.read_text(encoding="utf-8"))
    exp1_1 = _read_csv(EXP1_1_RESULTS)
    exp1_2 = _read_csv(EXP1_2_RESULTS)
    exp1_1_digest = _sha256(EXP1_1_RESULTS)
    exp1_2_digest = _sha256(EXP1_2_RESULTS)
    tool_digest = _sha256(Path(__file__))
    records: list[dict[str, Any]] = []
    for prefill_row in prefill:
        model_id = str(prefill_row["model_id"])
        seq_len = int(prefill_row["seq_len"])
        if seq_len not in TARGET_SEQUENCE_LENGTHS:
            raise ValueError(
                f"{prefill_row['case_id']}: unexpected S={seq_len}"
            )
        layers = ("mlp",) if model_id == "deepseek_v3" else ("attention", "mlp")
        operators: list[dict[str, Any]] = []
        for layer in layers:
            for operator in EXP1_1_OPERATORS:
                operators.append(
                    _exp1_1_item(
                        exp1_1, model_id, layer, operator, exp1_1_digest, seq_len
                    )
                )
        if model_id in EXP1_2_MODEL:
            for operator in EXP1_2_OPERATORS:
                operators.append(
                    _exp1_2_item(
                        exp1_2, model_id, operator, exp1_2_digest, seq_len
                    )
                )
        mean_speedup = sum(item["speedup"] for item in operators) / len(operators)
        system_speedup = float(prefill_row["prefill_cycles_base"]) / float(
            prefill_row["prefill_cycles_overlap"]
        )
        limitations = [
            "unweighted_operator_mean",
            "exp1_1_seq_len_nearest_same_mesh_proxy",
            "operator_mean_is_not_amdahl_weighted_prediction",
        ]
        if model_id in EXP1_2_MODEL:
            limitations.extend([
                "nested_tp_suboperator_and_moe_composite_scopes",
                "exp1_2_compact_four_rank_to_prefill_2x3_topology_proxy",
            ])
        if model_id == "deepseek_v3":
            limitations.append("deepseek_mla_attention_has_no_exp1_1_anchor")
        record: dict[str, Any] = {
            "schema_version": "exp2.prefill_operator_amdahl.v2",
            "case_id": f"prefill_amdahl__{prefill_row['case_id']}",
            "prefill_case_id": prefill_row["case_id"],
            "prefill_result_digest": prefill_row["result_digest"],
            "model": prefill_row["model"],
            "model_id": model_id,
            "seq_len": seq_len,
            "prefill_mesh": TARGET_MESH,
            "prefill_placement": prefill_row["placement"],
            "system_prefill_speedup": system_speedup,
            "system_prefill_speedup_metric": "prefill_cycles_base/prefill_cycles_overlap",
            "operator_speedup_mean": mean_speedup,
            "operator_speedup_count": len(operators),
            "operator_speedup_mean_method": "unweighted_arithmetic_mean",
            "operator_speedup_mean_scope": "descriptive_not_amdahl_prediction",
            "operator_to_system_ratio": mean_speedup / system_speedup,
            "selected_operators": operators,
            "tool_digest": tool_digest,
            "limitation_tags": limitations,
        }
        record["result_digest"] = _canonical_digest(record)
        records.append(record)
    return records


def write_records(records: list[dict[str, Any]]) -> None:
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_STEM.with_suffix(".json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = sorted({key for record in records for key in record})
    with OUTPUT_STEM.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in record.items()
            })


def build_and_write() -> list[dict[str, Any]]:
    records = build_records()
    write_records(records)
    return records


def main() -> int:
    records = build_and_write()
    print(json.dumps({
        "cases": len(records),
        "min_operator_speedup_mean": min(x["operator_speedup_mean"] for x in records),
        "max_operator_speedup_mean": max(x["operator_speedup_mean"] for x in records),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
