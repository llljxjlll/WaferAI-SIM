#!/usr/bin/env python3
"""Map exp2 training cases to exp1 operators and derive mean speedups."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
EXPS_ROOT = ROOT.parents[1]
EXP1_1_RESULTS = EXPS_ROOT / "exp1" / "exp1_1" / "results" / "results.csv"
EXP1_2_RESULTS = (
    EXPS_ROOT
    / "exp1"
    / "exp1_2"
    / "results"
    / "h128"
    / "architecture"
    / "results.csv"
)
TRAINING_RESULTS = ROOT / "results" / "training_e2e.json"
OUTPUT_STEM = ROOT / "results" / "training_operator_amdahl"

EXP1_1_MODEL = {
    "llama2_7b": "LLaMA-2-7B",
    "gpt3_175b": "GPT-3-175B",
    "llama3_8b": "LLaMA-3-8B",
    "llama3_1_405b": "LLaMA-3.1-405B",
    "mixtral_8x7b": "Mixtral-8x7B-single-expert",
    "deepseek_v3": "DeepSeek-V3-single-routed-expert",
}
EXP1_2_MODEL = {
    "mixtral_8x7b": "Mixtral-8x7B",
    "deepseek_v3": "DeepSeek-V3",
}
EXP1_1_SEQUENCE_PROXY = {2304: 2048, 36864: 32768}
EXP1_1_OPERATORS = ("AG_GEMM", "GEMM_RS")
EXP1_2_OPERATORS = ("DISPATCH_GEMM", "GEMM_COMBINE")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _one(rows: Iterable[dict[str, str]], description: str) -> dict[str, str]:
    selected = list(rows)
    if len(selected) != 1:
        raise ValueError(f"{description}: expected one source case, got {len(selected)}")
    return selected[0]


def _exp1_1_item(
    rows: list[dict[str, str]],
    model_id: str,
    layer: str,
    operator: str,
    target_seq_len: int,
    source_digest: str,
) -> dict[str, Any]:
    source_seq_len = EXP1_1_SEQUENCE_PROXY[target_seq_len]
    source_model = EXP1_1_MODEL[model_id]
    source = _one(
        (
            item
            for item in rows
            if item["mesh"] == "3x3"
            and item["operator"] == operator
            and item["model"] == source_model
            and item["layer"] == layer
            and int(item["seq_len"]) == source_seq_len
        ),
        f"exp1-1/{model_id}/{layer}/{operator}/s{target_seq_len}",
    )
    return {
        "source_experiment": "exp1-1",
        "source_path": str(EXP1_1_RESULTS.relative_to(EXPS_ROOT)),
        "source_file_digest": source_digest,
        "source_case_id": source["case_id"],
        "operator": operator,
        "operator_scope": f"tp_{layer}_suboperator",
        "layer": layer,
        "source_model": source_model,
        "source_mesh": "3x3",
        "target_mesh": "3x3",
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
    seq_len: int,
    source_digest: str,
) -> dict[str, Any]:
    source_model = EXP1_2_MODEL[model_id]
    source = _one(
        (
            item
            for item in rows
            if item["profile"] == "H128"
            and item["placement"] == "noncompact"
            and item["operator"] == operator
            and item["model"] == source_model
            and int(item["seq_len"]) == seq_len
        ),
        f"exp1-2/{model_id}/{operator}/s{seq_len}",
    )
    return {
        "source_experiment": "exp1-2",
        "source_path": str(EXP1_2_RESULTS.relative_to(EXPS_ROOT)),
        "source_file_digest": source_digest,
        "source_case_id": source["case_id"],
        "operator": operator,
        "operator_scope": "ep_routed_expert_composite",
        "layer": "moe_mlp",
        "source_model": source_model,
        "source_mesh": source["physical_mesh"],
        "source_placement": "noncompact",
        "target_placement": "tp3x3_ep2x2_noncompact",
        "placement_match": "noncompact_four_rank_proxy",
        "source_profile": "H128",
        "source_seq_len": seq_len,
        "target_seq_len": seq_len,
        "seq_match": "exact",
        "target_over_source_seq_ratio": 1.0,
        "speedup_metric": "actual_speedup=T00/T11",
        "speedup": float(source["actual_speedup"]),
    }


def build_records() -> list[dict[str, Any]]:
    training = json.loads(TRAINING_RESULTS.read_text(encoding="utf-8"))
    exp1_1 = _read_csv(EXP1_1_RESULTS)
    exp1_2 = _read_csv(EXP1_2_RESULTS)
    exp1_1_digest = _sha256(EXP1_1_RESULTS)
    exp1_2_digest = _sha256(EXP1_2_RESULTS)
    tool_digest = _sha256(Path(__file__))
    records: list[dict[str, Any]] = []
    for training_row in training:
        model_id = str(training_row["model_id"])
        seq_len = int(training_row["seq_len"])
        layers = ("mlp",) if model_id == "deepseek_v3" else ("attention", "mlp")
        operators: list[dict[str, Any]] = []
        for layer in layers:
            for operator in EXP1_1_OPERATORS:
                operators.append(
                    _exp1_1_item(
                        exp1_1, model_id, layer, operator, seq_len, exp1_1_digest
                    )
                )
        if model_id in EXP1_2_MODEL:
            for operator in EXP1_2_OPERATORS:
                operators.append(
                    _exp1_2_item(
                        exp1_2, model_id, operator, seq_len, exp1_2_digest
                    )
                )
        mean_speedup = sum(item["speedup"] for item in operators) / len(operators)
        record: dict[str, Any] = {
            "schema_version": "exp2.training_operator_amdahl.v1",
            "case_id": f"amdahl__{training_row['case_id']}",
            "training_case_id": training_row["case_id"],
            "training_result_digest": training_row["result_digest"],
            "model": training_row["model"],
            "model_id": model_id,
            "seq_len": seq_len,
            "training_mesh": "3x3",
            "training_placement": training_row["placement"],
            "system_speedup_full_train": training_row["speedup_full_train"],
            "operator_speedup_mean": mean_speedup,
            "operator_speedup_count": len(operators),
            "operator_speedup_mean_method": "unweighted_arithmetic_mean",
            "operator_speedup_mean_scope": "descriptive_not_amdahl_prediction",
            "operator_to_system_ratio": mean_speedup
            / float(training_row["speedup_full_train"]),
            "selected_operators": operators,
            "tool_digest": tool_digest,
            "limitation_tags": [
                "unweighted_operator_mean",
                "exp1_1_seq_len_nearest_same_mesh_proxy",
                "operator_mean_is_not_amdahl_weighted_prediction",
            ],
        }
        if model_id in EXP1_2_MODEL:
            record["limitation_tags"].append(
                "nested_tp_suboperator_and_ep_composite_scopes"
            )
        if model_id == "deepseek_v3":
            record["limitation_tags"].append(
                "deepseek_mla_attention_has_no_exp1_1_anchor"
            )
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
        "min_operator_speedup_mean": min(item["operator_speedup_mean"] for item in records),
        "max_operator_speedup_mean": max(item["operator_speedup_mean"] for item in records),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
