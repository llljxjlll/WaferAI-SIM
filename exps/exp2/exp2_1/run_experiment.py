#!/usr/bin/env python3
"""Run the exp2-1 analytical E2E experiment with explicit provenance gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from capacity_model import (
    HBM_CAPACITY_BYTES,
    audit_inference_case,
    audit_training_case,
    build_capacity_audit,
    kv_elements_per_token_per_layer,
)
from e2e_replay import (
    CLOCK_HZ,
    estimate_inference_case,
    estimate_training_case,
)
from model_manifests import load_model_manifests


TRAIN_SEQUENCE_LENGTHS = (2304, 36864)
PREFILL_SEQUENCE_LENGTHS = TRAIN_SEQUENCE_LENGTHS
DECODE_BATCH_SIZES = (64, 512)
ROUTING_SKEWS = (1.0, 1.25, 1.5)
PRIMARY_WEIGHT_MODE = "global_shared_read_only"
REQUEST_PREFILL_SEQUENCE_LENGTH = 2304
KV_LENGTH = 36864
REQUEST_OUTPUT_TOKENS = 512
DP_OR_EP_RANKS = 4
D_INSTANCE_COUNT = 2
P_INSTANCE_COUNT = 4


ROOT = Path(__file__).resolve().parent


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_digest(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def load_calibration_summary() -> tuple[dict[str, Any], dict[str, Any]]:
    closure_path = ROOT / "calibration" / "hardware_unit_closure.json"
    source_path = ROOT / "calibration" / "source_evidence.json"
    closure = _load_json(closure_path) if closure_path.is_file() else {}
    source = _load_json(source_path) if source_path.is_file() else {}

    signatures: list[str] = []
    if source_path.is_file():
        signatures.append(f"source_evidence:{_file_digest(source_path)}")
    for key in ("fresh_smokes", "retained_current_build_smokes"):
        value = source.get(key, [])
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict) or not item.get("family"):
                continue
            digest = item.get("source_sha256")
            evidence_path = item.get("evidence_path")
            if digest is None and evidence_path:
                candidate = ROOT / str(evidence_path)
                if candidate.is_file():
                    digest = _file_digest(candidate)
            if digest:
                signatures.append(f"{item['family']}:{digest}")
    for key in ("retained_evidence", "cycle_smokes", "smokes", "evidence"):
        value = source.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    signature = (
                        item.get("evidence_signature")
                        or item.get("case_id")
                        or item.get("id")
                    )
                    if signature:
                        signatures.append(str(signature))

    unit_pass = (
        closure.get("overall_pass") is True
        or closure.get("unit_closure_passed") is True
        or closure.get("simulator_unit_closure") is True
    )
    repeatability = (
        source.get("repeatability_passed") is True
        or source.get("all_repeatability_passed") is True
    )
    structural_repeatability = any(
        isinstance(item, dict)
        and item.get("status") == "pass"
        and int(item.get("executions", 0)) >= 2
        and len(set(item.get("makespan_cycles", []))) == 1
        for item in source.get("fresh_smokes", [])
    )
    repeatability = repeatability or structural_repeatability
    direct_validation = (
        unit_pass and source.get("direct_validation_passed") is True
        and source.get("p95_relative_error") is not None
    )
    evidence_for_replay = {
        "evidence_signatures": sorted(set(signatures)),
        "unit_closure_passed": unit_pass,
        "repeatability_passed": repeatability,
        "validation_passed": direct_validation,
        "p95_relative_error": float(source.get("p95_relative_error", 1.0)),
    }
    summary: dict[str, Any] = {
        "schema_version": "exp2.calibration_summary.v1",
        "target_unit_closure_passed": unit_pass,
        "repeatability_passed": repeatability,
        "direct_validation_passed": direct_validation,
        "p95_relative_error": source.get("p95_relative_error"),
        "evidence_signatures": sorted(set(signatures)),
        "hardware_unit_closure_path": (
            str(closure_path.relative_to(ROOT)) if closure_path.is_file() else None
        ),
        "source_evidence_path": (
            str(source_path.relative_to(ROOT)) if source_path.is_file() else None
        ),
        "hardware_unit_closure_digest": (
            _file_digest(closure_path) if closure_path.is_file() else None
        ),
        "source_evidence_digest": (
            _file_digest(source_path) if source_path.is_file() else None
        ),
        "publish_status": (
            "cycle_accurate_anchor_calibrated"
            if unit_pass and repeatability and direct_validation
            else "analytical_only_target_binding_unclosed"
        ),
    }
    summary["calibration_summary_digest"] = _canonical_digest(summary)
    return summary, evidence_for_replay


def _capacity_fields(audit: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "parameter_bytes",
        "optimizer_bytes",
        "gradient_bytes",
        "activation_peak_bytes",
        "kv_cache_bytes",
        "shared_weight_bytes",
        "replicated_weight_bytes",
        "total_resident_bytes",
        "hbm_capacity_bytes",
        "capacity_status",
        "capacity_headroom_bytes",
        "capacity_audit_digest",
        "memory_policy",
    )
    return {name: audit[name] for name in names if name in audit}



def _limitations_for_capacity(
    record: Mapping[str, Any], capacity_status: str
) -> list[str]:
    tags = set(record.get("limitation_tags", []))
    tags -= {"capacity_projection", "capacity_infeasible_projection"}
    if capacity_status == "capacity_infeasible_projection":
        tags |= {"capacity_projection", "capacity_infeasible_projection"}
    return sorted(str(tag) for tag in tags)


def _provenance(
    manifest: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    hardware_path = ROOT / "configs" / "target_hardware.json"
    simulation_path = ROOT / "configs" / "target_simulation.json"
    return {
        "model_manifest_digest": manifest["manifest_digest"],
        "hardware_digest": _file_digest(hardware_path),
        "simulation_digest": _file_digest(simulation_path),
        "tool_digest": _file_digest(Path(__file__)),
        "e2e_replay_tool_digest": _file_digest(ROOT / "e2e_replay.py"),
        "capacity_tool_digest": _file_digest(ROOT / "capacity_model.py"),
        "calibration_summary_digest": calibration["calibration_summary_digest"],
        "target_unit_closure": bool(calibration["target_unit_closure_passed"]),
        "calibration_status": (
            "target_cycle_validation_passed"
            if calibration["direct_validation_passed"]
            else "target_cycle_validation_pending"
        ),
    }


def _uncertainty_fraction(
    estimate_source: str, capacity_status: str
) -> float:
    value = 0.35 if estimate_source == "analytical_only_mla" else 0.25
    if capacity_status == "capacity_infeasible_projection":
        value += 0.10
    return min(value, 0.49)


def _speed_interval(base: float, overlap: float, uncertainty: float) -> tuple[float, float]:
    return (
        base * (1.0 - uncertainty) / (overlap * (1.0 + uncertainty)),
        base * (1.0 + uncertainty) / (overlap * (1.0 - uncertainty)),
    )


def _finish_record(
    record: dict[str, Any],
    manifest: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    # Re-finalizing a copied primary record must not make the new digest depend
    # on the previous digest (the lambda=1 MoE sensitivity case does this).
    record.pop("result_digest", None)
    limitations = set(record.get("limitation_tags", []))
    if not calibration["target_unit_closure_passed"]:
        limitations.add("target_hardware_unit_closure_failed")
    if not calibration["direct_validation_passed"]:
        limitations.add("no_target_direct_counterfactual_validation")
    record["limitation_tags"] = sorted(limitations)
    record.update(_provenance(manifest, calibration))
    record["workload_digest"] = _canonical_digest({
        key: record.get(key)
        for key in (
            "workload",
            "model_id",
            "seq_len",
            "batch_size",
            "kv_length",
            "output_tokens",
            "routing_skew",
            "placement",
            "weight_mode",
            "include_shared_experts",
            "model_manifest_digest",
            "tp",
            "dp",
            "ep",
        )
    })
    record["result_digest"] = _canonical_digest(record)
    return record


def build_training_records(
    manifests: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest in manifests.values():
        for seq_len in TRAIN_SEQUENCE_LENGTHS:
            capacity = audit_training_case(manifest, seq_len)
            estimate = estimate_training_case(
                manifest,
                seq_len,
                evidence=evidence,
                capacity_status=capacity["capacity_status"],
            )
            tokens_per_step = DP_OR_EP_RANKS * seq_len
            base = float(estimate["T_base_cycles"])
            overlap = float(estimate["T_overlap_cycles"])
            uncertainty = _uncertainty_fraction(
                str(estimate["estimate_source"]), capacity["capacity_status"]
            )
            low, high = _speed_interval(base, overlap, uncertainty)
            record = {
                **estimate,
                **_capacity_fields(capacity),
                **_full_training_fields(estimate, tokens_per_step, uncertainty),
                "case_id": f"train__{manifest['model_id']}__s{seq_len}",
                "workload": "training_full_step",
                "model": manifest["display_name"],
                "model_id": manifest["model_id"],
                "model_family": "moe" if manifest["routed_expert_count"] else "dense",
                "attention_type": manifest["attention_type"],
                "seq_len": seq_len,
                "batch_size": 1,
                "routing_skew": 1.0,
                "placement": (
                    "tp3x3_ep2x2_noncompact"
                    if manifest["routed_expert_count"]
                    else "tp3x3_dp2x2"
                ),
                "training_tokens_per_step": tokens_per_step,
                "T_base_seconds": base / CLOCK_HZ,
                "T_overlap_seconds": overlap / CLOCK_HZ,
                "training_tokens_per_s_base": tokens_per_step * CLOCK_HZ / base,
                "training_tokens_per_s_overlap": tokens_per_step * CLOCK_HZ / overlap,
                "speedup": base / overlap,
                "uncertainty_fraction": uncertainty,
                "uncertainty_low": low,
                "uncertainty_high": high,
                "status": capacity["capacity_status"],
                "limitation_tags": _limitations_for_capacity(
                    estimate, str(capacity["capacity_status"])
                ),
            }
            records.append(_finish_record(record, manifest, calibration))
    return records


def _decode_primary(
    estimate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    batch_size: int,
    capacity: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    base = float(estimate["base_phase_cycles"]["decode"])
    overlap = float(estimate["overlap_phase_cycles"]["decode"])
    lower = float(estimate["decode_theory_lower_cycles"])
    estimate_source = str(estimate["estimate_source"])
    uncertainty = _uncertainty_fraction(
        estimate_source, str(capacity["capacity_status"])
    )
    low, high = _speed_interval(base, overlap, uncertainty)
    prefill_base = float(estimate["base_phase_cycles"]["prefill"])
    prefill_overlap = float(estimate["overlap_phase_cycles"]["prefill"])
    handoff_base = float(estimate["base_phase_cycles"]["handoff"])
    handoff_overlap = float(estimate["overlap_phase_cycles"]["handoff"])
    wait_base = float(estimate["base_phase_cycles"]["handoff_wait"])
    wait_overlap = float(estimate["overlap_phase_cycles"]["handoff_wait"])
    record = {
        **estimate,
        **_capacity_fields(capacity),
        "case_id": f"decode__{manifest['model_id']}__b{batch_size}",
        "workload": "inference_decode_steady_step",
        "model": manifest["display_name"],
        "model_id": manifest["model_id"],
        "model_family": "moe" if manifest["routed_expert_count"] else "dense",
        "attention_type": manifest["attention_type"],
        "seq_len": 1,
        "batch_size": batch_size,
        "kv_length": KV_LENGTH,
        "routing_skew": 1.0,
        "placement": "pd_4p_2d_each_2x3",
        "weight_mode": PRIMARY_WEIGHT_MODE,
        "composite_T_base_cycles": estimate["T_base_cycles"],
        "composite_T_overlap_cycles": estimate["T_overlap_cycles"],
        "T_base_cycles": base,
        "T_overlap_cycles": overlap,
        "T_base_seconds": base / CLOCK_HZ,
        "T_overlap_seconds": overlap / CLOCK_HZ,
        "speedup": base / overlap,
        "theory_lower_cycles": lower,
        "theory_speedup": base / lower,
        "attainment": lower / overlap,
        "uncertainty_fraction": uncertainty,
        "uncertainty_low": low,
        "uncertainty_high": high,
        "system_decode_tokens_per_s_base": (
            D_INSTANCE_COUNT * batch_size * CLOCK_HZ / base
        ),
        "system_decode_tokens_per_s_overlap": (
            D_INSTANCE_COUNT * batch_size * CLOCK_HZ / overlap
        ),
        "decode_step_latency_seconds_base": base / CLOCK_HZ,
        "decode_step_latency_seconds_overlap": overlap / CLOCK_HZ,
        "TPOT_seconds_base": base / CLOCK_HZ,
        "TPOT_seconds_overlap": overlap / CLOCK_HZ,
        "prefill_cycles_base": prefill_base,
        "prefill_cycles_overlap": prefill_overlap,
        "handoff_cycles_base": handoff_base,
        "handoff_cycles_overlap": handoff_overlap,
        "handoff_wait_cycles_base": wait_base,
        "handoff_wait_cycles_overlap": wait_overlap,
        "TTFT_cycles_base": prefill_base + handoff_base + wait_base,
        "TTFT_cycles_overlap": prefill_overlap + handoff_overlap + wait_overlap,
        "status": capacity["capacity_status"],
        "limitation_tags": _limitations_for_capacity(
            estimate, str(capacity["capacity_status"])
        ),
    }
    return _finish_record(record, manifest, calibration)


def build_inference_records(
    manifests: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decode: list[dict[str, Any]] = []
    prefill_pd: list[dict[str, Any]] = []
    for manifest in manifests.values():
        by_batch: dict[int, dict[str, Any]] = {}
        for batch_size in DECODE_BATCH_SIZES:
            capacity = audit_inference_case(
                manifest, batch_size, PRIMARY_WEIGHT_MODE
            )
            estimate = estimate_inference_case(
                manifest,
                batch_size,
                prefill_seq=REQUEST_PREFILL_SEQUENCE_LENGTH,
                kv_length=KV_LENGTH,
                evidence=evidence,
                capacity_status=capacity["capacity_status"],
            )
            record = _decode_primary(
                estimate, manifest, batch_size, capacity, calibration
            )
            decode.append(record)
            by_batch[batch_size] = record

        for prefill_seq in PREFILL_SEQUENCE_LENGTHS:
            if prefill_seq == REQUEST_PREFILL_SEQUENCE_LENGTH:
                source = by_batch[64]
            else:
                capacity = audit_inference_case(
                    manifest, 64, PRIMARY_WEIGHT_MODE
                )
                estimate = estimate_inference_case(
                    manifest,
                    64,
                    prefill_seq=prefill_seq,
                    kv_length=KV_LENGTH,
                    evidence=evidence,
                    capacity_status=capacity["capacity_status"],
                )
                source = _decode_primary(
                    estimate, manifest, 64, capacity, calibration
                )
            prefill_kv_bytes = (
                P_INSTANCE_COUNT
                * prefill_seq
                * manifest["num_layers"]
                * kv_elements_per_token_per_layer(manifest)
                * 2
            )
            shared_weights = manifest["parameter_count"] * 2
            resident = shared_weights + prefill_kv_bytes
            status = (
                "capacity_feasible"
                if resident <= HBM_CAPACITY_BYTES
                else "capacity_infeasible_projection"
            )
            base = float(source["TTFT_cycles_base"])
            overlap = float(source["TTFT_cycles_overlap"])
            uncertainty = _uncertainty_fraction(
                str(source["estimate_source"]), status
            )
            low, high = _speed_interval(base, overlap, uncertainty)
            lower = sum(
                float(source["phase_theory_lower_cycles"].get(phase, 0.0))
                for phase in ("prefill", "handoff", "handoff_wait")
            )
            record = {
                "case_id": f"prefill_pd__{manifest['model_id']}__s{prefill_seq}",
                "workload": "inference_prefill_pd_ttft",
                "model": manifest["display_name"],
                "model_id": manifest["model_id"],
                "model_family": "moe" if manifest["routed_expert_count"] else "dense",
                "attention_type": manifest["attention_type"],
                "seq_len": prefill_seq,
                "batch_size": 1,
                "kv_length": 0,
                "routing_skew": 1.0,
                "placement": "pd_4p_2d_each_2x3",
                "weight_mode": PRIMARY_WEIGHT_MODE,
                "prefill_cycles_base": source["prefill_cycles_base"],
                "prefill_cycles_overlap": source["prefill_cycles_overlap"],
                "handoff_cycles_base": source["handoff_cycles_base"],
                "handoff_cycles_overlap": source["handoff_cycles_overlap"],
                "handoff_wait_cycles_base": source["handoff_wait_cycles_base"],
                "handoff_wait_cycles_overlap": source["handoff_wait_cycles_overlap"],
                "T_base_cycles": base,
                "T_overlap_cycles": overlap,
                "TTFT_seconds_base": base / CLOCK_HZ,
                "TTFT_seconds_overlap": overlap / CLOCK_HZ,
                "speedup": base / overlap,
                "theory_lower_cycles": lower,
                "theory_speedup": base / lower,
                "attainment": lower / overlap,
                "uncertainty_fraction": uncertainty,
                "uncertainty_low": low,
                "uncertainty_high": high,
                "parameter_bytes": shared_weights,
                "optimizer_bytes": 0,
                "gradient_bytes": 0,
                "activation_peak_bytes": 0,
                "kv_cache_bytes": prefill_kv_bytes,
                "shared_weight_bytes": shared_weights,
                "replicated_weight_bytes": 0,
                "total_resident_bytes": resident,
                "hbm_capacity_bytes": HBM_CAPACITY_BYTES,
                "capacity_headroom_bytes": HBM_CAPACITY_BYTES - resident,
                "capacity_status": status,
                "estimate_source": source["estimate_source"],
                "evidence_signatures": source["evidence_signatures"],
                "limitation_tags": sorted(
                    set(_limitations_for_capacity(source, status))
                    - {
                        "decode_shape_efficiency_analytical",
                        "decode_two_token_recurrence",
                        "decode_two_token_local_interval",
                        "decode_steady_state_convergence_not_checked",
                        "kv_append_hbm_write_analytical",
                    }
                ),
                "status": status,
            }
            prefill_pd.append(_finish_record(record, manifest, calibration))
    return decode, prefill_pd


def build_inference_request_records(
    manifests: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any],
    decode_records: Iterable[Mapping[str, Any]],
    prefill_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compose TTFT and fixed-context local TPOT into a request-level metric."""

    prefill_by_model = {
        str(item["model_id"]): dict(item)
        for item in prefill_records
        if int(item["seq_len"]) == REQUEST_PREFILL_SEQUENCE_LENGTH
    }
    records: list[dict[str, Any]] = []
    for decode_item in decode_records:
        decode = dict(decode_item)
        model_id = str(decode["model_id"])
        manifest = manifests[model_id]
        prefill = prefill_by_model[model_id]
        batch_size = int(decode["batch_size"])
        ttft_base = float(prefill["T_base_cycles"])
        ttft_overlap = float(prefill["T_overlap_cycles"])
        local_tpot_base = float(decode["T_base_cycles"])
        local_tpot_overlap = float(decode["T_overlap_cycles"])
        decode_total_base = REQUEST_OUTPUT_TOKENS * local_tpot_base
        decode_total_overlap = REQUEST_OUTPUT_TOKENS * local_tpot_overlap
        base = ttft_base + decode_total_base
        overlap = ttft_overlap + decode_total_overlap
        status = str(decode["capacity_status"])
        estimate_source = str(decode["estimate_source"])
        uncertainty = _uncertainty_fraction(estimate_source, status)
        low, high = _speed_interval(base, overlap, uncertainty)
        lower = (
            float(prefill["theory_lower_cycles"])
            + REQUEST_OUTPUT_TOKENS * float(decode["theory_lower_cycles"])
        )
        limitations = set(prefill.get("limitation_tags", [])) | set(
            decode.get("limitation_tags", [])
        )
        limitations |= {
            "composite_prefill_decode_formula",
            "constant_local_tpot_extrapolated_over_512_tokens",
            "decode_kv_growth_within_generation_not_replayed",
        }
        record = {
            **_capacity_fields(decode),
            "case_id": (
                f"request_e2e__{model_id}__b{batch_size}"
                f"__g{REQUEST_OUTPUT_TOKENS}"
            ),
            "workload": "inference_request_prefill_decode_composite",
            "model": decode["model"],
            "model_id": model_id,
            "model_family": decode["model_family"],
            "attention_type": decode["attention_type"],
            "seq_len": REQUEST_PREFILL_SEQUENCE_LENGTH,
            "batch_size": batch_size,
            "kv_length": KV_LENGTH,
            "output_tokens": REQUEST_OUTPUT_TOKENS,
            "routing_skew": decode["routing_skew"],
            "placement": decode["placement"],
            "weight_mode": decode["weight_mode"],
            "include_shared_experts": decode.get("include_shared_experts", False),
            "request_formula": "TTFT_cycles + output_tokens * local_TPOT_cycles",
            "decode_context_policy": "fixed_kv_36864_local_tpot_constant",
            "same_work_invariant": True,
            "TTFT_cycles_base": ttft_base,
            "TTFT_cycles_overlap": ttft_overlap,
            "local_TPOT_cycles_base": local_tpot_base,
            "local_TPOT_cycles_overlap": local_tpot_overlap,
            "decode_total_cycles_base": decode_total_base,
            "decode_total_cycles_overlap": decode_total_overlap,
            "T_base_cycles": base,
            "T_overlap_cycles": overlap,
            "T_base_seconds": base / CLOCK_HZ,
            "T_overlap_seconds": overlap / CLOCK_HZ,
            "request_latency_seconds_base": base / CLOCK_HZ,
            "request_latency_seconds_overlap": overlap / CLOCK_HZ,
            "request_output_tokens_per_s_base": (
                REQUEST_OUTPUT_TOKENS * CLOCK_HZ / base
            ),
            "request_output_tokens_per_s_overlap": (
                REQUEST_OUTPUT_TOKENS * CLOCK_HZ / overlap
            ),
            "ttft_fraction_base": ttft_base / base,
            "ttft_fraction_overlap": ttft_overlap / overlap,
            "decode_fraction_base": decode_total_base / base,
            "decode_fraction_overlap": decode_total_overlap / overlap,
            "speedup": base / overlap,
            "theory_lower_cycles": lower,
            "theory_speedup": base / lower,
            "attainment": lower / overlap,
            "uncertainty_fraction": uncertainty,
            "uncertainty_low": low,
            "uncertainty_high": high,
            "estimate_source": estimate_source,
            "evidence_signatures": decode["evidence_signatures"],
            "prefill_result_digest": prefill["result_digest"],
            "decode_result_digest": decode["result_digest"],
            "status": status,
            "limitation_tags": sorted(limitations),
        }
        for field in ("tp", "dp", "ep"):
            if field in decode:
                record[field] = decode[field]
        records.append(_finish_record(record, manifest, calibration))
    return records


def build_moe_skew_records(
    manifests: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any],
    evidence: Mapping[str, Any],
    training_primary: Iterable[Mapping[str, Any]],
    inference_primary: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    primary_train = {
        (str(item["model_id"]), int(item["seq_len"])): dict(item)
        for item in training_primary
    }
    primary_infer = {
        (str(item["model_id"]), int(item["batch_size"])): dict(item)
        for item in inference_primary
    }
    for manifest in manifests.values():
        if not manifest["routed_expert_count"]:
            continue
        for skew in ROUTING_SKEWS:
            for seq_len in TRAIN_SEQUENCE_LENGTHS:
                if skew == 1.0:
                    record = dict(primary_train[(manifest["model_id"], seq_len)])
                else:
                    capacity = audit_training_case(manifest, seq_len)
                    estimate = estimate_training_case(
                        manifest,
                        seq_len,
                        routing_skew=skew,
                        evidence=evidence,
                        capacity_status=capacity["capacity_status"],
                    )
                    base = float(estimate["T_base_cycles"])
                    overlap = float(estimate["T_overlap_cycles"])
                    uncertainty = _uncertainty_fraction(
                        str(estimate["estimate_source"]), capacity["capacity_status"]
                    )
                    low, high = _speed_interval(base, overlap, uncertainty)
                    record = {
                        **estimate,
                        **_capacity_fields(capacity),
                        **_full_training_fields(
                            estimate, DP_OR_EP_RANKS * seq_len, uncertainty
                        ),
                        "model": manifest["display_name"],
                        "model_id": manifest["model_id"],
                        "workload": "training_full_step",
                        "model_family": "moe",
                        "attention_type": manifest["attention_type"],
                        "seq_len": seq_len,
                        "training_tokens_per_step": DP_OR_EP_RANKS * seq_len,
                        "T_base_seconds": base / CLOCK_HZ,
                        "T_overlap_seconds": overlap / CLOCK_HZ,
                        "training_tokens_per_s_base": (
                            DP_OR_EP_RANKS * seq_len * CLOCK_HZ / base
                        ),
                        "training_tokens_per_s_overlap": (
                            DP_OR_EP_RANKS * seq_len * CLOCK_HZ / overlap
                        ),
                        "batch_size": 1,
                        "placement": "tp3x3_ep2x2_noncompact",
                        "speedup": base / overlap,
                        "uncertainty_fraction": uncertainty,
                        "uncertainty_low": low,
                        "uncertainty_high": high,
                        "status": capacity["capacity_status"],
                        "limitation_tags": _limitations_for_capacity(
                            estimate, str(capacity["capacity_status"])
                        ),
                    }
                record["case_id"] = (
                    f"skew__train__{manifest['model_id']}__s{seq_len}__l{skew}"
                )
                record["routing_skew"] = skew
                record["sensitivity_workload"] = "training"
                records.append(_finish_record(record, manifest, calibration))

            for batch_size in DECODE_BATCH_SIZES:
                if skew == 1.0:
                    record = dict(primary_infer[(manifest["model_id"], batch_size)])
                else:
                    capacity = audit_inference_case(
                        manifest, batch_size, PRIMARY_WEIGHT_MODE
                    )
                    estimate = estimate_inference_case(
                        manifest,
                        batch_size,
                        routing_skew=skew,
                        evidence=evidence,
                        capacity_status=capacity["capacity_status"],
                    )
                    record = _decode_primary(
                        estimate, manifest, batch_size, capacity, calibration
                    )
                record["case_id"] = (
                    f"skew__decode__{manifest['model_id']}__b{batch_size}__l{skew}"
                )
                record["routing_skew"] = skew
                record["sensitivity_workload"] = "inference_decode"
                records.append(_finish_record(record, manifest, calibration))
    return records


def write_records(records: list[dict[str, Any]], stem: Path) -> None:
    if not records:
        raise ValueError(f"cannot write empty result set: {stem}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = sorted({key for record in records for key in record})
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in record.items()
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results"
    )
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    manifests = load_model_manifests()
    calibration, evidence = load_calibration_summary()
    training = build_training_records(manifests, calibration, evidence)
    inference, prefill_pd = build_inference_records(
        manifests, calibration, evidence
    )
    inference_request = build_inference_request_records(
        manifests, calibration, inference, prefill_pd
    )
    skew = (
        []
        if args.skip_sensitivity
        else build_moe_skew_records(
            manifests, calibration, evidence, training, inference
        )
    )
    shared = (
        []
        if args.skip_sensitivity
        else build_shared_expert_records(
            manifests, calibration, evidence, training, inference
        )
    )

    capacity = build_capacity_audit()

    write_records(training, args.output_dir / "training_e2e")
    write_records(inference, args.output_dir / "inference_decode_e2e")
    write_records(
        prefill_pd, args.output_dir / "inference_prefill_pd_breakdown"
    )
    write_records(
        inference_request, args.output_dir / "inference_request_e2e"
    )
    if skew:
        write_records(skew, args.output_dir / "moe_skew_sensitivity")
    if shared:
        write_records(
            shared, args.output_dir / "deepseek_shared_expert_sensitivity")
    (args.output_dir / "capacity_audit.json").write_text(
        json.dumps(capacity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "training_cases": len(training),
        "inference_decode_cases": len(inference),
        "prefill_pd_cases": len(prefill_pd),
        "inference_request_cases": len(inference_request),
        "moe_skew_cases": len(skew),
        "deepseek_shared_expert_cases": len(shared),
        "capacity_training_cases": capacity["training_case_count"],
        "capacity_inference_cases": capacity["inference_case_count"],
        "capacity_infeasible_training": sum(
            item["capacity_status"] == "capacity_infeasible_projection"
            for item in capacity["training"]
        ),
        "capacity_infeasible_inference": sum(
            item["capacity_status"] == "capacity_infeasible_projection"
            for item in capacity["inference"]
        ),
        "publish_status": calibration["publish_status"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_shared_expert_records(
    manifests: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any],
    evidence: Mapping[str, Any],
    training_primary: Iterable[Mapping[str, Any]],
    inference_primary: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """DeepSeek routed-only primary path versus explicit shared-expert opt-in."""

    manifest = manifests["deepseek_v3"]
    primary_train = {
        int(item["seq_len"]): dict(item)
        for item in training_primary
        if item["model_id"] == manifest["model_id"]
    }
    primary_infer = {
        int(item["batch_size"]): dict(item)
        for item in inference_primary
        if item["model_id"] == manifest["model_id"]
    }
    records: list[dict[str, Any]] = []
    for include_shared in (False, True):
        mode = "routed_plus_shared" if include_shared else "routed_only"
        for seq_len in TRAIN_SEQUENCE_LENGTHS:
            if not include_shared:
                record = dict(primary_train[seq_len])
            else:
                capacity = audit_training_case(manifest, seq_len)
                estimate = estimate_training_case(
                    manifest,
                    seq_len,
                    include_shared_experts=True,
                    evidence=evidence,
                    capacity_status=capacity["capacity_status"],
                )
                base = float(estimate["T_base_cycles"])
                overlap = float(estimate["T_overlap_cycles"])
                uncertainty = _uncertainty_fraction(
                    str(estimate["estimate_source"]),
                    str(capacity["capacity_status"]),
                )
                low, high = _speed_interval(base, overlap, uncertainty)
                tokens = DP_OR_EP_RANKS * seq_len
                record = {
                    **estimate,
                    **_capacity_fields(capacity),
                    **_full_training_fields(estimate, tokens, uncertainty),
                    "workload": "training_full_step",
                    "model": manifest["display_name"],
                    "model_id": manifest["model_id"],
                    "model_family": "moe",
                    "attention_type": manifest["attention_type"],
                    "seq_len": seq_len,
                    "batch_size": 1,
                    "routing_skew": 1.0,
                    "placement": "tp3x3_ep2x2_noncompact",
                    "training_tokens_per_step": tokens,
                    "T_base_seconds": base / CLOCK_HZ,
                    "T_overlap_seconds": overlap / CLOCK_HZ,
                    "training_tokens_per_s_base": tokens * CLOCK_HZ / base,
                    "training_tokens_per_s_overlap": tokens * CLOCK_HZ / overlap,
                    "speedup": base / overlap,
                    "uncertainty_fraction": uncertainty,
                    "uncertainty_low": low,
                    "uncertainty_high": high,
                    "status": capacity["capacity_status"],
                    "limitation_tags": _limitations_for_capacity(
                        estimate, str(capacity["capacity_status"])
                    ),
                }
            record["case_id"] = f"shared__train__s{seq_len}__{mode}"
            record["include_shared_experts"] = include_shared
            record["shared_expert_mode"] = mode
            record["sensitivity_workload"] = "training"
            record["shared_expert_count"] = manifest["shared_expert_count"]
            records.append(_finish_record(record, manifest, calibration))

        for batch_size in DECODE_BATCH_SIZES:
            if not include_shared:
                record = dict(primary_infer[batch_size])
            else:
                capacity = audit_inference_case(
                    manifest, batch_size, PRIMARY_WEIGHT_MODE
                )
                estimate = estimate_inference_case(
                    manifest,
                    batch_size,
                    include_shared_experts=True,
                    evidence=evidence,
                    capacity_status=capacity["capacity_status"],
                )
                record = _decode_primary(
                    estimate, manifest, batch_size, capacity, calibration
                )
            record["case_id"] = f"shared__decode__b{batch_size}__{mode}"
            record["include_shared_experts"] = include_shared
            record["shared_expert_mode"] = mode
            record["sensitivity_workload"] = "inference_decode"
            record["shared_expert_count"] = manifest["shared_expert_count"]
            records.append(_finish_record(record, manifest, calibration))
    return records


def _full_training_fields(
    estimate: Mapping[str, Any],
    tokens_per_step: int,
    uncertainty: float,
) -> dict[str, Any]:
    base = float(estimate["T_base_cycles"])
    forward_only = float(estimate["T_forward_only_overlap_cycles"])
    full = float(estimate["T_full_train_overlap_cycles"])
    low, high = _speed_interval(base, full, uncertainty)
    return {
        "primary_training_state": "full_train_overlap",
        "T_forward_only_overlap_cycles": forward_only,
        "T_forward_only_overlap_seconds": forward_only / CLOCK_HZ,
        "T_full_train_overlap_cycles": full,
        "T_full_train_overlap_seconds": full / CLOCK_HZ,
        "speedup_forward_only": base / forward_only,
        "speedup_full_train": base / full,
        "training_tokens_per_s_forward_only": (
            tokens_per_step * CLOCK_HZ / forward_only
        ),
        "training_tokens_per_s_full_train": tokens_per_step * CLOCK_HZ / full,
        "uncertainty_full_train_low": low,
        "uncertainty_full_train_high": high,
    }


if __name__ == "__main__":
    raise SystemExit(main())
