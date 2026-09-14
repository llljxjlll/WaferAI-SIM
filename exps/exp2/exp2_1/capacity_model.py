"""64 GiB HBM capacity gates for exp2-1 training and PD inference.

All arithmetic is integer byte arithmetic.  The result is exact with respect to
the frozen model manifest and the explicitly named memory policy; it is not a
claim that a rounded public model name is an exact checkpoint tensor count.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

try:  # Support both standalone-script and package imports.
    from .model_manifests import canonical_bytes, load_model_manifests
except ImportError:  # pragma: no cover - exercised by the standalone test path
    from model_manifests import canonical_bytes, load_model_manifests


GIB = 1024**3
HBM_CAPACITY_BYTES = 4 * 16 * GIB
TRAIN_SEQUENCE_LENGTHS = (2304, 36864)
DECODE_BATCH_SIZES = (64, 512)
PREFILL_SEQUENCE_LENGTH = 2304
KV_LENGTH = 36864
P_INSTANCE_COUNT = 4
D_INSTANCE_COUNT = 2
TOTAL_INFERENCE_INSTANCE_COUNT = P_INSTANCE_COUNT + D_INSTANCE_COUNT


@dataclass(frozen=True)
class TrainingMemoryPolicy:
    weight_bytes_per_parameter: int = 2
    gradient_bytes_per_parameter: int = 2
    master_weight_bytes_per_parameter: int = 4
    adam_m_bytes_per_parameter: int = 4
    adam_v_bytes_per_parameter: int = 4
    activation_bytes_per_element: int = 2
    checkpoint_strategy: str = "per_transformer_block"
    optimizer: str = "AdamW"
    optimizer_sharding: str = "none"

    @property
    def optimizer_bytes_per_parameter(self) -> int:
        return (
            self.master_weight_bytes_per_parameter
            + self.adam_m_bytes_per_parameter
            + self.adam_v_bytes_per_parameter
        )


DEFAULT_TRAINING_POLICY = TrainingMemoryPolicy()


def _status(total: int) -> str:
    return "capacity_feasible" if total <= HBM_CAPACITY_BYTES else "capacity_infeasible_projection"


def _digest(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "capacity_audit_digest"}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _training_resident_parameter_count(manifest: Mapping[str, Any]) -> tuple[int, dict[str, int]]:
    breakdown = manifest["parameter_breakdown"]
    if manifest["routed_expert_count"] == 0:
        factors = {name: 4 for name in breakdown}
        # Four 3x3 TP groups are four unsharded DP replicas.
        return 4 * manifest["parameter_count"], factors

    # Four 3x3 groups are EP ranks. Routed experts are partitioned once across
    # EP=4; attention, embeddings, routers and shared experts are replicated.
    factors = {name: 4 for name in breakdown}
    factors["routed_experts"] = 1
    resident = sum(breakdown[name] * factors[name] for name in breakdown)
    return resident, factors


def _standard_kv_elements_per_token_per_layer(manifest: Mapping[str, Any]) -> int:
    return 2 * manifest["num_kv_heads"] * manifest["head_dim"]


def kv_elements_per_token_per_layer(manifest: Mapping[str, Any]) -> int:
    if manifest["attention_type"] == "MLA":
        dims = manifest["mla_dimensions"]
        # The compressed latent and decoupled RoPE key are retained.  Treating
        # MLA as standard 128-head KV would overstate cache by over 50x.
        return dims["kv_lora_rank"] + dims["qk_rope_head_dim"]
    return _standard_kv_elements_per_token_per_layer(manifest)


def _activation_peak_bytes(
    manifest: Mapping[str, Any], seq_len: int, policy: TrainingMemoryPolicy
) -> tuple[int, dict[str, int]]:
    tokens = seq_len  # batch_size=1/rank in the fixed experiment workload.
    hidden = manifest["hidden_size"]
    layers = manifest["num_layers"]
    checkpoint_elements = (layers + 1) * tokens * hidden

    if manifest["attention_type"] == "MLA":
        dims = manifest["mla_dimensions"]
        q_width = manifest["num_attention_heads"] * (
            dims["qk_nope_head_dim"] + dims["qk_rope_head_dim"]
        )
        kv_width = dims["kv_lora_rank"] + dims["qk_rope_head_dim"]
        attention_workspace_elements = tokens * (q_width + kv_width + hidden)
    else:
        kv_width = manifest["num_kv_heads"] * manifest["head_dim"]
        attention_workspace_elements = tokens * (2 * hidden + 2 * kv_width)

    if manifest["routed_expert_count"]:
        moe_intermediate = manifest.get("moe_intermediate_size", manifest["intermediate_size"])
        routed_workspace = tokens * manifest["top_k"] * (2 * hidden + 2 * moe_intermediate)
        shared_workspace = tokens * (
            hidden + 2 * moe_intermediate * manifest["shared_expert_count"]
        )
        mlp_workspace_elements = routed_workspace + shared_workspace
        activation_replication_factor = 1
    else:
        mlp_factor = 1 if manifest["mlp_type"] == "dense_gelu" else 2
        mlp_workspace_elements = tokens * (
            hidden + mlp_factor * manifest["intermediate_size"]
        )
        activation_replication_factor = 4

    workspace_elements = max(attention_workspace_elements, mlp_workspace_elements)
    elements_per_replica = checkpoint_elements + workspace_elements
    total = elements_per_replica * activation_replication_factor * policy.activation_bytes_per_element
    return total, {
        "checkpoint_elements_per_replica": checkpoint_elements,
        "attention_workspace_elements_per_replica": attention_workspace_elements,
        "mlp_workspace_elements_per_replica": mlp_workspace_elements,
        "workspace_peak_elements_per_replica": workspace_elements,
        "activation_replication_factor": activation_replication_factor,
        "activation_bytes_per_element": policy.activation_bytes_per_element,
    }


def audit_training_case(
    manifest: Mapping[str, Any],
    seq_len: int,
    policy: TrainingMemoryPolicy = DEFAULT_TRAINING_POLICY,
) -> dict[str, Any]:
    if seq_len not in TRAIN_SEQUENCE_LENGTHS:
        raise ValueError(f"seq_len must be one of {TRAIN_SEQUENCE_LENGTHS}")
    resident_params, placement_factors = _training_resident_parameter_count(manifest)
    parameter_bytes = resident_params * policy.weight_bytes_per_parameter
    optimizer_bytes = resident_params * policy.optimizer_bytes_per_parameter
    gradient_bytes = resident_params * policy.gradient_bytes_per_parameter
    activation_peak_bytes, activation_formula = _activation_peak_bytes(manifest, seq_len, policy)
    total = parameter_bytes + optimizer_bytes + gradient_bytes + activation_peak_bytes
    record: dict[str, Any] = {
        "case_id": f"train:{manifest['model_id']}:s{seq_len}",
        "workload": "training_full_step",
        "model": manifest["display_name"],
        "model_id": manifest["model_id"],
        "model_manifest_digest": manifest["manifest_digest"],
        "seq_len": seq_len,
        "batch_size_per_rank": 1,
        "placement": "dense_tp9_dp4" if not manifest["routed_expert_count"] else "moe_tp9_ep4",
        "parameter_bytes": parameter_bytes,
        "optimizer_bytes": optimizer_bytes,
        "gradient_bytes": gradient_bytes,
        "activation_peak_bytes": activation_peak_bytes,
        "kv_cache_bytes": 0,
        "shared_weight_bytes": 0,
        "replicated_weight_bytes": parameter_bytes,
        "total_resident_bytes": total,
        "hbm_capacity_bytes": HBM_CAPACITY_BYTES,
        "capacity_status": _status(total),
        "capacity_headroom_bytes": HBM_CAPACITY_BYTES - total,
        "resident_parameter_count": resident_params,
        "parameter_placement_factors": placement_factors,
        "memory_policy": {
            "weight_dtype": "FP16",
            "weight_bytes_per_parameter": policy.weight_bytes_per_parameter,
            "gradient_dtype": "FP16",
            "gradient_bytes_per_parameter": policy.gradient_bytes_per_parameter,
            "optimizer": policy.optimizer,
            "optimizer_sharding": policy.optimizer_sharding,
            "optimizer_state": "FP32 master weight + FP32 first moment + FP32 second moment",
            "optimizer_bytes_per_parameter": policy.optimizer_bytes_per_parameter,
            "activation_checkpoint": policy.checkpoint_strategy,
        },
        "activation_formula": activation_formula,
        "limitation_tags": ["capacity_projection"] if total > HBM_CAPACITY_BYTES else [],
    }
    record["capacity_audit_digest"] = _digest(record)
    return record


def audit_inference_case(
    manifest: Mapping[str, Any], batch_size: int, weight_mode: str
) -> dict[str, Any]:
    if batch_size not in DECODE_BATCH_SIZES:
        raise ValueError(f"batch_size must be one of {DECODE_BATCH_SIZES}")
    if weight_mode not in {"global_shared_read_only", "replicated_per_instance"}:
        raise ValueError("unknown inference weight mode")

    weight_bytes = manifest["parameter_count"] * 2  # target workload freezes FP16 weights.
    if weight_mode == "global_shared_read_only":
        shared_weight_bytes = weight_bytes
        replicated_weight_bytes = 0
        weight_copy_count = 1
    else:
        shared_weight_bytes = 0
        replicated_weight_bytes = TOTAL_INFERENCE_INSTANCE_COUNT * weight_bytes
        weight_copy_count = TOTAL_INFERENCE_INSTANCE_COUNT

    cache_sequences_tokens = (
        D_INSTANCE_COUNT * batch_size * KV_LENGTH
        + P_INSTANCE_COUNT * PREFILL_SEQUENCE_LENGTH
    )
    kv_elements = (
        cache_sequences_tokens
        * manifest["num_layers"]
        * kv_elements_per_token_per_layer(manifest)
    )
    kv_cache_bytes = kv_elements * 2
    total = shared_weight_bytes + replicated_weight_bytes + kv_cache_bytes
    record: dict[str, Any] = {
        "case_id": f"infer:{manifest['model_id']}:b{batch_size}:{weight_mode}",
        "workload": "pd_prefill_plus_decode_steady_residency",
        "model": manifest["display_name"],
        "model_id": manifest["model_id"],
        "model_manifest_digest": manifest["manifest_digest"],
        "seq_len": 1,
        "batch_size_per_decode_instance": batch_size,
        "kv_length": KV_LENGTH,
        "prefill_sequence_length_per_p_instance": PREFILL_SEQUENCE_LENGTH,
        "p_instance_count": P_INSTANCE_COUNT,
        "d_instance_count": D_INSTANCE_COUNT,
        "weight_mode": weight_mode,
        "weight_copy_count": weight_copy_count,
        "parameter_bytes": weight_bytes,
        "optimizer_bytes": 0,
        "gradient_bytes": 0,
        "activation_peak_bytes": 0,
        "kv_cache_bytes": kv_cache_bytes,
        "shared_weight_bytes": shared_weight_bytes,
        "replicated_weight_bytes": replicated_weight_bytes,
        "total_resident_bytes": total,
        "hbm_capacity_bytes": HBM_CAPACITY_BYTES,
        "capacity_status": _status(total),
        "capacity_headroom_bytes": HBM_CAPACITY_BYTES - total,
        "kv_cache_formula": {
            "resident_sequence_tokens": cache_sequences_tokens,
            "layers": manifest["num_layers"],
            "elements_per_token_per_layer": kv_elements_per_token_per_layer(manifest),
            "element_bytes": 2,
            "representation": "mla_compressed_latent_plus_rope_key"
            if manifest["attention_type"] == "MLA"
            else "standard_k_and_v",
        },
        "memory_policy": {
            "weight_dtype": "FP16",
            "kv_dtype": "FP16",
            "read_only_weight_sharing": weight_mode == "global_shared_read_only",
            "activation_workspace_excluded": "transient scratch is performance-model state, not resident-capacity state",
        },
        "limitation_tags": ["capacity_projection"] if total > HBM_CAPACITY_BYTES else [],
    }
    record["capacity_audit_digest"] = _digest(record)
    return record


def build_capacity_audit() -> dict[str, Any]:
    manifests = load_model_manifests()
    training = [
        audit_training_case(manifest, seq_len)
        for manifest in manifests.values()
        for seq_len in TRAIN_SEQUENCE_LENGTHS
    ]
    inference = [
        audit_inference_case(manifest, batch_size, weight_mode)
        for manifest in manifests.values()
        for batch_size in DECODE_BATCH_SIZES
        for weight_mode in ("global_shared_read_only", "replicated_per_instance")
    ]
    document: dict[str, Any] = {
        "schema_version": "exp2.capacity_audit.v1",
        "hbm_capacity_bytes": HBM_CAPACITY_BYTES,
        "hbm_layout": "4 x 16 GiB edge stacks; no implicit per-die HBM",
        "training_case_count": len(training),
        "inference_case_count": len(inference),
        "training": training,
        "inference": inference,
    }
    document["capacity_audit_digest"] = _digest(document)
    return document


__all__ = [
    "DECODE_BATCH_SIZES",
    "DEFAULT_TRAINING_POLICY",
    "HBM_CAPACITY_BYTES",
    "KV_LENGTH",
    "PREFILL_SEQUENCE_LENGTH",
    "TRAIN_SEQUENCE_LENGTHS",
    "TrainingMemoryPolicy",
    "audit_inference_case",
    "audit_training_case",
    "build_capacity_audit",
    "kv_elements_per_token_per_layer",
]
