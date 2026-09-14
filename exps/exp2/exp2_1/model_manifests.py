"""Load and validate the frozen exp2-1 model manifests.

The checked-in JSON files are the experiment inputs.  This module deliberately
does not query a model hub at run time: upstream URLs are provenance, while the
SHA-256 digest covers the local canonical content that was actually used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / "manifests" / "models"
SOURCE_DIR = ROOT / "manifests" / "sources"

SCHEMA_VERSION = "exp2.model_manifest.v1"
SOURCE_SCHEMA_VERSION = "exp2.model_source_snapshot.v1"
MODEL_ORDER = (
    "llama2_7b",
    "gpt3_175b",
    "llama3_8b",
    "llama3_1_405b",
    "mixtral_8x7b",
    "deepseek_v3",
)

REQUIRED_FIELDS = (
    "model_id",
    "display_name",
    "num_layers",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_kv_heads",
    "head_dim",
    "attention_type",
    "mlp_type",
    "norm_residual_rope",
    "moe_layer_frequency",
    "routed_expert_count",
    "shared_expert_count",
    "top_k",
    "parameter_count",
    "dtype",
    "parameter_breakdown",
    "sources",
    "assumptions",
)


class ManifestError(ValueError):
    """A frozen input is missing, inconsistent, or has a stale digest."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_digest(document: dict[str, Any], digest_field: str) -> str:
    payload = {key: value for key, value in document.items() if key != digest_field}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _require_positive_int(value: Any, field: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{field} must be an integer >= {minimum}, got {value!r}")


def validate_source_snapshot(document: dict[str, Any]) -> None:
    if document.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ManifestError("unexpected source snapshot schema")
    expected = content_digest(document, "snapshot_digest")
    if document.get("snapshot_digest") != expected:
        raise ManifestError("source snapshot digest mismatch")
    if not document.get("source_url") or not isinstance(document.get("fields"), dict):
        raise ManifestError("source snapshot requires source_url and fields")


def validate_manifest(
    document: dict[str, Any], *, source_dir: Path = SOURCE_DIR
) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("unexpected model manifest schema")
    missing = [field for field in REQUIRED_FIELDS if field not in document]
    if missing:
        raise ManifestError(f"missing required manifest fields: {missing}")
    for field in (
        "num_layers",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_kv_heads",
        "parameter_count",
    ):
        _require_positive_int(document[field], field)
    for field in (
        "moe_layer_frequency",
        "routed_expert_count",
        "shared_expert_count",
        "top_k",
    ):
        _require_positive_int(document[field], field, allow_zero=True)

    if document["attention_type"] == "MLA":
        if document["head_dim"] is not None:
            raise ManifestError("MLA must not be represented by one standard head_dim")
        components = document.get("mla_dimensions", {})
        for field in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
            _require_positive_int(components.get(field), f"mla_dimensions.{field}")
    else:
        _require_positive_int(document["head_dim"], "head_dim")
        if document["hidden_size"] != document["num_attention_heads"] * document["head_dim"]:
            raise ManifestError("hidden_size != num_attention_heads * head_dim")

    is_moe = document["mlp_type"] in {"sparse_moe_swiglu", "deepseek_moe_swiglu"}
    if is_moe != bool(document["routed_expert_count"]):
        raise ManifestError("MoE type and routed_expert_count disagree")
    if document["top_k"] > document["routed_expert_count"]:
        raise ManifestError("top_k exceeds routed expert count")

    breakdown = document["parameter_breakdown"]
    if not isinstance(breakdown, dict) or not breakdown:
        raise ManifestError("parameter_breakdown must be a non-empty object")
    for key, value in breakdown.items():
        _require_positive_int(value, f"parameter_breakdown.{key}", allow_zero=True)
    if sum(breakdown.values()) != document["parameter_count"]:
        raise ManifestError("parameter breakdown does not sum to parameter_count")

    expected = content_digest(document, "manifest_digest")
    if document.get("manifest_digest") != expected:
        raise ManifestError("model manifest digest mismatch")

    for source in document["sources"]:
        snapshot_name = source.get("local_snapshot")
        snapshot_digest = source.get("snapshot_digest")
        if not snapshot_name:
            continue
        path = source_dir / snapshot_name
        if not path.is_file():
            raise ManifestError(f"missing source snapshot: {path}")
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        validate_source_snapshot(snapshot)
        if snapshot["snapshot_digest"] != snapshot_digest:
            raise ManifestError(f"source digest mismatch for {snapshot_name}")


def load_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(document)
    return document


def load_model_manifests(
    model_ids: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    requested = tuple(model_ids) if model_ids is not None else MODEL_ORDER
    unknown = sorted(set(requested) - set(MODEL_ORDER))
    if unknown:
        raise ManifestError(f"unknown model ids: {unknown}")
    result = {model_id: load_manifest(MANIFEST_DIR / f"{model_id}.json") for model_id in requested}
    if tuple(result) != requested:
        raise ManifestError("manifest order changed unexpectedly")
    return result


def manifest_digest_map() -> dict[str, str]:
    return {
        model_id: manifest["manifest_digest"]
        for model_id, manifest in load_model_manifests().items()
    }


__all__ = [
    "MANIFEST_DIR",
    "MODEL_ORDER",
    "ManifestError",
    "canonical_bytes",
    "content_digest",
    "load_manifest",
    "load_model_manifests",
    "manifest_digest_map",
    "validate_manifest",
    "validate_source_snapshot",
]
