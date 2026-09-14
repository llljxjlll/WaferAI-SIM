#!/usr/bin/env python3
"""Read-only adapter for the frozen exp-2 workloads and control results.

The adapter deliberately does not import exp-2 Python modules.  The checked-in
JSON artefacts are the experiment boundary, and their raw SHA-256 plus per-row
``result_digest`` values are retained as provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


EXP4_ROOT = Path(__file__).resolve().parent
DEFAULT_EXP2_ROOT = EXP4_ROOT.parent / "exp2" / "exp2_1"
MODEL_ORDER = (
    "llama2_7b",
    "gpt3_175b",
    "llama3_8b",
    "llama3_1_405b",
    "mixtral_8x7b",
    "deepseek_v3",
)

RESULT_SPECS = {
    "training": ("training_e2e.json", "T_full_train_overlap_cycles"),
    "prefill": ("inference_prefill_pd_breakdown.json", "T_overlap_cycles"),
    "decode": ("inference_decode_e2e.json", "T_overlap_cycles"),
    "request": ("inference_request_e2e.json", "T_overlap_cycles"),
}


class Exp2AdapterError(ValueError):
    """The frozen exp-2 inputs do not satisfy the exp-4 join contract."""


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    workload_id: str
    kind: str
    model_id: str
    model: str
    model_family: str
    seq_len: int
    batch_size: int
    kv_len: int | None
    placement: str
    naive_field: str
    sw_opt_field: str
    naive_cycles: float
    sw_opt_cycles: float
    sw_opt_speedup: float
    naive_resource_service_cycles: dict[str, float]
    sw_opt_resource_service_cycles: dict[str, float]
    resource_service_status: str
    source_path: str
    source_file_digest: str
    source_result_digest: str
    source_workload_digest: str
    model_manifest_digest: str
    calibration_status: str
    limitation_tags: tuple[str, ...]
    source_status: str
    estimate_source: str
    adapter_digest: str

    def manifest_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["limitation_tags"] = list(self.limitation_tags)
        return result


@dataclass(frozen=True, slots=True)
class SwOptOnlyRow:
    workload_id: str
    kind: str
    model_id: str
    seq_len: int
    batch_size: int
    kv_len: int | None
    prefill_seq: int | None
    output_tokens: int | None
    naive_cycles: float
    sw_opt_cycles: float
    speedup: float
    source_result_digest: str
    source_file_digest: str
    calibration_status: str
    limitation_tags: tuple[str, ...]
    status: str
    evidence_level: str
    result_digest: str

    def manifest_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["limitation_tags"] = list(self.limitation_tags)
        return result


PrimitiveWorkload = WorkloadSpec


@dataclass(frozen=True, slots=True)
class Exp2Dataset:
    workloads: tuple[WorkloadSpec, ...]
    sw_opt_only_rows: tuple[SwOptOnlyRow, ...]
    provenance: dict[str, Any]


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise Exp2AdapterError(f"missing frozen exp-2 input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_number(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Exp2AdapterError(f"{field} is not numeric in {row.get('case_id')}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise Exp2AdapterError(f"{field} must be finite and positive")
    return value


def _validate_manifest_binding(exp2_root: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    manifest_dir = exp2_root / "manifests" / "models"
    observed: dict[str, str] = {}
    for model_id in MODEL_ORDER:
        path = manifest_dir / f"{model_id}.json"
        document = _load_json(path)
        if document.get("model_id") != model_id or not document.get("manifest_digest"):
            raise Exp2AdapterError(f"invalid model manifest: {path}")
        observed[model_id] = str(document["manifest_digest"])
    for row in rows:
        if row.get("model_manifest_digest") != observed.get(str(row.get("model_id"))):
            raise Exp2AdapterError(
                f"manifest digest mismatch in {row.get('case_id', '<unknown>')}"
            )


def _load_result_rows(kind: str, exp2_root: Path) -> tuple[list[dict[str, Any]], Path, str]:
    try:
        filename, _ = RESULT_SPECS[kind]
    except KeyError as exc:
        raise Exp2AdapterError(f"unknown workload kind: {kind}") from exc
    path = exp2_root / "results" / filename
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise Exp2AdapterError(f"expected a JSON row list: {path}")
    return rows, path, file_sha256(path)


def _primitive_from_row(
    kind: str, row: Mapping[str, Any], source_path: Path, source_file_digest: str
) -> WorkloadSpec:
    sw_field = RESULT_SPECS[kind][1]
    naive = _positive_number(row, "T_base_cycles")
    sw_opt = _positive_number(row, sw_field)
    speedup = naive / sw_opt
    stored_speedup_field = "speedup_full_train" if kind == "training" else "speedup"
    stored = _positive_number(row, stored_speedup_field)
    if not math.isclose(speedup, stored, rel_tol=1e-12, abs_tol=0.0):
        raise Exp2AdapterError(f"stale speedup in {row.get('case_id')}")
    payload = {
        "schema_version": "exp4.exp2_primitive.v1",
        "kind": kind,
        "source_result_digest": row["result_digest"],
        "source_file_digest": source_file_digest,
        "naive_field": "T_base_cycles",
        "sw_opt_field": sw_field,
        "naive_cycles": naive,
        "sw_opt_cycles": sw_opt,
    }
    base_services = {str(key): float(value) for key, value in row.get("base_resource_service_cycles", {}).items()}
    optimized_service_field = (
        "full_train_resource_service_cycles" if kind == "training"
        else "overlap_resource_service_cycles"
    )
    optimized_services = {
        str(key): float(value) for key, value in row.get(optimized_service_field, {}).items()
    }
    resource_status = (
        "source_resource_service_available"
        if base_services and optimized_services
        else "source_resource_service_not_published"
    )
    return WorkloadSpec(
        workload_id=str(row["case_id"]),
        kind=kind,
        model_id=str(row["model_id"]),
        model=str(row["model"]),
        model_family=str(row["model_family"]),
        seq_len=int(row["seq_len"]),
        batch_size=int(row["batch_size"]),
        kv_len=int(row["kv_length"]) if row.get("kv_length") is not None else None,
        placement=str(row["placement"]),
        naive_field="T_base_cycles",
        sw_opt_field=sw_field,
        naive_cycles=naive,
        sw_opt_cycles=sw_opt,
        sw_opt_speedup=speedup,
        naive_resource_service_cycles=base_services,
        sw_opt_resource_service_cycles=optimized_services,
        resource_service_status=resource_status,
        source_path=str(source_path),
        source_file_digest=source_file_digest,
        source_result_digest=str(row["result_digest"]),
        source_workload_digest=str(row["workload_digest"]),
        model_manifest_digest=str(row["model_manifest_digest"]),
        calibration_status=str(row["calibration_status"]),
        limitation_tags=tuple(str(tag) for tag in row.get("limitation_tags", ())),
        source_status=str(row["status"]),
        estimate_source=str(row["estimate_source"]),
        adapter_digest=canonical_digest(payload),
    )


def load_primitive_workloads(
    exp2_root: Path | str = DEFAULT_EXP2_ROOT,
) -> tuple[WorkloadSpec, ...]:
    """Load the exact 36 training/prefill/decode workload definitions."""

    root = Path(exp2_root).resolve()
    workloads: list[WorkloadSpec] = []
    raw_rows: list[dict[str, Any]] = []
    for kind in ("training", "prefill", "decode"):
        rows, path, digest = _load_result_rows(kind, root)
        raw_rows.extend(rows)
        workloads.extend(_primitive_from_row(kind, row, path, digest) for row in rows)
    _validate_manifest_binding(root, raw_rows)
    expected = {"training": 12, "prefill": 12, "decode": 12}
    counts = {kind: sum(item.kind == kind for item in workloads) for kind in expected}
    if counts != expected or len({item.workload_id for item in workloads}) != 36:
        raise Exp2AdapterError(f"unexpected primitive workload inventory: {counts}")
    order = {model_id: index for index, model_id in enumerate(MODEL_ORDER)}
    kind_order = {"training": 0, "prefill": 1, "decode": 2}
    return tuple(
        sorted(
            workloads,
            key=lambda item: (
                kind_order[item.kind], order[item.model_id], item.seq_len, item.batch_size
            ),
        )
    )


def _sw_opt_row(
    kind: str, row: Mapping[str, Any], source_file_digest: str
) -> SwOptOnlyRow:
    sw_field = RESULT_SPECS[kind][1]
    naive = _positive_number(row, "T_base_cycles")
    sw_opt = _positive_number(row, sw_field)
    speedup = naive / sw_opt
    payload = {
        "schema_version": "exp4.sw_opt_only.v1",
        "workload_id": row["case_id"],
        "kind": kind,
        "naive_cycles": naive,
        "sw_opt_cycles": sw_opt,
        "speedup": speedup,
        "source_result_digest": row["result_digest"],
        "source_file_digest": source_file_digest,
    }
    return SwOptOnlyRow(
        workload_id=str(row["case_id"]),
        kind=kind,
        model_id=str(row["model_id"]),
        seq_len=int(row["seq_len"]),
        batch_size=int(row["batch_size"]),
        kv_len=int(row["kv_length"]) if row.get("kv_length") is not None else None,
        prefill_seq=int(row["seq_len"]) if kind == "request" else None,
        output_tokens=int(row["output_tokens"]) if kind == "request" else None,
        naive_cycles=naive,
        sw_opt_cycles=sw_opt,
        speedup=speedup,
        source_result_digest=str(row["result_digest"]),
        source_file_digest=source_file_digest,
        calibration_status=str(row["calibration_status"]),
        limitation_tags=tuple(str(tag) for tag in row.get("limitation_tags", ())),
        status=str(row["status"]),
        evidence_level=str(row["estimate_source"]),
        result_digest=canonical_digest(payload),
    )


def load_sw_opt_only_rows(
    exp2_root: Path | str = DEFAULT_EXP2_ROOT, *, include_requests: bool = True
) -> tuple[SwOptOnlyRow, ...]:
    """Perform the exact exp-2 control-row join (36 primitive + optional 12 request)."""

    root = Path(exp2_root).resolve()
    kinds = ("training", "prefill", "decode", "request") if include_requests else (
        "training", "prefill", "decode"
    )
    joined: list[SwOptOnlyRow] = []
    raw_rows: list[dict[str, Any]] = []
    for kind in kinds:
        rows, _, digest = _load_result_rows(kind, root)
        raw_rows.extend(rows)
        joined.extend(_sw_opt_row(kind, row, digest) for row in rows)
    _validate_manifest_binding(root, raw_rows)
    expected = 48 if include_requests else 36
    if len(joined) != expected or len({row.workload_id for row in joined}) != expected:
        raise Exp2AdapterError("exp-2 exact row join is not one-to-one")
    return tuple(joined)


def source_inventory(exp2_root: Path | str = DEFAULT_EXP2_ROOT) -> dict[str, Any]:
    """Return immutable provenance for every source consumed by this adapter."""

    root = Path(exp2_root).resolve()
    paths = [root / "manifests" / "models" / f"{model}.json" for model in MODEL_ORDER]
    paths.extend(root / "results" / spec[0] for spec in RESULT_SPECS.values())
    paths.extend(
        (root / "results" / "capacity_audit.json", root / "results" / "calibration_summary.json")
    )
    files = {str(path): file_sha256(path) for path in paths}
    calibration = _load_json(root / "results" / "calibration_summary.json")
    payload = {
        "schema_version": "exp4.exp2_source_inventory.v1",
        "files": files,
        "calibration_summary_digest": calibration["calibration_summary_digest"],
        "publish_status": calibration["publish_status"],
    }
    payload["inventory_digest"] = canonical_digest(payload)
    return payload


def load_workloads(
    exp2_root: Path | str = DEFAULT_EXP2_ROOT,
) -> tuple[WorkloadSpec, ...]:
    """Stable main-pipeline entry point for the 36 primitive workloads."""
    return load_primitive_workloads(exp2_root)


def load_exp2(exp2_root: Path | str = DEFAULT_EXP2_ROOT) -> Exp2Dataset:
    """Load all exp-2 inputs needed by exp-4 without mutating exp-2."""
    return Exp2Dataset(
        workloads=load_workloads(exp2_root),
        sw_opt_only_rows=load_sw_opt_only_rows(exp2_root, include_requests=True),
        provenance=source_inventory(exp2_root),
    )


__all__ = [
    "DEFAULT_EXP2_ROOT",
    "EXP4_ROOT",
    "MODEL_ORDER",
    "Exp2AdapterError",
    "Exp2Dataset",
    "PrimitiveWorkload",
    "WorkloadSpec",
    "SwOptOnlyRow",
    "canonical_digest",
    "file_sha256",
    "load_exp2",
    "load_primitive_workloads",
    "load_sw_opt_only_rows",
    "load_workloads",
    "source_inventory",
]
