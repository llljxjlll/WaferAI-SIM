#!/usr/bin/env python3
"""Atomically freeze the narrow S2/S3 Lite timing baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json

from lite_runtime_evidence import build_lite_runtime_evidence


_NAMES = (
    "capability_matrix.json",
    "input_digests.json",
    "s2.runtime.0.log",
    "s2.runtime.1.log",
    "s2_runtime_report.json",
    "s3.runtime.0.log",
    "s3.runtime.1.log",
    "s3_runtime_report.json",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _marker_digest(
    data: bytes,
    prefixes: tuple[str, ...],
    done_literal: str,
) -> str:
    lines = []
    for line in data.decode("utf-8").splitlines():
        positions = tuple(
            position
            for prefix in prefixes
            if (position := line.find(prefix)) >= 0
        )
        if positions:
            lines.append(line[min(positions) :].split(" | ", 1)[0].rstrip(". "))
        elif done_literal in line:
            lines.append(done_literal)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _check_target(root: Path) -> None:
    if not root.is_absolute():
        raise RuntimeError("baseline root must be absolute")
    if root.exists() or root.is_symlink():
        raise RuntimeError("baseline root already exists")
    current = root.parent
    while current != current.parent:
        if current.is_symlink():
            raise RuntimeError("baseline ancestors must not be symlinks")
        current = current.parent


def freeze(root: Path, logs: tuple[Path, Path, Path, Path]) -> None:
    _check_target(root)
    if any(not path.is_file() or path.is_symlink() for path in logs):
        raise RuntimeError("all four raw runtime logs must be regular files")
    raw_logs = tuple(path.read_bytes() for path in logs)
    if any(b"[PROTO_WAIT]" in data or b" FAIL:" in data for data in raw_logs):
        raise RuntimeError("raw runtime log contains a failure marker")

    bundle = build_lite_runtime_evidence()
    bundle.validate()
    s2_prefixes = (
        "[PROGRAM_MEMORY] ", "[TRAIN_CE] ", "[TRAIN_CE_BACKWARD] ",
        "[TRAIN_SGD] ", "[SIM_RESULT] ", "[HOSTLANE] ", "[HOSTSIG] ",
        "[P5 P2P DRAIN] ", "[P5 P2P TIMING DRAIN] ", "[COLL_DRAIN] ",
        "[DRAIN] ",
    )
    s3_prefixes = (
        "[PROGRAM_IO] ", "[PROGRAM_IO_PROBE] ", "[PROGRAM_MEMORY] ",
        "[SIM_RESULT] ", "[HOSTLANE] ", "[HOSTSIG] ", "[P5 P2P DRAIN] ",
        "[P5 P2P TIMING DRAIN] ", "[COLL_DRAIN] ", "[DRAIN] ",
        "[D2D_TYPE] ", "[D2D_LINK] ",
    )
    s2_digests = tuple(
        _marker_digest(data, s2_prefixes, "End DONE reception")
        for data in raw_logs[:2]
    )
    s3_digests = tuple(
        _marker_digest(data, s3_prefixes, "End DONE reception.")
        for data in raw_logs[2:]
    )
    if s2_digests != (bundle.s2_report.runtime_marker_digest,) * 2:
        raise RuntimeError("S2 runtime marker repeat/report digest changed")
    if s3_digests != (bundle.s3_report.runtime_marker_digest,) * 2:
        raise RuntimeError("S3 runtime marker repeat/report digest changed")
    entries: dict[str, bytes] = {
        "s2_runtime_report.json": canonical_json(bundle.s2_report).encode("utf-8"),
        "s3_runtime_report.json": canonical_json(bundle.s3_report).encode("utf-8"),
        "capability_matrix.json": canonical_json(bundle.matrix).encode("utf-8"),
        "s2.runtime.0.log": raw_logs[0],
        "s2.runtime.1.log": raw_logs[1],
        "s3.runtime.0.log": raw_logs[2],
        "s3.runtime.1.log": raw_logs[3],
    }
    digests = {
        "schema_version": "wafer_frontend.lite_runtime_baseline_inputs/v1alpha1",
        "baseline_id": "lite-runtime-v1",
        "s2_report_id": bundle.s2_report.id,
        "s2_report_digest": canonical_digest(bundle.s2_report),
        "s3_report_id": bundle.s3_report.id,
        "s3_report_digest": canonical_digest(bundle.s3_report),
        "capability_matrix_id": bundle.matrix.id,
        "capability_matrix_digest": canonical_digest(bundle.matrix),
        "files": {name: _sha(data) for name, data in sorted(entries.items())},
        "review_decision": "approved-lite-s2-s3-timing-scope",
        "excluded_claims": [
            "full_training",
            "dynamic_moe_routing",
            "compute_functional",
            "model_functional",
        ],
    }
    entries["input_digests.json"] = (
        json.dumps(digests, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if tuple(sorted(entries)) != _NAMES or any(name.endswith(".npup") for name in entries):
        raise RuntimeError("baseline file matrix changed")

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for name, data in entries.items():
            path = staging / name
            with path.open("xb") as handle:
                handle.write(data)
        if tuple(sorted(path.name for path in staging.iterdir())) != _NAMES:
            raise RuntimeError("staging file matrix changed")
        os.rename(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--s2-runtime-0", required=True, type=Path)
    parser.add_argument("--s2-runtime-1", required=True, type=Path)
    parser.add_argument("--s3-runtime-0", required=True, type=Path)
    parser.add_argument("--s3-runtime-1", required=True, type=Path)
    args = parser.parse_args()
    freeze(
        args.output.resolve(),
        tuple(
            path.resolve()
            for path in (
                args.s2_runtime_0,
                args.s2_runtime_1,
                args.s3_runtime_0,
                args.s3_runtime_1,
            )
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
