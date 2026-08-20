#!/usr/bin/env python3
"""Production PD1 linked-manifest to C++ finalizer interoperability gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from llm.frontend.wafer_frontend.schema.serde import canonical_json

from stage1a_state_cases import build_pd1_case


_EXPECTED_ARTIFACT_BYTES = 4272
_EXPECTED_ARTIFACT_SHA256 = (
    "b7776e8798b10d7931f32806d6f5dcd170496f873c6dbe3f804b3ae4ab32b7c0"
)


def _executable(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _run(
    command: list[str],
    *,
    input_text: str | None = None,
    expect_success: bool,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if (result.returncode == 0) != expect_success:
        raise RuntimeError(
            f"command exit={result.returncode}, expected success={expect_success}: "
            f"{command}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", required=True, type=_executable)
    parser.add_argument("--finalizer", required=True, type=_executable)
    args = parser.parse_args()

    case = build_pd1_case()
    manifest_text = canonical_json(case.manifest)
    repeat_text = canonical_json(build_pd1_case().manifest)
    if manifest_text != repeat_text:
        raise RuntimeError("PD1 builder is not canonically deterministic")

    first_selftest = _run(
        [str(args.selftest), "--pd1-stdin"],
        input_text=manifest_text,
        expect_success=True,
    )
    second_selftest = _run(
        [str(args.selftest), "--pd1-stdin"],
        input_text=manifest_text,
        expect_success=True,
    )
    if first_selftest.stdout != second_selftest.stdout:
        raise RuntimeError("PD1 C++ selftest output changed across identical runs")
    if (
        "cores=2 records=30 relocations=44 transfer_fragments=4"
        not in first_selftest.stdout
    ):
        raise RuntimeError(
            f"PD1 C++ selftest exact counts changed: {first_selftest.stdout}"
        )

    with tempfile.TemporaryDirectory(prefix="npusim_pd1_finalizer_") as raw:
        root = Path(raw)
        manifest_path = root / "pd1.linked.json"
        manifest_path.write_text(manifest_text, encoding="utf-8")
        artifacts = (root / "pd1.first.npup", root / "pd1.second.npup")
        reports = (root / "pd1.first.report.json", root / "pd1.second.report.json")
        for artifact, report in zip(artifacts, reports, strict=True):
            _run(
                [
                    str(args.finalizer),
                    "--input",
                    str(manifest_path),
                    "--output",
                    str(artifact),
                    "--report",
                    str(report),
                ],
                expect_success=True,
            )

        first_bytes = artifacts[0].read_bytes()
        second_bytes = artifacts[1].read_bytes()
        if first_bytes != second_bytes:
            raise RuntimeError("production finalizer emitted unstable PD1 bytes")
        digest = hashlib.sha256(first_bytes).hexdigest()
        if (
            len(first_bytes) != _EXPECTED_ARTIFACT_BYTES
            or digest != _EXPECTED_ARTIFACT_SHA256
        ):
            raise RuntimeError(
                "PD1 artifact golden changed: "
                f"bytes={len(first_bytes)} sha256={digest}"
            )
        first_report = json.loads(reports[0].read_text(encoding="utf-8"))
        second_report = json.loads(reports[1].read_text(encoding="utf-8"))
        if first_report != second_report:
            raise RuntimeError("production finalizer report is not deterministic")
        expected_report = {
            "artifact_bytes": _EXPECTED_ARTIFACT_BYTES,
            "artifact_sha256": _EXPECTED_ARTIFACT_SHA256,
            "core_count": 2,
            "record_count": 30,
            "relocation_count": 44,
        }
        for key, expected in expected_report.items():
            if first_report.get(key) != expected:
                raise RuntimeError(
                    f"PD1 finalizer report {key}={first_report.get(key)!r}, "
                    f"expected {expected!r}"
                )

        stale = json.loads(manifest_text)
        stale["schema_version"] = (
            "wafer_frontend.linked_program_manifest/v1alpha7"
        )
        stale_path = root / "pd1.stale.linked.json"
        stale_path.write_text(
            json.dumps(stale, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        failure = _run(
            [
                str(args.finalizer),
                "--input",
                str(stale_path),
                "--validate-only",
            ],
            expect_success=False,
        )
        if "unsupported schema version" not in failure.stderr:
            raise RuntimeError(
                "old PD1 LinkedProgramManifest did not fail at the strict "
                f"version gate: {failure.stderr}"
            )

    print(
        "[PD1 FINALIZER] PASS: "
        f"bytes={_EXPECTED_ARTIFACT_BYTES} sha256={_EXPECTED_ARTIFACT_SHA256} "
        "cores=2 records=30 relocations=44 transfer_fragments=4 repeat=2 "
        "old_version_failclosed=1 length_token_tamper=2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
