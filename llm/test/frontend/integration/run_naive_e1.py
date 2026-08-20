#!/usr/bin/env python3
"""Run the checked-in N7 E1 production example and freeze its stable report."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from llm.frontend.wafer_frontend import (
    NaiveRunCase,
    NaiveRunReport,
    NaiveRunRequest,
    NaiveRunValidation,
    run_naive,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    load_json_dataclass,
)


_ARTIFACT_SHA256 = (
    "17f40263ea05b2591935a08361abccd3ea2680ce9872372f8dfdd1a1ff2c3c8a"
)


def _fail(message: str) -> None:
    raise RuntimeError(f"[NAIVE N7 E1] FAIL: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    args.runtime_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="naive-n7-e1-", dir=args.runtime_root
    ) as raw:
        output = Path(raw) / "result"
        result = run_naive(
            NaiveRunRequest(
                case=NaiveRunCase.E1,
                validation=NaiveRunValidation.TIMING,
                spec_path=args.spec,
                hardware_config_path=args.hardware,
                simulation_config_path=args.simulation,
                mapping_config_path=args.mapping,
                output_dir=output,
                npusim_path=args.npusim,
                finalizer_path=args.finalizer,
            )
        )
        report = load_json_dataclass(
            NaiveRunReport, result.report_path, path="report"
        )
        if (
            canonical_json(report) != canonical_json(result.report)
            or not report.id.startswith("naive_run_report_")
        ):
            _fail(f"unstable report identity {report.id!r}")
        artifact = report.artifact
        runtime = report.runtime
        metrics = report.static_metrics
        if not isinstance(artifact, dict) or (
            artifact["artifact_sha256"],
            artifact["artifact_bytes"],
            artifact["core_count"],
            artifact["record_count"],
            artifact["relocation_count"],
        ) != (_ARTIFACT_SHA256, 36392, 2, 278, 464):
            _fail(f"artifact golden changed: {artifact!r}")
        if not isinstance(runtime, dict) or (
            runtime["program_io_initializations"],
            runtime["program_io_probes"],
            runtime["ack_total"],
            runtime["done_total"],
            runtime["drain_residuals"],
            runtime["makespan_cycles"],
            runtime["repeat_signature_stable"],
            runtime["observed_transfer_bytes"],
            runtime["d2d_link_packets"],
        ) != (94, 2, 4, 2, 0, 8958, True, 1024, [[0, 32], [1, 32]]):
            _fail(f"runtime golden changed: {runtime!r}")
        if not isinstance(metrics, dict) or (
            metrics["unique_flow_count"],
            metrics["scheduled_binding_count"],
            metrics["per_core_sram_max_end"],
            metrics["analytic_transfer_bytes"],
            metrics["task_counts"].get("dma_in"),
            metrics["task_counts"].get("dma_out"),
            metrics["opcode_counts"].get("LSU_LOAD"),
            metrics["opcode_counts"].get("LSU_STORE"),
            metrics["fragment_count"],
        ) != (8, 76, {"0": 30976, "16": 30976}, 1024, 18, 4, 18, 4, 52):
            _fail(f"static metrics changed: {metrics!r}")
        if (output / "SUCCESS").read_text(encoding="utf-8").strip() != report.id:
            _fail("SUCCESS marker does not identify the committed report")
        required = {
            "compile/linked_manifest.json",
            "compile/pass_receipts.json",
            "program/program.npup",
            "program/program_io.json",
            "run/parsed_markers.json",
            "run_report.json",
            "SUCCESS",
        }
        actual = {
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file()
        }
        if not required.issubset(actual):
            _fail(f"published directory is incomplete: {sorted(required - actual)!r}")

    print(
        "[NAIVE N7 E1] PASS: report="
        f"{report.id} artifact=36392B records=278 relocs=464 "
        "initializations=94 probes=2 ACK=4 DONE=2 drains=0 "
        "dma_in=18 dma_out=4 LSU_LOAD=18 LSU_STORE=4 fragments=52 "
        f"makespan_cycles=8958 sha256={_ARTIFACT_SHA256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
