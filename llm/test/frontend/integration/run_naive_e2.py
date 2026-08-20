#!/usr/bin/env python3
"""Run the opt-in N7 E2 TP4/L2 production timing gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from llm.frontend.wafer_frontend import (
    NaiveRunCase,
    NaiveRunRequest,
    NaiveRunValidation,
    run_naive,
)


_ARTIFACT_SHA256 = (
    "5476081c966f4d63ea68166d84090f96bce230b6330085b4b91d6aa65e01a5c3"
)


def _fail(message: str) -> None:
    raise RuntimeError(f"[NAIVE N7 E2] FAIL: {message}")


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
        prefix="naive-n7-e2-", dir=args.runtime_root
    ) as raw:
        result = run_naive(
            NaiveRunRequest(
                case=NaiveRunCase.E2,
                validation=NaiveRunValidation.TIMING,
                spec_path=args.spec,
                hardware_config_path=args.hardware,
                simulation_config_path=args.simulation,
                mapping_config_path=args.mapping,
                output_dir=Path(raw) / "result",
                npusim_path=args.npusim,
                finalizer_path=args.finalizer,
                timeout_seconds=600,
            )
        )
        report = result.report
        artifact = report.artifact
        runtime = report.runtime
        metrics = report.static_metrics
        if not isinstance(artifact, dict) or (
            artifact["artifact_sha256"],
            artifact["artifact_bytes"],
            artifact["core_count"],
            artifact["record_count"],
            artifact["relocation_count"],
        ) != (_ARTIFACT_SHA256, 179280, 4, 1452, 2260):
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
        ) != (
            432,
            4,
            8,
            4,
            0,
            12952,
            True,
            8192,
            tuple((index, 64) for index in range(8)),
        ):
            _fail(f"runtime golden changed: {runtime!r}")
        if not isinstance(metrics, dict) or (
            metrics["unique_flow_count"],
            metrics["scheduled_binding_count"],
            metrics["per_core_sram_max_end"],
            metrics["analytic_transfer_bytes"],
            metrics["task_counts"].get("transit"),
            metrics["task_counts"].get("dma_in"),
            metrics["task_counts"].get("dma_out"),
            metrics["opcode_counts"].get("LSU_LOAD"),
            metrics["opcode_counts"].get("LSU_STORE"),
            metrics["fragment_count"],
        ) != (
            96,
            372,
            {"0": 34432, "4": 34432, "8": 34432, "12": 34432},
            8192,
            32,
            60,
            16,
            60,
            16,
            180,
        ):
            _fail(f"static metrics changed: {metrics!r}")

    print(
        "[NAIVE N7 E2] PASS: artifact=179280B records=1452 relocs=2260 "
        "initializations=432 probes=4 ACK=8 DONE=4 drains=0 "
        "dma_in=60 dma_out=16 LSU_LOAD=60 LSU_STORE=16 fragments=180 "
        f"makespan_cycles=12952 sha256={_ARTIFACT_SHA256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
