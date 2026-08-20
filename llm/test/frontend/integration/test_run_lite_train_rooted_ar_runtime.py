from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from run_lite_train_rooted_ar_runtime import (
    build_production_pre_runtime,
    observe_rooted_ar_runtime,
    run_official_rooted_ar,
    validate_runtime_repeat,
)


_ARTIFACT = b"rooted-ar-focused-artifact"
_SHA = hashlib.sha256(_ARTIFACT).hexdigest()


def _synthetic_output(pre) -> str:
    contract = pre.program_io
    assert contract is not None
    expectation = pre.expectation
    lines = [
        (
            f"[PROGRAM_IO] phase={phase} mode=timing checksum={_SHA} "
            "initializations=126 probes=2 pass=1"
        )
        for phase in ("resolved", "applied", "verify")
    ]
    blobs = {blob.id: blob for blob in contract.blobs}
    for probe in contract.output_probes:
        checksum = blobs[probe.blob_ref].sha256
        lines.append(
            f"[PROGRAM_IO_PROBE] id={probe.id} bytes={probe.length_bytes} "
            f"expected_checksum={checksum} checksum={checksum} "
            "valid=1 exact=1 pass=1"
        )
    lines.extend(
        (
            f"[PROGRAM_MEMORY] core={item.core} "
            f"lsu_issued={item.lsu_issued} "
            f"lsu_completed={item.lsu_completed} "
            f"lsu_hbm_read_bytes={item.hbm_read_bytes} "
            f"lsu_hbm_write_bytes={item.hbm_write_bytes} "
            f"lsu_sram_read_bytes={item.sram_read_bytes} "
            f"lsu_sram_write_bytes={item.sram_write_bytes} "
            "lsu_residual=0 dte_residual=0"
        )
        for item in expectation.memory
    )
    lines.extend(
        (
            f"[TRAIN_CE] core={item.core} invocations={item.invocations} "
            f"rank_rows={item.rank_rows} "
            f"label_read_bytes={item.label_read_bytes} "
            f"loss_write_bytes={item.loss_write_bytes}"
        )
        for item in reversed(expectation.ce_forward)
    )
    lines.extend(
        (
            f"[TRAIN_CE_BACKWARD] core={item.core} "
            f"invocations={item.invocations} rank_rows={item.rank_rows} "
            f"upstream_elements={item.upstream_elements} "
            f"logits_read_bytes={item.logits_read_bytes} "
            f"label_read_bytes={item.label_read_bytes} "
            f"upstream_read_bytes={item.upstream_read_bytes} "
            f"logits_grad_write_bytes={item.logits_grad_write_bytes}"
        )
        for item in reversed(expectation.ce_backward)
    )
    lines.extend(
        (
            f"[TRAIN_SGD] core={item.core} invocations={item.invocations} "
            f"element_count={item.element_count} "
            f"learning_rate_f64_bits={item.learning_rate_f64_bits} "
            f"sram_read_bytes={item.sram_read_bytes} "
            f"sram_write_bytes={item.sram_write_bytes}"
        )
        for item in expectation.sgd
    )
    lines.extend((
        "[SIM_RESULT] makespan_cycles=4567",
        "[HOSTLANE] ack_total=4 done_total=2 mismatch=0",
        "[HOSTSIG] ack=0:0:2,16:0:2, done=0:1,16:1,",
        "[P5 P2P DRAIN] core=0 residual=0",
        "[P5 P2P DRAIN] core=16 residual=0",
        "[P5 P2P TIMING DRAIN] residual=0",
        "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
        "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0 event=0",
        "[DRAIN] router_residual=0 d2d_link_residual=0",
        "[D2D_TYPE] request_in=2 request_out=2 ack_in=4 "
        "ack_out=4 data_in=256 data_out=256",
        "[D2D_LINK] idx=0 die0->die1 dir=EAST req_in=1 req_out=1 "
        "ack_in=2 ack_out=2 data_in=128 data_out=128",
        "[D2D_LINK] idx=1 die1->die0 dir=WEST req_in=1 req_out=1 "
        "ack_in=2 ack_out=2 data_in=128 data_out=128",
        "End DONE reception",
        "End DONE reception",
    ))
    return "\n".join(lines)


class RootedArRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pre = build_production_pre_runtime(_SHA)
        cls.output = _synthetic_output(cls.pre)

    def test_production_pre_runtime_and_strict_repeat(self) -> None:
        self.assertEqual(
            (
                len(self.pre.linked.manifest.fragments),
                len(self.pre.program_io.blobs),
                len(self.pre.program_io.initializations),
                len(self.pre.program_io.output_probes),
            ),
            (98, 24, 126, 2),
        )
        first = observe_rooted_ar_runtime(
            self.output, _SHA, self.pre.program_io, self.pre.expectation,
        )
        second = observe_rooted_ar_runtime(
            self.output, _SHA, self.pre.program_io, self.pre.expectation,
        )
        self.assertEqual(validate_runtime_repeat(first, second).repeat_count, 2)

    def test_marker_transport_control_and_repeat_tamper_fail_closed(self) -> None:
        cases = {
            "missing-ce-bwd": self.output.replace(
                next(line for line in self.output.splitlines()
                     if line.startswith("[TRAIN_CE_BACKWARD]")) + "\n", "",
            ),
            "download-packet": self.output.replace(
                "data_in=256 data_out=256", "data_in=255 data_out=256",
            ),
            "proto-wait": self.output + "\n[PROTO_WAIT] core=16",
            "missing-done": self.output.replace(
                "End DONE reception\n", "", 1,
            ),
            "extra-done": self.output + "\nEnd DONE reception",
            "probe-bytes": self.output.replace(
                "bytes=32", "bytes=16", 1,
            ),
            "sgd-before-overlay-proof": self.output.replace(
                "ack_total=4", "ack_total=3",
            ),
            "residual": self.output.replace(
                "core=16 residual=0", "core=16 residual=1",
            ),
        }
        for label, output in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                RuntimeError, "FAIL",
            ):
                observe_rooted_ar_runtime(
                    output, _SHA, self.pre.program_io, self.pre.expectation,
                )
        first = observe_rooted_ar_runtime(
            self.output, _SHA, self.pre.program_io, self.pre.expectation,
        )
        changed = observe_rooted_ar_runtime(
            self.output.replace("makespan_cycles=4567", "makespan_cycles=4568"),
            _SHA, self.pre.program_io, self.pre.expectation,
        )
        with self.assertRaisesRegex(RuntimeError, "repeat changed"):
            validate_runtime_repeat(first, changed)

    def test_injected_finalizer_resolver_and_runtime_commands(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rooted-ar-runner-test-") as raw:
            root = Path(raw)
            args = argparse.Namespace(
                finalizer=root / "finalizer",
                resolver=root / "resolver",
                npusim=root / "npusim",
                simulation=root / "simulation.json",
                runtime_root=root,
                timeout=10,
            )
            calls = []

            def runner(command, cwd, timeout):
                calls.append((command, cwd, timeout))
                executable = Path(command[0]).name
                if executable == "finalizer":
                    output = Path(command[command.index("--output") + 1])
                    report = Path(command[command.index("--report") + 1])
                    output.write_bytes(_ARTIFACT)
                    report.write_text(json.dumps({
                        "artifact_sha256": _SHA,
                        "artifact_bytes": len(_ARTIFACT),
                        "core_count": 2,
                        "record_count": 351,
                        "relocation_count": 662,
                        "linked_manifest_id": self.pre.linked.manifest.id,
                        "linked_manifest_digest":
                        canonical_digest(self.pre.linked.manifest),
                    }), encoding="utf-8")
                    return "finalized"
                if executable == "resolver":
                    return "PASS initializations=126 probes=2"
                if executable == "npusim":
                    return self.output
                raise AssertionError(command)

            observation, repeat = run_official_rooted_ar(
                args, command_runner=runner,
            )
            self.assertEqual(observation.makespan_cycles, 4567)
            self.assertEqual(repeat.repeat_count, 2)
            self.assertEqual(
                [Path(item[0][0]).name for item in calls],
                ["finalizer", "finalizer", "resolver", "npusim", "npusim"],
            )


if __name__ == "__main__":
    unittest.main()
