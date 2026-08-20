from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from lite_train_cases import build_s2_lite_production_case
from run_lite_train_runtime import (
    LiteTrainActionFact,
    build_pre_runtime_commands,
    build_production_pre_runtime,
    expectation_from_production_case,
    observe_lite_train_runtime,
    validate_runtime_repeat,
    validate_static_wgrad_dependency,
)


def _synthetic_output(expectation, *, makespan: int = 321) -> str:
    memory = expectation.memory
    forward = expectation.ce_forward
    backward = expectation.ce_backward
    sgd = expectation.sgd
    core = memory.core
    return "\n".join(
        (
            (
                f"[PROGRAM_MEMORY] core={core} lsu_issued={memory.lsu_issued} "
                f"lsu_completed={memory.lsu_completed} "
                f"lsu_hbm_read_bytes={memory.hbm_load_bytes} "
                f"lsu_hbm_write_bytes={memory.hbm_store_bytes} "
                f"lsu_sram_read_bytes={memory.sram_read_bytes} "
                f"lsu_sram_write_bytes={memory.sram_write_bytes} "
                "lsu_residual=0 dte_residual=0"
            ),
            (
                f"[TRAIN_CE] core={core} invocations={forward.invocations} "
                f"rank_rows={forward.rank_rows} "
                f"label_read_bytes={forward.label_read_bytes} "
                f"loss_write_bytes={forward.loss_write_bytes}"
            ),
            (
                f"[TRAIN_CE_BACKWARD] core={core} "
                f"invocations={backward.invocations} "
                f"rank_rows={backward.rank_rows} "
                f"upstream_elements={backward.upstream_elements} "
                f"logits_read_bytes={backward.logits_read_bytes} "
                f"label_read_bytes={backward.label_read_bytes} "
                f"upstream_read_bytes={backward.upstream_read_bytes} "
                f"logits_grad_write_bytes={backward.logits_grad_write_bytes}"
            ),
            (
                f"[TRAIN_SGD] core={core} invocations={sgd.invocations} "
                f"element_count={sgd.element_count} "
                f"learning_rate_f64_bits={sgd.learning_rate_f64_bits} "
                f"sram_read_bytes={sgd.sram_read_bytes} "
                f"sram_write_bytes={sgd.sram_write_bytes}"
            ),
            f"[SIM_RESULT] makespan_cycles={makespan}",
            (
                f"[HOSTLANE] ack_total={expectation.ack_total} "
                f"done_total={expectation.done_total} mismatch=0 "
                "per_lane_done=1,0,0,0,0,0,0,0"
            ),
            f"[HOSTSIG] done={core}:1, ack={core}:0:{expectation.ack_total},",
            f"[P5 P2P DRAIN] core={core} residual=0",
            "[P5 P2P TIMING DRAIN] residual=0",
            (
                "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
                "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0 event=0"
            ),
            "[DRAIN] router_residual=0 d2d_link_residual=0",
            "End DONE reception",
        )
    )


class LiteTrainRuntimeSkeletonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_s2_lite_production_case()
        cls.expectation, cls.static = expectation_from_production_case(cls.case)

    def test_production_static_chain_and_synthetic_repeat(self) -> None:
        self.assertEqual(
            (
                self.expectation.memory.lsu_issued,
                self.expectation.memory.hbm_load_bytes,
                self.expectation.memory.hbm_store_bytes,
            ),
            (17, 13_472, 1_024),
        )
        self.assertTrue(self.static.timing_execution)
        self.assertFalse(self.static.functional_execution)
        output = _synthetic_output(self.expectation)
        first = observe_lite_train_runtime(output, self.expectation)
        second = observe_lite_train_runtime(output, self.expectation)
        repeat = validate_runtime_repeat(first, second)
        self.assertEqual(repeat.repeat_count, 2)
        self.assertEqual(repeat.first_marker_digest, repeat.second_marker_digest)

    def test_marker_hbm_static_and_repeat_tamper_fail_closed(self) -> None:
        output = _synthetic_output(self.expectation)
        cases = {
            "missing_forward": output.replace(
                next(line for line in output.splitlines() if line.startswith("[TRAIN_CE]")) + "\n",
                "",
            ),
            "duplicate_backward": output + "\n" + next(
                line for line in output.splitlines() if line.startswith("[TRAIN_CE_BACKWARD]")
            ),
            "hbm_load": output.replace("lsu_hbm_read_bytes=13472", "lsu_hbm_read_bytes=13471"),
            "sgd_extra_field": output.replace("[TRAIN_SGD] core=0", "[TRAIN_SGD] core=0 forged=1"),
        }
        for label, candidate in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(RuntimeError, "FAIL"):
                observe_lite_train_runtime(candidate, self.expectation)

        facts = (
            LiteTrainActionFact("ce", "ce_backward", "dgrad", (), 0),
            LiteTrainActionFact("wgrad", "gemm", "wgrad", (), 1),
            LiteTrainActionFact("sgd", "optimizer_update", "update", ("wgrad",), 2),
        )
        with self.assertRaisesRegex(RuntimeError, "direct static dependency"):
            validate_static_wgrad_dependency(facts)

        first = observe_lite_train_runtime(output, self.expectation)
        second = observe_lite_train_runtime(
            _synthetic_output(self.expectation, makespan=322), self.expectation
        )
        with self.assertRaisesRegex(RuntimeError, "repeat changed"):
            validate_runtime_repeat(first, second)

    def test_proto_wait_and_control_tamper_fail_closed(self) -> None:
        output = _synthetic_output(self.expectation)
        with self.assertRaisesRegex(RuntimeError, "PROTO_WAIT"):
            observe_lite_train_runtime(output + "\n[PROTO_WAIT] core=0", self.expectation)
        for old, new in (
            ("ack_total=2", "ack_total=1"),
            ("router_residual=0", "router_residual=1"),
            ("End DONE reception", ""),
        ):
            with self.subTest(old=old), self.assertRaisesRegex(RuntimeError, "FAIL"):
                observe_lite_train_runtime(output.replace(old, new), self.expectation)

    def test_lazy_production_lower_link_and_nonexecuting_commands(self) -> None:
        pre_runtime = build_production_pre_runtime(
            case_builder=lambda: self.case
        )
        self.assertIsNone(pre_runtime.program_io)
        self.assertEqual(pre_runtime.static_wgrad, self.static)
        commands = build_pre_runtime_commands(
            finalizer=Path("/tools/finalizer"),
            resolver=Path("/tools/resolver"),
            manifest=Path("/work/linked.json"),
            artifact=Path("/work/program.npup"),
            program_io=Path("/work/program_io.json"),
            finalizer_report_root=Path("/work"),
        )
        self.assertEqual(len(commands.finalizer_runs), 2)
        self.assertIn("--input", commands.finalizer_runs[0])
        self.assertEqual(commands.resolver_run[1], "--resolve")
        self.assertNotEqual(commands.finalizer_runs[0], commands.finalizer_runs[1])


if __name__ == "__main__":
    unittest.main()
