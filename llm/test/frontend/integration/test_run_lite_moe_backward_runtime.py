from __future__ import annotations

import hashlib
import unittest

from llm.frontend.wafer_frontend.schema.program_io import ProgramIoTargetKind
from run_lite_moe_backward_runtime import (
    build_lite_moe_backward_production_chain,
    build_mock_expectation,
    observe_lite_moe_backward_runtime,
    validate_runtime_repeat,
)


_SHA = hashlib.sha256(b"s3-lite-moe-backward-mock").hexdigest()
_VERIFY_SHA = hashlib.sha256(b"s3-lite-moe-backward-verify").hexdigest()


def _synthetic_output(expectation) -> str:
    lines = [
        (
            f"[PROGRAM_IO] phase={phase} mode=timing "
            f"checksum={_VERIFY_SHA if phase == 'verify' else _SHA} "
            "initializations=32 probes=4 pass=1"
        )
        for phase in ("resolved", "applied", "verify")
    ]
    state_abis = {
        abi.id: abi
        for linked in expectation.case.linked.manifest.fragments
        for abi in getattr(linked, "fragment", linked).state_abi
    }
    for probe in expectation.program_io.output_probes:
        abi = state_abis[probe.target.state_abi_id]
        checksum = hashlib.sha256(probe.id.encode("utf-8")).hexdigest()
        lines.append(
            f"[PROGRAM_IO_PROBE] id={probe.id} "
            f"die={abi.die_id} address={abi.address + probe.offset_bytes} "
            f"bytes={probe.length_bytes} expected_checksum={checksum} "
            f"checksum={checksum} valid=1 exact=1 pass=1"
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
            f"[TRAIN_SGD] core={item.core} invocations={item.invocations} "
            f"element_count={item.element_count} "
            f"learning_rate_f64_bits={item.learning_rate_f64_bits} "
            f"sram_read_bytes={item.sram_read_bytes} "
            f"sram_write_bytes={item.sram_write_bytes}"
        )
        for item in expectation.sgd
    )
    lines.extend((
        "[D2D_TYPE] request_in=4 request_out=4 ack_in=8 "
        "ack_out=8 data_in=8 data_out=8",
        "[D2D_LINK] idx=0 die0->die1 dir=EAST req_in=2 req_out=2 "
        "ack_in=4 ack_out=4 data_in=4 data_out=4.",
        "[D2D_LINK] idx=1 die1->die0 dir=WEST req_in=2 req_out=2 "
        "ack_in=4 ack_out=4 data_in=4 data_out=4.",
        "[SIM_RESULT] makespan_cycles=9000",
        "[HOSTLANE] done_total=2 ack_total=4 mismatch=0 "
        "per_lane_done=1,0,0,0,1,0,0,0",
        "[HOSTSIG] done=0:1,16:1, ack=0:0:2,16:0:1,16:16:1,",
        "[P5 P2P DRAIN] core=0 residual=0",
        "[P5 P2P DRAIN] core=16 residual=0",
        "[P5 P2P TIMING DRAIN] residual=0",
        "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
        "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0 event=0",
        "[DRAIN] router_residual=0 d2d_link_residual=0",
        "End DONE reception.",
        "End DONE reception.",
    ))
    return "\n".join(lines)


class LiteMoeBackwardRuntimeParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expectation = build_mock_expectation(_SHA)
        cls.output = _synthetic_output(cls.expectation)

    def test_exact_mock_parser_and_repeat(self) -> None:
        self.assertEqual(
            (
                len(self.expectation.program_io.blobs),
                len(self.expectation.program_io.initializations),
                len(self.expectation.program_io.output_probes),
                sum(
                    item.length_bytes
                    for item in self.expectation.program_io.initializations
                ),
                sum(
                    item.length_bytes
                    for item in self.expectation.program_io.output_probes
                ),
                sum(
                    item.target.kind is ProgramIoTargetKind.SRAM
                    for item in self.expectation.program_io.initializations
                ),
                sum(
                    item.target.kind is ProgramIoTargetKind.HBM
                    for item in self.expectation.program_io.initializations
                ),
            ),
            (8, 32, 4, 25472, 4096, 28, 4),
        )
        first = observe_lite_moe_backward_runtime(
            self.output, self.expectation,
        )
        second = observe_lite_moe_backward_runtime(
            self.output, self.expectation,
        )
        repeat = validate_runtime_repeat(first, second)
        self.assertEqual(repeat.repeat_count, 2)
        self.assertEqual(first.makespan_cycles, 9000)
        self.assertEqual(first.d2d_type, (4, 4, 8, 8, 8, 8))
        self.assertEqual((first.wgrad_static_count, first.reduce_static_count), (8, 4))
        self.assertTrue(first.timing_execution)
        self.assertFalse(first.functional_execution)
        self.assertEqual(sum(item.hbm_write_bytes for item in first.memory), 4096)

    def test_self_contained_pre_and_actual_sha_production_builder(self) -> None:
        pre = build_lite_moe_backward_production_chain(None)
        actual = build_lite_moe_backward_production_chain(_SHA)
        self.assertIsNone(pre.program_io)
        self.assertIsNone(pre.expectation)
        self.assertEqual(actual.linked, pre.linked)
        self.assertIsNotNone(actual.program_io)
        self.assertIsNotNone(actual.expectation)
        assert actual.program_io is not None
        self.assertEqual(actual.program_io.program_artifact_sha256, _SHA)
        self.assertEqual(
            (
                len(actual.program_io.blobs),
                len(actual.program_io.initializations),
                len(actual.program_io.output_probes),
            ),
            (8, 32, 4),
        )

    def test_program_io_work_transport_control_and_wait_tamper(self) -> None:
        first_probe = self.expectation.program_io.output_probes[0]
        state_abis = {
            abi.id: abi
            for linked in self.expectation.case.linked.manifest.fragments
            for abi in getattr(linked, "fragment", linked).state_abi
        }
        first_state = state_abis[first_probe.target.state_abi_id]
        first_sgd = next(
            line for line in self.output.splitlines()
            if line.startswith("[TRAIN_SGD]")
        )
        mutations = (
            self.output.replace("initializations=32", "initializations=31", 1),
            self.output.replace(
                f"phase=verify mode=timing checksum={_VERIFY_SHA}",
                "phase=verify mode=timing checksum=not-a-sha",
                1,
            ),
            self.output.replace(
                f"id={first_probe.id}", "id=forged", 1,
            ),
            self.output.replace(
                f"id={first_probe.id} die={first_state.die_id}",
                f"id={first_probe.id} die={1 - first_state.die_id}",
                1,
            ),
            self.output.replace(
                f"address={first_state.address + first_probe.offset_bytes}",
                f"address={first_state.address + first_probe.offset_bytes + 64}",
                1,
            ),
            self.output.replace(
                "lsu_hbm_write_bytes=2048",
                "lsu_hbm_write_bytes=2047",
                1,
            ),
            self.output.replace(first_sgd + "\n", ""),
            self.output.replace("data_in=8 data_out=8", "data_in=7 data_out=8", 1),
            self.output.replace("done_total=2", "done_total=1", 1),
            self.output.replace("core=16 residual=0", "core=16 residual=1", 1),
            self.output + "\n[PROTO_WAIT] forged=1",
            self.output.replace("End DONE reception.\n", "", 1),
        )
        for output in mutations:
            with self.subTest(), self.assertRaisesRegex(RuntimeError, "FAIL"):
                observe_lite_moe_backward_runtime(output, self.expectation)

    def test_repeat_makespan_tamper_fails_closed(self) -> None:
        first = observe_lite_moe_backward_runtime(
            self.output, self.expectation,
        )
        second = observe_lite_moe_backward_runtime(
            self.output.replace("makespan_cycles=9000", "makespan_cycles=9001"),
            self.expectation,
        )
        with self.assertRaisesRegex(RuntimeError, "repeat changed"):
            validate_runtime_repeat(first, second)


if __name__ == "__main__":
    unittest.main()
