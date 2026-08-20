from __future__ import annotations

from dataclasses import replace
import hashlib
import struct
import unittest

from run_lite_train_dp4_tree_ar_runtime import (
    _DONE,
    Dp4CeBackward,
    Dp4CeForward,
    Dp4FlowExpectation,
    Dp4LinkTraffic,
    Dp4Memory,
    Dp4ProbeExpectation,
    Dp4RuntimeExpectation,
    Dp4Sgd,
    main,
    observe_dp4_tree_ar_runtime,
    validate_dp4_repeat,
)


_SHA = hashlib.sha256(b"s2-lite-dp4-tree-ar-artifact").hexdigest()
_VERIFY = hashlib.sha256(b"s2-lite-dp4-tree-ar-verify").hexdigest()
_CORES = (0, 4, 8, 12)


def _expectation() -> Dp4RuntimeExpectation:
    flow_pairs = ((1, 0), (3, 2), (2, 0), (0, 1), (0, 2), (2, 3))
    flows = tuple(
        Dp4FlowExpectation(f"flow{index}", source, destination, 2048, (source, destination))
        for index, (source, destination) in enumerate(flow_pairs)
    )
    links = tuple(
        Dp4LinkTraffic(source, destination, 1, 2, 128)
        for source, destination in flow_pairs
    )
    learning_bits = struct.unpack("<Q", struct.pack("<d", 0.001))[0]
    result = Dp4RuntimeExpectation(
        artifact_sha256=_SHA,
        active_cores=_CORES,
        flows=flows,
        links=links,
        memory=tuple(Dp4Memory(core, 17, 17, 4096, 1024, 1024, 4096) for core in _CORES),
        ce_forward=tuple(Dp4CeForward(core, 1, 8, 32, 32) for core in _CORES),
        ce_backward=tuple(Dp4CeBackward(core, 1, 8, 8, 512, 32, 32, 512) for core in _CORES),
        sgd=tuple(Dp4Sgd(core, 1, 512, learning_bits, 3072, 1024) for core in _CORES),
        initialization_count=200,
        probes=tuple(
            Dp4ProbeExpectation(
                f"probe{die}", die, die * 1048576, 1024,
                hashlib.sha256(f"probe{die}".encode()).hexdigest(),
            )
            for die in range(4)
        ),
    )
    result.validate()
    return result


def _output(expectation: Dp4RuntimeExpectation) -> str:
    lines = [
        (
            f"[PROGRAM_IO] phase={phase} mode=timing "
            f"checksum={_VERIFY if phase == 'verify' else _SHA} "
            f"initializations={expectation.initialization_count} probes=4 pass=1"
        )
        for phase in ("resolved", "applied", "verify")
    ]
    lines.extend(
        f"[PROGRAM_IO_PROBE] id={item.probe_id} die={item.die_id} "
        f"address={item.address} bytes={item.bytes} "
        f"expected_checksum={item.checksum} checksum={item.checksum} "
        "valid=1 exact=1 pass=1"
        for item in expectation.probes
    )
    lines.extend(
        f"[PROGRAM_MEMORY] core={item.core} lsu_issued={item.lsu_issued} "
        f"lsu_completed={item.lsu_completed} lsu_hbm_read_bytes={item.hbm_read_bytes} "
        f"lsu_hbm_write_bytes={item.hbm_write_bytes} lsu_sram_read_bytes={item.sram_read_bytes} "
        f"lsu_sram_write_bytes={item.sram_write_bytes} lsu_residual=0 dte_residual=0"
        for item in expectation.memory
    )
    lines.extend(
        f"[TRAIN_CE] core={item.core} invocations=1 rank_rows={item.rank_rows} "
        f"label_read_bytes={item.label_read_bytes} loss_write_bytes={item.loss_write_bytes}"
        for item in expectation.ce_forward
    )
    lines.extend(
        f"[TRAIN_CE_BACKWARD] core={item.core} invocations=1 rank_rows={item.rank_rows} "
        f"upstream_elements={item.upstream_elements} logits_read_bytes={item.logits_read_bytes} "
        f"label_read_bytes={item.label_read_bytes} upstream_read_bytes={item.upstream_read_bytes} "
        f"logits_grad_write_bytes={item.logits_grad_write_bytes}"
        for item in expectation.ce_backward
    )
    lines.extend(
        f"[TRAIN_SGD] core={item.core} invocations=1 element_count=512 "
        f"learning_rate_f64_bits={item.learning_rate_f64_bits} "
        f"sram_read_bytes={item.sram_read_bytes} sram_write_bytes={item.sram_write_bytes}"
        for item in expectation.sgd
    )
    lines.append("[D2D_TYPE] request_in=6 request_out=6 ack_in=12 ack_out=12 data_in=768 data_out=768")
    for index, item in enumerate(expectation.links):
        lines.append(
            f"[D2D_LINK] idx={index} die{item.source_die}->die{item.destination_die} "
            f"dir=EAST req_in={item.request_packets} req_out={item.request_packets} "
            f"ack_in={item.ack_packets} ack_out={item.ack_packets} "
            f"data_in={item.data_packets} data_out={item.data_packets}."
        )
    lines.extend((
        "[SIM_RESULT] makespan_cycles=12000",
        "[HOSTLANE] done_total=4 ack_total=8 mismatch=0 per_lane_done=1,1,1,1",
        "[HOSTSIG] done=0:1,4:1,8:1,12:1, ack=0:0:2,4:0:2,8:0:2,12:0:2,",
        *tuple(f"[P5 P2P DRAIN] core={core} residual=0" for core in _CORES),
        "[P5 P2P TIMING DRAIN] residual=0",
        "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 gather=0 reduce_rx=0 endpoints=0 dte_tokens=0 event=0",
        "[DRAIN] router_residual=0 d2d_link_residual=0",
        *(_DONE + "." for _ in _CORES),
    ))
    return "\n".join(lines)


class Dp4TreeArParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expectation = _expectation()
        cls.output = _output(cls.expectation)

    def test_exact_dynamic_parser_and_repeat(self) -> None:
        first = observe_dp4_tree_ar_runtime(self.output, self.expectation)
        second = observe_dp4_tree_ar_runtime(self.output, self.expectation)
        repeat = validate_dp4_repeat(first, second)
        self.assertEqual(repeat.repeat_count, 2)
        self.assertEqual(first.makespan_cycles, 12000)
        self.assertEqual(
            (first.request_packets, first.ack_packets, first.data_packets),
            (6, 12, 768),
        )
        self.assertEqual(sum(item.hbm_write_bytes for item in first.memory), 4096)
        self.assertTrue(first.timing_execution)
        self.assertFalse(first.functional_execution)

    def test_formula_and_runtime_tamper_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "six exact"):
            replace(
                self.expectation,
                flows=(replace(self.expectation.flows[0], bytes=2032), *self.expectation.flows[1:]),
            ).validate()
        mutations = (
            self.output.replace("data_in=768 data_out=768", "data_in=767 data_out=768", 1),
            self.output.replace("data_in=128 data_out=128", "data_in=127 data_out=128", 1),
            self.output.replace("ack_total=8", "ack_total=7", 1),
            self.output.replace("done_total=4", "done_total=3", 1),
            self.output.replace("lsu_hbm_write_bytes=1024", "lsu_hbm_write_bytes=1023", 1),
            self.output.replace("residual=0", "residual=1", 1),
            self.output + "\n[PROTO_WAIT] forged=1",
            self.output.replace("id=probe0", "id=forged", 1),
        )
        for output in mutations:
            with self.subTest(), self.assertRaisesRegex(RuntimeError, "FAIL"):
                observe_dp4_tree_ar_runtime(output, self.expectation)

    def test_repeat_makespan_tamper_fails_closed(self) -> None:
        first = observe_dp4_tree_ar_runtime(self.output, self.expectation)
        second = observe_dp4_tree_ar_runtime(
            self.output.replace("makespan_cycles=12000", "makespan_cycles=12001"),
            self.expectation,
        )
        with self.assertRaisesRegex(RuntimeError, "repeat changed"):
            validate_dp4_repeat(first, second)

    def test_cli_fails_closed_on_nonpositive_timeout(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "timeout must be positive"):
            main(("--timeout", "0"))


if __name__ == "__main__":
    unittest.main()
