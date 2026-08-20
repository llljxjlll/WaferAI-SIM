from __future__ import annotations

from dataclasses import replace
import hashlib
from io import StringIO
import unittest
from contextlib import redirect_stdout

from run_lite_moe_dp4_runtime_matrix import (
    Dp4MoeExpectation,
    Dp4MoeFlow,
    Dp4MoeMemory,
    Dp4MoeProbe,
    LiteMoeDp4Mode,
    build_lite_moe_dp4_production,
    main,
    observe_lite_moe_dp4_runtime,
    validate_lite_moe_dp4_repeat,
)


_SHA = hashlib.sha256(b"s3-lite-moe-dp4").hexdigest()
_VERIFY = hashlib.sha256(b"s3-lite-moe-dp4-verify").hexdigest()
_CORES = (0, 16, 32, 48)
_DISPATCH = (
    (1, 1, 0, (1, 0)),
    (2, 2, 1, (2, 3, 1)),
    (3, 3, 1, (3, 1)),
    (4, 0, 2, (0, 2)),
    (5, 1, 2, (1, 0, 2)),
    (6, 2, 3, (2, 3)),
)


def _expectation(mode: LiteMoeDp4Mode) -> Dp4MoeExpectation:
    dispatch = tuple(
        Dp4MoeFlow(f"dispatch.{token}", source, home, 32, path)
        for token, source, home, path in _DISPATCH
    )
    if mode is LiteMoeDp4Mode.DOWN_WGRAD:
        flows = tuple(replace(flow, id=flow.id.replace("dispatch", "grad")) for flow in dispatch)
        probes = tuple(
            Dp4MoeProbe(
                f"weight.{expert}", "hbm", expert, expert * 1048576,
                1024, hashlib.sha256(f"weight.{expert}".encode()).hexdigest(),
                "updated_weight",
            )
            for expert in range(4)
        )
        memory = tuple(Dp4MoeMemory(core, 2, 2, 1024, 1024, 1024, 1024) for core in _CORES)
        counts = (0, 0, 0, 8, 4, 4)
        case_id = "case.s3_lite.dp4_ep4.static_moe_down_wgrad"
        initializations = 34
    else:
        combine = tuple(
            Dp4MoeFlow(
                f"combine.{token}", home, source, 32, tuple(reversed(path))
            )
            for token, source, home, path in _DISPATCH
        )
        flows = dispatch + combine
        combined = tuple(
            Dp4MoeProbe(
                f"combined.{token}", "sram", _CORES[token % 4],
                65536 + token * 64, 32,
                hashlib.sha256(f"combined.{token}".encode()).hexdigest(),
                "combined",
            )
            for token in range(8)
        )
        if mode is LiteMoeDp4Mode.TRAIN_FORWARD:
            tape = tuple(
                Dp4MoeProbe(
                    f"tape.{token}", "sram", _CORES[(token // 2) % 4],
                    131072 + token * 64, 64,
                    hashlib.sha256(f"tape.{token}".encode()).hexdigest(),
                    "tape",
                )
                for token in range(8)
            )
            probes = combined + tape
            case_id = "case.s3_lite.dp4_ep4.static_moe_train_forward"
        else:
            probes = combined
            case_id = "case.s3_lite.dp4_ep4.static_moe_infer"
        memory = tuple(Dp4MoeMemory(core, 6, 6, 6144, 0, 0, 6144) for core in _CORES)
        counts = (
            24,
            8,
            8 if mode is LiteMoeDp4Mode.TRAIN_FORWARD else 0,
            0,
            0,
            0,
        )
        initializations = 88 if mode is LiteMoeDp4Mode.TRAIN_FORWARD else 80
    result = Dp4MoeExpectation(
        mode, case_id, _SHA, _CORES, flows, probes, memory, initializations,
        *counts,
    )
    result.validate()
    return result


def _output(expectation: Dp4MoeExpectation) -> str:
    lines = [
        f"[PROGRAM_IO] phase={phase} mode=timing "
        f"checksum={_VERIFY if phase == 'verify' else _SHA} "
        f"initializations={expectation.initialization_count} "
        f"probes={len(expectation.probes)} pass=1"
        for phase in ("resolved", "applied", "verify")
    ]
    for probe in expectation.probes:
        endpoint = "die" if probe.target_kind == "hbm" else "core"
        lines.append(
            f"[PROGRAM_IO_PROBE] id={probe.id} {endpoint}={probe.endpoint} "
            f"address={probe.address} bytes={probe.bytes} "
            f"expected_checksum={probe.checksum} checksum={probe.checksum} "
            "valid=1 exact=1 pass=1"
        )
    lines.extend(
        f"[PROGRAM_MEMORY] core={item.core} lsu_issued={item.lsu_issued} "
        f"lsu_completed={item.lsu_completed} lsu_hbm_read_bytes={item.hbm_read_bytes} "
        f"lsu_hbm_write_bytes={item.hbm_write_bytes} lsu_sram_read_bytes={item.sram_read_bytes} "
        f"lsu_sram_write_bytes={item.sram_write_bytes} lsu_residual=0 dte_residual=0"
        for item in expectation.memory
    )
    requests = sum(len(flow.die_path) - 1 for flow in expectation.flows)
    packets = sum(
        flow.packets * (len(flow.die_path) - 1)
        for flow in expectation.flows
    )
    lines.append(
        f"[D2D_TYPE] request_in={requests} request_out={requests} "
        f"ack_in={2 * requests} ack_out={2 * requests} "
        f"data_in={packets} data_out={packets}"
    )
    links: dict[tuple[int, int], list[int]] = {}
    for flow in expectation.flows:
        for source, destination in zip(flow.die_path, flow.die_path[1:]):
            counts = links.setdefault((source, destination), [0, 0, 0])
            counts[0] += 1
            counts[2] += flow.packets
    requests_by_link = {edge: counts[0] for edge, counts in links.items()}
    if expectation.mode is LiteMoeDp4Mode.DOWN_WGRAD:
        for source, destination in tuple(requests_by_link):
            links.setdefault((destination, source), [0, 0, 0])
        for (source, destination), counts in links.items():
            counts[1] = (
                requests_by_link.get((source, destination), 0)
                + requests_by_link.get((destination, source), 0)
            )
    else:
        for counts in links.values():
            counts[1] = 2 * counts[0]
    for index, ((source, destination), counts) in enumerate(sorted(links.items())):
        lines.append(
            f"[D2D_LINK] idx={index} die{source}->die{destination} dir=EAST "
            f"req_in={counts[0]} req_out={counts[0]} ack_in={counts[1]} "
            f"ack_out={counts[1]} data_in={counts[2]} data_out={counts[2]}."
        )
    lines.extend((
        "[HOSTLANE] done_total=4 ack_total=8 mismatch=0 per_lane_done=1,1,1,1",
        "[HOSTSIG] done=0:1,16:1,32:1,48:1, ack=0:0:2,16:0:2,32:0:2,48:0:2,",
        *(f"[P5 P2P DRAIN] core={core} residual=0" for core in _CORES),
        "[P5 P2P TIMING DRAIN] residual=0",
        "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 gather=0 reduce_rx=0 endpoints=0 dte_tokens=0 event=0",
        "[DRAIN] router_residual=0",
        "[DRAIN] d2d_link_residual=0",
        "[SIM_RESULT] makespan_cycles=9000",
        *("End DONE reception." for _ in _CORES),
    ))
    return "\n".join(lines)


class LiteMoeDp4RuntimeMatrixParserTest(unittest.TestCase):
    def test_three_mode_exact_parser_and_repeat(self) -> None:
        for mode in LiteMoeDp4Mode:
            with self.subTest(mode=mode.value):
                expectation = _expectation(mode)
                output = _output(expectation)
                first = observe_lite_moe_dp4_runtime(output, expectation)
                second = observe_lite_moe_dp4_runtime(output, expectation)
                validate_lite_moe_dp4_repeat(first, second)
                self.assertEqual(first.data_packets, 12 if mode is LiteMoeDp4Mode.DOWN_WGRAD else 24)
                self.assertEqual(
                    first.data_packet_hops,
                    16 if mode is LiteMoeDp4Mode.DOWN_WGRAD else 32,
                )
                self.assertTrue(first.timing_execution)
                self.assertFalse(first.functional_execution)

    def test_mode_formula_and_runtime_tamper_fail_closed(self) -> None:
        expectation = _expectation(LiteMoeDp4Mode.TRAIN_FORWARD)
        with self.assertRaisesRegex(RuntimeError, "flow byte/packet"):
            replace(
                expectation,
                flows=(replace(expectation.flows[0], bytes=16), *expectation.flows[1:]),
            ).validate()
        output = _output(expectation)
        mutations = (
            output.replace("data_in=32 data_out=32", "data_in=31 data_out=32", 1),
            output.replace("ack_total=8", "ack_total=7", 1),
            output.replace("residual=0", "residual=1", 1),
            output + "\n[PROTO_WAIT] forged=1",
            output.replace("id=tape.0", "id=forged", 1),
        )
        for mutation in mutations:
            with self.subTest(), self.assertRaisesRegex(RuntimeError, "FAIL"):
                observe_lite_moe_dp4_runtime(mutation, expectation)

    def test_cli_rejects_nonpositive_timeout_before_runner(self) -> None:
        called = False

        def fake_runner(_args):
            nonlocal called
            called = True
            return ()

        with self.assertRaisesRegex(RuntimeError, "timeout must be positive"):
            main(("--timeout", "0"), matrix_runner=fake_runner)
        self.assertFalse(called)

    def test_cli_injected_runner_prints_each_case_and_matrix_pass(self) -> None:
        observations = tuple(
            observe_lite_moe_dp4_runtime(_output(_expectation(mode)), _expectation(mode))
            for mode in LiteMoeDp4Mode
        )
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main((), matrix_runner=lambda _args: observations), 0)
        rendered = output.getvalue()
        self.assertEqual(rendered.count("] PASS:"), 3)
        for observation in observations:
            self.assertIn(observation.case_id, rendered)
        self.assertIn("[S3-LITE DP4 MATRIX] PASS", rendered)

    def test_three_production_builders_derive_exact_runtime_inputs(self) -> None:
        expected = {
            LiteMoeDp4Mode.INFER: (80, 8, 384, 24),
            LiteMoeDp4Mode.TRAIN_FORWARD: (88, 16, 384, 24),
            LiteMoeDp4Mode.DOWN_WGRAD: (34, 4, 192, 12),
        }
        for index, mode in enumerate(LiteMoeDp4Mode):
            with self.subTest(mode=mode.value):
                artifact_sha = str(index + 1) * 64
                result = build_lite_moe_dp4_production(mode, artifact_sha)
                result.validate()
                self.assertEqual(
                    (
                        result.expectation.initialization_count,
                        len(result.expectation.probes),
                        sum(item.bytes for item in result.expectation.flows),
                        sum(item.packets for item in result.expectation.flows),
                    ),
                    expected[mode],
                )
                self.assertEqual(
                    result.chain.program_io.program_artifact_sha256,
                    artifact_sha,
                )


if __name__ == "__main__":
    unittest.main()
