from __future__ import annotations

from pathlib import Path
import unittest

from run_lite_moe_runtime import (
    LiteMoeRuntimeExpectation,
    _PROGRAM_IO_PROBE,
    _rows,
    observe_lite_moe_runtime,
)


_ROOT = Path(__file__).resolve().parents[4]
_BASELINE = _ROOT / "notes/frontend/baselines/lite-runtime-v1"
_SHA = "c49a8202eac0fbbe89aa1b96ba6cba782136b6e8bdf09a6b5870817a27650361"


class LiteMoeRuntimeParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.first = (_BASELINE / "s3.runtime.0.log").read_text(encoding="utf-8")
        cls.second = (_BASELINE / "s3.runtime.1.log").read_text(encoding="utf-8")
        probes = _rows(cls.first, _PROGRAM_IO_PROBE)
        cls.expectation = LiteMoeRuntimeExpectation(
            cores=(0, 16),
            hbm_read_bytes_per_core=12288,
            dte_transfers_per_direction=4,
            transfer_bytes=32,
            artifact_sha256=_SHA,
            probe_ids_and_bytes=tuple(sorted(
                (row["id"], int(row["bytes"], 10)) for row in probes
            )),
        )

    def test_two_raw_runs_have_exact_typed_observation(self) -> None:
        first = observe_lite_moe_runtime(self.first, self.expectation)
        second = observe_lite_moe_runtime(self.second, self.expectation)
        self.assertEqual(first, second)
        self.assertEqual(first.makespan_cycles, 8342)
        self.assertEqual(first.d2d_type, (8, 8, 16, 16, 16, 16))

    def test_memory_d2d_control_and_wait_tamper_fail_closed(self) -> None:
        mutations = (
            self.first.replace("lsu_hbm_read_bytes=12288", "lsu_hbm_read_bytes=12287", 1),
            self.first.replace("request_in=8", "request_in=9", 1),
            self.first.replace("done_total=2", "done_total=3", 1),
            self.first + "\n[PROTO_WAIT] forged=1\n",
            self.first.replace("[P5 P2P TIMING DRAIN]", "[REMOVED]", 1),
        )
        for output in mutations:
            with self.subTest():
                with self.assertRaises(RuntimeError):
                    observe_lite_moe_runtime(output, self.expectation)


if __name__ == "__main__":
    unittest.main()
