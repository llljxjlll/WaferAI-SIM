"""Require physical absence and completion before step-zero state probing."""

from __future__ import annotations

import unittest

from .run_dense_sgd_external_authority_canary import _observe
from .test_run_dense_training_sequence_runtime_canary import _output


def _valid() -> str:
    bytes_count = 808
    prefix = "\n".join((
        "[EXTERNAL_AUTHORITY_PRELOAD] state_abis=15 "
        "hbm_initializations=0 present_bytes=0 source_bytes=808 pass=1",
        "[EXTERNAL_DMA_READY] program=source completed=1 "
        "external_read_bytes=808 hbm_write_bytes=808 pending=0",
        "[EXTERNAL_AUTHORITY_RESTORED] state_abis=15 "
        "payload_bytes=808 matched=1 pending=0 pass=1",
    ))
    drain = (
        "[EXTERNAL_DMA_DRAIN] probes=1 external_read_bytes=808 "
        "external_write_bytes=808 hbm_read_bytes=808 "
        "hbm_write_bytes=808 pending=0"
    )
    return prefix + "\n" + _output().replace(
        "[SIM_RESULT]", drain + "\n[SIM_RESULT]",
    )


class DenseExternalAuthorityObserverTest(unittest.TestCase):
    def _observe(self, text: str):
        return _observe(text, hbm_bytes=808, state_count=15,
                        matmul_records=41)

    def test_two_step_authoritative_sgd_marker_order(self) -> None:
        result = self._observe(_valid())
        self.assertEqual(result["state_versions"], [0, 1, 2])
        self.assertEqual(result["pending_requests"], 0)
        self.assertEqual(result["state_abis"], 15)
        self.assertFalse(result["functional"])

    def test_version_zero_before_real_restore_rejected(self) -> None:
        text = _valid()
        marker = "[EXTERNAL_AUTHORITY_RESTORED] state_abis=15 "
        start = text.index(marker)
        end = text.index("\n", start)
        restored = text[start:end]
        text = text[:start] + text[end + 1:]
        text = text.replace(
            "[DENSE_TRAINING_SEQUENCE_STATE] version=1",
            restored + "\n[DENSE_TRAINING_SEQUENCE_STATE] version=1", 1,
        )
        with self.assertRaisesRegex(RuntimeError, "before SGD compute"):
            self._observe(text)

    def test_host_preloaded_state_marker_rejected(self) -> None:
        text = _valid().replace(
            "hbm_initializations=0 present_bytes=0",
            "hbm_initializations=15 present_bytes=808",
        )
        with self.assertRaisesRegex(RuntimeError, "absence/restore"):
            self._observe(text)

    def test_unfinished_external_writeback_rejected(self) -> None:
        text = _valid().replace(
            "hbm_write_bytes=808 pending=0\n[SIM_RESULT]",
            "hbm_write_bytes=808 pending=1\n[SIM_RESULT]",
        )
        with self.assertRaisesRegex(RuntimeError, "absence/restore"):
            self._observe(text)


if __name__ == "__main__":
    unittest.main()
