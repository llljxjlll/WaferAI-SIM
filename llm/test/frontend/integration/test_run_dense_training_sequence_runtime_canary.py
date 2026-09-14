from __future__ import annotations

import unittest

from .run_dense_training_sequence_runtime_canary import observe_runtime


_DIGEST = "1" * 64


def _output() -> str:
    lines = [
        (
            "[DENSE_TRAINING_SEQUENCE_STATE] "
            f"version={version} bytes=808 digest={_DIGEST} "
            "content_changed=0 functional=0 pass=1"
        )
        for version in range(3)
    ]
    lines.extend(
        (
            f"[DENSE_SEQUENCE_SEGMENT] index={step} status=done "
            f"final={int(step == 1)}",
            (
                "[DENSE_TRAINING_SEQUENCE_STEP] "
                f"index={step} input_version={step} "
                f"output_version={step + 1} trainable_states=15 "
                "matmul_records=41 sgd_records=15 store_records=15 "
                "functional=0 pass=1"
            ),
            f"[DENSE_SEQUENCE_PROGRAM_IO] index={step} probes=15 pass=1",
        )
        for step in range(2)
    )
    flattened = []
    for item in lines:
        flattened.extend(item if type(item) is tuple else (item,))
    flattened.extend("[TRAIN_SGD] core=0" for _ in range(30))
    flattened.extend((
        "[DENSE_SEQUENCE_DRAIN] segments=2 one_shot=1",
        "[SIM_RESULT] makespan_cycles=6715",
    ))
    return "\n".join(flattened)


class DenseTrainingSequenceRuntimeParserTest(unittest.TestCase):
    def test_exact_two_step_observation(self) -> None:
        result = observe_runtime(
            _output(), state_count=15, hbm_bytes=808, matmul_records=41
        )
        self.assertEqual(result.versions, (0, 1, 2))
        self.assertEqual(result.sgd_invocations, 30)
        self.assertFalse(result.functional)

    def test_missing_middle_state_version_fails(self) -> None:
        missing = _output().replace(
            (
                "[DENSE_TRAINING_SEQUENCE_STATE] "
                f"version=1 bytes=808 digest={_DIGEST} "
                "content_changed=0 functional=0 pass=1\n"
            ),
            "",
        )
        with self.assertRaisesRegex(RuntimeError, "versions 0,1,2"):
            observe_runtime(
                missing, state_count=15, hbm_bytes=808, matmul_records=41
            )

    def test_old_output_version_fails(self) -> None:
        old = _output().replace(
            "index=1 input_version=1 output_version=2",
            "index=1 input_version=1 output_version=1",
        )
        with self.assertRaisesRegex(RuntimeError, "step closure"):
            observe_runtime(
                old, state_count=15, hbm_bytes=808, matmul_records=41
            )

    def test_functional_overclaim_fails(self) -> None:
        overclaim = _output().replace(
            "content_changed=0 functional=0 pass=1",
            "content_changed=0 functional=1 pass=1",
            1,
        )
        with self.assertRaisesRegex(RuntimeError, "functional"):
            observe_runtime(
                overclaim, state_count=15, hbm_bytes=808, matmul_records=41
            )


if __name__ == "__main__":
    unittest.main()
