from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest

from run_stage3_static_profiles import (
    _REVIEWED_RUNTIME_GOLDENS,
    _ReviewedRuntimeWitness,
    _expected_report_names,
    _publish_report_texts,
    _rows,
    _validate_control,
    _validate_d2d,
    _validate_reviewed_runtime,
)
from stage3_decode_cases import Stage3StaticCaseKind


def _control_output() -> str:
    return "\n".join(
        (
            "[SIM_RESULT] makespan_cycles=5805",
            "[HOSTLANE] ack_total=2 done_total=1 mismatch=0",
            "[HOSTSIG] done=0:1 ack=0:0:2",
            "[P5 P2P TIMING DRAIN] residual=0",
            "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
            "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0 event=0",
            "[DRAIN] router_residual=0",
            "[DRAIN] d2d_link_residual=0",
            "End DONE reception",
        )
    )


def _d2d_output() -> str:
    return "\n".join(
        (
            "[D2D_TYPE] request_in=0 request_out=0 ack_in=0 ack_out=0 "
            "data_in=0 data_out=0",
            "[D2D_LINK] req_in=0 req_out=0 ack_in=0 ack_out=0 "
            "data_in=0 data_out=0",
        )
    )


class Stage3ReviewedRuntimeTest(unittest.TestCase):
    def test_all_profiles_and_each_reviewed_field_are_fail_closed(self) -> None:
        for kind in Stage3StaticCaseKind:
            expected = _REVIEWED_RUNTIME_GOLDENS[kind]
            with self.subTest(kind=kind, accepted=True):
                _validate_reviewed_runtime(kind, expected)
            for field in fields(_ReviewedRuntimeWitness):
                value = getattr(expected, field.name)
                tampered = value + 1 if type(value) is int else value + "0"
                with self.subTest(kind=kind, field=field.name):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "reviewed runtime six-field golden changed",
                    ):
                        _validate_reviewed_runtime(
                            kind,
                            replace(expected, **{field.name: tampered}),
                        )

    def test_control_and_d2d_markers_are_exact(self) -> None:
        for kind in Stage3StaticCaseKind:
            with self.subTest(kind=kind):
                makespan, control = _validate_control(kind, _control_output())
                self.assertEqual(makespan, 5805)
                control.validate()
                _validate_d2d(kind, _d2d_output())

    def test_control_and_d2d_tampering_fail_closed(self) -> None:
        kind = Stage3StaticCaseKind.MIXED
        with self.assertRaisesRegex(RuntimeError, "ACK/DONE"):
            _validate_control(
                kind,
                _control_output().replace("ack_total=2", "ack_total=3"),
            )
        with self.assertRaisesRegex(RuntimeError, "D2D counts"):
            _validate_d2d(
                kind,
                _d2d_output().replace("data_out=0", "data_out=1", 1),
            )
        with self.assertRaisesRegex(RuntimeError, "D2D_BEHA"):
            _validate_d2d(kind, _d2d_output() + "\n[D2D_BEHA] data_in=0")

    def test_marker_rows_strip_runtime_log_punctuation(self) -> None:
        output = "prefix [SIM_RESULT] makespan_cycles=5805. | 11611 ns\n"
        self.assertEqual(
            _rows(output, "[SIM_RESULT] "),
            [{"makespan_cycles": "5805"}],
        )


class Stage3ReportRootTest(unittest.TestCase):
    @staticmethod
    def _entries(
        kind: Stage3StaticCaseKind,
    ) -> tuple[tuple[str, str], ...]:
        return tuple((name, "") for name in _expected_report_names(kind))

    def test_exact_set_is_atomically_published_without_npup(self) -> None:
        kind = Stage3StaticCaseKind.MIXED
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "evidence"
            _publish_report_texts(root, kind, self._entries(kind))
            self.assertEqual(
                tuple(sorted(path.name for path in root.iterdir())),
                _expected_report_names(kind),
            )
            self.assertFalse(
                any(path.suffix == ".npup" for path in root.iterdir())
            )

    def test_incomplete_existing_and_symlink_roots_fail_before_write(self) -> None:
        kind = Stage3StaticCaseKind.DECODE
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            with self.assertRaisesRegex(RuntimeError, "exact reviewed set"):
                _publish_report_texts(
                    parent / "incomplete",
                    kind,
                    self._entries(kind)[:-1],
                )
            self.assertFalse((parent / "incomplete").exists())

            existing = parent / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must not already exist"):
                _publish_report_texts(existing, kind, self._entries(kind))
            self.assertEqual(tuple(existing.iterdir()), ())

            destination = parent / "destination"
            destination.mkdir()
            symlink = parent / "symlink"
            symlink.symlink_to(destination, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
                _publish_report_texts(symlink, kind, self._entries(kind))
            self.assertEqual(tuple(destination.iterdir()), ())


if __name__ == "__main__":
    unittest.main()
