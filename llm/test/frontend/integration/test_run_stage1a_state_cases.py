from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest

from llm.frontend.wafer_frontend.schema import Stage1aCase
from run_stage1a_state_cases import (
    _INTEGRATION_GOLDENS,
    _IntegrationGolden,
    _validate_integration_golden,
    _parse_d2d,
    _publish_report_texts,
    _report_text_entries,
)


class Stage1aIntegrationGoldenTest(unittest.TestCase):
    def test_reviewed_goldens_are_accepted(self) -> None:
        for case, golden in _INTEGRATION_GOLDENS.items():
            with self.subTest(case=case.value):
                _validate_integration_golden(case, golden)

    def test_every_golden_field_is_fail_closed(self) -> None:
        for case, golden in _INTEGRATION_GOLDENS.items():
            for field in fields(_IntegrationGolden):
                value = getattr(golden, field.name)
                changed = value + 1 if isinstance(value, int) else "0" * 64
                tampered = replace(golden, **{field.name: changed})
                with self.subTest(case=case.value, field=field.name):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "reviewed integration golden changed",
                    ):
                        _validate_integration_golden(case, tampered)


def _d2d_output(
    *,
    data_in: int,
    data_out: int,
    data_flows: int,
    logical_packets: int,
    include_behavioral: bool = True,
) -> str:
    typed = (
        "[D2D_TYPE] request_in=0 request_out=0 ack_in=0 ack_out=0 "
        f"data_in={data_in} data_out={data_out}\n"
    )
    if not include_behavioral:
        return typed
    return (
        typed
        + f"[D2D_BEHA] data_flows={data_flows} "
        f"logical_data_packets={logical_packets} service_cycles=0 "
        "fixed_cycles=0 total_d2d_cycles=0\n"
    )


class Stage1aD2DEvidenceTest(unittest.TestCase):
    def test_exact_type_and_optional_behavioral_markers_are_observed(self) -> None:
        cases = (
            (Stage1aCase.P1, 0, 0, 0),
            (Stage1aCase.K1, 0, 0, 0),
            (Stage1aCase.PD1, 4, 2, 64),
        )
        for case, packets, flows, expected_bytes in cases:
            for include_behavioral in (False, True):
                with self.subTest(
                    case=case.value,
                    behavioral=include_behavioral,
                ):
                    evidence = _parse_d2d(
                        case,
                        _d2d_output(
                            data_in=packets,
                            data_out=packets,
                            data_flows=flows,
                            logical_packets=packets,
                            include_behavioral=include_behavioral,
                        ),
                        expected_data_flows=flows,
                    )
                    self.assertEqual(evidence.observed_bytes, expected_bytes)
                    self.assertEqual(evidence.data_flows, flows)

    def test_missing_duplicate_mismatch_and_wrong_counts_fail_closed(self) -> None:
        valid = _d2d_output(
            data_in=4,
            data_out=4,
            data_flows=2,
            logical_packets=4,
        )
        duplicate_type = (
            valid
            + "[D2D_TYPE] request_in=0 request_out=0 ack_in=0 ack_out=0 "
            "data_in=4 data_out=4\n"
        )
        duplicate_behavioral = (
            valid
            + "[D2D_BEHA] data_flows=2 logical_data_packets=4 "
            "service_cycles=0 fixed_cycles=0 total_d2d_cycles=0\n"
        )
        cases = (
            ("missing_type", ""),
            ("duplicate_type", duplicate_type),
            ("duplicate_behavioral", duplicate_behavioral),
            (
                "mismatch",
                _d2d_output(
                    data_in=3,
                    data_out=4,
                    data_flows=2,
                    logical_packets=4,
                ),
            ),
            (
                "packets",
                _d2d_output(
                    data_in=3,
                    data_out=3,
                    data_flows=2,
                    logical_packets=3,
                ),
            ),
            (
                "flows",
                _d2d_output(
                    data_in=4,
                    data_out=4,
                    data_flows=1,
                    logical_packets=4,
                ),
            ),
        )
        for label, output in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, "D2D"):
                    _parse_d2d(
                        Stage1aCase.PD1,
                        output,
                        expected_data_flows=2,
                    )


def _text_entries(case: Stage1aCase) -> tuple[tuple[str, str], ...]:
    runtime_log = (
        "[SIM_RESULT] makespan_cycles=1\n"
        "[D2D_TYPE] data_in=0 data_out=0\n"
        "combined stdout+stderr witness\n"
    )
    return _report_text_entries(
        case,
        oracle_text='{"oracle":true}',
        runtime_report_text='{"runtime":true}',
        finalizer_logs=("", ""),
        resolver_log="",
        runtime_logs=(runtime_log + "repeat=0\n", runtime_log + "repeat=1\n"),
        dramsys_negative_log=(
            "dramsys combined negative\n"
            if case is Stage1aCase.PD1
            else None
        ),
    )


class Stage1aReportRootEvidenceTest(unittest.TestCase):
    def test_exact_text_file_set_has_repeat_markers_and_no_npup(self) -> None:
        for case in Stage1aCase:
            with self.subTest(case=case.value):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw) / "evidence"
                    entries = _text_entries(case)
                    _publish_report_texts(root, case, entries)
                    stem = case.value.lower()
                    expected = {
                        f"{stem}.oracle.json",
                        f"{stem}.runtime.json",
                        f"{stem}.finalizer.0.log",
                        f"{stem}.finalizer.1.log",
                        f"{stem}.resolver.log",
                        f"{stem}.runtime.0.log",
                        f"{stem}.runtime.1.log",
                    }
                    if case is Stage1aCase.PD1:
                        expected.add(f"{stem}.dramsys-negative.log")
                    actual = {path.name for path in root.iterdir()}
                    self.assertEqual(actual, expected)
                    self.assertFalse(
                        any(name.endswith(".npup") for name in actual)
                    )
                    for index in range(2):
                        runtime = (
                            root / f"{stem}.runtime.{index}.log"
                        ).read_text(encoding="utf-8")
                        self.assertIn("[SIM_RESULT]", runtime)
                        self.assertIn("[D2D_TYPE]", runtime)

    def test_preexisting_file_and_symlink_fail_before_any_write(self) -> None:
        case = Stage1aCase.P1
        entries = _text_entries(case)
        for label, symlink in (("existing", False), ("symlink", True)):
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as raw:
                    root = Path(raw) / "evidence"
                    root.mkdir()
                    collision = root / "p1.runtime.0.log"
                    if symlink:
                        collision.symlink_to(Path(raw) / "missing-target")
                    else:
                        collision.write_text("do not replace", encoding="utf-8")
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "refusing to overwrite/symlink",
                    ):
                        _publish_report_texts(root, case, entries)
                    self.assertEqual(
                        {path.name for path in root.iterdir()},
                        {"p1.runtime.0.log"},
                    )

    def test_symlink_report_root_is_rejected(self) -> None:
        case = Stage1aCase.P1
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "destination"
            destination.mkdir()
            root = Path(raw) / "evidence"
            root.symlink_to(destination, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
                _publish_report_texts(root, case, _text_entries(case))
            self.assertEqual(tuple(destination.iterdir()), ())


if __name__ == "__main__":
    unittest.main()
