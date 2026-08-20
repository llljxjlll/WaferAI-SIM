from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest

from run_stage2_dense_forward import (
    _GOLDENS,
    _REVIEWED_RUNTIME_GOLDENS,
    _ReviewedRuntimeWitness,
    _StructureWitness,
    _parse_d2d,
    _expected_report_names,
    _publish_report_texts,
    _rows,
    _validate_reviewed_runtime,
    _validate_structure,
)


def _structure(tp_degree: int) -> _StructureWitness:
    expected = _GOLDENS[tp_degree]
    return _StructureWitness(
        action_count=expected.action_count,
        leaf_count=expected.leaf_count,
        record_count=expected.record_count,
        address_binding_count=expected.address_binding_count,
        relocation_count=expected.relocation_count,
    )


def _tp2_d2d() -> str:
    return "\n".join(
        (
            "[D2D_TYPE] request_in=16 request_out=16 ack_in=32 "
            "ack_out=32 data_in=128 data_out=128",
            "[D2D_LINK] idx=0 die0->die1 dir=E req_in=8 req_out=8 "
            "ack_in=16 ack_out=16 data_in=64 data_out=64",
            "[D2D_LINK] idx=1 die1->die0 dir=W req_in=8 req_out=8 "
            "ack_in=16 ack_out=16 data_in=64 data_out=64",
        )
    )


class Stage2RunnerStructureTest(unittest.TestCase):
    def test_all_tp_structures_are_accepted(self) -> None:
        for tp_degree in (1, 2, 4):
            with self.subTest(tp_degree=tp_degree):
                _validate_structure(tp_degree, _structure(tp_degree))

    def test_each_of_five_structure_fields_is_fail_closed(self) -> None:
        for tp_degree in (1, 2, 4):
            expected = _structure(tp_degree)
            for field in fields(_StructureWitness):
                with self.subTest(tp_degree=tp_degree, field=field.name):
                    tampered = replace(
                        expected,
                        **{field.name: getattr(expected, field.name) + 1},
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "five-field structure changed"
                    ):
                        _validate_structure(tp_degree, tampered)

    def test_reviewed_runtime_five_fields_and_each_tamper(self) -> None:
        for tp_degree in (1, 2, 4):
            expected = _REVIEWED_RUNTIME_GOLDENS[tp_degree]
            with self.subTest(tp_degree=tp_degree, accepted=True):
                _validate_reviewed_runtime(tp_degree, expected)
            for field in fields(_ReviewedRuntimeWitness):
                value = getattr(expected, field.name)
                tampered_value = value + 1 if type(value) is int else value + "0"
                with self.subTest(tp_degree=tp_degree, field=field.name):
                    with self.assertRaisesRegex(
                        RuntimeError, "reviewed runtime five-field golden changed"
                    ):
                        _validate_reviewed_runtime(
                            tp_degree,
                            replace(expected, **{field.name: tampered_value}),
                        )


class Stage2RunnerD2DTest(unittest.TestCase):
    def test_tp2_logical_and_physical_link_counts_are_exact(self) -> None:
        evidence = _parse_d2d(2, _tp2_d2d())
        self.assertEqual(
            tuple(
                (item.source_die_id, item.destination_die_id, item.data_packets)
                for item in evidence.links
            ),
            ((0, 1, 64), (1, 0, 64)),
        )
        self.assertEqual(
            (
                evidence.flow_count,
                evidence.logical_packet_count,
                evidence.physical_packet_count,
                evidence.request_packet_count,
                evidence.ack_packet_count,
                evidence.logical_bytes,
                evidence.byte_hop_bytes,
            ),
            (16, 128, 128, 16, 32, 2048, 2048),
        )

    def test_type_link_and_balance_tampering_fail_closed(self) -> None:
        valid = _tp2_d2d()
        cases = {
            "type": valid.replace("data_out=128", "data_out=127", 1),
            "link": valid.rsplit("\n", 1)[0],
            "balance": valid.replace("data_out=64", "data_out=63", 1),
            "request": valid.replace(
                "req_in=8 req_out=8", "req_in=9 req_out=9", 1),
        }
        for label, output in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, "D2D"):
                    _parse_d2d(2, output)

    def test_marker_rows_strip_runtime_log_punctuation(self) -> None:
        output = "prefix [SIM_RESULT] makespan_cycles=5805. | 11611 ns\n"
        self.assertEqual(
            _rows(output, "[SIM_RESULT] "),
            [{"makespan_cycles": "5805"}],
        )


class Stage2ReportRootTest(unittest.TestCase):
    @staticmethod
    def _entries(tp_degree: int = 1) -> tuple[tuple[str, str], ...]:
        return tuple((name, "") for name in _expected_report_names(tp_degree))

    def test_exact_set_is_atomically_published_without_npup(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "evidence"
            _publish_report_texts(root, 1, self._entries())
            self.assertEqual(
                tuple(sorted(path.name for path in root.iterdir())),
                _expected_report_names(1),
            )
            self.assertFalse(any(path.suffix == ".npup" for path in root.iterdir()))

    def test_incomplete_existing_and_symlink_roots_fail_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            with self.assertRaisesRegex(RuntimeError, "exact reviewed set"):
                _publish_report_texts(
                    parent / "incomplete",
                    1,
                    self._entries()[:-1],
                )
            self.assertFalse((parent / "incomplete").exists())

            existing = parent / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must not already exist"):
                _publish_report_texts(existing, 1, self._entries())
            self.assertEqual(tuple(existing.iterdir()), ())

            destination = parent / "destination"
            destination.mkdir()
            symlink = parent / "symlink"
            symlink.symlink_to(destination, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
                _publish_report_texts(symlink, 1, self._entries())
            self.assertEqual(tuple(destination.iterdir()), ())


if __name__ == "__main__":
    unittest.main()
