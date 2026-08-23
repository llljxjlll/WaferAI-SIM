from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from intra_die_resource_evidence import collect_resource_evidence, parse_resource_output


SAMPLE = """
\x1b[92m[PRIM]\x1b[0m Core 2 start compute primitive Matmul_f. | 10 ns
[PRIM] Core 2 end compute primitive Matmul_f. | 21 ns
[PRIM] Core 2 start compute primitive Lsu_mem. | 30 ns
[PRIM] Core 2 end compute primitive Lsu_mem. | 40 ns
[PROGRAM_MEMORY] core=2 lsu_issued=1 lsu_completed=1 lsu_hbm_read_bytes=64
[SYSTEM] [D2D] in_pkts=1 out_pkts=1 busy_cycles=7 stall_cycles=3.
"""


class IntraDieResourceEvidenceTest(unittest.TestCase):
    def test_observable_intervals_are_classified_without_overlap_claim(self) -> None:
        row = parse_resource_output(SAMPLE)
        core = row["per_core"][0]
        self.assertEqual(core["primitive_busy_ns"]["compute"], 11)
        self.assertEqual(core["primitive_busy_cycles"]["compute"], 6)
        self.assertEqual(core["primitive_busy_ns"]["dte_lsu"], 10)
        self.assertEqual(row["d2d"], {"busy_cycles": 7, "stall_cycles": 3})
        self.assertIsNone(row["overlap_cycles"])

    def test_three_repeats_must_have_identical_resource_signature(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw); run = output / "run"; run.mkdir()
            for index in range(3):
                (run / f"stdout.{index}.log").write_text(SAMPLE, encoding="utf-8")
            evidence = collect_resource_evidence(output, 3)
            self.assertTrue(evidence["repeat_signature_stable"])
            (run / "stdout.2.log").write_text(SAMPLE.replace("21 ns", "23 ns"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed across"):
                collect_resource_evidence(output, 3)

    def test_unmatched_primitive_interval_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unclosed"):
            parse_resource_output("[PRIM] Core 0 start compute primitive Matmul_f. | 1 ns")


if __name__ == "__main__":
    unittest.main()
