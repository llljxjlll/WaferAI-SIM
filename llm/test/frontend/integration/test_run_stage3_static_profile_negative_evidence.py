from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from llm.frontend.wafer_frontend.schema.serde import canonical_json
from run_stage3_static_profile_negative_evidence import (
    _validate_evidence,
    _write_new,
    build_negative_evidence,
)


class Stage3StaticProfileNegativeEvidenceTest(unittest.TestCase):
    def test_nine_production_witnesses_are_deterministic_and_strict(self) -> None:
        first = build_negative_evidence()
        second = build_negative_evidence()
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(len(first["witnesses"]), 9)
        self.assertTrue(all(item["passed"] for item in first["witnesses"]))

        reloaded = json.loads(canonical_json(first))
        _validate_evidence(reloaded)

        for field, value, message in (
            ("expected_message", "changed", "witness payload"),
            ("validator", "changed", "witness payload"),
        ):
            with self.subTest(field=field):
                tampered = copy.deepcopy(reloaded)
                tampered["witnesses"][0][field] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    _validate_evidence(tampered)

        tampered = copy.deepcopy(reloaded)
        tampered["witnesses"][0]["bindings"]["sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "witness payload"):
            _validate_evidence(tampered)

        tampered = copy.deepcopy(reloaded)
        tampered["id"] = "stage3_static_profile_negative_evidence_tampered"
        with self.assertRaisesRegex(RuntimeError, "stable id"):
            _validate_evidence(tampered)

    def test_json_publish_rejects_overwrite_symlink_and_npup(self) -> None:
        evidence = build_negative_evidence()
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            output = parent / "negative.json"
            _write_new(output, evidence)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                canonical_json(evidence) + "\n",
            )
            with self.assertRaisesRegex(RuntimeError, "overwrite/symlink"):
                _write_new(output, evidence)

            target = parent / "target.json"
            target.write_text("keep", encoding="utf-8")
            symlink = parent / "symlink.json"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "overwrite/symlink"):
                _write_new(symlink, evidence)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

            with self.assertRaisesRegex(RuntimeError, "not NPUP"):
                _write_new(parent / "negative.npup", evidence)
            self.assertFalse(
                any(path.suffix == ".npup" for path in parent.iterdir())
            )


if __name__ == "__main__":
    unittest.main()
