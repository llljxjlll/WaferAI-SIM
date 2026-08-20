from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from run_stage2_dense_forward_negative_evidence import (
    _expect_frontend_error,
    _validate_evidence,
    _write_new,
)


_DIGEST = "1" * 64
_KEYS = (
    "ce_projection_unsupported",
    "resolver_artifact_sha_mismatch",
    "runtime_artifact_sha_tamper",
    "runtime_d2d_packet_tamper",
    "runtime_functional_overclaim",
    "runtime_makespan_tamper",
    "runtime_repeat_marker_mismatch",
    "tp2_greedy_unsupported",
    "tp4_greedy_unsupported",
)


def _bindings() -> dict[str, tuple[dict[str, str], ...]]:
    return {
        "tools": ({"name": "python", "sha256": _DIGEST},),
        "sources": ({"path": "source.py", "sha256": _DIGEST},),
        "inputs": ({"name": "input", "sha256": _DIGEST},),
    }


class Stage2NegativeEvidenceTest(unittest.TestCase):
    def test_exact_error_type_and_message_are_required(self) -> None:
        def reject() -> None:
            raise SchemaError("exact", path="mutation")

        observed = _expect_frontend_error(
            key="exact",
            validator="reject",
            expected_type=SchemaError,
            expected_message="schema_error at mutation: exact",
            mutation=reject,
            bindings=_bindings(),
        )
        self.assertTrue(observed["passed"])
        with self.assertRaisesRegex(RuntimeError, "expected SchemaError"):
            _expect_frontend_error(
                key="wrong_message",
                validator="reject",
                expected_type=SchemaError,
                expected_message="schema_error at mutation: different",
                mutation=reject,
                bindings=_bindings(),
            )

    def test_nine_witness_binding_closure_is_strict(self) -> None:
        evidence = {
            "schema_version": (
                "wafer_frontend.stage2_dense_forward_negative_evidence/v1alpha1"
            ),
            "witnesses": tuple(
                {"key": key, "passed": True, "bindings": _bindings()}
                for key in _KEYS
            ),
        }
        _validate_evidence(evidence)
        bad = dict(evidence)
        bad["witnesses"] = evidence["witnesses"][:-1]
        with self.assertRaisesRegex(RuntimeError, "exactly nine"):
            _validate_evidence(bad)

    def test_json_publish_rejects_overwrite_symlink_and_npup(self) -> None:
        evidence = {"witnesses": 9}
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
            self.assertFalse(any(path.suffix == ".npup" for path in parent.iterdir()))


if __name__ == "__main__":
    unittest.main()
