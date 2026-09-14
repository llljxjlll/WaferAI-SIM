from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError

from flexible_mesh_release_dense import (
    _validate_dense_credit_closure,
)


class FlexibleDenseCreditClosureTest(unittest.TestCase):
    def test_requires_one_exact_balanced_credit_marker(self) -> None:
        _validate_dense_credit_closure("[CREDIT] data_balanced=1 ctrl_balanced=1.")
        for forged in (
            "",
            "[CREDIT] data_balanced=0 ctrl_balanced=1.",
            "[CREDIT] data_balanced=1 ctrl_balanced=0.",
            "[CREDIT] data_balanced=1 ctrl_balanced=1.\n"
            "[CREDIT] data_balanced=1 ctrl_balanced=1.",
        ):
            with self.subTest(forged=forged):
                with self.assertRaisesRegex(SchemaError, "credit closure"):
                    _validate_dense_credit_closure(forged)


if __name__ == "__main__":
    unittest.main()
