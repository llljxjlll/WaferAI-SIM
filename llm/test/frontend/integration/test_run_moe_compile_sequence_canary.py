from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest

from llm.test.frontend.integration.run_moe_compile_sequence_canary import main


class MoeCompileSequenceCanaryTest(unittest.TestCase):
    def test_compile_canary_covers_both_families(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(), 0)
        text = output.getvalue()
        self.assertIn("family=moe_inference_e2e units=6", text)
        self.assertIn("family=moe_training_e2e units=4", text)
        self.assertIn("status=PASS units=10", text)
        self.assertIn("runtime=runtime_not_materialized", text)


if __name__ == "__main__":
    unittest.main()
