"""The verified AdamW WGRAD leaf cannot be published as full Dense training."""
from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_adamw_compile_sequence import (
    compile_dense_adamw_step,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_full_train_admission import (
    require_dense_adamw_full_train_admission,
)
from llm.test.frontend.unit.test_dense_adamw_compile_sequence import _adamw_case


class DenseAdamwFullTrainAdmissionTest(unittest.TestCase):
    def test_current_public_two_step_leaf_is_not_full_training(self) -> None:
        source, physical = _adamw_case()
        linked = tuple(compile_dense_adamw_step(source, physical, i).manifest
                       for i in (0, 1))
        with self.assertRaisesRegex(
                SchemaError, "full forward/loss/backward/optimizer physical work missing"):
            require_dense_adamw_full_train_admission(linked)

    def test_single_step_cannot_claim_two_step_training(self) -> None:
        with self.assertRaisesRegex(SchemaError, "requires two independently linked"):
            require_dense_adamw_full_train_admission(())


if __name__ == "__main__":
    unittest.main()
