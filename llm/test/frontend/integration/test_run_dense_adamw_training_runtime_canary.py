"""Reject 16/17 missing or repeated E2E AdamW source operations."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from llm.test.frontend.unit.test_dense_adamw_compile_sequence import _adamw_case
from .run_dense_adamw_training_runtime_canary import _source_oracle


class DenseAdamwRuntimeSourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.physical = _adamw_case()

    def _oracle(self, operations):
        fake = SimpleNamespace(logical_graph=SimpleNamespace(
            operations=tuple(operations),
            state_versions=self.source.logical_graph.state_versions,
        ))
        return _source_oracle(fake, self.physical)

    def test_real_two_layer_17_per_step_passes(self):
        self._oracle(self.source.logical_graph.operations)

    def test_missing_one_gate_up_logic_fails(self):
        operations = tuple(
            item for item in self.source.logical_graph.operations
            if not (
                item.kind.value == "adamw_update"
                and item.step == 1
                and item.parameter_ref == "layer.1.mlp_up.weight"
            )
        )
        with self.assertRaisesRegex(RuntimeError, "17 logical parameters"):
            self._oracle(operations)

    def test_repeating_one_gate_up_update_fails(self):
        duplicate = next(
            item for item in self.source.logical_graph.operations
            if item.kind.value == "adamw_update"
            and item.step == 0
            and item.parameter_ref == "layer.0.mlp_gate.weight"
        )
        with self.assertRaisesRegex(RuntimeError, "17 logical parameters"):
            self._oracle((*self.source.logical_graph.operations, duplicate))


if __name__ == "__main__":
    unittest.main()
