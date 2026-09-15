"""Source capacity must cover physical Dense training states before offload."""

from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.training_state_abi_reconciliation import (
    inspect_physical_training_state_abis,
    require_offload_state_abi_source,
)

from .run_dense_training_sequence_runtime_canary import _offload_sequence
from ..unit.test_dense_training_compile_sequence import _sequence


class TrainingStateAbiReconciliationTest(unittest.TestCase):
    def test_one_die_accepted_external_authority_has_exact_source_bytes(self) -> None:
        sequence = _offload_sequence()
        sequence.validate()
        source = sequence.materialization
        linked = sequence.segments[0].linked_program.manifest
        witness = require_offload_state_abi_source(source, linked)
        self.assertEqual(len(witness), 1)
        self.assertEqual((witness[0].source_parameter_bytes,
                          witness[0].linked_state_bytes,
                          witness[0].linked_span_bytes,
                          witness[0].linked_padding_bytes,
                          witness[0].state_abi_count),
                         (17344, 17344, 17344, 0, 15))

    def test_real_tp4_four_die_linked_states_reject_unmaterialized_source(self) -> None:
        # A genuine two-step production link, rather than a fabricated StateABI.
        sequence = _sequence(1, 4)
        sequence.validate()
        source = sequence.materialization
        linked = sequence.segments[0].linked_program.manifest
        witness = inspect_physical_training_state_abis(source, linked)
        self.assertEqual(tuple(item.die_id for item in witness), (0, 1, 2, 3))
        self.assertEqual(tuple((item.source_parameter_bytes,
                                item.linked_state_bytes,
                                item.linked_span_bytes,
                                item.linked_padding_bytes,
                                item.state_abi_count)
                               for item in witness),
                         ((3112, 4768, 4928, 160, 15),) * 4)
        with self.assertRaises(SchemaError) as blocked:
            require_offload_state_abi_source(source, linked)
        self.assertEqual(blocked.exception.code,
                         "external_state_abi_source_mismatch")
        self.assertIn("die:0 declared=3112 linked=4768", str(blocked.exception))
        self.assertIn("die:3 declared=3112 linked=4768", str(blocked.exception))


if __name__ == "__main__":
    unittest.main()
