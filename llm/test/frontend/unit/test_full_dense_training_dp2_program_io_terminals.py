"""SGD terminal identity is replica-local; physical buffers remain independently verified."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import program_io
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.ir2 import StateUseAccess
from llm.frontend.wafer_frontend.schema.train_n6 import TrainLinkedProgram


def _replica(*, outputs: tuple[str, ...]) -> SimpleNamespace:
    nodes = tuple(SimpleNamespace(kind=OpKind.OPTIMIZER_UPDATE, outputs=(value,))
                  for value in outputs)
    values = tuple(SimpleNamespace(id=value, consumers=())
                   for value in ('loss::step0', *outputs))
    return SimpleNamespace(ir1=SimpleNamespace(nodes=nodes, values=values))


class DP2ProgramIoTerminalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = object.__new__(TrainLinkedProgram)

    def test_same_logical_sgd_outputs_in_two_replicas_have_distinct_physical_owners(self) -> None:
        replicas = ((0, _replica(outputs=('state::step0', 'state::step1'))),
                    (1, _replica(outputs=('state::step0', 'state::step1'))))
        with patch.object(program_io, '_lowering_contexts', return_value=replicas):
            self.assertEqual(program_io._terminal_value_ids(self.source), {'loss::step0'})

    def test_duplicate_sgd_output_within_one_replica_is_rejected(self) -> None:
        replicas = ((0, _replica(outputs=('same', 'same'))),
                    (1, _replica(outputs=('one', 'two'))))
        with patch.object(program_io, '_lowering_contexts', return_value=replicas):
            with self.assertRaisesRegex(SchemaError, 'within its DP replica'):
                program_io._terminal_value_ids(self.source)

    def test_two_dp_physical_states_must_not_alias_one_hbm_binding(self) -> None:
        def state(identifier: str, binding: str) -> SimpleNamespace:
            return SimpleNamespace(
                abi=SimpleNamespace(state_ref='same-logical-parameter',
                                    id=identifier, hbm_binding_ref=binding),
                first_access=StateUseAccess.READ,
                uses=((0, StateUseAccess.READ), (1, StateUseAccess.WRITE)),
            )
        copies = (state('physical-dp0', 'same-hbm-binding'),
                  state('physical-dp1', 'same-hbm-binding'))
        with patch.object(program_io, '_state_logical_signature', return_value=('fp16-shard',)):
            with self.assertRaisesRegex(SchemaError, 'physical StateABI ids and HBM bindings must be unique'):
                program_io._state_abi_groups(copies)
            proper = (copies[0], state('physical-dp1', 'distinct-hbm-binding'))
            self.assertEqual(len(program_io._state_abi_groups(proper)['same-logical-parameter']), 2)

    def test_missing_sgd_output_in_one_replica_is_rejected(self) -> None:
        malformed = _replica(outputs=('one', 'two'))
        malformed.ir1.nodes = (SimpleNamespace(kind=OpKind.OPTIMIZER_UPDATE, outputs=()),
                               malformed.ir1.nodes[1])
        with patch.object(program_io, '_lowering_contexts',
                          return_value=((0, malformed), (1, _replica(outputs=('one', 'two'))))):
            with self.assertRaisesRegex(SchemaError, 'within its DP replica'):
                program_io._terminal_value_ids(self.source)


if __name__ == '__main__':
    unittest.main()
