"""Standalone asynchronous RECV/WAIT retains exact source plan, rank and token."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import _fused_recv_wait_pairs
from llm.frontend.wafer_frontend.schema.ir2 import (
    OriginKind, SemanticTaskKind, StandaloneNodeOrigin,
)
from test_global_action_schema import _create_global, _with_recv_wait


class StandaloneRecvWaitPairTest(unittest.TestCase):
    def test_exact_standalone_pair_and_forged_provenance(self):
        ir1, projection, schedule_set = _with_recv_wait()
        dag = _create_global(ir1, projection, schedule_set)
        recv = next(action for action in dag.actions
                    if action.task_kind is SemanticTaskKind.RECV)
        wait = next(action for action in dag.actions
                    if action.task_kind is SemanticTaskKind.WAIT)
        plan = "physical_standalone_rs"
        recv = replace(recv, origin_ref=StandaloneNodeOrigin(
            OriginKind.STANDALONE_COLLECTIVE, plan, 0, "recv"))
        wait = replace(wait, origin_ref=StandaloneNodeOrigin(
            OriginKind.STANDALONE_COLLECTIVE, plan, 0, "wait"))
        actions = {action.id: action for action in (recv, wait)}
        by_wait, by_recv = _fused_recv_wait_pairs(actions, "test")
        self.assertEqual(by_wait[wait.id], recv)
        self.assertEqual(by_recv[recv.id], wait)
        for forged in (
            replace(wait, origin_ref=replace(
                wait.origin_ref, collective_plan_id="other_plan")),
            replace(wait, origin_ref=replace(wait.origin_ref, rank=1)),
            replace(wait, runtime_binding=replace(
                wait.runtime_binding, token_symbol="other_token")),
            replace(wait, deps=()),
        ):
            with self.subTest(forged=forged):
                with self.assertRaises(SchemaError):
                    _fused_recv_wait_pairs({recv.id: recv, wait.id: forged}, "test")


if __name__ == "__main__":
    unittest.main()
