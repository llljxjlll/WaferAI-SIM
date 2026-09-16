"""A single Train link must bind physical DTE peers across both DP replicas."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.linker import _runtime_definitions
from llm.frontend.wafer_frontend.passes.train_link_program import link_train
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    ManifestInputKind, RegionManifest, RuntimeSymbolKind,
)
from llm.frontend.wafer_frontend.schema.global_action import LogicalCoreRef
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind, StandaloneNodeOrigin
from llm.test.frontend.unit import test_full_dense_training_dp2_n6_lowering as fixture


class FullDenseDP2NativeLinkTest(unittest.TestCase):
    @classmethod
    @builder_validation_session()
    def setUpClass(cls) -> None:
        fixture.FullDenseDP2N6LoweringTest.setUpClass()
        cls.native = fixture.FullDenseDP2N6LoweringTest.native
        cls.source = fixture.FullDenseDP2N6LoweringTest.source
        cls.linked = link_train(cls.native)
        cls.actions = {
            action.id: action for replica in cls.source.replicas
            for action in replica.global_dag.actions
            if action.task_kind is not SemanticTaskKind.TRANSIT
        }
        cls.fragments = tuple(
            linked.fragment if isinstance(linked, RegionManifest) else linked
            for replica in cls.native.replicas for linked in replica.fragments
        )

    def test_one_manifest_shares_only_source_bound_cross_replica_dte_fsm(self) -> None:
        manifest = self.linked.manifest
        self.assertEqual(len(manifest.fragments), 1288)
        anchors = tuple(d for d in manifest.input_digests
                        if d.kind is ManifestInputKind.DENSE_DP2_ROUTE_PLAN)
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].artifact_id,
                         self.source.replicas[0].scheduled.projected.dp_gradient_routes.id)
        self.assertEqual(anchors[0].schema_version,
                         "wafer_frontend.dense_dp2_route_plan/v1alpha1")
        self.assertEqual(sum(fragment.producer_pass == "dense_dp_gradient_lowering"
                             for fragment in self.fragments), 120)
        expected = _runtime_definitions(self.actions, self.fragments)
        by_id = {definition.symbol.id: definition for definition in
                 manifest.runtime_symbol_definitions}
        self.assertTrue(all(by_id[definition.symbol.id] == definition
                            for definition in expected))

    def test_missing_remote_recv_forged_core_and_conflicting_fsm_fail(self) -> None:
        dp_plan = self.source.replicas[0].scheduled.projected.dp_gradient_routes
        dp_fragments = tuple(fragment for fragment in self.fragments
                             if fragment.producer_pass == "dense_dp_gradient_lowering")
        remote = next(fragment for fragment in dp_fragments
                      if fragment.source_global_dag_id ==
                      self.source.replicas[1].global_dag.id)
        with self.assertRaisesRegex(SchemaError, "DTE FSM must be shared"):
            _runtime_definitions(self.actions, tuple(fragment for fragment in
                                                    self.fragments if fragment.id != remote.id))
        remote_recv = next(action for action in self.source.replicas[1].global_dag.actions
                           if action.task_kind is SemanticTaskKind.RECV
                           and isinstance(action.origin_ref, StandaloneNodeOrigin)
                           and action.origin_ref.collective_plan_id == dp_plan.id)
        forged_actions = dict(self.actions)
        forged_actions[remote_recv.id] = replace(
            remote_recv,
            logical_core=LogicalCoreRef(remote_recv.logical_core.die_id,
                                        remote_recv.logical_core.local_core_id + 7),
        )
        with self.assertRaisesRegex(SchemaError, "exact physical core"):
            _runtime_definitions(forged_actions, self.fragments)
        local_symbols = {symbol.id for fragment in dp_fragments
                         if fragment.source_global_dag_id ==
                         self.source.replicas[0].global_dag.id
                         for symbol in fragment.runtime_symbols
                         if symbol.kind is RuntimeSymbolKind.DTE_FSM}
        symbol = next(symbol for symbol in remote.runtime_symbols
                      if symbol.kind is RuntimeSymbolKind.DTE_FSM
                      and symbol.id in local_symbols)
        tampered = replace(remote, runtime_symbols=tuple(
            replace(item, source_ref=item.source_ref + ".forged")
            if item.id == symbol.id else item
            for item in remote.runtime_symbols
        ))
        with self.assertRaisesRegex(SchemaError, "conflicting runtime symbol"):
            _runtime_definitions(self.actions, tuple(
                tampered if item.id == remote.id else item
                for item in self.fragments
            ))

    def test_missing_route_trust_anchor_fails_structural_manifest_validation(self) -> None:
        invalid = replace(
            self.linked.manifest,
            input_digests=tuple(d for d in self.linked.manifest.input_digests
                                if d.kind is not ManifestInputKind.DENSE_DP2_ROUTE_PLAN),
        )
        with self.assertRaisesRegex(SchemaError, "DP2 route trust anchor"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
