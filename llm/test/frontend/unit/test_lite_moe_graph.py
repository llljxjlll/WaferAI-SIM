from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_moe_graph import (
    LiteMoeIR0Validator,
    build_lite_moe_ir0_adapter,
)
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.ir0 import (
    IR0,
    LogicalNode,
    OpKind,
    P2PByteWorkload,
)
from llm.frontend.wafer_frontend.schema.lite_moe import LiteMoeTransferRole
from llm.frontend.wafer_frontend.schema.lite_moe_graph import (
    LITE_MOE_IR0_ADAPTER_SCHEMA_VERSION,
    LiteMoeIR0Adapter,
    LiteMoeP2PBinding,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)
from llm.test.frontend.integration.lite_moe_cases import (
    build_lite_moe_source_case,
)


def _graph(graph: IR0, **changes: object) -> IR0:
    fields = {
        "producer_pass": graph.producer_pass,
        "job": graph.job,
        "instances": graph.instances,
        "nodes": graph.nodes,
        "values": graph.values,
        "edges": graph.edges,
        "fusion_candidates": graph.fusion_candidates,
        "profile": graph.profile,
        "train": graph.train,
        "instance_profiles": graph.instance_profiles,
        "node_profiles": graph.node_profiles,
        "pd_plan_id": graph.pd_plan_id,
        "persistent_states": graph.persistent_states,
        "state_accesses": graph.state_accesses,
    }
    fields.update(changes)
    return IR0.create(**fields)


def _adapter(
    adapter: LiteMoeIR0Adapter, **changes: object
) -> LiteMoeIR0Adapter:
    fields = adapter._semantic_key()
    fields.update(changes)
    return LiteMoeIR0Adapter.create(**fields)


class LiteMoeGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = build_lite_moe_source_case()
        self.adapter = build_lite_moe_ir0_adapter(
            self.source.spec,
            self.source.moe_spec,
            self.source.oracle,
        )

    def _validate(self, adapter: LiteMoeIR0Adapter) -> None:
        LiteMoeIR0Validator.validate(
            adapter,
            self.source.spec,
            self.source.moe_spec,
            self.source.oracle,
        )

    def test_exact_graph_golden_round_trip_and_determinism(self) -> None:
        graph = self.adapter.graph
        self.assertEqual(
            LITE_MOE_IR0_ADAPTER_SCHEMA_VERSION,
            "wafer_frontend.s3_lite_static_moe_ir0_adapter/v1alpha1",
        )
        self.assertEqual(len(graph.nodes), 40)
        self.assertEqual(
            Counter(node.kind for node in graph.nodes),
            {OpKind.GEMM: 24, OpKind.ELEMENTWISE: 8, OpKind.P2P: 8},
        )
        self.assertEqual(len(graph.values), 60)
        self.assertEqual(len(graph.edges), 36)
        self.assertEqual(len(graph.persistent_states), 12)
        self.assertEqual(len(graph.state_accesses), 24)
        self.assertEqual(len(self.adapter.p2p_bindings), 8)
        self.assertEqual(
            sum(
                node.workload.bytes
                for node in graph.nodes
                if type(node.workload) is P2PByteWorkload
            ),
            256,
        )
        self.assertEqual(
            sum(
                2
                * node.workload.logical_shape[0]
                * node.workload.logical_shape[1]
                * node.workload.logical_shape[2]
                for node in graph.nodes
                if node.kind is OpKind.GEMM
            ),
            24576,
        )
        self.assertEqual(
            tuple(
                (binding.token_index, binding.role, binding.source_die_id,
                 binding.destination_die_id)
                for binding in self.adapter.p2p_bindings
            ),
            (
                (1, LiteMoeTransferRole.MOE_DISPATCH, 1, 0),
                (1, LiteMoeTransferRole.MOE_COMBINE, 0, 1),
                (3, LiteMoeTransferRole.MOE_DISPATCH, 1, 0),
                (3, LiteMoeTransferRole.MOE_COMBINE, 0, 1),
                (4, LiteMoeTransferRole.MOE_DISPATCH, 0, 1),
                (4, LiteMoeTransferRole.MOE_COMBINE, 1, 0),
                (6, LiteMoeTransferRole.MOE_DISPATCH, 0, 1),
                (6, LiteMoeTransferRole.MOE_COMBINE, 1, 0),
            ),
        )
        self._validate(self.adapter)
        with self.assertRaisesRegex(SchemaError, "requires a TP mesh axis"):
            DenseIR0Validator.validate(graph)
        second = build_lite_moe_ir0_adapter(
            self.source.spec, self.source.moe_spec, self.source.oracle
        )
        self.assertEqual(second, self.adapter)
        self.assertEqual(canonical_digest(second), canonical_digest(self.adapter))
        self.assertEqual(
            loads_dataclass(
                LiteMoeIR0Adapter,
                canonical_json(self.adapter),
                path="adapter",
            ),
            self.adapter,
        )

    def test_missing_duplicate_node_and_edge_fail_closed(self) -> None:
        graph = self.adapter.graph
        with self.assertRaises(SchemaError):
            _adapter(
                self.adapter,
                graph=_graph(graph, nodes=graph.nodes[:-1]),
            )
        with self.assertRaisesRegex(SchemaError, "duplicate id"):
            _adapter(
                self.adapter,
                graph=_graph(graph, nodes=(*graph.nodes, graph.nodes[0])),
            )
        with self.assertRaisesRegex(SchemaError, "data edges"):
            _adapter(
                self.adapter,
                graph=_graph(graph, edges=graph.edges[:-1]),
            )

    def test_binding_route_and_workload_tamper_fail_closed(self) -> None:
        first, second, *rest = self.adapter.p2p_bindings
        swapped = (
            LiteMoeP2PBinding.create(
                node_ref=second.node_ref,
                role=first.role,
                token_index=first.token_index,
                expert_index=first.expert_index,
                source_die_id=first.source_die_id,
                destination_die_id=first.destination_die_id,
            ),
            LiteMoeP2PBinding.create(
                node_ref=first.node_ref,
                role=second.role,
                token_index=second.token_index,
                expert_index=second.expert_index,
                source_die_id=second.source_die_id,
                destination_die_id=second.destination_die_id,
            ),
            *rest,
        )
        with self.assertRaisesRegex(SchemaError, "binding/route"):
            self._validate(_adapter(self.adapter, p2p_bindings=swapped))
        with self.assertRaisesRegex(SchemaError, "binding/route"):
            self._validate(
                _adapter(
                    self.adapter,
                    p2p_bindings=self.adapter.p2p_bindings[:-1],
                )
            )
        with self.assertRaisesRegex(SchemaError, "duplicate"):
            _adapter(
                self.adapter,
                p2p_bindings=(first, first, *self.adapter.p2p_bindings[2:]),
            )
        with self.assertRaisesRegex(SchemaError, "local tokens"):
            LiteMoeP2PBinding.create(
                node_ref="forged",
                role=LiteMoeTransferRole.MOE_DISPATCH,
                token_index=0,
                expert_index=0,
                source_die_id=0,
                destination_die_id=0,
            )

        graph = self.adapter.graph
        index = next(
            index for index, node in enumerate(graph.nodes)
            if node.kind is OpKind.P2P
        )
        node = graph.nodes[index]
        changed = replace(
            node,
            workload=P2PByteWorkload(bytes=48, dtype=node.workload.dtype),
        )
        changed_nodes = (
            *graph.nodes[:index], changed, *graph.nodes[index + 1:]
        )
        forged = _adapter(
            self.adapter,
            graph=_graph(graph, nodes=changed_nodes),
        )
        with self.assertRaisesRegex(SchemaError, "node contract"):
            self._validate(forged)

    def test_state_access_and_provenance_tamper_fail_closed(self) -> None:
        graph = self.adapter.graph
        removed = graph.persistent_states[0]
        reduced_graph = _graph(
            graph,
            persistent_states=graph.persistent_states[1:],
            state_accesses=tuple(
                access for access in graph.state_accesses
                if access.state_ref != removed.id
            ),
        )
        with self.assertRaisesRegex(SchemaError, "parameter-state"):
            self._validate(_adapter(self.adapter, graph=reduced_graph))
        reduced_accesses = _graph(
            graph, state_accesses=graph.state_accesses[1:]
        )
        with self.assertRaisesRegex(SchemaError, "parameter access"):
            self._validate(_adapter(self.adapter, graph=reduced_accesses))
        foreign = _adapter(
            self.adapter, source_experiment_digest="1" * 64
        )
        with self.assertRaisesRegex(SchemaError, "source provenance"):
            self._validate(foreign)

    def test_strict_serde_version_and_id_fail_closed(self) -> None:
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(self.adapter, schema_version="v0").validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(self.adapter, id="forged").validate()
        raw = json.loads(canonical_json(self.adapter))
        raw["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(LiteMoeIR0Adapter, json.dumps(raw), path="adapter")
        del raw["unexpected"]
        del raw["p2p_bindings"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(LiteMoeIR0Adapter, json.dumps(raw), path="adapter")


if __name__ == "__main__":
    unittest.main()
