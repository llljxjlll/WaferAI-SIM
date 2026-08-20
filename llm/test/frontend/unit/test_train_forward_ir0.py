from __future__ import annotations

from dataclasses import replace
import unittest

from _fixtures import valid_spec

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.train_forward import (
    build_train_forward_ir0,
)
from llm.frontend.wafer_frontend.passes.train_forward_oracle import (
    build_train_forward_oracle,
)
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import (
    AttentionMode,
    AttentionWorkload,
    CrossEntropyForwardWorkload,
    IR0,
    IR0_SCHEMA_VERSION,
    JobKind,
    OpKind,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.train_forward_oracle import (
    TRAIN_FORWARD_ORACLE_SCHEMA_VERSION,
    TrainForwardOracle,
)


def _tiny_train_spec() -> ExperimentSpec:
    raw = valid_spec()
    raw["model"].update(
        V=32,
        H=16,
        I=32,
        NH=4,
        KVH=4,
        DH=4,
        rotary_dim=4,
        L=2,
        max_position_embeddings=64,
    )
    raw["workload"] = {
        "mode": "train",
        "infer": None,
        "train": {
            "global_batch": 4,
            "micro_batch": 1,
            "seq_len": 8,
            "backward": False,
            "optimizer": "none",
            "structure": {
                "micro_batch_count": 2,
                "pp_schedule": "gpipe",
                "interleave_chunks": 1,
                "recompute": "none",
            },
        },
    }
    raw["parallel"]["instances"][0].update(
        role="train", tp=2, sp=True, dp=2, replicas=1, pp=1, ep=1
    )
    return from_data(ExperimentSpec, raw, path="spec")


def _rebuild(graph: IR0, **changes: object) -> IR0:
    fields: dict[str, object] = {
        "producer_pass": graph.producer_pass,
        "job": graph.job,
        "instances": graph.instances,
        "nodes": graph.nodes,
        "values": graph.values,
        "edges": graph.edges,
        "fusion_candidates": graph.fusion_candidates,
        "persistent_states": graph.persistent_states,
        "state_accesses": graph.state_accesses,
        "profile": graph.profile,
        "train": graph.train,
        "instance_profiles": graph.instance_profiles,
        "node_profiles": graph.node_profiles,
        "pd_plan_id": graph.pd_plan_id,
    }
    fields.update(changes)
    return IR0.create(**fields)  # type: ignore[arg-type]


class TrainForwardIr0Test(unittest.TestCase):
    def test_tiny_dp2_tp2_topology_and_typed_boundary(self) -> None:
        graph = build_train_forward_ir0(_tiny_train_spec())
        self.assertEqual(IR0_SCHEMA_VERSION, "wafer_frontend.ir0/v1alpha11")
        self.assertIs(graph.job, JobKind.TRAIN)
        self.assertEqual(
            (
                len(graph.nodes),
                len(graph.values),
                len(graph.edges),
                len(graph.fusion_candidates),
                len(graph.persistent_states),
                len(graph.state_accesses),
            ),
            (34, 51, 37, 4, 30, 30),
        )
        self.assertTrue(
            all(
                state.identity.kind is StateKind.PARAMETER
                for state in graph.persistent_states
            )
        )
        attention = tuple(
            node for node in graph.nodes if node.kind is OpKind.ATTENTION
        )
        self.assertEqual(len(attention), 2)
        self.assertTrue(
            all(
                isinstance(node.workload, AttentionWorkload)
                and node.workload.mode is AttentionMode.TRAIN_FORWARD
                and node.workload.logical_kv_read_bytes == 0
                and node.workload.logical_kv_write_bytes == 0
                for node in attention
            )
        )
        ce = tuple(node for node in graph.nodes if node.kind is OpKind.CE_FORWARD)
        self.assertEqual(len(ce), 1)
        self.assertIsInstance(ce[0].workload, CrossEntropyForwardWorkload)
        labels = next(value for value in graph.values if value.id.endswith(".labels"))
        loss = next(value for value in graph.values if value.id.endswith(".loss"))
        self.assertEqual((labels.shape, labels.dtype), ((8,), DType.INT32))
        self.assertEqual((loss.shape, loss.dtype), ((8,), DType.FP32))
        self.assertFalse(any(node.kind is OpKind.SAMPLING for node in graph.nodes))
        DenseIR0Validator.validate(graph, "train")

    def test_strict_round_trip_and_old_version_rejected(self) -> None:
        graph = build_train_forward_ir0(_tiny_train_spec())
        self.assertEqual(
            loads_dataclass(IR0, canonical_json(graph), path="graph"), graph
        )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(graph, schema_version="wafer_frontend.ir0/v1alpha9").validate()

    def test_ce_and_attention_tamper_fail_closed(self) -> None:
        graph = build_train_forward_ir0(_tiny_train_spec())
        bad_values = tuple(
            replace(value, dtype=DType.FP16)
            if value.id.endswith(".labels")
            else value
            for value in graph.values
        )
        with self.assertRaisesRegex(SchemaError, "cross entropy"):
            DenseIR0Validator.validate(_rebuild(graph, values=bad_values))

        attention_index = next(
            index
            for index, node in enumerate(graph.nodes)
            if node.kind is OpKind.ATTENTION
        )
        attention = graph.nodes[attention_index]
        assert isinstance(attention.workload, AttentionWorkload)
        bad_attention = replace(
            attention,
            workload=replace(
                attention.workload,
                logical_kv_write_bytes=4,
                rank_kv_write_bytes=2,
            ),
        )
        bad_nodes = list(graph.nodes)
        bad_nodes[attention_index] = bad_attention
        with self.assertRaisesRegex(SchemaError, "must equal 0"):
            _rebuild(graph, nodes=tuple(bad_nodes)).validate()

    def test_independent_oracle_frozen_dp2_tp2_goldens(self) -> None:
        spec = _tiny_train_spec()
        graph = build_train_forward_ir0(spec)
        oracle = build_train_forward_oracle(spec)
        self.assertEqual(
            TRAIN_FORWARD_ORACLE_SCHEMA_VERSION,
            "wafer_frontend.train_forward_oracle/v1alpha1",
        )
        self.assertEqual(
            (
                oracle.parameters.unique_tensor_count,
                oracle.parameters.unique_elements,
                oracle.parameters.unique_bytes,
                oracle.parameters.tp_placed_elements,
                oracle.parameters.tp_placed_bytes,
                oracle.parameters.dp_replicated_bytes,
            ),
            (15, 6224, 12448, 7328, 14656, 29312),
        )
        self.assertEqual(
            (
                oracle.gemm_flops_per_microbatch,
                oracle.attention_query_key_pairs_per_microbatch,
                oracle.attention_flops_per_microbatch,
                oracle.logical_forward_flops_per_microbatch,
                oracle.rank_forward_flops_per_microbatch,
                oracle.cluster_forward_flops_per_step,
            ),
            (90112, 72, 4608, 94720, 47360, 378880),
        )
        self.assertEqual(
            (
                oracle.collectives.node_count,
                oracle.collectives.logical_tensor_bytes_per_node,
                oracle.collectives.group_payload_bytes_per_microbatch,
                oracle.collectives.cluster_step_group_payload_bytes,
            ),
            (8, 256, 2048, 8192),
        )
        self.assertEqual(
            (
                oracle.ce.logical_rows,
                oracle.ce.rank_rows,
                oracle.ce.logical_label_bytes,
                oracle.ce.rank_label_bytes,
                oracle.ce.logical_loss_bytes,
                oracle.ce.rank_loss_bytes,
            ),
            (8, 4, 32, 16, 32, 16),
        )
        oracle.validate_against_ir0(spec, graph)
        self.assertEqual(
            loads_dataclass(
                TrainForwardOracle,
                canonical_json(oracle),
                path="oracle",
            ),
            oracle,
        )

    def test_tp1_quotient_and_oracle_tamper_fail_closed(self) -> None:
        spec = _tiny_train_spec()
        instance = spec.parallel.instances[0]
        spec = replace(
            spec,
            parallel=replace(
                spec.parallel,
                instances=(replace(instance, tp=1, sp=False),),
            ),
        )
        spec.validate()
        graph = build_train_forward_ir0(spec)
        oracle = build_train_forward_oracle(spec)
        self.assertEqual(
            (
                len(graph.nodes),
                len(graph.values),
                len(graph.edges),
                len(graph.fusion_candidates),
                oracle.graph.collective_node_count,
            ),
            (26, 43, 29, 0, 0),
        )
        oracle.validate_against_ir0(spec, graph)

        semantic = oracle._semantic_key()
        semantic["parameters"] = replace(
            oracle.parameters,
            unique_tensor_count=oracle.parameters.unique_tensor_count + 1,
        )
        forged = TrainForwardOracle.create(**semantic)
        with self.assertRaisesRegex(SchemaError, "source spec"):
            forged.validate_against_spec(spec)
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                oracle,
                schema_version="wafer_frontend.train_forward_oracle/v1alpha0",
            ).validate()


if __name__ == "__main__":
    unittest.main()
