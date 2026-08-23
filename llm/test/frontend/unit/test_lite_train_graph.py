from __future__ import annotations

from dataclasses import replace
import unittest

from llm.test.frontend.unit._fixtures import valid_spec

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_train import (
    build_s2_lite_lm_head_train_oracle,
)
from llm.frontend.wafer_frontend.passes.lite_train_graph import (
    build_s2_lite_lm_head_train_ir0,
)
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.action import (
    FUSION_PLAN_SCHEMA_VERSION,
    STANDALONE_COLLECTIVE_PLAN_SCHEMA_VERSION,
    canonical_compute_operand_roles,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.experiment import (
    ExperimentSpec,
    TrainOptimizer,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    EdgeKind,
    IR0,
    IR0_SCHEMA_VERSION,
    OpKind,
    OpPhase,
    StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.lite_train import (
    S2_LITE_LM_HEAD_TRAIN_CASE_ID,
    S2LiteLmHeadTrainContract,
    S2LiteTrainCoverage,
    S2LiteTrainStage,
)
from llm.frontend.wafer_frontend.schema.lite_train_graph import (
    S2_LITE_LM_HEAD_TRAIN_IR0_SCHEMA_VERSION,
    S2LiteLmHeadTrainIR0,
)
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PERSISTENT_STATE_DECL_SCHEMA_VERSION,
    PERSISTENT_STATE_IDENTITY_SCHEMA_VERSION,
    PERSISTENT_STATE_MANIFEST_SCHEMA_VERSION,
    PersistentStateAccess,
    StateKind,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)


def _spec() -> ExperimentSpec:
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
            "global_batch": 1,
            "micro_batch": 1,
            "seq_len": 8,
            "backward": False,
            "optimizer": "none",
            "structure": {
                "micro_batch_count": 1,
                "pp_schedule": "gpipe",
                "interleave_chunks": 1,
                "recompute": "none",
            },
        },
    }
    raw["parallel"]["instances"][0].update(
        id="T0",
        role="train",
        tp=1,
        sp=False,
        dp=1,
        replicas=1,
        pp=1,
        ep=1,
    )
    return from_data(ExperimentSpec, raw, path="spec")


def _sources():
    spec = _spec()
    contract = S2LiteLmHeadTrainContract.create(
        case_id=S2_LITE_LM_HEAD_TRAIN_CASE_ID,
        source_spec_digest=canonical_digest(spec),
        coverage=S2LiteTrainCoverage.LM_HEAD_ONLY,
        backbone_frozen=True,
        embedding_frozen=True,
        optimizer=TrainOptimizer.SGD,
        learning_rate=0.001,
        momentum=0.0,
        dp_degree=1,
        tp_degree=1,
        pp_degree=1,
        ep_degree=1,
        micro_batch_count=1,
        step_count=1,
        micro_batch_size=1,
        sequence_length=8,
        hidden_size=16,
        vocabulary_size=32,
        activation_dtype=DType.FP16,
        label_dtype=DType.INT32,
        loss_gradient_dtype=DType.FP32,
        weight_dtype=DType.FP16,
        weight_gradient_dtype=DType.FP32,
        stages=(
            S2LiteTrainStage.CE_BACKWARD,
            S2LiteTrainStage.LM_HEAD_WGRAD,
            S2LiteTrainStage.SGD_UPDATE,
        ),
    )
    return spec, contract, build_s2_lite_lm_head_train_oracle(contract)


def _build() -> S2LiteLmHeadTrainIR0:
    return build_s2_lite_lm_head_train_ir0(*_sources())


def _rebuild_graph(graph: IR0, **changes: object) -> IR0:
    semantic: dict[str, object] = {
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
    semantic.update(changes)
    return IR0.create(**semantic)  # type: ignore[arg-type]


class S2LiteLmHeadTrainGraphTest(unittest.TestCase):
    def test_tp1_exact_topology_workloads_roles_and_state(self) -> None:
        result = _build()
        graph = result.graph
        self.assertEqual(IR0_SCHEMA_VERSION, "wafer_frontend.ir0/v1alpha12")
        self.assertEqual(
            S2_LITE_LM_HEAD_TRAIN_IR0_SCHEMA_VERSION,
            "wafer_frontend.s2_lite_lm_head_train_ir0/v1alpha1",
        )
        self.assertEqual(
            (
                PERSISTENT_STATE_IDENTITY_SCHEMA_VERSION,
                PERSISTENT_STATE_DECL_SCHEMA_VERSION,
                PERSISTENT_STATE_MANIFEST_SCHEMA_VERSION,
            ),
            (
                "wafer_frontend.persistent_state_identity/v1alpha2",
                "wafer_frontend.persistent_state_decl/v1alpha2",
                "wafer_frontend.persistent_state_manifest/v1alpha2",
            ),
        )
        self.assertEqual(
            (
                FUSION_PLAN_SCHEMA_VERSION,
                STANDALONE_COLLECTIVE_PLAN_SCHEMA_VERSION,
            ),
            (
                "wafer_frontend.fusion_plan/v1alpha10",
                "wafer_frontend.standalone_collective_plan/v1alpha9",
            ),
        )
        self.assertEqual(
            (
                len(result.base_graph.nodes),
                len(result.base_graph.values),
                len(result.base_graph.edges),
                len(result.base_graph.persistent_states),
                len(result.base_graph.state_accesses),
            ),
            (26, 43, 29, 15, 15),
        )
        self.assertEqual(
            (
                len(graph.nodes),
                len(graph.values),
                len(graph.edges),
                len(graph.fusion_candidates),
                len(graph.persistent_states),
                len(graph.state_accesses),
            ),
            (29, 47, 34, 0, 15, 16),
        )
        by_kind = {kind: tuple(node for node in graph.nodes if node.kind is kind) for kind in OpKind}
        ce_backward = by_kind[OpKind.CE_BACKWARD][0]
        update = by_kind[OpKind.OPTIMIZER_UPDATE][0]
        wgrad = next(node for node in by_kind[OpKind.GEMM] if node.phase is OpPhase.WGRAD)
        self.assertEqual(
            (ce_backward.phase, wgrad.phase, update.phase),
            (OpPhase.DGRAD, OpPhase.WGRAD, OpPhase.UPDATE),
        )
        self.assertEqual(
            canonical_compute_operand_roles(
                ce_backward.kind, ce_backward.workload, tiled=False
            ),
            (("logits", "labels", "loss_gradient"), ("logits_gradient",)),
        )
        self.assertEqual(
            canonical_compute_operand_roles(update.kind, update.workload, tiled=False),
            (("weight", "weight_gradient"), ("updated_weight",)),
        )
        self.assertEqual(
            canonical_compute_operand_roles(wgrad.kind, wgrad.workload, tiled=False),
            (("lhs", "rhs"), ("output",)),
        )
        trainable = tuple(
            state
            for state in graph.persistent_states
            if state.identity.kind is StateKind.TRAINABLE_PARAMETER
        )
        self.assertEqual(len(trainable), 1)
        self.assertEqual(trainable[0].identity.tensor_ref, "T0.lm_head.weight")
        self.assertIs(trainable[0].access, PersistentStateAccess.READ_WRITE)
        self.assertTrue(
            all(
                state.identity.kind is StateKind.PARAMETER
                and state.access is PersistentStateAccess.READ_ONLY
                for state in graph.persistent_states
                if state is not trainable[0]
            )
        )
        trainable_accesses = tuple(
            access
            for access in graph.state_accesses
            if access.state_ref == trainable[0].id
        )
        self.assertEqual(
            tuple((access.node_ref, access.mode) for access in trainable_accesses),
            (
                ("T0.lm_head", StateAccessMode.READ),
                ("T0.sgd_update", StateAccessMode.READ_WRITE),
            ),
        )
        self.assertEqual(
            tuple(edge.kind for edge in graph.edges).count(EdgeKind.CONTROL), 1
        )
        DenseIR0Validator.validate(graph, "graph")

    def test_strict_carrier_roundtrip_stable_id_and_old_versions(self) -> None:
        result = _build()
        self.assertEqual(
            loads_dataclass(
                S2LiteLmHeadTrainIR0,
                canonical_json(result),
                path="carrier",
            ),
            result,
        )
        self.assertEqual(_build().id, result.id)
        with self.subTest("carrier-old-version"):
            with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
                replace(
                    result,
                    schema_version="wafer_frontend.s2_lite_lm_head_train_ir0/v1alpha0",
                ).validate()
        with self.subTest("ir0-old-version"):
            with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
                replace(
                    result.graph,
                    schema_version="wafer_frontend.ir0/v1alpha10",
                ).validate()
        with self.subTest("stable-id"):
            with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
                replace(result, id="s2_lite_lm_head_train_ir0_forged").validate()

    def test_lineage_phase_control_and_state_tamper_fail_closed(self) -> None:
        graph = _build().graph
        ce_backward = next(node for node in graph.nodes if node.kind is OpKind.CE_BACKWARD)
        update = next(node for node in graph.nodes if node.kind is OpKind.OPTIMIZER_UPDATE)
        with self.subTest("ce-phase"):
            nodes = tuple(
                replace(node, phase=OpPhase.FWD) if node.id == ce_backward.id else node
                for node in graph.nodes
            )
            with self.assertRaises(SchemaError):
                DenseIR0Validator.validate(_rebuild_graph(graph, nodes=nodes))
        with self.subTest("missing-control"):
            edges = tuple(edge for edge in graph.edges if edge.kind is not EdgeKind.CONTROL)
            with self.assertRaisesRegex(SchemaError, "control"):
                DenseIR0Validator.validate(_rebuild_graph(graph, edges=edges))
        with self.subTest("optimizer-input-order"):
            nodes = tuple(
                replace(node, inputs=tuple(reversed(node.inputs)))
                if node.id == update.id
                else node
                for node in graph.nodes
            )
            with self.assertRaises(SchemaError):
                DenseIR0Validator.validate(_rebuild_graph(graph, nodes=nodes))
        with self.subTest("trainable-state-read-only"):
            states = tuple(
                replace(state, access=PersistentStateAccess.READ_ONLY)
                if state.identity.kind is StateKind.TRAINABLE_PARAMETER
                else state
                for state in graph.persistent_states
            )
            with self.assertRaisesRegex(SchemaError, "trainable parameter"):
                _rebuild_graph(graph, persistent_states=states).validate()

    def test_source_digest_geometry_and_oracle_mismatch_fail_closed(self) -> None:
        spec, contract, oracle = _sources()
        other_contract = S2LiteLmHeadTrainContract.create(
            **{
                **contract._semantic_key(),
                "source_spec_digest": "2" * 64,
            }
        )
        other_oracle = build_s2_lite_lm_head_train_oracle(other_contract)
        with self.subTest("source-digest"):
            with self.assertRaisesRegex(SchemaError, "source digest"):
                build_s2_lite_lm_head_train_ir0(spec, other_contract, other_oracle)
        with self.subTest("oracle-source"):
            with self.assertRaisesRegex(SchemaError, "source contract"):
                build_s2_lite_lm_head_train_ir0(spec, contract, other_oracle)


if __name__ == "__main__":
    unittest.main()
