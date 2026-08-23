from __future__ import annotations

import json
import importlib
import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.schema.common import DType, MeshAxisName, ProfileKey, Sharding, TensorValue
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    CollectiveRole,
    CollectiveWorkload,
    CrossEntropyForwardWorkload,
    CrossEntropyReduction,
    EmbeddingTablePlacement,
    EmbeddingWorkload,
    ElementwiseWorkload,
    GemmPartition,
    GemmWorkload,
    GreedySampleWorkload,
    IR0_SCHEMA_VERSION,
    LogicalNode,
    NodeEffects,
    NodeMath,
    NumericalPolicy,
    NormWorkload,
    EffectKind,
    OpKind,
    OpPhase,
    PackedQkvLayout,
    P2PByteWorkload,
    ReduceOp,
    RopeQkWorkload,
    SampleRowSelection,
    SamplingMode,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1_SCHEMA_VERSION, PhysicalNode
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateLifetime,
    StateKind,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass


def _profile() -> ProfileKey:
    return ProfileKey(
        prefill_tokens=4,
        decode_tokens=0,
        num_seqs=1,
        context_sum=4,
        context_max=4,
        kv_pages=1,
        expert_load=None,
    )


def _embedding() -> EmbeddingWorkload:
    return EmbeddingWorkload(
        profile=_profile(),
        logical_index_shape=(4,),
        rank_index_shape=(2,),
        logical_table_shape=(32, 8),
        rank_table_shape=(32, 8),
        logical_output_shape=(4, 8),
        rank_output_shape=(2, 8),
        table_placement=EmbeddingTablePlacement.REPLICATED,
        index_dtype=DType.INT32,
        table_dtype=DType.FP16,
        output_dtype=DType.FP16,
    )


def _rope() -> RopeQkWorkload:
    return RopeQkWorkload(
        profile=_profile(),
        logical_input_shape=(4, 32),
        rank_input_shape=(4, 16),
        logical_output_shape=(4, 32),
        rank_output_shape=(4, 16),
        packed_layout=PackedQkvLayout.Q_K_V,
        num_heads=4,
        num_kv_heads=2,
        rank_num_heads=2,
        rank_num_kv_heads=1,
        head_dim=4,
        rotary_dim=4,
        rope_theta=10000.0,
        max_position_embeddings=128,
        dtype=DType.FP16,
    )


def _sampling() -> GreedySampleWorkload:
    return GreedySampleWorkload(
        profile=_profile(),
        mode=SamplingMode.GREEDY,
        row_selection=SampleRowSelection.LAST_PER_SEQUENCE,
        tp_degree=1,
        logical_logits_shape=(4, 32),
        rank_logits_shape=(4, 32),
        logical_output_shape=(1,),
        rank_output_shape=(1,),
        sample_count=1,
        comparisons=31,
        logits_dtype=DType.FP16,
        output_dtype=DType.INT32,
    )


def _ce() -> CrossEntropyForwardWorkload:
    return CrossEntropyForwardWorkload(
        profile=_profile(),
        reduction=CrossEntropyReduction.NONE,
        logical_logits_shape=(4, 32),
        rank_logits_shape=(2, 32),
        logical_label_shape=(4,),
        rank_label_shape=(2,),
        logical_loss_shape=(4,),
        rank_loss_shape=(2,),
        logits_dtype=DType.FP16,
        label_dtype=DType.INT32,
        loss_dtype=DType.FP32,
    )


class Stage2ForwardSchemaTest(unittest.TestCase):
    def test_versions_and_strict_round_trips(self) -> None:
        self.assertEqual(IR0_SCHEMA_VERSION, "wafer_frontend.ir0/v1alpha12")
        self.assertEqual(IR1_SCHEMA_VERSION, "wafer_frontend.ir1/v1alpha14")
        public_schema = importlib.import_module("llm.frontend.wafer_frontend.schema")
        for name in (
            "DType", "EmbeddingTablePlacement", "PackedQkvLayout", "SamplingMode",
            "SampleRowSelection", "CrossEntropyReduction", "EmbeddingWorkload",
            "RopeQkWorkload", "GreedySampleWorkload", "CrossEntropyForwardWorkload",
        ):
            self.assertIn(name, public_schema.__all__)
            self.assertIsNotNone(getattr(public_schema, name))
        for workload in (_embedding(), _rope(), _sampling(), _ce()):
            with self.subTest(type=type(workload).__name__):
                workload.validate("workload")
                self.assertEqual(
                    loads_dataclass(type(workload), canonical_json(workload), path="workload"),
                    workload,
                )
                raw = json.loads(canonical_json(workload))
                raw["unexpected"] = 1
                with self.assertRaisesRegex(SchemaError, "unknown field"):
                    loads_dataclass(type(workload), json.dumps(raw), path="workload")
                del raw["unexpected"]
                del raw[next(iter(raw))]
                with self.assertRaisesRegex(SchemaError, "missing required field"):
                    loads_dataclass(type(workload), json.dumps(raw), path="workload")

    def test_embedding_replicated_table_sp_rows_and_mixed_dtypes(self) -> None:
        workload = _embedding()
        workload.validate("embedding")
        for update, message in (
            ({"rank_index_shape": (3,), "rank_output_shape": (3, 8)}, "exactly divide"),
            ({"rank_table_shape": (16, 8)}, "replicated embedding"),
            ({"index_dtype": DType.FP16}, "indices require INT32"),
            ({"table_dtype": DType.INT32}, "table requires FP16"),
            ({"output_dtype": DType.INT32}, "output requires FP16"),
        ):
            with self.subTest(update=update), self.assertRaisesRegex(SchemaError, message):
                replace(workload, **update).validate("embedding")

    def test_rope_packed_head_geometry_and_position_contract(self) -> None:
        workload = _rope()
        workload.validate("rope")
        for update, message in (
            ({"packed_layout": "q_k_v"}, "packed Q_K_V"),
            ({"num_kv_heads": 3}, "must divide"),
            ({"rank_num_heads": 1}, "same TP degree"),
            ({"rotary_dim": 2}, "rotary_dim == head_dim"),
            ({"rotary_dim": 3}, "rotary_dim == head_dim"),
            ({"rope_theta": 0.0}, "finite positive"),
            ({"rope_theta": 10000}, "finite positive"),
            ({"max_position_embeddings": 3}, "cover profile.context_max"),
            ({"rank_input_shape": (4, 20), "rank_output_shape": (4, 20)}, "must equal"),
        ):
            with self.subTest(update=update), self.assertRaisesRegex(SchemaError, message):
                replace(workload, **update).validate("rope")

    def test_greedy_is_tp1_terminal_and_counts_comparisons(self) -> None:
        workload = _sampling()
        workload.validate("sampling")
        self.assertEqual(workload.sample_count, workload.profile.num_seqs)
        self.assertEqual(workload.comparisons, 31)
        with self.assertRaisesRegex(UnsupportedFeatureError, "TP-sharded"):
            replace(workload, tp_degree=2).validate("sampling")
        with self.assertRaisesRegex(SchemaError, "must equal 31"):
            replace(workload, comparisons=32).validate("sampling")
        with self.assertRaisesRegex(SchemaError, "sample ids require INT32"):
            replace(workload, output_dtype=DType.FP32).validate("sampling")

    def test_cross_entropy_none_and_mixed_dtypes(self) -> None:
        workload = _ce()
        workload.validate("ce")
        for reduction in (CrossEntropyReduction.SUM, CrossEntropyReduction.MEAN):
            with self.subTest(reduction=reduction), self.assertRaisesRegex(
                UnsupportedFeatureError, "not implemented"
            ):
                replace(workload, reduction=reduction).validate("ce")
        for update, message in (
            ({"rank_logits_shape": (2, 16)}, "vocabulary axis must be replicated"),
            ({"label_dtype": DType.FP16}, "labels require INT32"),
            ({"loss_dtype": DType.FP16}, "loss requires FP32"),
        ):
            with self.subTest(update=update), self.assertRaisesRegex(SchemaError, message):
                replace(workload, **update).validate("ce")

    def test_direct_non_enum_contract_fields_fail_as_schema_errors(self) -> None:
        invalid = (
            replace(_embedding(), table_placement="replicated"),
            replace(_embedding(), index_dtype="int32"),
            replace(_rope(), packed_layout="q_k_v"),
            replace(_sampling(), mode="greedy"),
            replace(_sampling(), row_selection="last_per_sequence"),
            replace(_ce(), reduction="none"),
        )
        for workload in invalid:
            with self.subTest(type=type(workload).__name__), self.assertRaises(SchemaError):
                workload.validate("workload")

    def test_new_kind_exact_maps_and_union_serde(self) -> None:
        for kind, workload in (
            (OpKind.EMBEDDING, _embedding()),
            (OpKind.ROPE, _rope()),
            (OpKind.SAMPLING, _sampling()),
            (OpKind.CE_FORWARD, _ce()),
        ):
            logical = LogicalNode(
                id=f"logical_{kind.value}", instance_id="instance", kind=kind,
                phase=OpPhase.FWD, stage=0, mesh_ref="mesh", inputs=("input",),
                outputs=("output",), workload=workload,
                math=NodeMath(DType.FP32, NumericalPolicy.TOLERANCE),
                effects=NodeEffects(EffectKind.PURE, None, None), impl_ref="naive",
            )
            physical = PhysicalNode(
                id=f"physical_{kind.value}", origin_node_id=logical.id,
                instance_id="instance", kind=kind, phase=OpPhase.FWD, stage=0,
                mesh_ref="mesh", execution_group_ref="group", inputs=("input",),
                outputs=("output",), workload=workload, math=logical.math,
                effects=logical.effects, impl_ref="naive",
            )
            with self.subTest(kind=kind):
                logical.validate("logical")
                physical.validate("physical")
                self.assertEqual(
                    loads_dataclass(LogicalNode, canonical_json(logical), path="logical"),
                    logical,
                )
                with self.assertRaisesRegex(SchemaError, "requires one of"):
                    replace(logical, workload=_embedding() if kind is not OpKind.EMBEDDING else _rope()).validate("logical")
                with self.assertRaisesRegex(SchemaError, "must be an OpKind"):
                    replace(logical, kind=kind.value).validate("logical")
                with self.assertRaisesRegex(SchemaError, "must be an OpKind"):
                    replace(physical, kind=kind.value).validate("physical")

    def test_int32_tensor_allowed_but_legacy_compute_and_state_reject(self) -> None:
        value = TensorValue(
            id="tokens", shape=(4,), dtype=DType.INT32, logical_layout="M",
            sharding=Sharding("mesh", (MeshAxisName.SP,), ()), producer=None,
            consumers=("embedding",), alias_set=None,
        )
        value.validate("value")
        with self.assertRaisesRegex(SchemaError, "must be a DType"):
            replace(value, dtype="int32").validate("value")
        for legacy in (
            GemmWorkload((4, 8, 16), (4, 8, 16), GemmPartition.REPLICATED, DType.INT32),
            NormWorkload((4, 8), (4, 8), (4, 8), (4, 8), DType.INT32),
            ElementwiseWorkload(((4, 8),), (4, 8), ((4, 8),), (4, 8), DType.INT32),
            P2PByteWorkload(16, DType.INT32),
            CollectiveWorkload(
                collective=CollectiveKind.REDUCE_SCATTER,
                reduce_op=ReduceOp.SUM,
                mesh_axes=(MeshAxisName.TP,),
                participant_count=2,
                reduction_mesh_axes=(MeshAxisName.TP,),
                scatter_tensor_axis=1,
                gather_tensor_axis=None,
                logical_tensor_bytes=32,
                rank_input_bytes=32,
                rank_output_bytes=16,
                rank_logical_payload_bytes=16,
                group_logical_payload_bytes=32,
                dtype=DType.INT32,
                role=CollectiveRole.ACTIVATION,
                input_layout="MH",
                output_layout="MH_shard_tp",
            ),
            NodeMath(DType.INT32, NumericalPolicy.TOLERANCE),
        ):
            with self.subTest(type=type(legacy).__name__), self.assertRaisesRegex(SchemaError, "FP16 or FP32"):
                legacy.validate("legacy")

        identity = PersistentStateIdentity.create(
            kind=StateKind.PARAMETER, instance_ref="instance", mesh_ref="mesh",
            request_ref=None, layer_index=None, tensor_ref="weight",
            shard_index=0, generation=0,
        )
        with self.assertRaisesRegex(SchemaError, "unsupported persistent-state dtype"):
            PersistentStateDecl.create(
                identity=identity, shape=(4,), dtype=DType.INT32, layout="H",
                lifetime=PersistentStateLifetime.PERSISTENT,
                access=PersistentStateAccess.READ_ONLY,
            )


if __name__ == "__main__":
    unittest.main()
