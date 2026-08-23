from __future__ import annotations

import dataclasses
import importlib
import json
import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.schema.common import DType, MeshAxisName
from llm.frontend.wafer_frontend.schema.ir0 import (
    AttentionMode,
    AttentionWorkload,
    CollectiveKind,
    ElementwiseWorkload,
    EffectKind,
    FusionImpl,
    GemmPartition,
    GemmWorkload,
    IR0,
    IR0_SCHEMA_VERSION,
    InstanceProfileBinding,
    LogicalRole,
    NodeEffects,
    NodeMath,
    NodeProfileBinding,
    NormWorkload,
    NumericalPolicy,
    OpKind,
    P2PByteWorkload,
    ReduceOp,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    loads_dataclass,
)

from _fixtures import static_profile, valid_ir0


def rebuild(ir0: IR0, **updates: object) -> IR0:
    fields = {
        "producer_pass": ir0.producer_pass,
        "job": ir0.job,
        "instances": ir0.instances,
        "nodes": ir0.nodes,
        "values": ir0.values,
        "edges": ir0.edges,
        "fusion_candidates": ir0.fusion_candidates,
        "profile": ir0.profile,
        "train": ir0.train,
        "instance_profiles": ir0.instance_profiles,
        "node_profiles": ir0.node_profiles,
        "pd_plan_id": ir0.pd_plan_id,
        "persistent_states": ir0.persistent_states,
        "state_accesses": ir0.state_accesses,
    }
    fields.update(updates)
    return IR0.create(**fields)  # type: ignore[arg-type]


class IR0SchemaTest(unittest.TestCase):
    def test_canonical_round_trip_and_stable_id(self) -> None:
        ir0 = valid_ir0()
        self.assertEqual(IR0_SCHEMA_VERSION, "wafer_frontend.ir0/v1alpha12")
        decoded = loads_dataclass(IR0, canonical_json(ir0), path="ir0")
        self.assertEqual(decoded, ir0)
        self.assertEqual(decoded.id, valid_ir0().id)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decoded.id = "changed"  # type: ignore[misc]

    def test_n2a_workloads_are_public_schema_exports(self) -> None:
        public_schema = importlib.import_module(
            "llm.frontend.wafer_frontend.schema"
        )
        for name in (
            "GemmPartition",
            "GemmWorkload",
            "NormWorkload",
            "ElementwiseWorkload",
            "P2PByteWorkload",
            "AttentionWorkload",
            "CollectiveWorkload",
            "NodeMath",
            "state_access_tensor_view",
        ):
            self.assertIn(name, public_schema.__all__)
            self.assertIsNotNone(getattr(public_schema, name))

    def test_ir0_rejects_physical_placement_fields(self) -> None:
        raw = json.loads(canonical_json(valid_ir0()))
        raw["placement"] = {"die_id": 0}
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            from_data(IR0, raw, path="ir0")
        raw = json.loads(canonical_json(valid_ir0()))
        raw["nodes"][0]["die_region"] = [0]
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            from_data(IR0, raw, path="ir0")

    def test_duplicate_and_dangling_node_ids_are_rejected(self) -> None:
        ir0 = valid_ir0()
        duplicate = replace(ir0.nodes[1], id=ir0.nodes[0].id)
        with self.assertRaisesRegex(SchemaError, "duplicate id"):
            rebuild(ir0, nodes=(ir0.nodes[0], duplicate)).validate()
        dangling = replace(ir0.nodes[0], inputs=("missing_value",))
        with self.assertRaisesRegex(SchemaError, "dangling value"):
            rebuild(ir0, nodes=(dangling, ir0.nodes[1])).validate()

    def test_producer_consumer_and_edge_tables_must_agree(self) -> None:
        ir0 = valid_ir0()
        missing_consumer = replace(ir0.values[1], consumers=())
        with self.assertRaisesRegex(SchemaError, "missing this node"):
            rebuild(
                ir0,
                values=(ir0.values[0], missing_consumer, ir0.values[2]),
            ).validate()
        wrong_producer = replace(ir0.values[1], producer=ir0.nodes[1].id)
        with self.assertRaises(SchemaError):
            rebuild(
                ir0,
                values=(ir0.values[0], wrong_producer, ir0.values[2]),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "data edges"):
            rebuild(ir0, edges=()).validate()

    def test_fusion_boundary_and_uint64_are_strict(self) -> None:
        ir0 = valid_ir0()
        candidate = replace(ir0.fusion_candidates[0], boundary_outputs=("v_partial",))
        with self.assertRaisesRegex(SchemaError, "boundary outputs"):
            rebuild(ir0, fusion_candidates=(candidate,)).validate()
        bad_workload = replace(
            ir0.nodes[0].workload, logical_shape=(-1, 128, 256)
        )
        bad_node = replace(ir0.nodes[0], workload=bad_workload)
        with self.assertRaisesRegex(SchemaError, "unsigned 64-bit"):
            rebuild(ir0, nodes=(bad_node, ir0.nodes[1])).validate()

        selected = replace(ir0.fusion_candidates[0], impl=FusionImpl.NAIVE)
        with self.assertRaisesRegex(SchemaError, "cannot select"):
            rebuild(ir0, fusion_candidates=(selected,)).validate()

    def test_gemm_workload_distinguishes_logical_and_rank_shapes(self) -> None:
        row = GemmWorkload(
            logical_shape=(32, 128, 256),
            rank_shape=(32, 128, 128),
            partition=GemmPartition.ROW_PARALLEL,
            dtype=DType.FP16,
        )
        row.validate("row")
        GemmWorkload(
            logical_shape=(32, 256, 128),
            rank_shape=(32, 128, 128),
            partition=GemmPartition.COLUMN_PARALLEL,
            dtype=DType.FP16,
        ).validate("column")
        GemmWorkload(
            logical_shape=(32, 128, 256),
            rank_shape=(32, 128, 256),
            partition=GemmPartition.REPLICATED,
            dtype=DType.FP16,
        ).validate("replicated")
        for invalid in (
            replace(row, rank_shape=(16, 128, 128)),
            replace(row, rank_shape=(32, 64, 128)),
            replace(row, rank_shape=(32, 128, 129)),
            replace(row, logical_shape=(32, 128, 0)),
        ):
            with self.assertRaises(SchemaError):
                invalid.validate("gemm")

    def test_structured_norm_elementwise_and_p2p_workloads(self) -> None:
        NormWorkload(
            (32, 256), (32, 256), (16, 256), (16, 256), DType.FP16
        ).validate("norm")
        ElementwiseWorkload(
            ((32, 1024),),
            (32, 512),
            ((32, 512),),
            (32, 256),
            DType.FP16,
        ).validate("swiglu")
        ElementwiseWorkload(
            ((32, 256), (32, 256)),
            (32, 256),
            ((16, 256), (16, 256)),
            (16, 256),
            DType.FP16,
        ).validate("residual")
        P2PByteWorkload(4096, DType.FP16).validate("p2p")
        with self.assertRaisesRegex(SchemaError, "preserve"):
            NormWorkload(
                (32, 256), (32, 128), (16, 256), (16, 128), DType.FP16
            ).validate("norm")
        with self.assertRaisesRegex(SchemaError, "at least one"):
            ElementwiseWorkload((), (32, 256), (), (16, 256), DType.FP16).validate(
                "elementwise"
            )
        with self.assertRaisesRegex(SchemaError, "greater than zero"):
            P2PByteWorkload(0, DType.FP16).validate("p2p")

    def test_collective_logical_payload_and_axes_are_exact(self) -> None:
        rs = valid_ir0().nodes[1].workload
        rs.validate("rs")
        self.assertEqual(rs.logical_tensor_bytes, 8192)
        self.assertEqual(rs.rank_logical_payload_bytes, 4096)
        all_gather = replace(
            rs,
            collective=CollectiveKind.ALL_GATHER,
            reduce_op=None,
            reduction_mesh_axes=(),
            scatter_tensor_axis=None,
            gather_tensor_axis=0,
            rank_input_bytes=4096,
            rank_output_bytes=8192,
        )
        all_gather.validate("all_gather")
        for invalid in (
            replace(rs, participant_count=3),
            replace(rs, reduction_mesh_axes=()),
            replace(rs, scatter_tensor_axis=None),
            replace(rs, rank_logical_payload_bytes=8192),
            replace(all_gather, gather_tensor_axis=None),
            replace(all_gather, reduce_op=ReduceOp.SUM),
        ):
            with self.assertRaises(SchemaError):
                invalid.validate("collective")
        with self.assertRaises(UnsupportedFeatureError):
            replace(rs, collective=CollectiveKind.ALL_TO_ALL).validate("collective")

    def test_attention_workload_and_ir0_require_explicit_kv_state(self) -> None:
        base = valid_ir0()
        gemm = base.nodes[0]
        workload = AttentionWorkload(
            profile=base.profile,
            mode=AttentionMode.PREFILL,
            causal=True,
            query_tokens=32,
            context_sum=32,
            context_max=32,
            hidden_size=256,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            rank_num_heads=2,
            rank_num_kv_heads=1,
            query_key_pairs=528,
            logical_kv_read_bytes=0,
            logical_kv_write_bytes=16384,
            rank_kv_read_bytes=0,
            rank_kv_write_bytes=8192,
            dtype=DType.FP16,
        )
        workload.validate("attention")
        attention = replace(
            gemm,
            kind=OpKind.ATTENTION,
            workload=workload,
            effects=NodeEffects(
                EffectKind.STATEFUL, "kv_effect_layer_0", "kv_alias_layer_0"
            ),
        )
        input_value = replace(base.values[0], consumers=(attention.id,))
        output_value = replace(base.values[1], consumers=())
        graph = rebuild(
            base,
            nodes=(attention,),
            values=(input_value, output_value),
            edges=(),
            fusion_candidates=(),
        )
        graph.validate()
        pure_attention = replace(
            attention, effects=NodeEffects(EffectKind.PURE, None, None)
        )
        with self.assertRaisesRegex(SchemaError, "stateful KV-cache"):
            rebuild(graph, nodes=(pure_attention,)).validate()
        wrong_profile = replace(
            workload,
            profile=replace(static_profile(), context_sum=64, context_max=64),
            context_sum=64,
            context_max=64,
        )
        with self.assertRaisesRegex(SchemaError, "must belong to its instance"):
            rebuild(graph, nodes=(replace(attention, workload=wrong_profile),)).validate()
        with self.assertRaisesRegex(SchemaError, "same TP degree"):
            replace(workload, rank_num_kv_heads=2).validate("attention")

    def test_multi_instance_profile_bindings_are_exact(self) -> None:
        base = valid_ir0()
        source = base.instances[0]
        attention_workload = AttentionWorkload(
            profile=base.profile,
            mode=AttentionMode.PREFILL,
            causal=True,
            query_tokens=32,
            context_sum=32,
            context_max=32,
            hidden_size=256,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            rank_num_heads=2,
            rank_num_kv_heads=1,
            query_key_pairs=528,
            logical_kv_read_bytes=0,
            logical_kv_write_bytes=16384,
            rank_kv_read_bytes=0,
            rank_kv_write_bytes=8192,
            dtype=DType.FP16,
        )
        attention = replace(
            base.nodes[0],
            kind=OpKind.ATTENTION,
            workload=attention_workload,
            effects=NodeEffects(
                EffectKind.STATEFUL,
                "kv_effect_layer_0",
                "kv_alias_layer_0",
            ),
        )
        input_value = replace(base.values[0], consumers=(attention.id,))
        output_value = replace(base.values[1], consumers=())
        decode = replace(
            source,
            id="z_decode",
            role=LogicalRole.DECODE,
            meshes=(replace(source.meshes[0], id="z_decode.mesh.tp"),),
        )
        decode_profile = replace(
            base.profile,
            prefill_tokens=0,
            decode_tokens=1,
            context_sum=33,
            context_max=33,
        )
        bindings = tuple(
            sorted(
                (
                    InstanceProfileBinding(source.id, base.profile),
                    InstanceProfileBinding(decode.id, decode_profile),
                ),
                key=lambda item: item.instance_ref,
            )
        )
        graph = rebuild(
            base,
            instances=(source, decode),
            nodes=(attention,),
            values=(input_value, output_value),
            edges=(),
            fusion_candidates=(),
            instance_profiles=bindings,
            pd_plan_id="stage4_pd_plan_fixture",
            profile=bindings[0].profile,
        )
        graph.validate()
        decoded = loads_dataclass(IR0, canonical_json(graph), path="ir0")
        self.assertEqual(decoded, graph)

        fused_bindings = tuple(
            sorted(
                (
                    InstanceProfileBinding(source.id, base.profile),
                    InstanceProfileBinding(source.id, decode_profile),
                ),
                key=lambda item: (item.instance_ref, item.profile.stable_id()),
            )
        )
        fused = rebuild(
            graph,
            instances=(source,),
            instance_profiles=fused_bindings,
            node_profiles=(NodeProfileBinding(attention.id, base.profile),),
            profile=fused_bindings[0].profile,
        )
        fused.validate()
        self.assertEqual(len(fused.instance_profiles), 2)

        invalid_bindings = (
            bindings[:1],
            (bindings[0], bindings[0]),
            tuple(reversed(bindings)),
            (
                bindings[0],
                replace(bindings[1], instance_ref="missing_instance"),
            ),
        )
        for invalid in invalid_bindings:
            with self.subTest(invalid=invalid):
                with self.assertRaises(SchemaError):
                    rebuild(
                        graph,
                        instance_profiles=invalid,
                        profile=invalid[0].profile,
                    ).validate()

        wrong_source_profile = replace(bindings[0], profile=decode_profile)
        with self.assertRaisesRegex(SchemaError, "must belong to its instance"):
            rebuild(
                graph,
                instance_profiles=(wrong_source_profile, bindings[1]),
                profile=decode_profile,
            ).validate()

    def test_legacy_workload_and_node_math_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            from_data(
                GemmWorkload,
                {"m": 32, "n": 128, "k": 256, "dtype": "fp16"},
                path="gemm",
            )
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            from_data(
                NodeMath,
                {"reduction_axes": [1], "accumulation_dtype": "fp32"},
                path="math",
            )

    def test_missing_required_profile_and_strict_enum_fail_decode(self) -> None:
        raw = json.loads(canonical_json(valid_ir0()))
        del raw["profile"]
        with self.assertRaisesRegex(SchemaError, "profile"):
            from_data(IR0, raw, path="ir0")
        raw = json.loads(canonical_json(valid_ir0()))
        raw["nodes"][0]["kind"] = "GEMM"
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            from_data(IR0, raw, path="ir0")


if __name__ == "__main__":
    unittest.main()
