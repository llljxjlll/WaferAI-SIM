from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes import DenseIR0Validator as PublicValidator
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.common import DType, MeshAxisName, Sharding
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    DeviceMesh,
    EdgeKind,
    GraphEdge,
    IR0,
    MeshAxis,
    OpKind,
    P2PByteWorkload,
)
from llm.frontend.wafer_frontend.schema.serde import from_data

from _fixtures import valid_spec


def _graph(*, tp: int) -> IR0:
    raw = valid_spec()
    raw["parallel"]["instances"][0].update(tp=tp, sp=tp > 1)  # type: ignore[index]
    if tp == 4:
        raw["model"]["KVH"] = 4  # type: ignore[index]
    spec = from_data(ExperimentSpec, raw, path="spec")
    return logical_expand(build_ir0(spec)).entries[0].graph


def _rebuild(graph: IR0, **updates: object) -> IR0:
    fields: dict[str, object] = {
        "producer_pass": graph.producer_pass,
        "job": graph.job,
        "instances": graph.instances,
        "nodes": graph.nodes,
        "values": graph.values,
        "edges": graph.edges,
        "fusion_candidates": graph.fusion_candidates,
        "profile": graph.profile,
        "train": graph.train,
    }
    fields.update(updates)
    return IR0.create(**fields)  # type: ignore[arg-type]


def _replace_node(graph: IR0, node_id: str, replacement: object) -> IR0:
    return _rebuild(
        graph,
        nodes=tuple(replacement if node.id == node_id else node for node in graph.nodes),
    )


def _replace_value(graph: IR0, value_id: str, replacement: object) -> IR0:
    return _rebuild(
        graph,
        values=tuple(replacement if value.id == value_id else value for value in graph.values),
    )


class DenseIR0ValidatorTest(unittest.TestCase):
    def test_current_tp1_and_tp2_graphs_pass(self) -> None:
        self.assertIs(PublicValidator, DenseIR0Validator)
        for tp in (1, 2):
            with self.subTest(tp=tp):
                graph = _graph(tp=tp)
                DenseIR0Validator.validate(graph)

    def test_tp4_multilayer_and_multiprofile_graphs_pass(self) -> None:
        DenseIR0Validator.validate(_graph(tp=4))

        raw = valid_spec()
        raw["parallel"]["instances"][0].update(tp=2, sp=True)  # type: ignore[index]
        raw["model"]["L"] = 2  # type: ignore[index]
        layer_bundle = logical_expand(build_ir0(from_data(ExperimentSpec, raw, path="spec")))
        self.assertEqual(len(layer_bundle.entries[0].graph.nodes), 33)
        DenseIR0Validator.validate(layer_bundle.entries[0].graph)

        raw = valid_spec()
        raw["parallel"]["instances"][0].update(tp=2, sp=True, role="both")  # type: ignore[index]
        first = dict(raw["workload"]["infer"]["profile"])  # type: ignore[index]
        second = dict(first)
        second.update(
            prefill_tokens=64,
            context_sum=64,
            context_max=64,
            kv_pages=4,
        )
        raw["workload"]["infer"] = {  # type: ignore[index]
            "source": "shape_dist",
            "output": "logits",
            "shape_dist": {
                "profiles": (
                    {"key": second, "weight": 0.75},
                    {"key": first, "weight": 0.25},
                )
            },
        }
        profile_bundle = logical_expand(build_ir0(from_data(ExperimentSpec, raw, path="spec")))
        self.assertEqual(len(profile_bundle.entries), 2)
        for entry in profile_bundle.entries:
            DenseIR0Validator.validate(entry.graph)

    def test_gemm_value_shape_rank_shape_and_dtype_are_exact(self) -> None:
        graph = _graph(tp=2)
        qkv = next(node for node in graph.nodes if node.id.endswith(".qkv"))
        workload = qkv.workload

        bad_logical = replace(
            workload,
            logical_shape=(32, 510, 256),
            rank_shape=(32, 255, 256),
        )
        with self.assertRaisesRegex(SchemaError, "logical_shape"):
            DenseIR0Validator.validate(
                _replace_node(graph, qkv.id, replace(qkv, workload=bad_logical))
            )

        bad_rank = replace(workload, rank_shape=(32, 128, 256))
        with self.assertRaisesRegex(SchemaError, "rank_shape"):
            DenseIR0Validator.validate(
                _replace_node(graph, qkv.id, replace(qkv, workload=bad_rank))
            )

        bad_dtype = replace(workload, dtype=DType.FP32)
        with self.assertRaisesRegex(SchemaError, "dtypes"):
            DenseIR0Validator.validate(
                _replace_node(graph, qkv.id, replace(qkv, workload=bad_dtype))
            )

    def test_sharding_axis_divisibility_and_endpoint_mesh_are_exact(self) -> None:
        graph = _graph(tp=2)
        input_value = next(value for value in graph.values if value.id.endswith(".embedding_out"))

        unknown_axis = replace(
            input_value,
            sharding=replace(
                input_value.sharding,
                dim_map=(MeshAxisName.DP, None),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "absent from the referenced mesh"):
            DenseIR0Validator.validate(_replace_value(graph, input_value.id, unknown_axis))

        indivisible = replace(input_value, shape=(31, input_value.shape[1]))
        with self.assertRaisesRegex(SchemaError, "divide evenly"):
            DenseIR0Validator.validate(_replace_value(graph, input_value.id, indivisible))

        other_mesh = DeviceMesh(
            "other_tp_mesh", (MeshAxis(MeshAxisName.TP, 2),)
        )
        instance = replace(
            graph.instances[0], meshes=graph.instances[0].meshes + (other_mesh,)
        )
        wrong_mesh_value = replace(
            input_value,
            sharding=Sharding(
                other_mesh.id,
                input_value.sharding.dim_map,
                input_value.sharding.partial,
            ),
        )
        wrong_mesh_graph = _rebuild(
            graph,
            instances=(instance,),
            values=tuple(
                wrong_mesh_value if value.id == input_value.id else value
                for value in graph.values
            ),
        )
        with self.assertRaisesRegex(SchemaError, "producer/consumer"):
            DenseIR0Validator.validate(wrong_mesh_graph)

    def test_tensor_value_dtype_tamper_is_schema_error_not_key_error(self) -> None:
        graph = _graph(tp=2)
        input_value = next(value for value in graph.values if value.id.endswith(".embedding_out"))
        forged = replace(input_value, dtype="bf16")
        with self.assertRaisesRegex(SchemaError, "must be a DType"):
            DenseIR0Validator.validate(_replace_value(graph, input_value.id, forged))

    def test_collective_axis_layout_bytes_and_sharding_delta_are_exact(self) -> None:
        graph = _graph(tp=2)
        ag = next(
            node
            for node in graph.nodes
            if node.kind is OpKind.COLLECTIVE
            and node.workload.collective is CollectiveKind.ALL_GATHER
        )
        workload = ag.workload

        for changed, message in (
            (replace(workload, gather_tensor_axis=2), "out of range"),
            (replace(workload, input_layout="forged_layout"), "layouts"),
            (
                replace(
                    workload,
                    logical_tensor_bytes=32768,
                    rank_input_bytes=16384,
                    rank_output_bytes=32768,
                    rank_logical_payload_bytes=16384,
                    group_logical_payload_bytes=32768,
                ),
                "bytes",
            ),
            (
                replace(workload, rank_logical_payload_bytes=8193),
                "payload",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    DenseIR0Validator.validate(
                        _replace_node(graph, ag.id, replace(ag, workload=changed))
                    )

        output = next(value for value in graph.values if value.id == ag.outputs[0])
        wrong_delta = replace(
            output,
            sharding=replace(
                output.sharding, dim_map=(MeshAxisName.TP, None)
            ),
        )
        with self.assertRaisesRegex(SchemaError, "sharding delta"):
            DenseIR0Validator.validate(_replace_value(graph, output.id, wrong_delta))

        rs = next(
            node
            for node in graph.nodes
            if node.kind is OpKind.COLLECTIVE
            and node.workload.collective is CollectiveKind.REDUCE_SCATTER
        )
        wrong_rs_axis = replace(rs.workload, scatter_tensor_axis=1)
        with self.assertRaisesRegex(SchemaError, "TP-partial"):
            DenseIR0Validator.validate(
                _replace_node(graph, rs.id, replace(rs, workload=wrong_rs_axis))
            )

    def test_norm_elementwise_and_attention_cross_contracts_are_exact(self) -> None:
        graph = _graph(tp=2)
        norm = next(node for node in graph.nodes if node.id.endswith(".norm1"))
        bad_norm_workload = replace(
            norm.workload,
            rank_activation_shape=(8, 256),
            rank_output_shape=(8, 256),
        )
        with self.assertRaisesRegex(SchemaError, "RMSNorm workload/value"):
            DenseIR0Validator.validate(
                _replace_node(graph, norm.id, replace(norm, workload=bad_norm_workload))
            )
        with self.assertRaisesRegex(SchemaError, "rms_norm"):
            DenseIR0Validator.validate(
                _replace_node(graph, norm.id, replace(norm, impl_ref="layer_norm"))
            )

        swiglu = next(node for node in graph.nodes if node.id.endswith(".swiglu"))
        bad_swiglu_workload = replace(
            swiglu.workload,
            logical_output_shape=swiglu.workload.logical_input_shape,
            rank_output_shape=swiglu.workload.rank_input_shape,
        )
        with self.assertRaisesRegex(SchemaError, "halve"):
            DenseIR0Validator.validate(
                _replace_node(
                    graph, swiglu.id, replace(swiglu, workload=bad_swiglu_workload)
                )
            )

        attention = next(node for node in graph.nodes if node.kind is OpKind.ATTENTION)
        bad_pairs = replace(
            attention.workload,
            query_key_pairs=attention.workload.query_key_pairs + 1,
        )
        with self.assertRaisesRegex(SchemaError, "must equal"):
            DenseIR0Validator.validate(
                _replace_node(graph, attention.id, replace(attention, workload=bad_pairs))
            )
        bad_heads = replace(
            attention.workload,
            rank_num_heads=attention.workload.num_heads,
            rank_num_kv_heads=attention.workload.num_kv_heads,
            rank_kv_read_bytes=attention.workload.logical_kv_read_bytes,
            rank_kv_write_bytes=attention.workload.logical_kv_write_bytes,
        )
        with self.assertRaisesRegex(SchemaError, "derive from TP"):
            DenseIR0Validator.validate(
                _replace_node(graph, attention.id, replace(attention, workload=bad_heads))
            )

    def test_dense_mvp_rejects_p2p_nodes(self) -> None:
        graph = _graph(tp=1)
        norm = graph.nodes[0]
        p2p = replace(
            norm,
            kind=OpKind.P2P,
            workload=P2PByteWorkload(bytes=1024, dtype=DType.FP16),
            impl_ref="p2p",
        )
        with self.assertRaisesRegex(UnsupportedFeatureError, "does not support P2P"):
            DenseIR0Validator.validate(_replace_node(graph, norm.id, p2p))

    def test_partial_cannot_have_an_external_postprocess_consumer(self) -> None:
        graph = _graph(tp=2)
        partial = next(value for value in graph.values if value.id.endswith(".o_partial"))
        residual = next(node for node in graph.nodes if node.id.endswith(".residual1"))
        rs_output = next(value for value in graph.values if value.id.endswith(".rs1_out"))
        changed_partial = replace(
            partial, consumers=(partial.consumers[0], residual.id)
        )
        changed_rs_output = replace(rs_output, consumers=())
        changed_residual = replace(
            residual,
            inputs=tuple(
                partial.id if value_id == rs_output.id else value_id
                for value_id in residual.inputs
            ),
        )
        changed_candidate = replace(
            graph.fusion_candidates[0],
            boundary_outputs=(partial.id, rs_output.id),
        )
        broken = _rebuild(
            graph,
            nodes=tuple(
                changed_residual if node.id == residual.id else node
                for node in graph.nodes
            ),
            values=tuple(
                changed_partial
                if value.id == partial.id
                else changed_rs_output
                if value.id == rs_output.id
                else value
                for value in graph.values
            ),
            edges=tuple(
                GraphEdge(
                    edge.id,
                    edge.kind,
                    partial.producer,
                    edge.destination_node,
                    partial.id,
                )
                if edge.value_id == rs_output.id
                else edge
                for edge in graph.edges
            ),
            fusion_candidates=(changed_candidate, *graph.fusion_candidates[1:]),
        )
        with self.assertRaisesRegex(SchemaError, "exactly one reduction consumer"):
            DenseIR0Validator.validate(broken)

    def test_control_dependencies_are_unique_and_acyclic(self) -> None:
        graph = _graph(tp=1)
        first = graph.nodes[0]
        last = graph.nodes[-1]
        duplicate_edges = graph.edges + (
            GraphEdge("control_0", EdgeKind.CONTROL, first.id, last.id, None),
            GraphEdge("control_1", EdgeKind.CONTROL, first.id, last.id, None),
        )
        with self.assertRaisesRegex(SchemaError, "duplicate control"):
            DenseIR0Validator.validate(_rebuild(graph, edges=duplicate_edges))

        cycle_edges = graph.edges + (
            GraphEdge("control_cycle", EdgeKind.CONTROL, last.id, first.id, None),
        )
        with self.assertRaisesRegex(SchemaError, "cycle"):
            DenseIR0Validator.validate(_rebuild(graph, edges=cycle_edges))


if __name__ == "__main__":
    unittest.main()
