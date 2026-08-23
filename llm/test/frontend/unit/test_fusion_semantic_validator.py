from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.passes.validate_fusion import (
    FusionSemanticValidator,
    validate,
)
from llm.frontend.wafer_frontend.schema.common import MeshAxisName
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import (
    EdgeKind,
    DeviceMesh,
    EffectKind,
    FusionImpl,
    FusionOrigin,
    FusionPattern,
    GraphEdge,
    IR0,
    MeshAxis,
    NodeEffects,
    NumericalPolicy,
    OpPhase,
)
from llm.frontend.wafer_frontend.schema.serde import from_data

from _fixtures import valid_spec


def _graph(*, tp: int = 2, layers: int = 1) -> IR0:
    raw = valid_spec()
    raw["parallel"]["instances"][0].update(tp=tp, sp=tp > 1)  # type: ignore[index]
    raw["model"]["L"] = layers  # type: ignore[index]
    return logical_expand(
        build_ir0(from_data(ExperimentSpec, raw, path="spec"))
    ).entries[0].graph

def _gemm_rs_candidates(graph: IR0):
    return tuple(
        candidate
        for candidate in graph.fusion_candidates
        if candidate.semantic_contract.pattern is FusionPattern.GEMM_RS
    )



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
        "persistent_states": graph.persistent_states,
        "state_accesses": graph.state_accesses,
    }
    fields.update(updates)
    return IR0.create(**fields)  # type: ignore[arg-type]


def _replace_candidate(graph: IR0, original: object, replacement: object) -> IR0:
    return _rebuild(
        graph,
        fusion_candidates=tuple(
            replacement if candidate is original else candidate
            for current, candidate in enumerate(graph.fusion_candidates)
        ),
    )


def _replace_node(graph: IR0, node_id: str, replacement: object) -> IR0:
    return _rebuild(
        graph,
        nodes=tuple(
            replacement if node.id == node_id else node for node in graph.nodes
        ),
    )


def _replace_value(graph: IR0, value_id: str, replacement: object) -> IR0:
    return _rebuild(
        graph,
        values=tuple(
            replacement if value.id == value_id else value
            for value in graph.values
        ),
    )


def _boundaries(graph: IR0, members: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    member_set = set(members)
    inputs = tuple(
        value.id
        for value in graph.values
        if member_set.intersection(value.consumers)
        and (value.producer is None or value.producer not in member_set)
    )
    outputs = tuple(
        value.id
        for value in graph.values
        if value.producer in member_set
        and (
            not value.consumers
            or any(consumer not in member_set for consumer in value.consumers)
        )
    )
    return inputs, outputs


def _edges_from_values(graph: IR0, values: tuple[object, ...]) -> tuple[GraphEdge, ...]:
    controls = tuple(edge for edge in graph.edges if edge.kind is EdgeKind.CONTROL)
    data = tuple(
        GraphEdge(
            id=f"mutation.edge.{index}",
            kind=EdgeKind.DATA,
            source_node=value.producer,
            destination_node=consumer,
            value_id=value.id,
        )
        for index, (value, consumer) in enumerate(
            (value, consumer)
            for value in values
            if value.producer is not None
            for consumer in value.consumers
        )
    )
    return data + controls


class FusionSemanticValidatorTest(unittest.TestCase):
    def test_current_tp1_tp2_and_l2_graphs_pass(self) -> None:
        for tp, layers, candidates in ((1, 1, 0), (2, 1, 4), (2, 2, 8)):
            with self.subTest(tp=tp, layers=layers):
                graph = _graph(tp=tp, layers=layers)
                self.assertEqual(len(graph.fusion_candidates), candidates)
                FusionSemanticValidator.validate(graph)
                validate(graph)

    def test_contract_fields_and_numerical_policy_are_exact(self) -> None:
        graph = _graph()
        candidate = _gemm_rs_candidates(graph)[0]
        contract = candidate.semantic_contract
        cases = (
            (replace(contract, tile_domain=("N", "M")), "tile_domain"),
            (replace(contract, reduction_axes=(1,)), "reduction_axes"),
            (replace(contract, input_layouts=tuple(reversed(contract.input_layouts))), "input_layouts"),
            (replace(contract, output_layout="forged_layout"), "output_layout"),
            (
                replace(contract, numerical_policy=NumericalPolicy.TOLERANCE),
                "numerical policy",
            ),
        )
        for replacement, message in cases:
            with self.subTest(message=message):
                changed = replace(candidate, semantic_contract=replacement)
                with self.assertRaisesRegex(SchemaError, message):
                    FusionSemanticValidator.validate(
                        _replace_candidate(graph, candidate, changed)
                    )

        rs = next(node for node in graph.nodes if node.id == candidate.members[1])
        changed_rs = replace(
            rs,
            math=replace(rs.math, numerical_policy=NumericalPolicy.TOLERANCE),
        )
        with self.assertRaisesRegex(SchemaError, "numerical policy"):
            FusionSemanticValidator.validate(_replace_node(graph, rs.id, changed_rs))

    def test_member_order_directness_boundary_order_origin_and_impl_are_exact(self) -> None:
        graph = _graph()
        candidate = _gemm_rs_candidates(graph)[0]

        reversed_members = replace(candidate, members=tuple(reversed(candidate.members)))
        with self.assertRaisesRegex(SchemaError, "sole partial output|ordered"):
            FusionSemanticValidator.validate(
                _replace_candidate(graph, candidate, reversed_members)
            )

        swapped_boundary = replace(
            candidate,
            boundary_inputs=tuple(reversed(candidate.boundary_inputs)),
        )
        with self.assertRaisesRegex(SchemaError, "input order"):
            FusionSemanticValidator.validate(
                _replace_candidate(graph, candidate, swapped_boundary)
            )

        wrong_output = replace(
            candidate,
            boundary_outputs=_gemm_rs_candidates(graph)[1].boundary_outputs,
        )
        with self.assertRaisesRegex(SchemaError, "boundary_outputs"):
            FusionSemanticValidator.validate(
                _replace_candidate(graph, candidate, wrong_output)
            )

        down_rs = _gemm_rs_candidates(graph)[1]
        disconnected_members = (candidate.members[0], down_rs.members[1])
        boundary_inputs, boundary_outputs = _boundaries(graph, disconnected_members)
        disconnected = replace(
            candidate,
            members=disconnected_members,
            boundary_inputs=boundary_inputs,
            boundary_outputs=boundary_outputs,
        )
        with self.assertRaisesRegex(SchemaError, "sole partial output"):
            FusionSemanticValidator.validate(
                _replace_candidate(graph, candidate, disconnected)
            )

        declared = replace(candidate, origin=FusionOrigin.DECLARED)
        with self.assertRaisesRegex(SchemaError, "DISCOVERED"):
            FusionSemanticValidator.validate(_replace_candidate(graph, candidate, declared))

        selected = replace(candidate, impl=FusionImpl.NAIVE)
        with self.assertRaisesRegex(SchemaError, "implementation"):
            FusionSemanticValidator.validate(_replace_candidate(graph, candidate, selected))

    def test_member_stage_phase_and_mesh_scope_are_closed(self) -> None:
        graph = _graph()
        candidate = _gemm_rs_candidates(graph)[0]
        rs = next(node for node in graph.nodes if node.id == candidate.members[1])
        for replacement, message in (
            (replace(rs, stage=1), r"share instance, stage, phase.*mesh"),
            (replace(rs, phase=OpPhase.DGRAD), r"share instance, stage, phase.*mesh"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    FusionSemanticValidator.validate(
                        _replace_node(graph, rs.id, replacement)
                    )

        other_mesh = DeviceMesh(
            "other_tp_mesh", (MeshAxis(MeshAxisName.TP, 2),)
        )
        instance = replace(
            graph.instances[0], meshes=graph.instances[0].meshes + (other_mesh,)
        )
        cross_mesh = _rebuild(
            graph,
            instances=(instance,),
            nodes=tuple(
                replace(node, mesh_ref=other_mesh.id) if node.id == rs.id else node
                for node in graph.nodes
            ),
        )
        with self.assertRaisesRegex(SchemaError, "mesh"):
            FusionSemanticValidator.validate(cross_mesh)

    def test_partial_nonlinear_external_consumer_effect_and_alias_are_rejected(self) -> None:
        graph = _graph()
        candidate = _gemm_rs_candidates(graph)[0]
        gemm = next(node for node in graph.nodes if node.id == candidate.members[0])
        rs = next(node for node in graph.nodes if node.id == candidate.members[1])
        partial = next(value for value in graph.values if value.id == gemm.outputs[0])
        swiglu = next(node for node in graph.nodes if node.id.endswith(".swiglu"))
        swiglu_input = next(value for value in graph.values if value.id == swiglu.inputs[0])
        swiglu_output = next(value for value in graph.values if value.id == swiglu.outputs[0])

        changed_values = tuple(
            replace(value, consumers=(swiglu.id,))
            if value.id == partial.id
            else replace(value, consumers=())
            if value.id == swiglu_input.id
            else replace(value, consumers=value.consumers + (rs.id,))
            if value.id == swiglu_output.id
            else value
            for value in graph.values
        )
        changed_nodes = tuple(
            replace(node, inputs=(partial.id,))
            if node.id == swiglu.id
            else replace(node, inputs=(swiglu_output.id,))
            if node.id == rs.id
            else node
            for node in graph.nodes
        )
        nonlinear = _rebuild(
            graph,
            nodes=changed_nodes,
            values=changed_values,
            edges=_edges_from_values(graph, changed_values),
            fusion_candidates=(),
        )
        with self.assertRaisesRegex(SchemaError, "directly into one SUM ReduceScatter"):
            FusionSemanticValidator.validate(nonlinear)

        residual = next(node for node in graph.nodes if node.id.endswith(".residual1"))
        external = replace(partial, consumers=partial.consumers + (residual.id,))
        external_residual = replace(
            residual, inputs=residual.inputs + (partial.id,)
        )
        external_values = tuple(
            external if value.id == partial.id else value for value in graph.values
        )
        external_graph = _rebuild(
            graph,
            nodes=tuple(
                external_residual if node.id == residual.id else node
                for node in graph.nodes
            ),
            values=external_values,
            edges=_edges_from_values(graph, external_values),
            fusion_candidates=(),
        )
        with self.assertRaisesRegex(SchemaError, "exactly one reduction consumer"):
            FusionSemanticValidator.validate(external_graph)

        effectful = replace(
            gemm,
            effects=NodeEffects(EffectKind.INPLACE, "effect", "alias"),
        )
        with self.assertRaisesRegex(SchemaError, "pure"):
            FusionSemanticValidator.validate(_replace_node(graph, gemm.id, effectful))

        boundary = next(
            value for value in graph.values if value.id == candidate.boundary_inputs[0]
        )
        aliased = replace(boundary, alias_set="unsafe_alias")
        with self.assertRaisesRegex(SchemaError, "alias-safe"):
            FusionSemanticValidator.validate(_replace_value(graph, boundary.id, aliased))

    def test_generic_convexity_is_required(self) -> None:
        graph = _graph()
        candidate = _gemm_rs_candidates(graph)[0]
        gemm_id, rs_id = candidate.members
        norm = next(node for node in graph.nodes if node.id.endswith(".swiglu"))
        source = next(value for value in graph.values if value.id == norm.inputs[0])
        output = next(value for value in graph.values if value.id == norm.outputs[0])
        bridge_id = "convexity_bridge"
        bridge_input = replace(
            source,
            id="convexity_bridge_input",
            producer=None,
            consumers=(bridge_id,),
        )
        bridge_output = replace(
            output,
            id="convexity_bridge_output",
            producer=bridge_id,
            consumers=(),
        )
        bridge = replace(
            norm,
            id=bridge_id,
            inputs=(bridge_input.id,),
            outputs=(bridge_output.id,),
        )
        controls = (
            GraphEdge("convexity_control_0", EdgeKind.CONTROL, gemm_id, bridge_id, None),
            GraphEdge("convexity_control_1", EdgeKind.CONTROL, bridge_id, rs_id, None),
        )
        convexity = _rebuild(
            graph,
            nodes=graph.nodes + (bridge,),
            values=graph.values + (bridge_input, bridge_output),
            edges=graph.edges + controls,
        )
        with self.assertRaisesRegex(SchemaError, "convex"):
            FusionSemanticValidator.validate(convexity)

    def test_candidates_have_complete_nonoverlapping_one_to_one_coverage(self) -> None:
        graph = _graph()
        missing = _rebuild(graph, fusion_candidates=graph.fusion_candidates[1:])
        with self.assertRaisesRegex(SchemaError, "every direct"):
            FusionSemanticValidator.validate(missing)

        duplicate = replace(
            graph.fusion_candidates[0],
            id=f"{graph.fusion_candidates[0].id}.duplicate",
        )
        duplicated = _rebuild(
            graph,
            fusion_candidates=graph.fusion_candidates + (duplicate,),
        )
        with self.assertRaisesRegex(SchemaError, "share member"):
            FusionSemanticValidator.validate(duplicated)


if __name__ == "__main__":
    unittest.main()
