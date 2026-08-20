from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    DenseLogicalBundleValidator as PublicValidator,
)
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.passes.validate_logical_bundle import (
    DenseLogicalBundleValidator,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import IR0
from llm.frontend.wafer_frontend.schema.logical import (
    ExpandedIR0Bundle,
    ExpandedProfileIR0,
    IR0Template,
    ProfileEntry,
)
from llm.frontend.wafer_frontend.schema.serde import from_data

from _fixtures import valid_spec


def _template_bundle(
    *, tp: int = 2, layers: int = 1, multi_profile: bool = False
) -> tuple[IR0Template, ExpandedIR0Bundle]:
    raw = valid_spec()
    raw["parallel"]["instances"][0].update(  # type: ignore[index]
        tp=tp,
        sp=tp > 1,
        role="prefill",
    )
    raw["model"]["L"] = layers  # type: ignore[index]
    if multi_profile:
        prefill = dict(raw["workload"]["infer"]["profile"])  # type: ignore[index]
        second_prefill = {
            "prefill_tokens": 64,
            "decode_tokens": 0,
            "num_seqs": 1,
            "context_sum": 64,
            "context_max": 64,
            "kv_pages": 4,
            "expert_load": None,
        }
        raw["workload"]["infer"] = {  # type: ignore[index]
            "source": "shape_dist",
            "output": "logits",
            "shape_dist": {
                "profiles": (
                    {"key": prefill, "weight": 0.4},
                    {"key": second_prefill, "weight": 0.6},
                )
            },
        }
    template = build_ir0(from_data(ExperimentSpec, raw, path="spec"))
    return template, logical_expand(template)


def _rebuild_template(template: IR0Template, **updates: object) -> IR0Template:
    fields: dict[str, object] = {
        "job": template.job,
        "model": template.model,
        "instance": template.instance,
        "sequence_parallel": template.sequence_parallel,
        "layer": template.layer,
        "infer_output": template.infer_output,
        "profiles": template.profiles,
    }
    fields.update(updates)
    return IR0Template.create(**fields)  # type: ignore[arg-type]


def _rebuild_graph(graph: IR0, **updates: object) -> IR0:
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
    }
    fields.update(updates)
    return IR0.create(**fields)  # type: ignore[arg-type]


def _rebuild_bundle(
    template: IR0Template,
    graphs: tuple[IR0, ...],
) -> ExpandedIR0Bundle:
    entries = tuple(
        ExpandedProfileIR0.create(
            source_template_id=template.id,
            weight=profile.weight,
            graph=graph,
        )
        for profile, graph in zip(template.profiles, graphs)
    )
    return ExpandedIR0Bundle.create(source_template=template, entries=entries)


class DenseLogicalBundleValidatorTest(unittest.TestCase):
    def test_public_export(self) -> None:
        self.assertIs(PublicValidator, DenseLogicalBundleValidator)

    def test_tp1_tp2_l2_and_multiprofile_bundles_pass(self) -> None:
        for tp, layers, multi_profile in (
            (1, 1, False),
            (2, 2, False),
            (2, 2, True),
        ):
            with self.subTest(tp=tp, layers=layers, multi=multi_profile):
                template, bundle = _template_bundle(
                    tp=tp, layers=layers, multi_profile=multi_profile
                )
                DenseLogicalBundleValidator.validate(template, bundle)
                self.assertEqual(
                    len({entry.graph.id for entry in bundle.entries}),
                    len(bundle.entries),
                )

    def test_changed_template_layers_model_or_instance_rejects_old_bundle(self) -> None:
        template, bundle = _template_bundle(tp=2)
        changed = (
            _rebuild_template(
                template, model=replace(template.model, num_layers=2)
            ),
            _rebuild_template(
                template, model=replace(template.model, intermediate_size=1024)
            ),
            _rebuild_template(
                template, instance=replace(template.instance, id="P_changed")
            ),
        )
        for candidate in changed:
            with self.subTest(template_id=candidate.id):
                with self.assertRaisesRegex(SchemaError, "source_template_id"):
                    DenseLogicalBundleValidator.validate(candidate, bundle)

    def test_extra_node_value_and_missing_candidate_are_rejected(self) -> None:
        template, bundle = _template_bundle(tp=2)
        graph = bundle.entries[0].graph

        extra_node = replace(
            graph.nodes[0],
            id=f"{template.instance.id}.layer0.forged_node",
            inputs=(),
            outputs=(),
        )
        with self.assertRaisesRegex(SchemaError, "canonical per-layer IDs"):
            DenseLogicalBundleValidator.validate(
                template,
                _rebuild_bundle(
                    template,
                    (_rebuild_graph(graph, nodes=graph.nodes + (extra_node,)),),
                ),
            )

        extra_value = replace(
            graph.values[0],
            id=f"{template.instance.id}.layer0.forged_value",
            producer=None,
            consumers=(),
        )
        with self.assertRaisesRegex(SchemaError, "canonical per-layer IDs"):
            DenseLogicalBundleValidator.validate(
                template,
                _rebuild_bundle(
                    template,
                    (_rebuild_graph(graph, values=graph.values + (extra_value,)),),
                ),
            )

        without_candidate = _rebuild_graph(
            graph, fusion_candidates=graph.fusion_candidates[:-1]
        )
        with self.assertRaisesRegex(SchemaError, "candidate count"):
            DenseLogicalBundleValidator.validate(
                template, _rebuild_bundle(template, (without_candidate,))
            )

    def test_layer_chain_cannot_restart_from_the_original_input(self) -> None:
        template, bundle = _template_bundle(tp=2, layers=2)
        graph = bundle.entries[0].graph
        initial = next(value for value in graph.values if value.id.endswith("embedding_out"))
        previous = next(value for value in graph.values if value.id.endswith("layer0.output"))
        norm1 = next(node for node in graph.nodes if node.id.endswith("layer1.norm1"))
        residual1 = next(node for node in graph.nodes if node.id.endswith("layer1.residual1"))
        changed_initial = replace(
            initial, consumers=initial.consumers + (norm1.id, residual1.id)
        )
        changed_previous = replace(previous, consumers=())
        changed_norm = replace(norm1, inputs=(initial.id, norm1.inputs[1]))
        changed_residual = replace(
            residual1,
            inputs=(initial.id, *residual1.inputs[1:]),
        )
        nodes = tuple(
            changed_norm
            if node.id == norm1.id
            else changed_residual
            if node.id == residual1.id
            else node
            for node in graph.nodes
        )
        values = tuple(
            changed_initial
            if value.id == initial.id
            else changed_previous
            if value.id == previous.id
            else value
            for value in graph.values
        )
        edges = tuple(
            edge for edge in graph.edges if edge.value_id != previous.id
        )
        broken = _rebuild_graph(graph, nodes=nodes, values=values, edges=edges)
        with self.assertRaisesRegex(
            SchemaError,
            "edge count|previous layer output|consumer node does not list|data edges do not match",
        ):
            DenseLogicalBundleValidator.validate(
                template, _rebuild_bundle(template, (broken,))
            )

    def test_weight_shape_producer_and_per_layer_kv_identity_are_exact(self) -> None:
        template, bundle = _template_bundle(tp=2, layers=2)
        graph = bundle.entries[0].graph
        weight = next(value for value in graph.values if value.id.endswith("layer0.w_qkv"))
        bad_shape = replace(weight, shape=(weight.shape[0], weight.shape[1] // 2))
        with self.assertRaisesRegex(SchemaError, "weight producer/shape"):
            DenseLogicalBundleValidator.validate(
                template,
                _rebuild_bundle(
                    template,
                    (_rebuild_graph(
                        graph,
                        values=tuple(
                            bad_shape if value.id == weight.id else value
                            for value in graph.values
                        ),
                    ),),
                ),
            )

        norm = next(node for node in graph.nodes if node.id.endswith("layer0.norm1"))
        qkv = next(node for node in graph.nodes if node.id.endswith("layer0.qkv"))
        produced_weight = replace(weight, producer=norm.id, consumers=())
        producing_norm = replace(norm, outputs=norm.outputs + (weight.id,))
        qkv_without_weight = replace(qkv, inputs=(qkv.inputs[0],))
        producer_graph = _rebuild_graph(
            graph,
            nodes=tuple(
                producing_norm
                if node.id == norm.id
                else qkv_without_weight
                if node.id == qkv.id
                else node
                for node in graph.nodes
            ),
            values=tuple(
                produced_weight if value.id == weight.id else value
                for value in graph.values
            ),
        )
        with self.assertRaisesRegex(SchemaError, "weight producer/shape"):
            DenseLogicalBundleValidator.validate(
                template, _rebuild_bundle(template, (producer_graph,))
            )

        attention0 = next(node for node in graph.nodes if node.id.endswith("layer0.attention"))
        attention1 = next(node for node in graph.nodes if node.id.endswith("layer1.attention"))
        reused = replace(attention1, effects=attention0.effects)
        reused_graph = _rebuild_graph(
            graph,
            nodes=tuple(
                reused if node.id == attention1.id else node for node in graph.nodes
            ),
        )
        with self.assertRaisesRegex(SchemaError, "KV provenance"):
            DenseLogicalBundleValidator.validate(
                template, _rebuild_bundle(template, (reused_graph,))
            )

    def test_swapped_or_duplicate_profile_graphs_are_rejected(self) -> None:
        template, bundle = _template_bundle(tp=2, multi_profile=True)
        graphs = tuple(entry.graph for entry in bundle.entries)
        for replacement in (tuple(reversed(graphs)), (graphs[0], graphs[0])):
            with self.subTest(graph_ids=tuple(graph.id for graph in replacement)):
                with self.assertRaises(SchemaError):
                    DenseLogicalBundleValidator.validate(
                        template, _rebuild_bundle(template, replacement)
                    )

    def test_reweight_preserves_graphs_but_updates_provenance(self) -> None:
        template, bundle = _template_bundle(tp=2, layers=2, multi_profile=True)
        by_key = {
            profile.key: weight
            for profile, weight in zip(template.profiles, (0.25, 0.75))
        }
        profiles = tuple(
            ProfileEntry.create(key=profile.key, weight=by_key[profile.key])
            for profile in template.profiles
        )
        reweighted = _rebuild_template(template, profiles=profiles)
        graphs = tuple(entry.graph for entry in bundle.entries)
        rebound = _rebuild_bundle(reweighted, graphs)
        DenseLogicalBundleValidator.validate(reweighted, rebound)
        self.assertEqual(
            tuple(entry.graph.id for entry in rebound.entries),
            tuple(entry.graph.id for entry in bundle.entries),
        )
        self.assertNotEqual(reweighted.id, template.id)
        self.assertEqual(rebound.source_template_id, reweighted.id)


if __name__ == "__main__":
    unittest.main()
