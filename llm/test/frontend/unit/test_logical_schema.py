from __future__ import annotations

import dataclasses
import unittest
from dataclasses import fields, replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType, MeshAxisName, ProfileKey
from llm.frontend.wafer_frontend.schema.experiment import InferOutput
from llm.frontend.wafer_frontend.schema.ir0 import IR0, JobKind
from llm.frontend.wafer_frontend.schema.logical import (
    EXPANDED_IR0_BUNDLE_SCHEMA_VERSION,
    IR0_TEMPLATE_SCHEMA_VERSION,
    DenseLayerKind,
    DenseLayerTemplate,
    DenseModelShape,
    ElementwiseKind,
    ExpandedIR0Bundle,
    ExpandedProfileIR0,
    IR0Template,
    NormKind,
    ProfileEntry,
    SequenceParallelSpec,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass

from _fixtures import valid_ir0


def second_profile() -> ProfileKey:
    return ProfileKey(
        prefill_tokens=64,
        decode_tokens=0,
        num_seqs=1,
        context_sum=64,
        context_max=64,
        kv_pages=4,
        expert_load=None,
    )


def expanded_graph(profile: ProfileKey, *, producer_pass: str = "logical_expand") -> IR0:
    base = valid_ir0()
    result = IR0.create(
        producer_pass=producer_pass,
        job=base.job,
        instances=base.instances,
        nodes=base.nodes,
        values=base.values,
        edges=base.edges,
        fusion_candidates=base.fusion_candidates,
        profile=profile,
        train=base.train,
    )
    result.validate()
    return result


def valid_template(*, profile_count: int = 2) -> IR0Template:
    base = valid_ir0()
    weighted = [(base.profile, 0.25), (second_profile(), 0.75)]
    if profile_count == 1:
        weighted = [(base.profile, 1.0)]
    profiles = tuple(
        sorted(
            (ProfileEntry.create(key=key, weight=weight) for key, weight in weighted),
            key=lambda entry: entry.profile_id,
        )
    )
    result = IR0Template.create(
        job=JobKind.INFER,
        model=DenseModelShape(
            vocab_size=512,
            hidden_size=256,
            intermediate_size=512,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            rotary_dim=64,
            dtype=DType.FP16,
            tie_word_embeddings=False,
            rms_norm_epsilon=1e-5,
            rope_theta=10000.0,
            max_position_embeddings=4096,
        ),
        instance=base.instances[0],
        sequence_parallel=SequenceParallelSpec(True, MeshAxisName.TP),
        layer=DenseLayerTemplate(
            id="dense_layer",
            kind=DenseLayerKind.LLAMA_DENSE_BLOCK_V1,
            norm=NormKind.RMS_NORM,
            activation=ElementwiseKind.SWIGLU,
            has_bias=False,
        ),
        infer_output=InferOutput.LOGITS,
        profiles=profiles,
    )
    result.validate()
    return result


def valid_bundle(template: IR0Template | None = None) -> ExpandedIR0Bundle:
    source = template or valid_template()
    entries = tuple(
        ExpandedProfileIR0.create(
            source_template_id=source.id,
            weight=profile.weight,
            graph=expanded_graph(profile.key),
        )
        for profile in source.profiles
    )
    result = ExpandedIR0Bundle.create(source_template=source, entries=entries)
    result.validate()
    return result


class LogicalSchemaTest(unittest.TestCase):
    def test_template_is_profile_independent_versioned_and_stable(self) -> None:
        template = valid_template()
        decoded = loads_dataclass(IR0Template, canonical_json(template), path="template")
        self.assertEqual(decoded, template)
        self.assertEqual(decoded.id, valid_template().id)
        field_names = {field.name for field in fields(IR0Template)}
        self.assertNotIn("profile", field_names)
        self.assertNotIn("nodes", field_names)
        self.assertNotIn("values", field_names)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decoded.id = "changed"  # type: ignore[misc]

    def test_stage2_versions_and_tiny_parameter_oracle_are_exact(self) -> None:
        template = valid_template(profile_count=1)
        self.assertEqual(
            IR0_TEMPLATE_SCHEMA_VERSION,
            "wafer_frontend.ir0_template/v1alpha3",
        )
        self.assertEqual(
            EXPANDED_IR0_BUNDLE_SCHEMA_VERSION,
            "wafer_frontend.expanded_ir0_bundle/v1alpha5",
        )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                template,
                schema_version="wafer_frontend.ir0_template/v1alpha2",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                valid_bundle(template),
                schema_version="wafer_frontend.expanded_ir0_bundle/v1alpha4",
            ).validate()

        tiny = replace(
            template.model,
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            head_dim=4,
            rotary_dim=4,
        )
        self.assertEqual(tiny.parameter_elements(), 6224)
        self.assertEqual(tiny.parameter_bytes(), 12448)

    def test_template_requires_sorted_normalized_profiles_and_tp_reuse(self) -> None:
        template = valid_template()
        reversed_template = IR0Template.create(
            job=template.job,
            model=template.model,
            instance=template.instance,
            sequence_parallel=template.sequence_parallel,
            layer=template.layer,
            infer_output=template.infer_output,
            profiles=tuple(reversed(template.profiles)),
        )
        with self.assertRaisesRegex(SchemaError, "strictly increasing"):
            reversed_template.validate()

        bad_weights = tuple(
            ProfileEntry.create(key=entry.key, weight=0.4)
            for entry in template.profiles
        )
        bad_weight_template = IR0Template.create(
            job=template.job,
            model=template.model,
            instance=template.instance,
            sequence_parallel=template.sequence_parallel,
            layer=template.layer,
            infer_output=template.infer_output,
            profiles=bad_weights,
        )
        with self.assertRaisesRegex(SchemaError, "sum to 1"):
            bad_weight_template.validate()

        bad_sp = IR0Template.create(
            job=template.job,
            model=template.model,
            instance=template.instance,
            sequence_parallel=SequenceParallelSpec(True, None),
            layer=template.layer,
            infer_output=template.infer_output,
            profiles=template.profiles,
        )
        with self.assertRaisesRegex(SchemaError, "reuse_axis"):
            bad_sp.validate()

    def test_bundle_round_trip_has_one_independent_graph_per_profile(self) -> None:
        bundle = valid_bundle()
        decoded = loads_dataclass(
            ExpandedIR0Bundle, canonical_json(bundle), path="bundle"
        )
        self.assertEqual(decoded, bundle)
        self.assertEqual(decoded.id, valid_bundle().id)
        self.assertEqual(
            tuple(entry.profile_id for entry in decoded.entries),
            tuple(profile.profile_id for profile in decoded.source_profiles),
        )
        self.assertEqual(len({entry.graph.id for entry in decoded.entries}), 2)
        self.assertTrue(
            all(entry.graph.producer_pass == "logical_expand" for entry in decoded.entries)
        )

    def test_bundle_rejects_profile_weight_and_provenance_mismatches(self) -> None:
        template = valid_template()
        bundle = valid_bundle(template)

        wrong_weight = ExpandedProfileIR0.create(
            source_template_id=template.id,
            weight=bundle.entries[0].weight + 0.125,
            graph=bundle.entries[0].graph,
        )
        candidate = ExpandedIR0Bundle.create(
            source_template=template,
            entries=(wrong_weight, bundle.entries[1]),
        )
        with self.assertRaisesRegex(SchemaError, "template profile weight"):
            candidate.validate()

        wrong_source = ExpandedProfileIR0.create(
            source_template_id="ir0_template_wrong",
            weight=bundle.entries[0].weight,
            graph=bundle.entries[0].graph,
        )
        candidate = ExpandedIR0Bundle.create(
            source_template=template,
            entries=(wrong_source, bundle.entries[1]),
        )
        with self.assertRaisesRegex(SchemaError, "source_template_id"):
            candidate.validate()

        candidate = ExpandedIR0Bundle.create(
            source_template=template,
            entries=tuple(reversed(bundle.entries)),
        )
        with self.assertRaises(SchemaError):
            candidate.validate()

    def test_logical_expand_cannot_be_a_no_op_or_mislabelled_graph(self) -> None:
        template = valid_template(profile_count=1)
        graph = expanded_graph(template.profiles[0].key, producer_pass="build_ir0")
        entry = ExpandedProfileIR0.create(
            source_template_id=template.id,
            weight=1.0,
            graph=graph,
        )
        with self.assertRaisesRegex(SchemaError, "logical_expand"):
            entry.validate()

        base = valid_ir0()
        empty = IR0.create(
            producer_pass="logical_expand",
            job=base.job,
            instances=base.instances,
            nodes=(),
            values=(),
            edges=(),
            fusion_candidates=(),
            profile=template.profiles[0].key,
        )
        empty.validate()
        entry = ExpandedProfileIR0.create(
            source_template_id=template.id,
            weight=1.0,
            graph=empty,
        )
        with self.assertRaisesRegex(SchemaError, "non-empty graph"):
            entry.validate()

    def test_stable_ids_reject_semantic_mutation(self) -> None:
        template = valid_template(profile_count=1)
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(template, model=replace(template.model, num_layers=2)).validate()
        bundle = valid_bundle(template)
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(bundle, id="expanded_ir0_bundle_wrong").validate()


if __name__ == "__main__":
    unittest.main()
