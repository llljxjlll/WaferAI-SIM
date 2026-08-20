from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes import build_ir0 as public_build_ir0
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.schema.common import MeshAxisName
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec, InferOutput
from llm.frontend.wafer_frontend.schema.ir0 import JobKind, LogicalRole
from llm.frontend.wafer_frontend.schema.logical import (
    DenseLayerKind,
    ElementwiseKind,
    IR0Template,
    NormKind,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)

from _fixtures import valid_spec


def decode(raw: dict[str, object]) -> ExperimentSpec:
    return from_data(ExperimentSpec, raw, path="spec")


def profile(
    *,
    prefill_tokens: int,
    decode_tokens: int,
    num_seqs: int,
    context_sum: int,
    context_max: int,
    kv_pages: int = 2,
) -> dict[str, object]:
    return {
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "num_seqs": num_seqs,
        "context_sum": context_sum,
        "context_max": context_max,
        "kv_pages": kv_pages,
        "expert_load": None,
    }


def set_parallel(raw: dict[str, object], *, tp: int, sp: bool) -> None:
    raw["parallel"]["instances"][0].update(tp=tp, sp=sp)  # type: ignore[index]


def set_static_profile(raw: dict[str, object], value: dict[str, object]) -> None:
    raw["workload"]["infer"] = {  # type: ignore[index]
        "source": "static_profile",
        "output": "logits",
        "profile": value,
    }


def set_distribution(
    raw: dict[str, object], entries: list[tuple[dict[str, object], float]]
) -> None:
    raw["workload"]["infer"] = {  # type: ignore[index]
        "source": "shape_dist",
        "output": "logits",
        "shape_dist": {
            "profiles": [
                {"key": key, "weight": weight} for key, weight in entries
            ]
        },
    }


class BuildIR0Test(unittest.TestCase):
    def test_static_prefill_builds_only_a_stable_profile_independent_template(self) -> None:
        self.assertIs(public_build_ir0, build_ir0)
        spec = decode(valid_spec())
        template = build_ir0(spec)
        template.validate()

        self.assertIs(template.job, JobKind.INFER)
        self.assertEqual(
            (
                template.model.hidden_size,
                template.model.vocab_size,
                template.model.intermediate_size,
                template.model.num_layers,
                template.model.num_heads,
                template.model.num_kv_heads,
                template.model.head_dim,
            ),
            (256, 512, 512, 1, 4, 2, 64),
        )
        self.assertEqual(template.model.rotary_dim, 64)
        self.assertFalse(template.model.tie_word_embeddings)
        self.assertEqual(template.model.rms_norm_epsilon, 1e-5)
        self.assertEqual(template.model.rope_theta, 10000.0)
        self.assertEqual(template.model.max_position_embeddings, 4096)
        self.assertIs(template.infer_output, InferOutput.LOGITS)
        self.assertEqual(template.instance.id, "P0")
        self.assertIs(template.instance.role, LogicalRole.PREFILL)
        self.assertEqual(len(template.instance.meshes), 1)
        mesh = template.instance.meshes[0]
        self.assertEqual(len(mesh.axes), 1)
        self.assertIs(mesh.axes[0].name, MeshAxisName.TP)
        self.assertEqual(mesh.axes[0].size, 2)
        self.assertTrue(template.sequence_parallel.enabled)
        self.assertIs(template.sequence_parallel.reuse_axis, MeshAxisName.TP)
        self.assertIs(template.layer.kind, DenseLayerKind.LLAMA_DENSE_BLOCK_V1)
        self.assertIs(template.layer.norm, NormKind.RMS_NORM)
        self.assertIs(template.layer.activation, ElementwiseKind.SWIGLU)
        self.assertFalse(template.layer.has_bias)
        self.assertEqual(len(template.profiles), 1)
        self.assertEqual(template.profiles[0].weight, 1.0)
        self.assertEqual(template.profiles[0].key, spec.workload.infer.profile)

        template_fields = {field.name for field in fields(IR0Template)}
        for forbidden in ("nodes", "values", "edges", "fusion_candidates"):
            self.assertNotIn(forbidden, template_fields)
        encoded = canonical_json(template)
        self.assertEqual(loads_dataclass(IR0Template, encoded), template)
        self.assertEqual(build_ir0(spec).id, template.id)
        self.assertEqual(canonical_json(build_ir0(spec)), encoded)

    def test_shape_distribution_is_canonically_sorted_without_weight_changes(self) -> None:
        first = profile(
            prefill_tokens=32,
            decode_tokens=0,
            num_seqs=1,
            context_sum=32,
            context_max=32,
        )
        second = profile(
            prefill_tokens=64,
            decode_tokens=0,
            num_seqs=1,
            context_sum=64,
            context_max=64,
            kv_pages=4,
        )
        raw_a = valid_spec()
        set_distribution(raw_a, [(second, 0.75), (first, 0.25)])
        raw_b = valid_spec()
        set_distribution(raw_b, [(first, 0.25), (second, 0.75)])

        template_a = build_ir0(decode(raw_a))
        template_b = build_ir0(decode(raw_b))
        self.assertEqual(template_a, template_b)
        self.assertEqual(
            tuple(entry.profile_id for entry in template_a.profiles),
            tuple(sorted(entry.profile_id for entry in template_a.profiles)),
        )
        expected_weights = {
            entry.key.stable_id(): entry.weight
            for entry in decode(raw_a).workload.infer.shape_dist.profiles  # type: ignore[union-attr]
        }
        self.assertEqual(
            {entry.profile_id: entry.weight for entry in template_a.profiles},
            expected_weights,
        )

    def test_tp_sp_support_matrix_is_fail_closed(self) -> None:
        for tp, sp, supported in (
            (1, False, True),
            (1, True, False),
            (2, False, False),
            (2, True, True),
        ):
            with self.subTest(tp=tp, sp=sp):
                raw = valid_spec()
                set_parallel(raw, tp=tp, sp=sp)
                spec = decode(raw)
                if supported:
                    template = build_ir0(spec)
                    self.assertEqual(template.instance.parallel.tp, tp)
                    self.assertIs(template.instance.parallel.sp, sp)
                else:
                    with self.assertRaises(UnsupportedFeatureError):
                        build_ir0(spec)

    def test_mixed_and_ambiguous_attention_profiles_are_rejected(self) -> None:
        cases = (
            profile(
                prefill_tokens=16,
                decode_tokens=16,
                num_seqs=17,
                context_sum=64,
                context_max=32,
            ),
            profile(
                prefill_tokens=32,
                decode_tokens=0,
                num_seqs=2,
                context_sum=32,
                context_max=32,
            ),
            profile(
                prefill_tokens=32,
                decode_tokens=0,
                num_seqs=1,
                context_sum=64,
                context_max=64,
            ),
            profile(
                prefill_tokens=32,
                decode_tokens=0,
                num_seqs=1,
                context_sum=64,
                context_max=32,
            ),
            profile(
                prefill_tokens=0,
                decode_tokens=32,
                num_seqs=16,
                context_sum=65536,
                context_max=4096,
            ),
        )
        for key in cases:
            with self.subTest(key=key):
                raw = valid_spec()
                raw["parallel"]["instances"][0]["role"] = "both"  # type: ignore[index]
                set_static_profile(raw, key)
                with self.assertRaises(UnsupportedFeatureError):
                    build_ir0(decode(raw))

    def test_role_compatibility_and_both_profile_manifest(self) -> None:
        prefill = profile(
            prefill_tokens=32,
            decode_tokens=0,
            num_seqs=1,
            context_sum=32,
            context_max=32,
        )
        decode_key = profile(
            prefill_tokens=0,
            decode_tokens=32,
            num_seqs=32,
            context_sum=131072,
            context_max=4096,
            kv_pages=128,
        )
        raw = valid_spec()
        raw["parallel"]["instances"][0]["role"] = "both"  # type: ignore[index]
        set_distribution(raw, [(prefill, 0.4), (decode_key, 0.6)])
        template = build_ir0(decode(raw))
        self.assertIs(template.instance.role, LogicalRole.BOTH)
        self.assertEqual(len(template.profiles), 2)

        incompatible = valid_spec()
        incompatible["parallel"]["instances"][0]["role"] = "prefill"  # type: ignore[index]
        set_static_profile(incompatible, decode_key)
        with self.assertRaises(SchemaError):
            build_ir0(decode(incompatible))

    def test_build_validates_and_does_not_mutate_the_input(self) -> None:
        spec = decode(valid_spec())
        before = canonical_digest(spec)
        build_ir0(spec)
        self.assertEqual(canonical_digest(spec), before)

        invalid = replace(spec, hardware=replace(spec.hardware, ref=""))
        with self.assertRaisesRegex(SchemaError, "hardware.ref"):
            build_ir0(invalid)

        raw = valid_spec()
        before_raw = deepcopy(raw)
        build_ir0(decode(raw))
        self.assertEqual(raw, before_raw)


if __name__ == "__main__":
    unittest.main()
