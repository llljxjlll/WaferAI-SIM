from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend import (
    RectMeshCompileChain,
    RectMeshCompileMode,
    RectMeshFallbackReason,
    RectMeshSpec,
    compile_rect_mesh,
)
from llm.frontend.wafer_frontend.errors import (
    SchemaError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    ExperimentSpec,
    ExplicitGroupPlacement,
    InterDiePolicyName,
    ParallelSpec,
    PlacementSpec,
    PlacementStrategy,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.schema.rect_mesh_compile import (
    RECT_MESH_ARTIFACT_FILE_LIMIT_BYTES,
    RECT_MESH_ARTIFACT_RECORD_LIMIT,
    RectMeshArtifactFilePreflight,
    RectMeshCompileCapabilityReport,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    loads_dataclass,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware

from _fixtures import valid_spec
from test_compiler_policy_wiring import _tiny_tp2_spec
from test_n6_pipeline import _e1_compile_inputs


class RectMeshCompilerCapabilitySchemaTest(unittest.TestCase):
    def test_report_is_stable_closed_and_capacity_checked(self) -> None:
        report = RectMeshCompileCapabilityReport.create(
            producer_pass="rect_mesh_compiler_schema",
            mesh=RectMeshSpec(2, 3),
            requested_mode=RectMeshCompileMode.AUTO,
            selected_chain=RectMeshCompileChain.NAIVE_FIXED_V1,
            fallback_reasons=(
                RectMeshFallbackReason.STANDARD_CHAIN_UNAVAILABLE,
            ),
            profile_count=1,
            manifest_count=1,
            fragment_count=3,
            symbolic_record_count=17,
        )
        self.assertEqual(
            loads_dataclass(
                RectMeshCompileCapabilityReport,
                canonical_json(report),
            ),
            report,
        )
        self.assertEqual(
            report.artifact_record_limit,
            RECT_MESH_ARTIFACT_RECORD_LIMIT,
        )
        self.assertEqual(
            report.artifact_file_limit_bytes,
            RECT_MESH_ARTIFACT_FILE_LIMIT_BYTES,
        )
        self.assertIs(
            report.artifact_file_preflight,
            RectMeshArtifactFilePreflight.DEFERRED_TO_FINALIZER,
        )
        self.assertFalse(report.standard_chain_complete)
        self.assertFalse(report.dense_workload_complete)
        with self.assertRaisesRegex(
            SchemaError, "production artifact limit"
        ) as raised:
            replace(
                report,
                symbolic_record_count=RECT_MESH_ARTIFACT_RECORD_LIMIT + 1,
            ).validate()
        self.assertEqual(
            raised.exception.code,
            RectMeshFallbackReason.ARTIFACT_RECORD_BUDGET.value,
        )


class RectMeshCompilerDispatchTest(unittest.TestCase):
    def test_all_one_hundred_shapes_cross_the_public_parameter_bridge(self) -> None:
        for rows in range(1, 11):
            for columns in range(1, 11):
                with self.subTest(rows=rows, columns=columns):
                    ranks = rows * columns
                    raw = valid_spec()
                    model = raw["model"]
                    assert isinstance(model, dict)
                    model.update(
                        {
                            "V": max(128, ranks),
                            "H": ranks,
                            "I": ranks,
                            "NH": ranks,
                            "KVH": ranks,
                            "DH": 1,
                            "rotary_dim": 1,
                            "L": 1,
                        }
                    )
                    infer = raw["workload"]["infer"]  # type: ignore[index]
                    assert isinstance(infer, dict)
                    profile = infer["profile"]
                    assert isinstance(profile, dict)
                    profile.update(
                        {
                            "prefill_tokens": ranks,
                            "context_sum": ranks,
                            "context_max": ranks,
                        }
                    )
                    parallel = raw["parallel"]
                    assert isinstance(parallel, dict)
                    instances = parallel["instances"]
                    assert isinstance(instances, list)
                    instance = instances[0]
                    assert isinstance(instance, dict)
                    instance["tp"] = ranks
                    spec = from_data(ExperimentSpec, raw, path="spec")
                    fabric = physical_fabric_from_data(
                        minimal_hardware(columns, rows)
                    )
                    with self.assertRaises(
                        UnsupportedFeatureError
                    ) as raised:
                        compile_rect_mesh(
                            spec,
                            fabric,
                            rect_mesh=RectMeshSpec(rows, columns),
                            hbm_address_spaces=(),
                            mode=RectMeshCompileMode.STANDARD,
                        )
                    self.assertEqual(
                        raised.exception.code,
                        RectMeshFallbackReason
                        .STANDARD_CHAIN_UNAVAILABLE.value,
                    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.fabric, cls.hbm_address_spaces = _e1_compile_inputs()
        cls.mesh = RectMeshSpec(rows=1, columns=2)
        source = _tiny_tp2_spec()
        cls.requested = replace(
            source,
            policy=replace(
                source.policy,
                inter_die=InterDiePolicyName.SWIZZLE_TOPO,
            ),
        )
        cls.result = compile_rect_mesh(
            cls.requested,
            cls.fabric,
            rect_mesh=cls.mesh,
            hbm_address_spaces=cls.hbm_address_spaces,
            mode=RectMeshCompileMode.AUTO,
            producer_pass="rect_mesh_compiler_test",
        )

    def test_auto_falls_back_to_one_executable_whole_workload_artifact(self) -> None:
        result = self.result
        result.validate()
        self.assertEqual(result.requested_spec, self.requested)
        self.assertIs(
            result.compilation.spec.policy.inter_die,
            InterDiePolicyName.NAIVE,
        )
        report = result.capability_report
        self.assertIs(
            report.selected_chain,
            RectMeshCompileChain.NAIVE_FIXED_V1,
        )
        self.assertEqual(
            report.fallback_reasons,
            (RectMeshFallbackReason.STANDARD_CHAIN_UNAVAILABLE,),
        )
        self.assertEqual((report.profile_count, report.manifest_count), (1, 1))
        manifest = result.compilation.linked.entries[0].manifest
        kinds = Counter(
            (
                fragment.kind
                if hasattr(fragment, "kind")
                else fragment.fragment.kind
            ).value
            for fragment in manifest.fragments
        )
        self.assertGreater(kinds["coarse"], 0)
        self.assertGreater(kinds["isa_region"], 0)
        self.assertGreater(kinds["standalone_collective"], 0)
        self.assertEqual(
            report.symbolic_record_count,
            sum(len(stream.records) for stream in manifest.core_streams),
        )

    def test_forced_standard_fails_before_partial_compilation(self) -> None:
        with self.assertRaises(UnsupportedFeatureError) as raised:
            compile_rect_mesh(
                self.requested,
                self.fabric,
                rect_mesh=self.mesh,
                hbm_address_spaces=self.hbm_address_spaces,
                mode=RectMeshCompileMode.STANDARD,
            )
        self.assertEqual(
            raised.exception.code,
            RectMeshFallbackReason.STANDARD_CHAIN_UNAVAILABLE.value,
        )

    def test_mesh_tp_and_explicit_placement_fail_closed(self) -> None:
        with self.assertRaises(SchemaError) as transposed:
            compile_rect_mesh(
                self.requested,
                self.fabric,
                rect_mesh=RectMeshSpec(rows=2, columns=1),
                hbm_address_spaces=self.hbm_address_spaces,
                mode=RectMeshCompileMode.STANDARD,
            )
        self.assertEqual(
            transposed.exception.code,
            RectMeshFallbackReason.INVALID_MESH.value,
        )

        instance = self.requested.parallel.instances[0]
        tp_one = replace(
            self.requested,
            parallel=ParallelSpec((replace(instance, tp=1),)),
        )
        with self.assertRaises(SchemaError) as sharding:
            compile_rect_mesh(
                tp_one,
                self.fabric,
                rect_mesh=self.mesh,
                hbm_address_spaces=self.hbm_address_spaces,
                mode=RectMeshCompileMode.STANDARD,
            )
        self.assertEqual(
            sharding.exception.code,
            RectMeshFallbackReason.INCOMPATIBLE_SHARDING.value,
        )

        reversed_placement = replace(
            self.requested,
            placement=PlacementSpec(
                strategy=PlacementStrategy.EXPLICIT,
                groups=(
                    ExplicitGroupPlacement(
                        instance_id=instance.id,
                        mesh_ref=f"{instance.id}.mesh.tp",
                        die_ids=(1, 0),
                    ),
                ),
            ),
        )
        with self.assertRaises(SchemaError) as placement:
            compile_rect_mesh(
                reversed_placement,
                self.fabric,
                rect_mesh=self.mesh,
                hbm_address_spaces=self.hbm_address_spaces,
                mode=RectMeshCompileMode.STANDARD,
            )
        self.assertEqual(
            placement.exception.code,
            RectMeshFallbackReason.INVALID_PLACEMENT.value,
        )


if __name__ == "__main__":
    unittest.main()
