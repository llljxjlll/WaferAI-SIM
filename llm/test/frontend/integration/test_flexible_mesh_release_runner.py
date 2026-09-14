from __future__ import annotations

import hashlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.schema._validation_session import (
    builder_validation_session,
    mark_validation_complete,
    validation_seen,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import CommandFragment
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseBinding,
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseFamily,
    FlexibleMeshReleaseTool,
    FlexibleMeshReleaseToolKind,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec

from flexible_mesh_release_dense import FlexibleDenseReleaseAdapter
from flexible_mesh_release_meshslice import FlexibleMeshSliceReleaseAdapter
from flexible_mesh_release_moe import FlexibleMoeReleaseAdapter
from flexible_mesh_release_profiles import release_trace_model_digest
from run_flexible_mesh_release import FlexibleMeshReleaseRunner


_ROOT = Path(__file__).resolve().parents[4]
_BUILD = _ROOT / "build-debug-final"
_SIMULATION = _ROOT / "llm/test/simulation_config/default_spec.json"
_MAPPING = (
    _BUILD
    / "flexible_mesh_runtime_case_d6669b3d3de260ce"
    / "mapping.spec"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _UnavailableAdapter:
    def __init__(self, family: FlexibleMeshReleaseFamily) -> None:
        self.family = family


class _ValidationProbe:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def validate(self) -> None:
        domain = "release_runner_validation_probe"
        if validation_seen(self, domain):
            return
        self.calls += 1
        if self.fail:
            raise RuntimeError("probe validation failed")
        mark_validation_complete(self, domain)


class BuilderValidationSessionTest(unittest.TestCase):
    def test_nested_adapter_runner_session_is_identity_safe_and_resets(self) -> None:
        case = FlexibleMeshReleaseCase.create(
            family=FlexibleMeshReleaseFamily.DENSE_TRAIN,
            mesh=RectMeshSpec(1, 1),
            trace_model_digest=release_trace_model_digest(
                FlexibleMeshReleaseFamily.DENSE_TRAIN
            ),
            runtime_profile_version="flexible-mesh-timing-v2",
        )
        adapter = FlexibleDenseReleaseAdapter(
            hardware_template_json=(
                _ROOT / "llm/test/program/p5_large_hardware.json"
            ).read_text(encoding="utf-8"),
            mapping_text=(
                _ROOT / "llm/test/default/mapping.spec"
            ).read_text(encoding="utf-8"),
        )
        original_validate = CommandFragment.validate
        expensive_calls: dict[int, int] = {}

        def counted_validate(fragment: CommandFragment, *args, **kwargs) -> None:
            if not validation_seen(fragment, "command_fragment"):
                key = id(fragment)
                expensive_calls[key] = expensive_calls.get(key, 0) + 1
            original_validate(fragment, *args, **kwargs)

        # Match the production nesting: runner session outside adapter session.
        with patch.object(CommandFragment, "validate", new=counted_validate):
            with builder_validation_session():
                materialized = adapter.materialize(case)
                materialized.validate(case)

        self.assertTrue(materialized.manifest.fragments)
        self.assertEqual(
            {
                fragment.id: expensive_calls.get(id(fragment), 0)
                for fragment in materialized.manifest.fragments
            },
            {fragment.id: 1 for fragment in materialized.manifest.fragments},
        )

        first = _ValidationProbe()
        second = _ValidationProbe()
        failing = _ValidationProbe(fail=True)
        with builder_validation_session():
            with builder_validation_session():
                first.validate()
                first.validate()
            second.validate()
            with self.assertRaisesRegex(RuntimeError, "probe validation failed"):
                failing.validate()
            with self.assertRaisesRegex(RuntimeError, "probe validation failed"):
                failing.validate()
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertEqual(failing.calls, 2)
        self.assertFalse(validation_seen(first, "release_runner_validation_probe"))

        reset = _ValidationProbe()
        with self.assertRaisesRegex(RuntimeError, "abort outer session"):
            with builder_validation_session():
                reset.validate()
                self.assertTrue(
                    validation_seen(reset, "release_runner_validation_probe")
                )
                raise RuntimeError("abort outer session")
        self.assertFalse(validation_seen(reset, "release_runner_validation_probe"))
        with builder_validation_session():
            reset.validate()
        self.assertEqual(reset.calls, 2)


class ProgramIoAdapterRebindTest(unittest.TestCase):
    def test_all_release_adapters_are_exact_sha_rebinds(self) -> None:
        hardware = (
            _ROOT / "llm/test/program/p5_large_hardware.json"
        ).read_text(encoding="utf-8")
        mapping = (
            _ROOT / "llm/test/default/mapping.spec"
        ).read_text(encoding="utf-8")
        families_and_adapters = (
            (
                FlexibleMeshReleaseFamily.DENSE_TRAIN,
                FlexibleDenseReleaseAdapter(
                    hardware_template_json=hardware, mapping_text=mapping,
                ),
            ),
            (
                FlexibleMeshReleaseFamily.MOE_INFERENCE,
                FlexibleMoeReleaseAdapter(
                    FlexibleMeshReleaseFamily.MOE_INFERENCE,
                    hardware_template_json=hardware, mapping_text=mapping,
                ),
            ),
            (
                FlexibleMeshReleaseFamily.MESHSLICE_AG,
                FlexibleMeshSliceReleaseAdapter(
                    FlexibleMeshReleaseFamily.MESHSLICE_AG,
                    hardware_template_json=hardware, mapping_text=mapping,
                ),
            ),
        )
        actual_sha = "ab" * 32
        for family, adapter in families_and_adapters:
            with self.subTest(family=family.value):
                case = FlexibleMeshReleaseCase.create(
                    family=family,
                    mesh=RectMeshSpec(1, 1),
                    trace_model_digest=release_trace_model_digest(family),
                    runtime_profile_version="flexible-mesh-timing-v3-one-shot",
                )
                materialized = adapter.materialize(case)
                preflight = adapter.build_program_io(materialized, "0" * 64)
                actual = adapter.build_program_io(materialized, actual_sha)
                rebound = preflight.bind_program_artifact_sha256(actual_sha)
                self.assertEqual(rebound, actual)
                rebound.validate_against(materialized.manifest)

@unittest.skipUnless(
    os.environ.get("NPUSIM_FLEXIBLE_RELEASE_CANARY") == "1",
    "requires built production tools",
)
class FlexibleMeshReleaseRunnerCanary(unittest.TestCase):
    def test_meshslice_local_runs_two_complete_toolchains(self) -> None:
        mapping_text = _MAPPING.read_text(encoding="utf-8")
        case = FlexibleMeshReleaseCase.create(
            family=FlexibleMeshReleaseFamily.MESHSLICE_AG,
            mesh=RectMeshSpec(1, 1),
            trace_model_digest=release_trace_model_digest(
                FlexibleMeshReleaseFamily.MESHSLICE_AG
            ),
            runtime_profile_version="timing-v1",
        )
        meshslice = FlexibleMeshSliceReleaseAdapter(
            FlexibleMeshReleaseFamily.MESHSLICE_AG,
            mapping_text=mapping_text,
        )
        prepared = meshslice.materialize(case)
        tools = tuple(
            FlexibleMeshReleaseTool(
                kind=kind,
                binary_path=str(path.resolve()),
                version="build-debug-final",
                sha256=_sha(path),
                allowlisted_sha256=(_sha(path),),
            )
            for kind, path in zip(
                FlexibleMeshReleaseToolKind,
                (
                    _BUILD / "npusim_program_finalizer",
                    _BUILD / "npusim_program_io_selftest",
                    _BUILD / "npusim",
                ),
            )
        )
        binding = FlexibleMeshReleaseBinding.create(
            runtime_profile_version="timing-v1",
            environment_profile_version="canary-v1",
            tools=tools,
            hardware_config_sha256=hashlib.sha256(
                prepared.hardware_json.encode("utf-8")
            ).hexdigest(),
            simulation_config_sha256=_sha(_SIMULATION),
            mapping_config_sha256=hashlib.sha256(
                mapping_text.encode("utf-8")
            ).hexdigest(),
        )
        adapters = tuple(
            meshslice if family is FlexibleMeshReleaseFamily.MESHSLICE_AG
            else _UnavailableAdapter(family)
            for family in FlexibleMeshReleaseFamily
        )
        runner = FlexibleMeshReleaseRunner(
            binding=binding,
            adapters=adapters,
            finalizer=_BUILD / "npusim_program_finalizer",
            resolver=_BUILD / "npusim_program_io_selftest",
            npusim=_BUILD / "npusim",
            simulation=_SIMULATION,
            runtime_root=_BUILD,
            timeout=120,
        )
        evidence = runner.run_case(case)
        self.assertTrue(evidence.runtime_verified)
        self.assertTrue(evidence.repeatability_verified)
        self.assertEqual(len(evidence.executions), 2)
        self.assertEqual(
            len(
                {
                    identity
                    for execution in evidence.executions
                    for identity in execution.run.identities
                }
            ),
            8,
        )


if __name__ == "__main__":
    unittest.main()
