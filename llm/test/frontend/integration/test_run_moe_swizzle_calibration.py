from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import SwiGluWorkload
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MoeCalibrationKind,
    MoeCalibrationStatus,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration_program import (
    MoeSwizzleCalibrationStandardLinkedProgram,
)
from llm.test.frontend.integration.run_moe_swizzle_calibration import (
    MoeSwizzleCalibrationExecutable,
    MoeSwizzleCalibrationFailureStage,
    MoeSwizzleCalibrationKey,
    MoeSwizzleCalibrationStageFailure,
    MoeSwizzleSwiGluCalibrationArtifact,
    canonical_moe_swizzle_calibration_keys,
    prepare_moe_swizzle_calibration_dramsys_runtime,
    run_moe_swizzle_isolated_calibration,
)


def _field(command: tuple[str, ...], name: str) -> str:
    prefix = f"--{name}="
    return next(item[len(prefix):] for item in command if item.startswith(prefix))


@dataclass
class _Provider:
    root: Path

    def materialize(self, key: MoeSwizzleCalibrationKey, output_root: Path) -> MoeSwizzleCalibrationExecutable:
        artifact_root = output_root / "artifact"
        artifact_root.mkdir(parents=False, exist_ok=False)
        program_payload = b"program"
        linked_program = object.__new__(MoeSwizzleCalibrationStandardLinkedProgram)
        object.__setattr__(linked_program, "id", "production_test_provider")
        object.__setattr__(
            linked_program,
            "source",
            _FakeSource(key.kind, key.shape, 7),
        )
        object.__setattr__(linked_program, "manifest", _FakeManifest("manifest"))
        object.__setattr__(
            linked_program,
            "program_io",
            _FakeProgramIo(
                "program-io", hashlib.sha256(program_payload).hexdigest()
            ),
        )
        paths: dict[str, Path] = {}
        for name, payload in (
            ("program", program_payload),
            (
                "linked_manifest",
                (canonical_json(linked_program.manifest) + "\n").encode("utf-8"),
            ),
            (
                "program_io",
                (canonical_json(linked_program.program_io) + "\n").encode("utf-8"),
            ),
            ("hardware_config", b"hardware"),
            ("simulation_config", b"simulation"),
            ("mapping_config", b"mapping"),
        ):
            path = artifact_root / name
            path.write_bytes(payload)
            paths[name] = path
        artifact = None
        if key.kind is MoeCalibrationKind.SWIGLU_GROUP:
            if key.shape is None:
                raise AssertionError("typed SWIGLU_GROUP key lost its shape")
            _, _, flattened = key.shape
            artifact = MoeSwizzleSwiGluCalibrationArtifact(
                SwiGluWorkload(
                    (1, 2 * flattened), (1, flattened),
                    (1, 2 * flattened), (1, flattened), DType.FP16,
                ),
                1, 4 * flattened, 2 * flattened, 0, 0,
            )
        return MoeSwizzleCalibrationExecutable(
            key, linked_program, linked_program.id, 7, **paths,
            swiglu_group_artifact=artifact,
        )


@dataclass(frozen=True)
class _FakeSource:
    kind: MoeCalibrationKind
    shape: tuple[int, int, int] | None
    target_runtime_core_id: int


@dataclass(frozen=True)
class _FakeManifest:
    marker: str


@dataclass(frozen=True)
class _FakeProgramIo:
    marker: str
    program_artifact_sha256: str


def _fake_npusim(root: Path) -> Path:
    tool = root / "build/npusim"
    tool.parent.mkdir(parents=True)
    tool.write_bytes(b"matching-npusim")
    dram = root / "DRAMSys/configs/hbm2-example.json"
    dram.parent.mkdir(parents=True)
    dram.write_text("{}", encoding="utf-8")
    return tool


def _marker(command: tuple[str, ...], _: Path, __: int) -> tuple[int, str]:
    kind = _field(command, "moe-swizzle-calibration-kind")
    shape = _field(command, "moe-swizzle-calibration-shape")
    dtype = "fp16" if shape != "none" else "none"
    return 0, (
        "[MOE_SWIZZLE_CALIBRATION] "
        f"kind={kind} sample={_field(command, 'moe-swizzle-calibration-sample')} "
        f"repeat={_field(command, 'moe-swizzle-calibration-repeat')} "
        f"cycles=17 shape={shape} dtype={dtype} "
        f"tool_sha256={_field(command, 'moe-swizzle-calibration-tool-sha256')} "
        f"hardware_sha256={_field(command, 'moe-swizzle-calibration-hardware-sha256')} "
        f"simulation_sha256={_field(command, 'moe-swizzle-calibration-simulation-sha256')} "
        f"mapping_sha256={_field(command, 'moe-swizzle-calibration-mapping-sha256')}"
    )


@patch.object(
    MoeSwizzleCalibrationStandardLinkedProgram,
    "validate",
    lambda self, path="": None,
)
class MoeSwizzleCalibrationRunnerTest(unittest.TestCase):
    def test_canonical_matrix_and_provider_driven_protocol_are_exact(self) -> None:
        keys = canonical_moe_swizzle_calibration_keys()
        self.assertEqual(len(keys), 168)
        self.assertEqual(len(set(keys)), 168)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = _fake_npusim(root)
            result = run_moe_swizzle_isolated_calibration(
                provider=_Provider(root / "provider"),
                npusim=tool,
                runtime_root=root / "runtime",
                execute=_marker,
                worker_count=4,
            )
            self.assertIs(result.profile.status, MoeCalibrationStatus.MEASURED)
            self.assertEqual(len(result.profile.samples), 168)
            self.assertEqual(result.run_count, 168)
            self.assertEqual(len(result.artifacts), 168)
            self.assertTrue(all(
                item.production_source_ref == "production_test_provider"
                for item in result.artifacts
            ))
            self.assertEqual(
                result.artifacts[0].program_sha256,
                hashlib.sha256(b"program").hexdigest(),
            )
            self.assertEqual(
                result.matching_tool_sha256,
                hashlib.sha256(b"matching-npusim").hexdigest(),
            )
            with self.assertRaisesRegex(SchemaError, "exact 168 executions"):
                replace(result, run_count=167).validate()
            with self.assertRaisesRegex(SchemaError, "exact 168 executions"):
                replace(
                    result,
                    raw_output_paths=result.raw_output_paths + (result.raw_output_paths[0],),
                ).validate()
            with self.assertRaisesRegex(SchemaError, "exact 168 executions"):
                replace(result, artifacts=result.artifacts[:143]).validate()
            for field in (
                "production_source_ref",
                "program_sha256",
                "linked_manifest_sha256",
                "program_io_sha256",
            ):
                tampered = replace(
                    result.artifacts[1],
                    **{field: "different_source" if field == "production_source_ref" else "f" * 64},
                )
                with self.assertRaisesRegex(SchemaError, "provenance drifted"):
                    replace(
                        result,
                        artifacts=(
                            result.artifacts[0],
                            tampered,
                            *result.artifacts[2:],
                        ),
                    ).validate()


    def test_one_worker_and_four_workers_are_semantically_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = _fake_npusim(root)
            one = run_moe_swizzle_isolated_calibration(
                provider=_Provider(root / "provider-one"),
                npusim=tool,
                runtime_root=root / "runtime-one",
                worker_count=1,
                execute=_marker,
            )
            four = run_moe_swizzle_isolated_calibration(
                provider=_Provider(root / "provider-four"),
                npusim=tool,
                runtime_root=root / "runtime-four",
                worker_count=4,
                execute=_marker,
            )
            self.assertEqual(one.profile, four.profile)
            self.assertEqual(
                tuple((item.key, item.production_source_ref, item.program_sha256, item.linked_manifest_sha256, item.program_io_sha256) for item in one.artifacts),
                tuple((item.key, item.production_source_ref, item.program_sha256, item.linked_manifest_sha256, item.program_io_sha256) for item in four.artifacts),
            )
            self.assertEqual(
                tuple(path.parent.name for path in four.raw_output_paths),
                tuple(
                    f"sample-{ordinal:03d}-{key.kind.value}"
                    for ordinal, key in enumerate(canonical_moe_swizzle_calibration_keys())
                ),
            )

    def test_worker_count_and_parallel_first_failure_are_failclosed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = _fake_npusim(root)
            for index, worker_count in enumerate((0, 65)):
                with self.assertRaisesRegex(SchemaError, "worker count"):
                    run_moe_swizzle_isolated_calibration(
                        provider=_Provider(root / f"provider-bad-{index}"),
                        npusim=tool,
                        runtime_root=root / f"runtime-bad-{index}",
                        worker_count=worker_count,
                        execute=_marker,
                    )

            def fail_two(
                command: tuple[str, ...], _: Path, seconds: int
            ) -> tuple[int, str]:
                sample_name = Path(_field(command, "program")).parent.parent.name
                ordinal = int(sample_name.split("-", 2)[1])
                if ordinal == 3:
                    return 9, "minimum ordinal exit\n"
                if ordinal == 7:
                    raise subprocess.TimeoutExpired(
                        command, seconds, output=b"later timeout\n"
                    )
                return _marker(command, _, seconds)

            runtime = root / "runtime-parallel-failure"
            with self.assertRaises(MoeSwizzleCalibrationStageFailure) as caught:
                run_moe_swizzle_isolated_calibration(
                    provider=_Provider(root / "provider-parallel-failure"),
                    npusim=tool,
                    runtime_root=runtime,
                    worker_count=4,
                    execute=fail_two,
                )
            failure = caught.exception
            self.assertEqual(failure.ordinal, 3)
            self.assertIs(
                failure.stage, MoeSwizzleCalibrationFailureStage.EXECUTE_EXIT
            )
            self.assertEqual(failure.returncode, 9)
            self.assertEqual(
                failure.raw_output_path.read_text(encoding="utf-8"),
                "minimum ordinal exit\n",
            )
            key7 = canonical_moe_swizzle_calibration_keys()[7]
            later_raw = (
                runtime
                / f"sample-007-{key7.kind.value}"
                / "raw.stdout"
            )
            self.assertEqual(
                later_raw.read_text(encoding="utf-8"), "later timeout\n"
            )

    def test_runner_paths_must_be_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = _fake_npusim(root)
            with self.assertRaisesRegex(SchemaError, "absolute regular file"):
                run_moe_swizzle_isolated_calibration(
                    provider=_Provider(root / "provider"),
                    npusim=Path("npusim"),
                    runtime_root=root / "runtime",
                    execute=_marker,
                )
            with self.assertRaisesRegex(SchemaError, "absolute directory"):
                run_moe_swizzle_isolated_calibration(
                    provider=_Provider(root / "provider"),
                    npusim=tool,
                    runtime_root=Path("runtime"),
                    execute=_marker,
                )

    def test_dramsys_runtime_link_is_exact_and_failclosed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            tool = root / "build/npusim"
            tool.parent.mkdir(parents=True)
            tool.write_bytes(b"matching-npusim")
            runtime = root / "runtime"
            runtime.mkdir()
            with self.assertRaisesRegex(SchemaError, "target/configs are missing"):
                prepare_moe_swizzle_calibration_dramsys_runtime(
                    npusim=tool, runtime_root=runtime,
                )
            required = root / "DRAMSys/configs/hbm2-example.json"
            required.parent.mkdir(parents=True)
            with self.assertRaisesRegex(SchemaError, "target/configs are missing"):
                prepare_moe_swizzle_calibration_dramsys_runtime(
                    npusim=tool, runtime_root=runtime,
                )
            required.write_text("{}", encoding="utf-8")
            wrong = root / "wrong-DRAMSys"
            wrong.mkdir()
            link = runtime / "DRAMSys"
            link.symlink_to(wrong, target_is_directory=True)
            with self.assertRaisesRegex(SchemaError, "target exact matching"):
                prepare_moe_swizzle_calibration_dramsys_runtime(
                    npusim=tool, runtime_root=runtime,
                )
            link.unlink()
            observed = prepare_moe_swizzle_calibration_dramsys_runtime(
                npusim=tool, runtime_root=runtime,
            )
            self.assertTrue(observed.is_symlink())
            self.assertEqual(observed.resolve(), (root / "DRAMSys").resolve())
            self.assertEqual(
                prepare_moe_swizzle_calibration_dramsys_runtime(
                    npusim=tool, runtime_root=runtime,
                ),
                observed,
            )

    def test_executable_paths_must_be_absolute_regular_files(self) -> None:
        key = canonical_moe_swizzle_calibration_keys()[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = _Provider(root).materialize(key, root)
            executable.validate()
            for name in (
                "program", "linked_manifest", "program_io",
                "hardware_config", "simulation_config", "mapping_config",
            ):
                with self.assertRaisesRegex(
                    SchemaError, "absolute existing regular file"
                ):
                    replace(executable, **{name: Path(getattr(executable, name).name)}).validate()
            with self.assertRaisesRegex(
                SchemaError, "absolute existing regular file"
            ):
                replace(executable, program=root).validate()

    def test_timeout_preserves_partial_raw_and_is_typed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = _fake_npusim(root)

            def timeout(
                command: tuple[str, ...], _: Path, seconds: int
            ) -> tuple[int, str]:
                raise subprocess.TimeoutExpired(
                    command, seconds, output=b"partial runtime stdout\n"
                )

            with self.assertRaises(MoeSwizzleCalibrationStageFailure) as caught:
                run_moe_swizzle_isolated_calibration(
                    provider=_Provider(root / "provider"),
                    npusim=tool,
                    runtime_root=root / "runtime",
                    execute=timeout,
                )
            failure = caught.exception
            self.assertIs(
                failure.stage, MoeSwizzleCalibrationFailureStage.EXECUTE_TIMEOUT
            )
            self.assertEqual(failure.ordinal, 0)
            self.assertEqual(
                failure.key, canonical_moe_swizzle_calibration_keys()[0]
            )
            self.assertIsNone(failure.returncode)
            self.assertEqual(
                failure.raw_output_path.read_text(encoding="utf-8"),
                "partial runtime stdout\n",
            )

    def test_swiglu_artifact_contract_is_typed_and_failclosed(self) -> None:
        key = next(
            item for item in canonical_moe_swizzle_calibration_keys()
            if item.kind is MoeCalibrationKind.SWIGLU_GROUP
        )
        with tempfile.TemporaryDirectory() as temporary:
            executable = _Provider(Path(temporary)).materialize(key, Path(temporary))
            executable.validate()
            artifact = executable.swiglu_group_artifact
            self.assertIsNotNone(artifact)
            with self.assertRaisesRegex(SchemaError, "production artifact contract"):
                replace(executable, swiglu_group_artifact=None).validate()
            for tampered in (
                replace(artifact, swiglu_record_count=2),
                replace(artifact, input_bytes=artifact.input_bytes + 2),
                replace(artifact, output_bytes=artifact.output_bytes + 2),
                replace(artifact, dte_record_count=1),
                replace(artifact, endpoint_session_count=1),
            ):
                with self.assertRaisesRegex(SchemaError, "one SWIGLU record"):
                    replace(executable, swiglu_group_artifact=tampered).validate()

    def test_missing_dedicated_marker_fails_closed_at_first_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = _fake_npusim(root)
            with self.assertRaisesRegex(SchemaError, "exactly one sample"):
                run_moe_swizzle_isolated_calibration(
                    provider=_Provider(root / "provider"),
                    npusim=tool,
                    runtime_root=root / "runtime",
                    execute=lambda command, cwd, timeout: (0, "performance_cycle 7"),
                )


if __name__ == "__main__":
    unittest.main()
