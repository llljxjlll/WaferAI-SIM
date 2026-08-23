#!/usr/bin/env python3
"""Run the exact 168-sample MoE Swizzle isolated calibration matrix.

The runner never derives a primitive, byte count, FLOP count, or artifact.  A
production provider must materialize every isolated sample through the normal
frontend chain; this module only executes matching artifacts and parses the
dedicated runtime marker.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Callable, Protocol

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import SwiGluWorkload
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES,
    MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES,
    MoeCalibrationKind,
    MoeCalibrationStatus,
    MoeSwizzleCalibrationProfile,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration_program import (
    MoeSwizzleCalibrationStandardLinkedProgram,
)
from llm.test.frontend.integration.moe_swizzle_runtime_markers import (
    parse_moe_swizzle_calibration,
)


MOE_SWIZZLE_GROUP_GEMM_CALIBRATION_SHAPES = (
    MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES
)
MOE_SWIZZLE_SWIGLU_GROUP_CALIBRATION_SHAPES = (
    MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES
)


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationKey:
    kind: MoeCalibrationKind
    sample_index: int
    repeat_index: int
    shape: tuple[int, int, int] | None

    def validate(self, path: str = "moe_swizzle_calibration_key") -> None:
        if type(self.kind) is not MoeCalibrationKind:
            raise SchemaError("must use a typed kind", path=f"{path}.kind")
        if self.sample_index not in range(3) or self.repeat_index not in range(2):
            raise SchemaError("sample/repeat is outside exact 3x2 coverage", path=path)
        if self.kind is MoeCalibrationKind.GROUP_GEMM:
            if self.shape not in MOE_SWIZZLE_GROUP_GEMM_CALIBRATION_SHAPES:
                raise SchemaError("GroupGEMM shape is outside the frozen matrix", path=f"{path}.shape")
        elif self.kind is MoeCalibrationKind.SWIGLU_GROUP:
            if self.shape not in MOE_SWIZZLE_SWIGLU_GROUP_CALIBRATION_SHAPES:
                raise SchemaError("SWIGLU_GROUP shape is outside the frozen matrix", path=f"{path}.shape")
        elif self.shape is not None:
            raise SchemaError("fixed sample forbids shape", path=f"{path}.shape")


def canonical_moe_swizzle_calibration_keys() -> tuple[MoeSwizzleCalibrationKey, ...]:
    keys = tuple(
        MoeSwizzleCalibrationKey(MoeCalibrationKind.GROUP_GEMM, sample, repeat, shape)
        for shape in MOE_SWIZZLE_GROUP_GEMM_CALIBRATION_SHAPES
        for sample in range(3)
        for repeat in range(2)
    ) + tuple(
        MoeSwizzleCalibrationKey(MoeCalibrationKind.SWIGLU_GROUP, sample, repeat, shape)
        for shape in MOE_SWIZZLE_SWIGLU_GROUP_CALIBRATION_SHAPES
        for sample in range(3)
        for repeat in range(2)
    ) + tuple(
        MoeSwizzleCalibrationKey(kind, sample, repeat, None)
        for kind in MoeCalibrationKind
        if kind not in (MoeCalibrationKind.GROUP_GEMM, MoeCalibrationKind.SWIGLU_GROUP)
        for sample in range(3)
        for repeat in range(2)
    )
    if len(keys) != 168 or len(set(keys)) != 168:
        raise AssertionError("canonical MoE calibration matrix is not exact 168")
    return keys


@dataclass(frozen=True, slots=True)
class MoeSwizzleSwiGluCalibrationArtifact:
    workload: SwiGluWorkload
    swiglu_record_count: int
    input_bytes: int
    output_bytes: int
    dte_record_count: int
    endpoint_session_count: int

    def validate_against(
        self,
        key: MoeSwizzleCalibrationKey,
        path: str = "moe_swizzle_swiglu_calibration_artifact",
    ) -> None:
        if key.kind is not MoeCalibrationKind.SWIGLU_GROUP or key.shape is None:
            raise SchemaError("artifact requires a typed SWIGLU_GROUP key", path=path)
        key.validate(f"{path}.key")
        m, intermediate, flattened = key.shape
        if intermediate != 32 or flattened != m * intermediate:
            raise SchemaError("SWIGLU_GROUP shape must close N=M*I with I=32", path=f"{path}.key.shape")
        self.workload.validate(f"{path}.workload")
        expected_workload = SwiGluWorkload(
            (1, 2 * flattened),
            (1, flattened),
            (1, 2 * flattened),
            (1, flattened),
            DType.FP16,
        )
        if self.workload != expected_workload:
            raise SchemaError("SWIGLU_GROUP workload does not match the exact ISA shape", path=f"{path}.workload")
        for name in (
            "swiglu_record_count", "input_bytes", "output_bytes",
            "dte_record_count", "endpoint_session_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > (1 << 64) - 1:
                raise SchemaError("must fit uint64", path=f"{path}.{name}")
        if (
            self.swiglu_record_count != 1
            or self.input_bytes != 4 * flattened
            or self.output_bytes != 2 * flattened
            or self.dte_record_count != 0
            or self.endpoint_session_count != 0
        ):
            raise SchemaError(
                "isolated SWIGLU_GROUP requires one SWIGLU record, exact FP16 bytes, and no DTE/session",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationExecutable:
    key: MoeSwizzleCalibrationKey
    linked_program: MoeSwizzleCalibrationStandardLinkedProgram
    production_source_ref: str
    runtime_core: int
    program: Path
    linked_manifest: Path
    program_io: Path
    hardware_config: Path
    simulation_config: Path
    mapping_config: Path
    swiglu_group_artifact: MoeSwizzleSwiGluCalibrationArtifact | None = None

    def validate(self, path: str = "moe_swizzle_calibration_executable") -> None:
        self.key.validate(f"{path}.key")
        if type(self.linked_program) is not MoeSwizzleCalibrationStandardLinkedProgram:
            raise SchemaError(
                "requires an exact typed linked program",
                path=f"{path}.linked_program",
            )
        self.linked_program.validate(f"{path}.linked_program")
        if self.linked_program.program_io is None:
            raise SchemaError(
                "typed linked program requires actual-SHA ProgramIo",
                path=f"{path}.linked_program.program_io",
            )
        if type(self.production_source_ref) is not str or not self.production_source_ref:
            raise SchemaError("requires a production source ref", path=f"{path}.production_source_ref")
        if type(self.runtime_core) is not int or self.runtime_core < 0 or self.runtime_core > 65535:
            raise SchemaError("runtime core must fit uint16", path=f"{path}.runtime_core")
        for name in (
            "program", "linked_manifest", "program_io", "hardware_config",
            "simulation_config", "mapping_config",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Path)
                or not value.is_absolute()
                or not value.is_file()
            ):
                raise SchemaError(
                    "must be an absolute existing regular file",
                    path=f"{path}.{name}",
                )
        if (
            (self.linked_program.source.kind, self.linked_program.source.shape)
            != (self.key.kind, self.key.shape)
            or self.linked_program.source.target_runtime_core_id
            != self.runtime_core
            or self.production_source_ref != self.linked_program.id
            or self.linked_manifest.read_text(encoding="utf-8")
            != canonical_json(self.linked_program.manifest) + "\n"
            or self.program_io.read_text(encoding="utf-8")
            != canonical_json(self.linked_program.program_io) + "\n"
            or hashlib.sha256(self.program.read_bytes()).hexdigest()
            != self.linked_program.program_io.program_artifact_sha256
        ):
            raise SchemaError(
                "typed linked program/path/SHA provenance is not exact",
                path=path,
            )
        if self.key.kind is MoeCalibrationKind.SWIGLU_GROUP:
            if self.swiglu_group_artifact is None:
                raise SchemaError(
                    "SWIGLU_GROUP requires a production artifact contract",
                    path=f"{path}.swiglu_group_artifact",
                )
            self.swiglu_group_artifact.validate_against(
                self.key, f"{path}.swiglu_group_artifact"
            )
        elif self.swiglu_group_artifact is not None:
            raise SchemaError(
                "non-SWIGLU calibration forbids a SWIGLU artifact contract",
                path=f"{path}.swiglu_group_artifact",
            )


class MoeSwizzleCalibrationProvider(Protocol):
    def materialize(
        self, key: MoeSwizzleCalibrationKey, output_root: Path
    ) -> MoeSwizzleCalibrationExecutable: ...


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationArtifactEvidence:
    key: MoeSwizzleCalibrationKey
    production_source_ref: str
    program_sha256: str
    linked_manifest_sha256: str
    program_io_sha256: str

    def validate(
        self, path: str = "moe_swizzle_calibration_artifact_evidence"
    ) -> None:
        self.key.validate(f"{path}.key")
        if (
            type(self.production_source_ref) is not str
            or not self.production_source_ref
        ):
            raise SchemaError(
                "requires a production source ref",
                path=f"{path}.production_source_ref",
            )
        for name in (
            "program_sha256", "linked_manifest_sha256", "program_io_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise SchemaError(
                    "must be lowercase SHA-256", path=f"{path}.{name}"
                )


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationRunEvidence:
    profile: MoeSwizzleCalibrationProfile
    matching_tool_sha256: str
    hardware_sha256: str
    simulation_sha256: str
    mapping_sha256: str
    run_count: int
    raw_output_paths: tuple[Path, ...]
    artifacts: tuple[MoeSwizzleCalibrationArtifactEvidence, ...]

    def validate(self, path: str = "moe_swizzle_calibration_run_evidence") -> None:
        self.profile.validate(f"{path}.profile")
        if self.profile.status is not MoeCalibrationStatus.MEASURED:
            raise SchemaError("actual runner requires a complete MEASURED profile", path=f"{path}.profile.status")
        if (
            self.run_count != 168
            or len(self.raw_output_paths) != 168
            or len(self.artifacts) != 168
        ):
            raise SchemaError("actual runner requires exact 168 executions", path=path)
        for name in (
            "matching_tool_sha256", "hardware_sha256", "simulation_sha256", "mapping_sha256",
        ):
            value = getattr(self, name)
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise SchemaError("must be lowercase SHA-256", path=f"{path}.{name}")
        if any(not item.is_file() for item in self.raw_output_paths):
            raise SchemaError("raw output evidence is missing", path=f"{path}.raw_output_paths")
        canonical_keys = canonical_moe_swizzle_calibration_keys()
        if tuple(item.key for item in self.artifacts) != canonical_keys:
            raise SchemaError(
                "artifact evidence must follow the exact canonical key order",
                path=f"{path}.artifacts",
            )
        invariant: dict[
            tuple[MoeCalibrationKind, tuple[int, int, int] | None],
            tuple[str, str, str, str],
        ] = {}
        for index, item in enumerate(self.artifacts):
            item.validate(f"{path}.artifacts[{index}]")
            family = (item.key.kind, item.key.shape)
            witness = (
                item.production_source_ref,
                item.program_sha256,
                item.linked_manifest_sha256,
                item.program_io_sha256,
            )
            previous = invariant.setdefault(family, witness)
            if witness != previous:
                raise SchemaError(
                    "artifact/source provenance drifted across exact 3x2 repeats",
                    path=f"{path}.artifacts[{index}]",
                )


class MoeSwizzleCalibrationFailureStage(str, Enum):
    EXECUTE_EXIT = "execute_exit"
    EXECUTE_TIMEOUT = "execute_timeout"


class MoeSwizzleCalibrationStageFailure(RuntimeError):
    def __init__(
        self,
        *,
        stage: MoeSwizzleCalibrationFailureStage,
        ordinal: int,
        key: MoeSwizzleCalibrationKey,
        raw_output_path: Path,
        returncode: int | None,
    ) -> None:
        self.stage = stage
        self.ordinal = ordinal
        self.key = key
        self.raw_output_path = raw_output_path
        self.returncode = returncode
        super().__init__(
            f"first calibration failure stage={stage.value} ordinal={ordinal} "
            f"key={key} returncode={returncode} raw={raw_output_path}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _run_subprocess(command: tuple[str, ...], cwd: Path, timeout: int) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout


def prepare_moe_swizzle_calibration_dramsys_runtime(
    *, npusim: Path, runtime_root: Path
) -> Path:
    """Close DRAMSys' fixed ``../DRAMSys`` lookup under isolated sample cwd."""

    target = (npusim.parent.parent / "DRAMSys").resolve()
    configs = target / "configs"
    required = configs / "hbm2-example.json"
    if not target.is_dir() or not configs.is_dir() or not required.is_file():
        raise SchemaError(
            "matching npusim DRAMSys target/configs are missing",
            path="moe_swizzle_calibration_runner.dramsys",
        )
    link = runtime_root / "DRAMSys"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != target:
            raise SchemaError(
                "DRAMSys link must target exact matching npusim tree",
                path="moe_swizzle_calibration_runner.dramsys",
            )
    else:
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as error:
            raise SchemaError(
                "could not create matching DRAMSys runtime link",
                path="moe_swizzle_calibration_runner.dramsys",
            ) from error
    if (
        not link.is_symlink()
        or link.resolve() != target
        or not (link / "configs/hbm2-example.json").is_file()
    ):
        raise SchemaError(
            "DRAMSys runtime link did not close exact matching configs",
            path="moe_swizzle_calibration_runner.dramsys",
        )
    return link


@dataclass(frozen=True, slots=True)
class _MoeSwizzleCalibrationExecution:
    ordinal: int
    key: MoeSwizzleCalibrationKey
    output: str
    raw_output_path: Path
    artifact: MoeSwizzleCalibrationArtifactEvidence
    config_digests: tuple[str, str, str]


def _execute_moe_swizzle_calibration_sample(
    *,
    ordinal: int,
    key: MoeSwizzleCalibrationKey,
    provider: MoeSwizzleCalibrationProvider,
    npusim: Path,
    runtime_root: Path,
    timeout_seconds: int,
    execute: Callable[[tuple[str, ...], Path, int], tuple[int, str]],
    tool_digest: str,
) -> _MoeSwizzleCalibrationExecution:
    sample_root = runtime_root / f"sample-{ordinal:03d}-{key.kind.value}"
    sample_root.mkdir(parents=False, exist_ok=False)
    executable = provider.materialize(key, sample_root)
    executable.validate(f"calibration.executables[{ordinal}]")
    if executable.key != key:
        raise SchemaError(
            "provider returned the wrong canonical key",
            path=f"calibration.executables[{ordinal}].key",
        )
    config_digests = tuple(
        _sha256(path)
        for path in (
            executable.hardware_config,
            executable.simulation_config,
            executable.mapping_config,
        )
    )
    hardware_sha, simulation_sha, mapping_sha = config_digests
    artifact = MoeSwizzleCalibrationArtifactEvidence(
        key,
        executable.production_source_ref,
        _sha256(executable.program),
        _sha256(executable.linked_manifest),
        _sha256(executable.program_io),
    )
    artifact.validate(f"calibration.artifacts[{ordinal}]")
    shape = "none" if key.shape is None else "x".join(
        str(item) for item in key.shape
    )
    command = (
        str(npusim),
        f"--program={executable.program}",
        f"--linked-manifest={executable.linked_manifest}",
        f"--program-io={executable.program_io}",
        f"--hardware-config={executable.hardware_config}",
        f"--simulation-config={executable.simulation_config}",
        f"--mapping-config={executable.mapping_config}",
        "--moe-swizzle-runtime-markers",
        f"--moe-swizzle-calibration-kind={key.kind.value}",
        f"--moe-swizzle-calibration-core={executable.runtime_core}",
        f"--moe-swizzle-calibration-sample={key.sample_index}",
        f"--moe-swizzle-calibration-repeat={key.repeat_index}",
        f"--moe-swizzle-calibration-shape={shape}",
        f"--moe-swizzle-calibration-tool-sha256={tool_digest}",
        f"--moe-swizzle-calibration-hardware-sha256={hardware_sha}",
        f"--moe-swizzle-calibration-simulation-sha256={simulation_sha}",
        f"--moe-swizzle-calibration-mapping-sha256={mapping_sha}",
    )
    raw_path = sample_root / "raw.stdout"
    try:
        returncode, output = execute(command, sample_root, timeout_seconds)
    except subprocess.TimeoutExpired as error:
        partial = error.stdout if error.stdout is not None else error.output
        if partial is None:
            output = ""
        elif isinstance(partial, bytes):
            output = partial.decode("utf-8", errors="replace")
        else:
            output = partial
        raw_path.write_text(output, encoding="utf-8")
        raise MoeSwizzleCalibrationStageFailure(
            stage=MoeSwizzleCalibrationFailureStage.EXECUTE_TIMEOUT,
            ordinal=ordinal,
            key=key,
            raw_output_path=raw_path,
            returncode=None,
        ) from error
    raw_path.write_text(output, encoding="utf-8")
    if returncode != 0:
        raise MoeSwizzleCalibrationStageFailure(
            stage=MoeSwizzleCalibrationFailureStage.EXECUTE_EXIT,
            ordinal=ordinal,
            key=key,
            raw_output_path=raw_path,
            returncode=returncode,
        )
    one = parse_moe_swizzle_calibration(
        output,
        tool_sha256=tool_digest,
        hardware_sha256=hardware_sha,
        simulation_sha256=simulation_sha,
        mapping_sha256=mapping_sha,
    )
    if len(one.samples) != 1:
        raise SchemaError(
            "each isolated execution must emit exactly one sample",
            path=f"calibration.outputs[{ordinal}]",
        )
    sample = one.samples[0]
    if (sample.kind, sample.sample_index, sample.repeat_index, sample.shape) != (
        key.kind, key.sample_index, key.repeat_index, key.shape,
    ):
        raise SchemaError(
            "dedicated marker key drifted from provider key",
            path=f"calibration.outputs[{ordinal}]",
        )
    return _MoeSwizzleCalibrationExecution(
        ordinal, key, output, raw_path, artifact, config_digests
    )


def run_moe_swizzle_isolated_calibration(
    *,
    provider: MoeSwizzleCalibrationProvider,
    npusim: Path,
    runtime_root: Path,
    timeout_seconds: int = 300,
    worker_count: int = 1,
    execute: Callable[[tuple[str, ...], Path, int], tuple[int, str]] = _run_subprocess,
) -> MoeSwizzleCalibrationRunEvidence:
    if (
        not isinstance(npusim, Path)
        or not npusim.is_absolute()
        or not npusim.is_file()
        or type(timeout_seconds) is not int
        or timeout_seconds <= 0
    ):
        raise SchemaError(
            "npusim must be an absolute regular file and timeout positive",
            path="moe_swizzle_calibration_runner",
        )
    if (
        type(worker_count) is not int
        or worker_count <= 0
        or worker_count > 64
    ):
        raise SchemaError(
            "worker count must be in [1,64]",
            path="moe_swizzle_calibration_runner.worker_count",
        )
    if not isinstance(runtime_root, Path) or not runtime_root.is_absolute():
        raise SchemaError(
            "runtime root must be an absolute directory",
            path="moe_swizzle_calibration_runner.runtime_root",
        )
    try:
        runtime_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SchemaError(
            "runtime root must be an absolute directory",
            path="moe_swizzle_calibration_runner.runtime_root",
        ) from error
    if not runtime_root.is_dir() or any(runtime_root.iterdir()):
        raise SchemaError(
            "runtime root must be an empty absolute directory",
            path="moe_swizzle_calibration_runner.runtime_root",
        )
    prepare_moe_swizzle_calibration_dramsys_runtime(
        npusim=npusim, runtime_root=runtime_root,
    )

    tool_digest = _sha256(npusim)
    keys = canonical_moe_swizzle_calibration_keys()

    def one(ordinal: int) -> _MoeSwizzleCalibrationExecution:
        return _execute_moe_swizzle_calibration_sample(
            ordinal=ordinal,
            key=keys[ordinal],
            provider=provider,
            npusim=npusim,
            runtime_root=runtime_root,
            timeout_seconds=timeout_seconds,
            execute=execute,
            tool_digest=tool_digest,
        )

    if worker_count == 1:
        completed = tuple(one(ordinal) for ordinal in range(len(keys)))
    else:
        results: dict[int, _MoeSwizzleCalibrationExecution] = {}
        failures: dict[int, BaseException] = {}
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="moe-swizzle-calibration",
        ) as executor:
            futures = {
                executor.submit(one, ordinal): ordinal
                for ordinal in range(len(keys))
            }
            for future in as_completed(futures):
                ordinal = futures[future]
                try:
                    results[ordinal] = future.result()
                except Exception as error:
                    failures[ordinal] = error
        if failures:
            raise failures[min(failures)]
        completed = tuple(results[ordinal] for ordinal in range(len(keys)))

    expected_configs = completed[0].config_digests
    expected_artifacts: dict[
        tuple[MoeCalibrationKind, tuple[int, int, int] | None],
        tuple[str, str, str, str],
    ] = {}
    for item in completed:
        if item.config_digests != expected_configs:
            raise SchemaError(
                "calibration config SHA drifted between samples",
                path=f"calibration.executables[{item.ordinal}]",
            )
        family = (item.key.kind, item.key.shape)
        witness = (
            item.artifact.production_source_ref,
            item.artifact.program_sha256,
            item.artifact.linked_manifest_sha256,
            item.artifact.program_io_sha256,
        )
        previous = expected_artifacts.setdefault(family, witness)
        if witness != previous:
            raise SchemaError(
                "artifact/source provenance drifted across exact 3x2 repeats",
                path=f"calibration.artifacts[{item.ordinal}]",
            )

    hardware_sha, simulation_sha, mapping_sha = expected_configs
    profile = parse_moe_swizzle_calibration(
        "\n".join(item.output for item in completed),
        tool_sha256=tool_digest,
        hardware_sha256=hardware_sha,
        simulation_sha256=simulation_sha,
        mapping_sha256=mapping_sha,
    )
    result = MoeSwizzleCalibrationRunEvidence(
        profile,
        tool_digest,
        hardware_sha,
        simulation_sha,
        mapping_sha,
        len(completed),
        tuple(item.raw_output_path for item in completed),
        tuple(item.artifact for item in completed),
    )
    result.validate()
    return result


__all__ = [
    "MOE_SWIZZLE_GROUP_GEMM_CALIBRATION_SHAPES",
    "MOE_SWIZZLE_SWIGLU_GROUP_CALIBRATION_SHAPES",
    "MoeSwizzleCalibrationArtifactEvidence",
    "MoeSwizzleCalibrationExecutable",
    "MoeSwizzleCalibrationFailureStage",
    "MoeSwizzleCalibrationKey",
    "MoeSwizzleCalibrationProvider",
    "MoeSwizzleCalibrationRunEvidence",
    "MoeSwizzleCalibrationStageFailure",
    "MoeSwizzleSwiGluCalibrationArtifact",
    "canonical_moe_swizzle_calibration_keys",
    "prepare_moe_swizzle_calibration_dramsys_runtime",
    "run_moe_swizzle_isolated_calibration",
]
