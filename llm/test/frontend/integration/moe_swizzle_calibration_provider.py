"""Production artifact-family provider for actual MoE Swizzle calibration.

The provider owns no byte/FLOP derivation.  Kind builders must lower and
finalize one isolated production artifact family, including its linked
manifest and actual-SHA ProgramIo sidecar.  The provider caches the 28 frozen
families and projects their immutable paths onto the exact 3x2 run keys.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
import hashlib
import subprocess
from threading import Lock
from typing import Mapping, Protocol

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.moe_swizzle_calibration_standard import (
    build_moe_swizzle_calibration_standard_linked_program,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_calibration_program_io import (
    build_moe_swizzle_calibration_program_io,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import SwiGluWorkload
from llm.frontend.wafer_frontend.schema.serde import canonical_json, load_json_value
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MoeCalibrationKind,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration_program import (
    MoeSwizzleCalibrationStandardLinkedProgram,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_standard import (
    MoeSwizzleStandardLinkedProgram,
)
from llm.test.frontend.integration.run_moe_swizzle_calibration import (
    MoeSwizzleCalibrationExecutable,
    MoeSwizzleCalibrationKey,
    MoeSwizzleSwiGluCalibrationArtifact,
    canonical_moe_swizzle_calibration_keys,
)


_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
MOE_SWIZZLE_CALIBRATION_SIMULATION_CONFIG = (
    _WORKSPACE_ROOT / "llm/test/sram/simulation.json"
)


def _resolve_runtime_config_reference(
    raw: object, *, runtime_reference_root: Path, path: str
) -> str:
    if type(raw) is not str or not raw:
        raise SchemaError("must be a non-empty path", path=path)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = runtime_reference_root / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise SchemaError(
            f"runtime reference does not resolve to a file: {resolved}",
            path=path,
        )
    return str(resolved)


def materialize_moe_swizzle_calibration_runtime_configs(
    *,
    output_root: Path,
    hardware_config: Path,
    simulation_config: Path,
    runtime_reference_root: Path,
) -> tuple[Path, Path]:
    """Write one canonical cwd-independent hardware/simulation config pair."""

    for name, value, directory in (
        ("output_root", output_root, True),
        ("hardware_config", hardware_config, False),
        ("simulation_config", simulation_config, False),
        ("runtime_reference_root", runtime_reference_root, True),
    ):
        if (
            not isinstance(value, Path)
            or not value.is_absolute()
            or (directory and not value.is_dir())
            or (not directory and not value.is_file())
        ):
            raise SchemaError(
                "requires an absolute existing directory"
                if directory else "requires an absolute existing file",
                path=f"moe_swizzle_calibration_runtime_configs.{name}",
            )
    hardware = load_json_value(
        hardware_config,
        path="moe_swizzle_calibration_runtime_configs.hardware",
    )
    simulation = load_json_value(
        simulation_config,
        path="moe_swizzle_calibration_runtime_configs.simulation",
    )
    if type(hardware) is not dict or type(simulation) is not dict:
        raise SchemaError(
            "hardware/simulation inputs must be JSON objects",
            path="moe_swizzle_calibration_runtime_configs",
        )
    gpu = simulation.get("gpu")
    if type(gpu) is not dict:
        raise SchemaError(
            "requires a gpu object",
            path="moe_swizzle_calibration_runtime_configs.simulation.gpu",
        )
    gpu["dram_config_file"] = _resolve_runtime_config_reference(
        gpu.get("dram_config_file"),
        runtime_reference_root=runtime_reference_root,
        path=(
            "moe_swizzle_calibration_runtime_configs"
            ".simulation.gpu.dram_config_file"
        ),
    )
    memory_system = hardware.get("memory_system")
    stacks = (
        memory_system.get("hbm_stacks")
        if type(memory_system) is dict else None
    )
    if type(stacks) is not list or len(stacks) != 4:
        raise SchemaError(
            "C2 calibration requires exactly four HBM stacks",
            path=(
                "moe_swizzle_calibration_runtime_configs"
                ".hardware.memory_system.hbm_stacks"
            ),
        )
    for index, stack in enumerate(stacks):
        if type(stack) is not dict:
            raise SchemaError(
                "must be an object",
                path=(
                    "moe_swizzle_calibration_runtime_configs"
                    f".hardware.memory_system.hbm_stacks[{index}]"
                ),
            )
        stack["channel_dram_config"] = _resolve_runtime_config_reference(
            stack.get("channel_dram_config"),
            runtime_reference_root=runtime_reference_root,
            path=(
                "moe_swizzle_calibration_runtime_configs.hardware"
                f".memory_system.hbm_stacks[{index}].channel_dram_config"
            ),
        )
    runtime_inputs = output_root / "runtime_inputs"
    runtime_inputs.mkdir(parents=False, exist_ok=False)
    hardware_output = runtime_inputs / "hardware.json"
    simulation_output = runtime_inputs / "simulation.json"
    hardware_output.write_text(canonical_json(hardware) + "\n", encoding="utf-8")
    simulation_output.write_text(canonical_json(simulation) + "\n", encoding="utf-8")
    return hardware_output, simulation_output


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationFamilyKey:
    kind: MoeCalibrationKind
    shape: tuple[int, int, int] | None

    def validate(
        self, path: str = "moe_swizzle_calibration_family_key"
    ) -> None:
        if type(self.kind) is not MoeCalibrationKind:
            raise SchemaError("must use a typed kind", path=f"{path}.kind")
        MoeSwizzleCalibrationKey(self.kind, 0, 0, self.shape).validate(path)


def canonical_moe_swizzle_calibration_family_keys(
) -> tuple[MoeSwizzleCalibrationFamilyKey, ...]:
    seen: dict[
        tuple[MoeCalibrationKind, tuple[int, int, int] | None],
        MoeSwizzleCalibrationFamilyKey,
    ] = {}
    for key in canonical_moe_swizzle_calibration_keys():
        family = MoeSwizzleCalibrationFamilyKey(key.kind, key.shape)
        seen.setdefault((family.kind, family.shape), family)
    result = tuple(seen.values())
    if (
        len(result) != 28
        or {item.kind for item in result} != set(MoeCalibrationKind)
    ):
        raise AssertionError("production calibration family matrix is not exact 28")
    return result


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationArtifactFamily:
    family: MoeSwizzleCalibrationFamilyKey
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

    def validate(
        self, path: str = "moe_swizzle_calibration_artifact_family"
    ) -> None:
        self.family.validate(f"{path}.family")
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
        if (
            type(self.production_source_ref) is not str
            or not self.production_source_ref
        ):
            raise SchemaError(
                "requires a production source ref",
                path=f"{path}.production_source_ref",
            )
        if (
            type(self.runtime_core) is not int
            or self.runtime_core < 0
            or self.runtime_core > 65535
        ):
            raise SchemaError(
                "runtime core must fit uint16", path=f"{path}.runtime_core"
            )
        for name in (
            "program",
            "linked_manifest",
            "program_io",
            "hardware_config",
            "simulation_config",
            "mapping_config",
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
            != (self.family.kind, self.family.shape)
            or self.production_source_ref != self.linked_program.id
            or self.runtime_core
            != self.linked_program.source.target_runtime_core_id
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
        representative = MoeSwizzleCalibrationKey(
            self.family.kind, 0, 0, self.family.shape
        )
        if self.family.kind is MoeCalibrationKind.SWIGLU_GROUP:
            if self.swiglu_group_artifact is None:
                raise SchemaError(
                    "SWIGLU_GROUP family requires its typed production audit",
                    path=f"{path}.swiglu_group_artifact",
                )
            self.swiglu_group_artifact.validate_against(
                representative, f"{path}.swiglu_group_artifact"
            )
        elif self.swiglu_group_artifact is not None:
            raise SchemaError(
                "non-SWIGLU family forbids a SWIGLU audit",
                path=f"{path}.swiglu_group_artifact",
            )

    def executable_for(
        self,
        key: MoeSwizzleCalibrationKey,
        path: str = "moe_swizzle_calibration_artifact_family",
    ) -> MoeSwizzleCalibrationExecutable:
        self.validate(path)
        key.validate(f"{path}.key")
        if (key.kind, key.shape) != (self.family.kind, self.family.shape):
            raise SchemaError(
                "requested run key belongs to another artifact family",
                path=f"{path}.key",
            )
        result = MoeSwizzleCalibrationExecutable(
            key=key,
            linked_program=self.linked_program,
            production_source_ref=self.production_source_ref,
            runtime_core=self.runtime_core,
            program=self.program,
            linked_manifest=self.linked_manifest,
            program_io=self.program_io,
            hardware_config=self.hardware_config,
            simulation_config=self.simulation_config,
            mapping_config=self.mapping_config,
            swiglu_group_artifact=self.swiglu_group_artifact,
        )
        result.validate(f"{path}.executable")
        return result


class MoeSwizzleCalibrationKindBuilder(Protocol):
    def build_family(
        self,
        family: MoeSwizzleCalibrationFamilyKey,
        output_root: Path,
    ) -> MoeSwizzleCalibrationArtifactFamily: ...


@dataclass(frozen=True, slots=True)
class ProductionMoeSwizzleCalibrationArtifactBuilder:
    """Finalize one cached family and rebuild ProgramIo with its actual SHA."""

    source: MoeSwizzleStandardLinkedProgram
    finalizer: Path
    hardware_config: Path
    simulation_config: Path
    mapping_config: Path

    def _validate(self) -> None:
        if type(self.source) is not MoeSwizzleStandardLinkedProgram:
            raise SchemaError(
                "requires a frozen whole MoE source",
                path="moe_swizzle_calibration_artifact_builder.source",
            )
        for name in (
            "finalizer",
            "hardware_config",
            "simulation_config",
            "mapping_config",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Path)
                or not value.is_absolute()
                or not value.is_file()
            ):
                raise SchemaError(
                    "requires an absolute existing file",
                    path=f"moe_swizzle_calibration_artifact_builder.{name}",
                )

    @staticmethod
    def _write(path: Path, value: object) -> None:
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")

    def build_family(
        self,
        family: MoeSwizzleCalibrationFamilyKey,
        output_root: Path,
    ) -> MoeSwizzleCalibrationArtifactFamily:
        self._validate()
        family.validate("moe_swizzle_calibration_artifact_builder.family")
        if not output_root.is_absolute() or not output_root.is_dir():
            raise SchemaError(
                "output root must be an existing absolute directory",
                path="moe_swizzle_calibration_artifact_builder.output_root",
            )
        linked = build_moe_swizzle_calibration_standard_linked_program(
            self.source, family.kind, family.shape,
        )
        manifest_path = output_root / "linked_manifest.json"
        wrapper_path = output_root / "linked_wrapper.json"
        artifact_path = output_root / "program.npup"
        report_path = output_root / "finalizer.json"
        program_io_path = output_root / "program_io.json"
        self._write(manifest_path, linked.manifest)
        self._write(wrapper_path, linked)
        completed = subprocess.run(
            (
                str(self.finalizer),
                "--input", str(manifest_path),
                "--output", str(artifact_path),
                "--report", str(report_path),
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not artifact_path.is_file():
            raise SchemaError(
                "production finalizer rejected the isolated calibration manifest: "
                + completed.stderr[-1000:],
                path="moe_swizzle_calibration_artifact_builder.finalizer",
            )
        artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        program_io = build_moe_swizzle_calibration_program_io(
            linked, artifact_sha,
        )
        exact = MoeSwizzleCalibrationStandardLinkedProgram.create(
            source=linked.source,
            fragment=linked.fragment,
            manifest=linked.manifest,
            program_io=program_io,
        )
        self._write(program_io_path, program_io)
        self._write(output_root / "finalized_wrapper.json", exact)
        swiglu = None
        if family.kind is MoeCalibrationKind.SWIGLU_GROUP:
            assert family.shape is not None
            _m, _intermediate, flattened = family.shape
            swiglu = MoeSwizzleSwiGluCalibrationArtifact(
                workload=SwiGluWorkload(
                    (1, 2 * flattened),
                    (1, flattened),
                    (1, 2 * flattened),
                    (1, flattened),
                    DType.FP16,
                ),
                swiglu_record_count=1,
                input_bytes=4 * flattened,
                output_bytes=2 * flattened,
                dte_record_count=0,
                endpoint_session_count=0,
            )
        result = MoeSwizzleCalibrationArtifactFamily(
            family=family,
            linked_program=exact,
            production_source_ref=exact.id,
            runtime_core=linked.source.target_runtime_core_id,
            program=artifact_path,
            linked_manifest=manifest_path,
            program_io=program_io_path,
            hardware_config=self.hardware_config,
            simulation_config=self.simulation_config,
            mapping_config=self.mapping_config,
            swiglu_group_artifact=swiglu,
        )
        result.validate("moe_swizzle_calibration_artifact_builder.result")
        return result


def build_production_moe_swizzle_calibration_provider(
    *,
    artifact_root: Path,
    source: MoeSwizzleStandardLinkedProgram,
    finalizer: Path,
    hardware_config: Path,
    mapping_config: Path,
    simulation_config: Path = MOE_SWIZZLE_CALIBRATION_SIMULATION_CONFIG,
) -> "ProductionMoeSwizzleCalibrationProvider":
    """Create the exact registry with the official LiteMoE HBM2+DTE config."""

    if not artifact_root.is_absolute():
        raise SchemaError(
            "artifact root must be absolute",
            path="moe_swizzle_calibration_provider.artifact_root",
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    hardware_config, simulation_config = (
        materialize_moe_swizzle_calibration_runtime_configs(
            output_root=artifact_root,
            hardware_config=hardware_config,
            simulation_config=simulation_config,
            runtime_reference_root=finalizer.parent,
        )
    )
    builder = ProductionMoeSwizzleCalibrationArtifactBuilder(
        source,
        finalizer,
        hardware_config,
        simulation_config,
        mapping_config,
    )
    return ProductionMoeSwizzleCalibrationProvider(
        artifact_root=artifact_root,
        builders={kind: builder for kind in MoeCalibrationKind},
    )


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationKindBinding:
    kind: MoeCalibrationKind
    builder: MoeSwizzleCalibrationKindBuilder

    def validate(
        self, path: str = "moe_swizzle_calibration_kind_binding"
    ) -> None:
        if type(self.kind) is not MoeCalibrationKind:
            raise SchemaError("must use a typed kind", path=f"{path}.kind")
        if not callable(getattr(self.builder, "build_family", None)):
            raise SchemaError(
                "builder must implement build_family",
                path=f"{path}.builder",
            )


class ProductionMoeSwizzleCalibrationProvider:
    def __init__(
        self,
        *,
        artifact_root: Path,
        builders: Mapping[
            MoeCalibrationKind, MoeSwizzleCalibrationKindBuilder
        ],
    ) -> None:
        if (
            not isinstance(artifact_root, Path)
            or not artifact_root.is_absolute()
        ):
            raise SchemaError(
                "artifact root must be absolute",
                path="moe_swizzle_calibration_provider.artifact_root",
            )
        if set(builders) != set(MoeCalibrationKind):
            raise SchemaError(
                "provider requires exactly all 14 kind builders",
                path="moe_swizzle_calibration_provider.builders",
            )
        self._artifact_root = artifact_root
        self._bindings = tuple(
            MoeSwizzleCalibrationKindBinding(kind, builders[kind])
            for kind in MoeCalibrationKind
        )
        for index, binding in enumerate(self._bindings):
            binding.validate(
                f"moe_swizzle_calibration_provider.builders[{index}]"
            )
        self._families = canonical_moe_swizzle_calibration_family_keys()
        self._family_ordinals = {
            family: index for index, family in enumerate(self._families)
        }
        self._cache: dict[
            MoeSwizzleCalibrationFamilyKey,
            Future[MoeSwizzleCalibrationArtifactFamily],
        ] = {}
        self._lock = Lock()
        try:
            self._artifact_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SchemaError(
                "artifact root must be a writable directory",
                path="moe_swizzle_calibration_provider.artifact_root",
            ) from error
        if not self._artifact_root.is_dir():
            raise SchemaError(
                "artifact root must be a directory",
                path="moe_swizzle_calibration_provider.artifact_root",
            )

    @property
    def bindings(self) -> tuple[MoeSwizzleCalibrationKindBinding, ...]:
        return self._bindings

    def _build_or_wait(
        self, family: MoeSwizzleCalibrationFamilyKey
    ) -> MoeSwizzleCalibrationArtifactFamily:
        with self._lock:
            future = self._cache.get(family)
            owner = future is None
            if future is None:
                future = Future()
                self._cache[family] = future
        if owner:
            try:
                ordinal = self._family_ordinals[family]
                suffix = (
                    "none"
                    if family.shape is None
                    else "x".join(str(item) for item in family.shape)
                )
                family_root = self._artifact_root / (
                    f"family-{ordinal:02d}-{family.kind.value}-{suffix}"
                )
                family_root.mkdir(parents=False, exist_ok=False)
                built = self._bindings[
                    tuple(MoeCalibrationKind).index(family.kind)
                ].builder.build_family(family, family_root)
                if type(built) is not MoeSwizzleCalibrationArtifactFamily:
                    raise SchemaError(
                        "kind builder returned the wrong typed family",
                        path="moe_swizzle_calibration_provider.builder",
                    )
                built.validate(
                    f"moe_swizzle_calibration_provider.families[{ordinal}]"
                )
                if built.family != family:
                    raise SchemaError(
                        "kind builder returned another family",
                        path=(
                            "moe_swizzle_calibration_provider"
                            f".families[{ordinal}].family"
                        ),
                    )
                future.set_result(built)
            except Exception as error:
                future.set_exception(error)
        return future.result()

    def materialize(
        self, key: MoeSwizzleCalibrationKey, output_root: Path
    ) -> MoeSwizzleCalibrationExecutable:
        key.validate("moe_swizzle_calibration_provider.key")
        if (
            not isinstance(output_root, Path)
            or not output_root.is_absolute()
            or not output_root.is_dir()
        ):
            raise SchemaError(
                "sample output root must be an absolute directory",
                path="moe_swizzle_calibration_provider.output_root",
            )
        family = MoeSwizzleCalibrationFamilyKey(key.kind, key.shape)
        built = self._build_or_wait(family)
        return built.executable_for(
            key, "moe_swizzle_calibration_provider.family"
        )


__all__ = [
    "MoeSwizzleCalibrationArtifactFamily",
    "MoeSwizzleCalibrationFamilyKey",
    "MoeSwizzleCalibrationKindBinding",
    "MoeSwizzleCalibrationKindBuilder",
    "MOE_SWIZZLE_CALIBRATION_SIMULATION_CONFIG",
    "ProductionMoeSwizzleCalibrationArtifactBuilder",
    "ProductionMoeSwizzleCalibrationProvider",
    "build_production_moe_swizzle_calibration_provider",
    "canonical_moe_swizzle_calibration_family_keys",
    "materialize_moe_swizzle_calibration_runtime_configs",
]
