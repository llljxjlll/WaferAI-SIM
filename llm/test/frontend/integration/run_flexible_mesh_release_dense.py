"""Run all or an exact mesh subset of the Flexible-Dense release matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import LinkedProgramManifest
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseBinding,
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseCaseEvidence,
    FlexibleMeshReleaseFamily,
    FlexibleMeshReleaseTool,
    FlexibleMeshReleaseToolKind,
)
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)

from flexible_mesh_release_dense import FlexibleDenseReleaseAdapter
from flexible_mesh_release_profiles import release_trace_model_digest
from run_flexible_mesh_release import FlexibleMeshReleaseRunner


_ROOT = Path(__file__).resolve().parents[4]
_DENSE_FAMILY = FlexibleMeshReleaseFamily.DENSE_TRAIN


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


class _UnavailableAdapter:
    def __init__(self, family: FlexibleMeshReleaseFamily) -> None:
        self.family = family


def _cases(args: argparse.Namespace) -> tuple[FlexibleMeshReleaseCase, ...]:
    return tuple(
        FlexibleMeshReleaseCase.create(
            family=_DENSE_FAMILY,
            mesh=RectMeshSpec(rows=rows, columns=columns),
            trace_model_digest=release_trace_model_digest(_DENSE_FAMILY),
            runtime_profile_version=args.runtime_profile_version,
        )
        for rows in range(1, 11)
        for columns in range(1, 11)
    )


def _parse_mesh(value: str) -> tuple[int, int]:
    row_text, separator, column_text = value.partition("x")
    try:
        rows = int(row_text)
        columns = int(column_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("mesh must be canonical HxW") from error
    if (
        separator != "x"
        or value != f"{rows}x{columns}"
        or not 1 <= rows <= 10
        or not 1 <= columns <= 10
    ):
        raise argparse.ArgumentTypeError("mesh must be canonical HxW with 1<=H,W<=10")
    return rows, columns


def _selected_cases(args: argparse.Namespace) -> tuple[FlexibleMeshReleaseCase, ...]:
    cases = _cases(args)
    mesh_filter = getattr(args, "mesh", None)
    if mesh_filter:
        selected_meshes = set(mesh_filter)
        cases = tuple(
            case for case in cases
            if (case.mesh.rows, case.mesh.columns) in selected_meshes
        )
    return tuple(
        case for index, case in enumerate(cases)
        if index % args.shard_count == args.shard_index
    )


def _checkpoint_name(args: argparse.Namespace) -> str:
    mesh_filter = getattr(args, "mesh", None)
    scope = ""
    if mesh_filter:
        canonical_meshes = tuple(sorted(set(mesh_filter)))
        scope = f"mesh_filter_{canonical_digest(canonical_meshes)[:12]}_"
    return (
        f"dense_release_{scope}shard_{args.shard_index}_of_{args.shard_count}.json"
    )


def _write_or_validate_binding(
    path: Path, binding: FlexibleMeshReleaseBinding,
) -> None:
    if path.is_file():
        cached = loads_dataclass(
            FlexibleMeshReleaseBinding,
            path.read_text(encoding="utf-8"),
            path="release_binding",
        )
        if cached != binding:
            raise SchemaError("release binding drifted", path="release_binding")
        return
    path.write_text(canonical_json(binding), encoding="utf-8")


def _tool(
    kind: FlexibleMeshReleaseToolKind,
    path: Path,
    version: str,
) -> FlexibleMeshReleaseTool:
    resolved = path.resolve()
    digest = _sha_file(resolved)
    return FlexibleMeshReleaseTool(
        kind=kind,
        binary_path=str(resolved),
        version=version,
        sha256=digest,
        allowlisted_sha256=(digest,),
    )


def _binding(args: argparse.Namespace) -> FlexibleMeshReleaseBinding:
    return FlexibleMeshReleaseBinding.create(
        runtime_profile_version=args.runtime_profile_version,
        environment_profile_version=args.environment_profile_version,
        tools=tuple(
            _tool(kind, path, args.tool_version)
            for kind, path in zip(
                FlexibleMeshReleaseToolKind,
                (args.finalizer, args.resolver, args.npusim),
            )
        ),
        hardware_config_sha256=_sha_file(args.hardware_template),
        simulation_config_sha256=_sha_file(args.simulation),
        mapping_config_sha256=_sha_file(args.mapping),
    )


def _validate_cached(
    path: Path,
    *,
    case: FlexibleMeshReleaseCase,
    binding: FlexibleMeshReleaseBinding,
    adapter: FlexibleDenseReleaseAdapter,
) -> FlexibleMeshReleaseCaseEvidence:
    evidence = loads_dataclass(
        FlexibleMeshReleaseCaseEvidence,
        path.read_text(encoding="utf-8"),
        path="case_evidence",
    )
    evidence.validate_against(binding)
    if evidence.case != case:
        raise SchemaError("cached case drifted", path="case_evidence.case")

    materialized = adapter.materialize(case)
    expected_hardware_sha = adapter.expected_hardware_sha256(case)
    if adapter.mapping_sha256 != binding.mapping_config_sha256:
        raise SchemaError("mapping profile drifted", path="binding.mapping_config_sha256")
    directory = path.parent
    for execution in evidence.executions:
        if (
            execution.spec_digest != materialized.spec_digest
            or execution.plan_digest != materialized.plan_digest
            or execution.manifest_digest != canonical_digest(materialized.manifest)
            or execution.hardware_config_sha256 != expected_hardware_sha
            or execution.mapping_config_sha256 != adapter.mapping_sha256
        ):
            raise SchemaError("cached materialization drifted", path="case_evidence.executions")
        run = directory / f"execution_{execution.execution_index}"
        artifact = (run / "program.npup").read_bytes()
        if (
            _sha_bytes(artifact) != execution.artifact_sha256
            or len(artifact) != execution.artifact_file_bytes
            or _sha_file(run / "hardware.json") != expected_hardware_sha
            or _sha_file(run / "mapping.spec") != adapter.mapping_sha256
        ):
            raise SchemaError("cached runtime inputs drifted", path="case_evidence.executions")
        manifest_bytes = (run / "linked.json").read_bytes()
        expected_manifest_bytes = canonical_json(materialized.manifest).encode("utf-8")
        if (
            manifest_bytes != expected_manifest_bytes
            or len(manifest_bytes) != execution.capacity.linked_manifest_file_bytes
        ):
            raise SchemaError("cached manifest bytes drifted", path="case_evidence.executions")
        manifest = loads_dataclass(
            LinkedProgramManifest,
            manifest_bytes.decode("utf-8"),
            path="linked_manifest",
        )
        if manifest != materialized.manifest:
            raise SchemaError("cached manifest drifted", path="case_evidence.executions")
        contract = loads_dataclass(
            ProgramIoContract,
            (run / "program_io.json").read_text(encoding="utf-8"),
            path="program_io",
        )
        contract.validate_against(manifest)
        expected_contract = adapter.build_program_io(
            materialized, execution.artifact_sha256,
        )
        if (
            contract != expected_contract
            or canonical_digest(contract) != execution.program_io_digest
        ):
            raise SchemaError("cached ProgramIO drifted", path="case_evidence.executions")
        raw_execution = loads_dataclass(
            type(execution),
            (run / "execution_evidence.json").read_text(encoding="utf-8"),
            path="execution_evidence",
        )
        if raw_execution != execution:
            raise SchemaError("cached execution evidence drifted", path="case_evidence.executions")
        resolver_output = (run / "resolver.stdout.txt").read_text(encoding="utf-8")
        semantic_resolver = resolver_output.replace(str(run.resolve()), "<EXECUTION>")
        if _sha_bytes(semantic_resolver.encode("utf-8")) != execution.resolver_digest:
            raise SchemaError("cached resolver stdout drifted", path="case_evidence.executions")
        observation = adapter.observe(
            case,
            materialized,
            contract,
            execution.artifact_sha256,
            (run / "npusim.stdout.txt").read_text(encoding="utf-8"),
        )
        if (
            observation.makespan_cycles != execution.makespan_cycles
            or observation.marker_digest != execution.marker_digest
            or observation.residual != execution.residual
            or observation.rank_coverage != execution.rank_coverage
            or observation.core_coverage != execution.core_coverage
            or observation.completion_markers != execution.completion_markers
        ):
            raise SchemaError("cached observation drifted", path="case_evidence.executions")
        report = json.loads((run / "finalizer.json").read_text(encoding="utf-8"))
        if (
            report.get("artifact_sha256") != execution.artifact_sha256
            or report.get("artifact_bytes") != execution.artifact_file_bytes
            or report.get("record_count") != execution.capacity.exact_record_count
            or report.get("linked_manifest_id") != materialized.manifest.id
            or report.get("linked_manifest_digest") != canonical_digest(materialized.manifest)
        ):
            raise SchemaError("cached finalizer report drifted", path="case_evidence.executions")
    return evidence


def run(args: argparse.Namespace) -> tuple[FlexibleMeshReleaseCaseEvidence, ...]:
    binding = _binding(args)
    mapping_text = args.mapping.read_text(encoding="utf-8")
    hardware_template_json = args.hardware_template.read_text(encoding="utf-8")
    dense_adapter = FlexibleDenseReleaseAdapter(
        hardware_template_json=hardware_template_json,
        mapping_text=mapping_text,
    )
    if dense_adapter.mapping_sha256 != binding.mapping_config_sha256:
        raise SchemaError("mapping profile drifted", path="binding.mapping_config_sha256")
    adapters = tuple(
        dense_adapter if family is _DENSE_FAMILY else _UnavailableAdapter(family)
        for family in FlexibleMeshReleaseFamily
    )
    runner = FlexibleMeshReleaseRunner(
        binding=binding,
        adapters=adapters,
        finalizer=args.finalizer,
        resolver=args.resolver,
        npusim=args.npusim,
        simulation=args.simulation,
        runtime_root=args.runtime_root,
        timeout=args.timeout,
    )
    selected = _selected_cases(args)
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    _write_or_validate_binding(
        args.runtime_root / "release_binding.json", binding,
    )
    checkpoint = (
        args.runtime_root
        / _checkpoint_name(args)
    )
    results: list[FlexibleMeshReleaseCaseEvidence] = []
    for ordinal, case in enumerate(selected, start=1):
        evidence_path = args.runtime_root / case.id / "case_evidence.json"
        evidence = None
        if args.resume and evidence_path.is_file():
            try:
                evidence = _validate_cached(
                    evidence_path,
                    case=case,
                    binding=binding,
                    adapter=dense_adapter,
                )
                print(
                    f"[{ordinal}/{len(selected)}] resume {case.family.value} "
                    f"{case.mesh.rows}x{case.mesh.columns}",
                    flush=True,
                )
            except (SchemaError, OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                print(
                    f"[{ordinal}/{len(selected)}] cached evidence rejected: {error}",
                    flush=True,
                )
        if evidence is None:
            print(
                f"[{ordinal}/{len(selected)}] run {case.family.value} "
                f"{case.mesh.rows}x{case.mesh.columns}",
                flush=True,
            )
            evidence = runner.run_case(case)
        results.append(evidence)
        checkpoint.write_text(canonical_json(tuple(results)), encoding="utf-8")
    print(
        f"complete shard={args.shard_index}/{args.shard_count} cases={len(results)} "
        f"executions={2 * len(results)} binding={binding.id}",
        flush=True,
    )
    return tuple(results)


def _parse_args() -> argparse.Namespace:
    build = _ROOT / "build-release-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mesh",
        action="append",
        type=_parse_mesh,
        help="run only this exact HxW mesh; repeat for multiple meshes",
    )
    parser.add_argument(
        "--runtime-root", type=Path,
        default=build / "flexible-dense-release-matrix-unified",
    )
    parser.add_argument("--finalizer", type=Path, default=build / "npusim_program_finalizer")
    parser.add_argument("--resolver", type=Path, default=build / "npusim_program_io_selftest")
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument(
        "--hardware-template", type=Path,
        default=_ROOT / "llm/test/program/p5_large_hardware.json",
    )
    parser.add_argument(
        "--simulation", type=Path,
        default=_ROOT / "llm/test/program/p5_behavioral_simulation.json",
    )
    parser.add_argument(
        "--mapping", type=Path,
        default=_ROOT / "llm/test/default/mapping.spec",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--runtime-profile-version", default="flexible-mesh-timing-v3-one-shot")
    parser.add_argument("--environment-profile-version", default="p5-rect-release-v1")
    parser.add_argument("--tool-version", default="build-release-final")
    args = parser.parse_args()
    if (
        args.shard_count <= 0
        or not 0 <= args.shard_index < args.shard_count
        or args.timeout <= 0
    ):
        parser.error("invalid shard or timeout")
    if args.mesh and len(args.mesh) != len(set(args.mesh)):
        parser.error("--mesh values must be unique")
    for name in (
        "finalizer", "resolver", "npusim", "hardware_template", "simulation", "mapping",
    ):
        if not getattr(args, name).is_file():
            parser.error(f"--{name.replace('_', '-')} is not a file")
    return args


if __name__ == "__main__":
    run(_parse_args())
