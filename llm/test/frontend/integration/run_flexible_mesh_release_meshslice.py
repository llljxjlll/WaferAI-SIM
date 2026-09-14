"""Run all three strict MeshSlice release families in resumable shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseCaseEvidence,
    FlexibleMeshReleaseFamily,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json

from flexible_mesh_release_meshslice import FlexibleMeshSliceReleaseAdapter
from flexible_mesh_release_profiles import release_trace_model_digest
from run_flexible_mesh_release import FlexibleMeshReleaseRunner
from run_flexible_mesh_release_dense import (
    _binding,
    _UnavailableAdapter,
    _validate_cached,
    _write_or_validate_binding,
)


_ROOT = Path(__file__).resolve().parents[4]
_FAMILIES = (
    FlexibleMeshReleaseFamily.MESHSLICE_AG,
    FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK,
    FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK,
)
_FAMILY_SELECTION = {
    "all": _FAMILIES,
    "ag": (FlexibleMeshReleaseFamily.MESHSLICE_AG,),
    "rs": (FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK,),
    "ar": (FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK,),
}


def _families(args: argparse.Namespace) -> tuple[FlexibleMeshReleaseFamily, ...]:
    family = getattr(args, "family", "all")
    try:
        return _FAMILY_SELECTION[family]
    except KeyError as error:
        raise SchemaError("unsupported MeshSlice family", path="family") from error


def _cases(args: argparse.Namespace) -> tuple[FlexibleMeshReleaseCase, ...]:
    return tuple(
        FlexibleMeshReleaseCase.create(
            family=family,
            mesh=RectMeshSpec(rows=rows, columns=columns),
            trace_model_digest=release_trace_model_digest(family),
            runtime_profile_version=args.runtime_profile_version,
        )
        for family in _families(args)
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
        raise argparse.ArgumentTypeError(
            "mesh must be canonical HxW with 1<=H,W<=10"
        )
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
        f"meshslice_release_{args.family}_{scope}"
        f"shard_{args.shard_index}_of_{args.shard_count}.json"
    )


def run(args: argparse.Namespace) -> tuple[FlexibleMeshReleaseCaseEvidence, ...]:
    binding = _binding(args)
    mapping_text = args.mapping.read_text(encoding="utf-8")
    hardware_template_json = args.hardware_template.read_text(encoding="utf-8")
    meshslice = {
        family: FlexibleMeshSliceReleaseAdapter(
            family,
            mapping_text=mapping_text,
            hardware_template_json=hardware_template_json,
        )
        for family in _families(args)
    }
    for adapter in meshslice.values():
        if adapter.mapping_sha256 != binding.mapping_config_sha256:
            raise SchemaError(
                "mapping profile drifted",
                path="binding.mapping_config_sha256",
            )
    adapters = tuple(
        meshslice.get(family, _UnavailableAdapter(family))
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
    checkpoint = args.runtime_root / _checkpoint_name(args)
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
                    adapter=meshslice[case.family],
                )
                print(
                    f"[{ordinal}/{len(selected)}] resume {case.family.value} "
                    f"{case.mesh.rows}x{case.mesh.columns}",
                    flush=True,
                )
            except (
                SchemaError, OSError, ValueError, TypeError,
                json.JSONDecodeError,
            ) as error:
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
        checkpoint.write_text(
            canonical_json(tuple(results)), encoding="utf-8"
        )
    print(
        f"complete shard={args.shard_index}/{args.shard_count} "
        f"cases={len(results)} executions={2 * len(results)} "
        f"binding={binding.id}",
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
        "--family", choices=tuple(_FAMILY_SELECTION), default="all"
    )
    parser.add_argument(
        "--mesh",
        action="append",
        type=_parse_mesh,
        help="run only this exact HxW mesh; repeat for multiple meshes",
    )
    parser.add_argument(
        "--runtime-root", type=Path,
        default=build / "flexible-meshslice-release-matrix-unified",
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
        "finalizer", "resolver", "npusim", "hardware_template",
        "simulation", "mapping",
    ):
        if not getattr(args, name).is_file():
            parser.error(f"--{name.replace('_', '-')} is not a file")
    return args


if __name__ == "__main__":
    run(_parse_args())
