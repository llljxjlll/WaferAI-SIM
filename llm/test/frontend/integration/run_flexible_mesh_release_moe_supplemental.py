"""Run independent, timing-only Flexible-MoE supplemental trace evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseCaseEvidence,
    FlexibleMeshReleaseFamily,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_json

from flexible_mesh_release_moe import FlexibleMoeReleaseAdapter
from flexible_mesh_release_supplemental_coverage import (
    supplemental_coverage_catalog,
)
from flexible_moe_supplemental_profiles import (
    FlexibleMoeSupplementalTrace,
    REPRESENTATIVE_SHAPES,
    SUPPLEMENTAL_PROFILES,
    supplemental_profile,
    trusted_spec_builders,
)
from run_flexible_mesh_release import FlexibleMeshReleaseRunner
from run_flexible_mesh_release_dense import _write_or_validate_binding
from run_flexible_mesh_release_moe import (
    _MOE_FAMILIES,
    _UnavailableAdapter,
    _binding,
    _validate_cached,
)


_ROOT = Path(__file__).resolve().parents[4]


def _cases(args: argparse.Namespace) -> tuple[FlexibleMeshReleaseCase, ...]:
    families = {
        "inference": _MOE_FAMILIES[:1],
        "train": _MOE_FAMILIES[1:],
        "both": _MOE_FAMILIES,
    }[args.family]
    traces = (
        tuple(FlexibleMoeSupplementalTrace)
        if args.trace == "all"
        else (
            FlexibleMoeSupplementalTrace.ALL_LOCAL,
            FlexibleMoeSupplementalTrace.HOT_EMPTY,
        )
        if args.trace == "required"
        else (FlexibleMoeSupplementalTrace(args.trace),)
    )
    if any(shape not in REPRESENTATIVE_SHAPES for shape in args.shapes):
        raise SchemaError("shape is not in the representative allowlist", path="shapes")
    return tuple(
        FlexibleMeshReleaseCase.create(
            family=family,
            mesh=RectMeshSpec(rows, columns),
            trace_model_digest=supplemental_profile(family, trace).digest,
            runtime_profile_version=args.runtime_profile_version,
        )
        for family in families
        for trace in traces
        for rows, columns in args.shapes
    )


def _supplemental_report(
    results: tuple[FlexibleMeshReleaseCaseEvidence, ...],
    *,
    binding,
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    executions = tuple(
        execution for evidence in results for execution in evidence.executions
    )
    used_digests = {item.case.trace_model_digest for item in results}
    profiles = tuple(
        {
            "family": profile.family.value,
            "trace": profile.trace.value,
            "digest": profile.digest,
        }
        for profile in SUPPLEMENTAL_PROFILES
        if profile.digest in used_digests
    )
    capacity_fields = (
        tuple(executions[0].capacity.__dataclass_fields__) if executions else ()
    )
    residual_fields = (
        tuple(executions[0].residual.__dataclass_fields__) if executions else ()
    )
    return {
        "schema_version": "wafer_frontend.flexible_moe_supplemental_report/v1alpha1",
        "primary_completion_eligible": False,
        "timing_only": True,
        "binding_id": binding.id,
        "binding_digest": binding.digest,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "case_count": len(results),
        "execution_count": 2 * len(results),
        "failed_case_ids": [],
        "runtime_verified_case_ids": [item.case.id for item in results],
        "repeatability_verified_case_ids": [item.case.id for item in results],
        "profiles": profiles,
        "planned_coverage": supplemental_coverage_catalog(),
        "selected_families": sorted({item.case.family.value for item in results}),
        "selected_traces": sorted({profile["trace"] for profile in profiles}),
        "shapes": sorted({
            (item.case.mesh.rows, item.case.mesh.columns) for item in results
        }),
        "capacity_max": {
            name: max(getattr(item.capacity, name) for item in executions)
            for name in capacity_fields
        },
        "residual_max": {
            name: max(getattr(item.residual, name) for item in executions)
            for name in residual_fields
        },
        "makespan_cycles_by_case": {
            item.case.id: item.executions[0].makespan_cycles for item in results
        },
    }


def run(args: argparse.Namespace) -> tuple[FlexibleMeshReleaseCaseEvidence, ...]:
    binding = _binding(args)
    mapping_text = args.mapping.read_text(encoding="utf-8")
    hardware_template_json = args.hardware_template.read_text(encoding="utf-8")
    moe_adapters = {
        family: FlexibleMoeReleaseAdapter(
            family,
            hardware_template_json=hardware_template_json,
            mapping_text=mapping_text,
            trusted_spec_builders=trusted_spec_builders(family),
        )
        for family in _MOE_FAMILIES
    }
    runner = FlexibleMeshReleaseRunner(
        binding=binding,
        adapters=tuple(
            moe_adapters.get(family, _UnavailableAdapter(family))
            for family in FlexibleMeshReleaseFamily
        ),
        finalizer=args.finalizer,
        resolver=args.resolver,
        npusim=args.npusim,
        simulation=args.simulation,
        runtime_root=args.runtime_root,
        timeout=args.timeout,
    )
    selected = tuple(
        case for index, case in enumerate(_cases(args))
        if index % args.shard_count == args.shard_index
    )
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    _write_or_validate_binding(args.runtime_root / "release_binding.json", binding)
    results = []
    for ordinal, case in enumerate(selected, start=1):
        evidence_path = args.runtime_root / case.id / "case_evidence.json"
        evidence = None
        if args.resume and evidence_path.is_file():
            evidence = _validate_cached(
                evidence_path,
                case=case,
                binding=binding,
                adapter=moe_adapters[case.family],
            )
            print(f"[{ordinal}/{len(selected)}] resume {case.family.value} {case.mesh.rows}x{case.mesh.columns}", flush=True)
        if evidence is None:
            print(f"[{ordinal}/{len(selected)}] run {case.family.value} {case.mesh.rows}x{case.mesh.columns}", flush=True)
            evidence = runner.run_case(case)
        results.append(evidence)
    report = _supplemental_report(
        tuple(results),
        binding=binding,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    report_path = args.runtime_root / f"moe_supplemental_report_{args.shard_index}_of_{args.shard_count}.json"
    report_path.write_text(canonical_json(report), encoding="utf-8")
    print(f"complete supplemental cases={len(results)} executions={2 * len(results)} report={report_path}", flush=True)
    return tuple(results)


def _parse_shapes(value: str) -> tuple[tuple[int, int], ...]:
    try:
        result = tuple(tuple(map(int, item.split("x"))) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shapes must use ROWSxCOLUMNS") from error
    if not result or any(len(item) != 2 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("shapes must be non-empty and unique")
    if any(item not in REPRESENTATIVE_SHAPES for item in result):
        raise argparse.ArgumentTypeError("shape is not in the representative allowlist")
    return result


def _parse_args() -> argparse.Namespace:
    build = _ROOT / "build-release-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("inference", "train", "both"), default="both")
    parser.add_argument(
        "--trace",
        choices=(
            "required",
            "all",
            *(item.value for item in FlexibleMoeSupplementalTrace),
        ),
        default="required",
        help=(
            "required runs all-local and hot-empty; balanced is already in "
            "the primary matrix"
        ),
    )
    parser.add_argument(
        "--shapes",
        type=_parse_shapes,
        default=REPRESENTATIVE_SHAPES,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=build / "flexible-moe-supplemental")
    parser.add_argument("--finalizer", type=Path, default=build / "npusim_program_finalizer")
    parser.add_argument("--resolver", type=Path, default=build / "npusim_program_io_selftest")
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument("--hardware-template", type=Path, default=_ROOT / "llm/test/program/p5_large_hardware.json")
    parser.add_argument("--simulation", type=Path, default=_ROOT / "llm/test/program/p5_behavioral_simulation.json")
    parser.add_argument("--mapping", type=Path, default=_ROOT / "llm/test/default/mapping.spec")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--runtime-profile-version", default="flexible-mesh-timing-v3-one-shot")
    parser.add_argument("--environment-profile-version", default="p5-rect-release-v1")
    parser.add_argument("--tool-version", default="build-release-final")
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count or args.timeout <= 0:
        parser.error("invalid shard or timeout")
    for name in ("finalizer", "resolver", "npusim", "hardware_template", "simulation", "mapping"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name.replace('_', '-')} is not a file")
    return args


if __name__ == "__main__":
    run(_parse_args())
