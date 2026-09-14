#!/usr/bin/env python3
"""exp1-1 case table and analytical lower-bound calculations.

The simulator invocation is deliberately kept separate from this small, importable
section.  ``iter_cases()`` is the single source of truth for the 176 logical cases.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Callable, Iterable, Literal, Mapping


Operator = Literal["AG_GEMM", "GEMM_RS"]
Layer = Literal["attention", "mlp"]

DTYPE_BYTES = 2
# plan.md specifies 16 cores/die and approximately 8 TFLOP/s per core.
DEFAULT_DIE_CORES = 16
DEFAULT_DIE_FLOPS_PER_SECOND = DEFAULT_DIE_CORES * 8e12
DEFAULT_EFFECTIVE_BANDWIDTH_BYTES_PER_SECOND = 256e9


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    hidden_size: int
    intermediate_size: int
    num_heads: int | None
    num_kv_heads: int | None
    head_dim: int | None

    @property
    def qkv_width(self) -> int | None:
        if (
            self.num_heads is None
            or self.num_kv_heads is None
            or self.head_dim is None
        ):
            return None
        return (self.num_heads + 2 * self.num_kv_heads) * self.head_dim

    @property
    def layers(self) -> tuple[Layer, ...]:
        # DeepSeek-V3 uses MLA; plan.md therefore excludes its attention layer.
        return ("mlp",) if self.qkv_width is None else ("attention", "mlp")


@dataclass(frozen=True, slots=True)
class MeshSpec:
    name: str
    rows: int
    columns: int
    sequence_lengths: tuple[int, int]

    @property
    def dies(self) -> int:
        return self.rows * self.columns

    @property
    def orientations(self) -> tuple[tuple[int, int], ...]:
        """Legal (Px, Py) logical grids, including a physical-axis transpose."""
        normal = (self.rows, self.columns)
        transpose = (self.columns, self.rows)
        return (normal,) if normal == transpose else (normal, transpose)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("LLaMA-2-7B", 4096, 11008, 32, 32, 128),
    ModelSpec("GPT-3-175B", 12288, 49152, 96, 96, 128),
    ModelSpec("LLaMA-3-8B", 4096, 14336, 32, 8, 128),
    ModelSpec("LLaMA-3.1-405B", 16384, 53248, 128, 8, 128),
    ModelSpec("Mixtral-8x7B-single-expert", 4096, 14336, 32, 8, 128),
    ModelSpec("DeepSeek-V3-single-routed-expert", 7168, 2048, None, None, None),
)

MESHES: tuple[MeshSpec, ...] = (
    MeshSpec("1x4", 1, 4, (2048, 32768)),
    MeshSpec("2x3", 2, 3, (2048, 32768)),
    MeshSpec("3x3", 3, 3, (2048, 32768)),
    MeshSpec("6x6", 6, 6, (36864, 147456)),
)

OPERATORS: tuple[Operator, ...] = ("AG_GEMM", "GEMM_RS")
BRANCHES = ("swizzle", "naive")


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    mesh: MeshSpec
    operator: Operator
    model: ModelSpec
    layer: Layer
    seq_len: int
    # These are the rank-local GEMM dimensions stated in plan.md.
    m: int
    n: int
    k: int

    @property
    def case_id(self) -> str:
        model = self.model.name.lower().replace(".", "_").replace("-", "_")
        return f"{self.mesh.name}_{self.operator.lower()}_{model}_{self.layer}_s{self.seq_len}"

    @property
    def logical_mnk(self) -> tuple[int, int, int]:
        """Return the unsharded GEMM dimensions used by plan.md's formulas.

        plan.md's shape tables list rank-local N (AG) or K (RS), while its
        theoretical equations divide an unsharded N or K by D.  Keeping both
        representations avoids dividing twice.  O-proj intentionally follows
        the current table: its unsharded K is the complete QKV width, not H.
        """
        if self.layer == "mlp":
            if self.operator == "AG_GEMM":
                return (self.seq_len, self.model.intermediate_size, self.model.hidden_size)
            return (self.seq_len, self.model.hidden_size, self.model.intermediate_size)

        qkv = self.model.qkv_width
        if qkv is None:  # guarded by iter_cases(), retained for direct construction
            raise ValueError(f"{self.model.name} has no attention shape")
        if self.operator == "AG_GEMM":
            return (self.seq_len, qkv, self.model.hidden_size)
        return (self.seq_len, self.model.hidden_size, qkv)

    @property
    def runtime_mnk(self) -> tuple[int, int, int]:
        """Return the padded unsharded GEMM actually sent to the frontend."""
        logical_m, logical_n, logical_k = self.logical_mnk
        d = self.mesh.dies
        align = lambda value, multiple=d: (
            (value + multiple - 1) // multiple
        ) * multiple
        hidden_alignment = math.lcm(d, 16)
        runtime_m = align(logical_m)
        if self.operator == "AG_GEMM":
            # SwizzleScalePoint encodes this projection width as 2*I, so I is
            # padded to TP and the resulting GEMM N has granularity 2*TP.
            runtime_n = 2 * align((logical_n + 1) // 2)
            runtime_k = align(logical_k, hidden_alignment)
        else:
            runtime_n = align(logical_n, hidden_alignment)
            runtime_k = align(logical_k)
        return runtime_m, runtime_n, runtime_k

    @property
    def flops(self) -> int:
        """Logical (all-die) GEMM FLOPs, used for plotted performance."""
        m, n, k = self.logical_mnk
        return 2 * m * n * k

    @property
    def runtime_flops(self) -> int:
        m, n, k = self.runtime_mnk
        return 2 * m * n * k

    @property
    def rank_flops(self) -> int:
        return 2 * self.m * self.n * self.k

    @property
    def collective_bytes(self) -> float:
        """Per-rank 1D-ring transferred bytes from the plan's model."""
        m, n, k = self.logical_mnk
        d = self.mesh.dies
        tensor_elements = m * k if self.operator == "AG_GEMM" else m * n
        return (d - 1) / d * tensor_elements * DTYPE_BYTES


def make_case(
    mesh: MeshSpec,
    operator: Operator,
    model: ModelSpec,
    layer: Layer,
    seq_len: int,
) -> ExperimentCase:
    """Build exactly the rank-local shape written in plan.md.

    Rank-local dimensions are derived from the TP-padded runtime shape.  The
    original dimensions remain available through ``logical_mnk`` for theory
    and effective-performance reporting.
    """
    if layer == "attention" and model.qkv_width is None:
        raise ValueError(f"{model.name} uses MLA and has no attention case")
    shell = ExperimentCase(mesh, operator, model, layer, seq_len, 0, 0, 0)
    runtime_m, runtime_n, runtime_k = shell.runtime_mnk
    d = mesh.dies
    if operator == "AG_GEMM":
        m, n, k = runtime_m, runtime_n // d, runtime_k
    else:
        m, n, k = runtime_m, runtime_n, runtime_k // d
    return ExperimentCase(mesh, operator, model, layer, seq_len, m, n, k)


def iter_cases() -> Iterable[ExperimentCase]:
    for mesh in MESHES:
        for operator in OPERATORS:
            for model in MODELS:
                for layer in model.layers:
                    for seq_len in mesh.sequence_lengths:
                        yield make_case(mesh, operator, model, layer, seq_len)


@dataclass(frozen=True, slots=True)
class AlgorithmTheory:
    algorithm: str
    px: int | None
    py: int | None
    compute_seconds: float
    communication_seconds: float

    @property
    def time_seconds(self) -> float:
        return max(self.compute_seconds, self.communication_seconds)


@dataclass(frozen=True, slots=True)
class TheoryResult:
    one_d: AlgorithmTheory
    two_d_candidates: tuple[AlgorithmTheory, ...]
    selected: AlgorithmTheory

    @property
    def time_seconds(self) -> float:
        return self.selected.time_seconds


def calculate_theory(
    case: ExperimentCase,
    *,
    die_flops_per_second: float = DEFAULT_DIE_FLOPS_PER_SECOND,
    effective_bandwidth_bytes_per_second: float = DEFAULT_EFFECTIVE_BANDWIDTH_BYTES_PER_SECOND,
    dtype_bytes: int = DTYPE_BYTES,
    runtime_mnk: tuple[int, int, int] | None = None,
) -> TheoryResult:
    """Evaluate the 1D and both legal 2D orientations from plan.md.

    runtime_mnk lets callers evaluate the bound on the exact padded work
    executed by the simulator.  The default preserves the logical-shape
    calculation used by the standalone case-table interface.
    """
    if die_flops_per_second <= 0 or effective_bandwidth_bytes_per_second <= 0:
        raise ValueError("compute rate and bandwidth must be positive")
    m, n, k = case.logical_mnk if runtime_mnk is None else runtime_mnk
    if min(m, n, k) <= 0:
        raise ValueError("runtime dimensions must be positive")
    d = case.mesh.dies
    rate = die_flops_per_second
    bandwidth = effective_bandwidth_bytes_per_second

    compute_1d = 2 * m * n * k / d / rate
    if case.operator == "AG_GEMM":
        communication_1d = (d - 1) / d * m * k * dtype_bytes / bandwidth
    else:
        communication_1d = (d - 1) / d * m * n * dtype_bytes / bandwidth
    one_d = AlgorithmTheory("1d_ring", None, None, compute_1d, communication_1d)

    two_d: list[AlgorithmTheory] = []
    for px, py in case.mesh.orientations:
        if px * py != d:
            raise AssertionError("logical grid must contain every die")
        compute_2d = 2 * (m / px) * (n / py) * k / rate
        if case.operator == "AG_GEMM":
            communication_2d = max(
                m * k * dtype_bytes / (px * bandwidth),
                k * n * dtype_bytes / (py * bandwidth),
            )
        else:
            communication_2d = (
                (px - 1) / px * m * (n / py) * dtype_bytes / bandwidth
            )
        two_d.append(
            AlgorithmTheory(
                "2d_row_column", px, py, compute_2d, communication_2d
            )
        )

    # A lower elapsed time is a higher performance upper bound.  This corrects
    # plan.md's final textual "max(T_1D,T_2D)" while preserving every row formula.
    selected = min((one_d, *two_d), key=lambda item: item.time_seconds)
    return TheoryResult(one_d, tuple(two_d), selected)


def _case_record(
    case: ExperimentCase,
    *,
    runtime_mnk: tuple[int, int, int] | None = None,
) -> dict[str, object]:
    runtime_mnk = case.runtime_mnk if runtime_mnk is None else runtime_mnk
    theory = calculate_theory(case, runtime_mnk=runtime_mnk)
    logical_m, logical_n, logical_k = case.logical_mnk
    runtime_m, runtime_n, runtime_k = runtime_mnk
    if case.operator == "AG_GEMM":
        runtime_tensor_elements = runtime_m * runtime_k
    else:
        runtime_tensor_elements = runtime_m * runtime_n
    runtime_collective_bytes = (
        (case.mesh.dies - 1) / case.mesh.dies
        * runtime_tensor_elements
        * DTYPE_BYTES
    )
    return {
        "case_id": case.case_id,
        "mesh": case.mesh.name,
        "operator": case.operator,
        "model": case.model.name,
        "layer": case.layer,
        "seq_len": case.seq_len,
        "M": case.m,
        "N": case.n,
        "K": case.k,
        "logical_M": logical_m,
        "logical_N": logical_n,
        "logical_K": logical_k,
        "runtime_M": runtime_m,
        "runtime_N": runtime_n,
        "runtime_K": runtime_k,
        "padding_M": runtime_m - logical_m,
        "padding_N": runtime_n - logical_n,
        "padding_K": runtime_k - logical_k,
        "flops": case.flops,
        "logical_flops": case.flops,
        "runtime_flops": case.runtime_flops,
        "runtime_flops_over_logical_flops": case.runtime_flops / case.flops,
        "rank_flops": case.rank_flops,
        "collective_bytes": case.collective_bytes,
        "logical_collective_bytes": case.collective_bytes,
        "runtime_collective_bytes": runtime_collective_bytes,
        "theory_time": theory.time_seconds,
        "theory_naive_time": (
            DEFAULT_DIE_CORES * theory.one_d.compute_seconds
            + theory.one_d.communication_seconds
        ),
        "theory_speedup": (
            DEFAULT_DIE_CORES * theory.one_d.compute_seconds
            + theory.one_d.communication_seconds
        ) / theory.time_seconds,
        "theory_algorithm": theory.selected.algorithm,
        "theory_Px": theory.selected.px,
        "theory_Py": theory.selected.py,
    }


def _self_check(cases: tuple[ExperimentCase, ...]) -> None:
    if len(cases) != 176:
        raise AssertionError(f"expected 176 logical cases, got {len(cases)}")
    if len({case.case_id for case in cases}) != len(cases):
        raise AssertionError("case_id values are not unique")
    expected_per_mesh_operator = 22
    for mesh in MESHES:
        for operator in OPERATORS:
            count = sum(
                case.mesh == mesh and case.operator == operator for case in cases
            )
            if count != expected_per_mesh_operator:
                raise AssertionError(
                    f"{mesh.name}/{operator}: expected 22 cases, got {count}"
                )



def _smoke_cases(cases: tuple[ExperimentCase, ...]) -> tuple[ExperimentCase, ...]:
    """Select the plan's small/short and large/long case per operator."""
    points = (("1x4", 2048), ("6x6", 147456))
    return tuple(
        next(
            case for case in cases
            if case.mesh.name == mesh
            and case.operator == operator
            and case.model == MODELS[0]
            and case.layer == "attention"
            and case.seq_len == seq_len
        )
        for operator in OPERATORS
        for mesh, seq_len in points
    )


RuntimeCall = Callable[[ExperimentCase, str, float], object]


def _load_runtime_adapter() -> tuple[RuntimeCall | None, str | None]:
    """Import the optional sibling adapter only for actual run modes."""
    module_names = ["runtime_adapter"]
    if __package__:
        module_names.append(f"{__package__}.runtime_adapter")
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                return None, f"runtime adapter dependency is missing: {exc}"
            continue
        except Exception as exc:
            return None, f"runtime adapter could not be imported: {exc}"
        run_branch = getattr(module, "run_branch", None)
        if callable(run_branch):
            return run_branch, None
        return None, "runtime_adapter must define run_branch(case, branch, timeout)"
    return None, (
        "runtime_adapter.py is not available yet; --list and --dry-run remain usable"
    )


def _result_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)  # type: ignore[arg-type]
    return {} if value is None else {"value": value}


def _run_one_case(
    case: ExperimentCase, run_branch: RuntimeCall, timeout: float
) -> dict[str, object]:
    record = _case_record(case)
    branch_results: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    for branch in BRANCHES:
        try:
            result = _result_mapping(run_branch(case, branch, timeout))
            result.setdefault("status", "ok")
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        branch_results[branch] = result
        if result.get("status") != "ok":
            errors.append(f"{branch}: {result.get('error', result.get('status'))}")
        record[f"{branch}_time"] = result.get("time_seconds", result.get("time"))
        record[f"{branch}_cycles"] = result.get("makespan_cycles", result.get("cycles"))
        record[f"{branch}_algorithm"] = result.get("algorithm")
        record[f"{branch}_status"] = result.get("status")
        record[f"{branch}_error"] = result.get("error", "")
    branch_statuses = {str(result.get("status", "failed")) for result in branch_results.values()}
    record["status"] = (
        "ok" if branch_statuses == {"ok"}
        else "unsupported" if branch_statuses <= {"ok", "unsupported"}
        else "failed"
    )
    record["error"] = "; ".join(errors)
    record["branch_results"] = branch_results
    return record


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_results(records: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preferred = list(_case_record(next(iter(iter_cases())))) + [
        "swizzle_time", "swizzle_cycles", "swizzle_algorithm", "swizzle_status",
        "swizzle_error", "naive_time", "naive_cycles", "naive_algorithm",
        "naive_status", "naive_error", "status", "error", "branch_results",
    ]
    all_fields = {key for record in records for key in record}
    fields = [key for key in preferred if key in all_fields]
    fields.extend(sorted(all_fields - set(fields)))
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(value) for key, value in record.items()})


def _run_cases(
    cases: tuple[ExperimentCase, ...], *, run_branch: RuntimeCall,
    jobs: int, timeout: float, output_dir: Path,
) -> list[dict[str, object]]:
    if jobs == 1:
        records = [_run_one_case(case, run_branch, timeout) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            records = list(executor.map(
                lambda case: _run_one_case(case, run_branch, timeout), cases
            ))
    _write_results(records, output_dir)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="print all cases as JSONL")
    group.add_argument("--dry-run", action="store_true", help="validate and summarize cases")
    group.add_argument("--smoke", action="store_true", help="run one smallest case per operator")
    group.add_argument("--all", action="store_true", help="run all 176 logical cases")
    parser.add_argument("--jobs", type=int, default=8, help="concurrent cases (default: 8)")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="per-branch timeout in seconds (default: 3600)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    cases = tuple(iter_cases())
    _self_check(cases)
    if args.list:
        for case in cases:
            print(json.dumps(_case_record(case), ensure_ascii=False, sort_keys=True))
    elif args.dry_run:
        print(json.dumps({
            "logical_cases": len(cases), "simulator_runs": 2 * len(cases),
            "cases_per_mesh_operator": 22,
            "meshes": [asdict(mesh) for mesh in MESHES],
            "operators": list(OPERATORS),
        }, ensure_ascii=False, indent=2))
    else:
        run_branch, adapter_error = _load_runtime_adapter()
        if run_branch is None:
            print(f"error: {adapter_error}", file=sys.stderr)
            return 2
        selected = _smoke_cases(cases) if args.smoke else cases
        records = _run_cases(
            selected, run_branch=run_branch, jobs=args.jobs,
            timeout=args.timeout, output_dir=args.output_dir,
        )
        failed = sum(record["status"] != "ok" for record in records)
        print(json.dumps({
            "logical_cases": len(records), "successful": len(records) - failed,
            "failed": failed,
            "results_json": str(args.output_dir / "results.json"),
            "results_csv": str(args.output_dir / "results.csv"),
        }, ensure_ascii=False, indent=2))
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
