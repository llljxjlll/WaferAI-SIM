#!/usr/bin/env python3
"""Single source of truth for the Exp3.1 logical cases and GPU GEMM shapes.

The lookup order is always ``(M, N, K)`` for ``A[M,K] @ B[K,N]``.  Dense
GEMM+RS uses the exact semantic padding rule from Exp1.1.  MoE cases use the
two real model profiles selected by the current Exp3.1 plan; the gate and up
projections share one shape and therefore carry an execution count of two.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import yaml


Shape = tuple[int, int, int]
OperatorFamily = Literal["gemm_rs", "dispatch_gemm"]

SCHEMA_VERSION = 1
SEQUENCE_LENGTHS = (2304, 36864)
SHAPE_GROUPS = (
    "gemm_rs_coarse",
    "gemm_rs_1d_ring_c_eq_d",
    "gemm_rs_2d_row_column",
    "dispatch_gemm_coarse",
    "dispatch_gemm_source_expert_chunk",
)


@dataclass(frozen=True, slots=True)
class MeshSpec:
    dies: int
    rows: int
    columns: int

    def __post_init__(self) -> None:
        if self.dies != self.rows * self.columns:
            raise ValueError("mesh dimensions must multiply to dies")


MESHES = (
    MeshSpec(6, 2, 3),
    MeshSpec(9, 3, 3),
    MeshSpec(36, 6, 6),
)


@dataclass(frozen=True, slots=True)
class DenseLayerSpec:
    model: str
    layer: str
    output_size: int
    input_size: int


DENSE_LAYERS = (
    DenseLayerSpec("LLaMA-2-7B", "o_proj", 4096, 12288),
    DenseLayerSpec("LLaMA-2-7B", "down_proj", 4096, 11008),
    DenseLayerSpec("GPT-3-175B", "o_proj", 12288, 36864),
    DenseLayerSpec("GPT-3-175B", "down_proj", 12288, 49152),
)


@dataclass(frozen=True, slots=True)
class MoeModelSpec:
    model: str
    hidden_size: int
    intermediate_size: int
    experts: int
    topk: int


MOE_MODELS = (
    MoeModelSpec("Mixtral-8x7B", 4096, 14336, 8, 2),
    MoeModelSpec("DeepSeek-V3", 7168, 2048, 256, 8),
)


@dataclass(frozen=True, slots=True)
class LogicalCase:
    """One of the 48 logical experiment points.

    Upper-case topology aliases are intentional: the experiment notation uses
    ``D``, ``Px``, ``Py`` and ``S``, and the runner can consume them directly.
    ``logical_shape`` and ``runtime_shape`` describe the coarse/global work;
    the exact GPU lookup keys are exposed separately.
    """

    case_id: str
    operator_family: OperatorFamily
    stage: str
    model: str
    D: int
    Px: int
    Py: int
    S: int
    logical_shape: Shape
    runtime_shape: Shape
    coarse_key: Shape
    ring_key: Shape | None
    rc_key: Shape | None
    chunk_key: Shape | None
    gemm_execution_count: int
    valid_flops: int
    padded_flops: int
    padding: Mapping[str, object]
    hidden_size: int | None = None
    intermediate_size: int | None = None
    experts: int | None = None
    topk: int | None = None

    @property
    def layer(self) -> str:
        """Dense-layer/MoE-stage compatibility alias used by the runner."""

        return self.stage

    @property
    def die_count(self) -> int:
        return self.D

    @property
    def mesh_rows(self) -> int:
        return self.Px

    @property
    def mesh_columns(self) -> int:
        return self.Py

    @property
    def seq_len(self) -> int:
        return self.S

    @property
    def coarse_shape(self) -> Shape:
        return self.coarse_key

    @property
    def ring_shape(self) -> Shape | None:
        return self.ring_key

    @property
    def row_column_shape(self) -> Shape | None:
        return self.rc_key

    @property
    def source_expert_chunk_shape(self) -> Shape | None:
        return self.chunk_key

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["layer"] = self.layer
        # JSON arrays make the generated artifact language-neutral.
        for key in (
            "logical_shape",
            "runtime_shape",
            "coarse_key",
            "ring_key",
            "rc_key",
            "chunk_key",
        ):
            value = result[key]
            result[key] = None if value is None else list(value)
        return result


@dataclass(frozen=True, slots=True)
class GpuShapeReference:
    group: str
    shape: Shape
    case_id: str
    algorithm: str
    gemm_execution_count: int

    @property
    def execution_count(self) -> int:
        return self.gemm_execution_count

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "shape": list(self.shape),
            "case_id": self.case_id,
            "algorithm": self.algorithm,
            "gemm_execution_count": self.gemm_execution_count,
        }


def align_up(value: int, alignment: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("value must be a positive integer")
    if type(alignment) is not int or alignment <= 0:
        raise ValueError("alignment must be a positive integer")
    return ((value + alignment - 1) // alignment) * alignment


def dense_runtime_shape(logical_shape: Shape, dies: int) -> Shape:
    """Apply Exp1.1 GEMM_RS semantic padding to an unsharded shape."""

    m, n, k = logical_shape
    return (
        align_up(m, dies),
        align_up(n, math.lcm(dies, 16)),
        align_up(k, dies),
    )


def _dense_case(mesh: MeshSpec, layer: DenseLayerSpec, seq_len: int) -> LogicalCase:
    logical = (seq_len, layer.output_size, layer.input_size)
    runtime = dense_runtime_shape(logical, mesh.dies)
    m, n, k = runtime
    coarse = (m, n, k // mesh.dies)
    ring = (m, n // mesh.dies, k // mesh.dies)
    rc = (m, n // mesh.columns, k // mesh.rows)
    count = 1
    return LogicalCase(
        case_id=(
            f"d{mesh.dies}_gemm_rs_{layer.model.lower().replace('-', '_')}"
            f"_{layer.layer}_s{seq_len}"
        ),
        operator_family="gemm_rs",
        stage=layer.layer,
        model=layer.model,
        D=mesh.dies,
        Px=mesh.rows,
        Py=mesh.columns,
        S=seq_len,
        logical_shape=logical,
        runtime_shape=runtime,
        coarse_key=coarse,
        ring_key=ring,
        rc_key=rc,
        chunk_key=None,
        gemm_execution_count=count,
        valid_flops=2 * math.prod(logical) * count,
        padded_flops=2 * math.prod(runtime) * count,
        padding={
            "rule": "exp1_1_gemm_rs_semantic_padding",
            "M_alignment": mesh.dies,
            "N_alignment": math.lcm(mesh.dies, 16),
            "K_alignment": mesh.dies,
            "delta": [runtime[index] - logical[index] for index in range(3)],
        },
    )


def _moe_case(
    mesh: MeshSpec, model: MoeModelSpec, stage: str, seq_len: int
) -> LogicalCase:
    routed_tokens = seq_len * model.topk
    if routed_tokens % mesh.dies:
        raise ValueError("coarse routed-token count is not integral")
    if routed_tokens % (mesh.dies * model.experts):
        raise ValueError("source-expert routed-token chunk is not integral")
    coarse_m = routed_tokens // mesh.dies
    chunk_m = routed_tokens // (mesh.dies * model.experts)
    if stage == "up_gate":
        n, k, count = model.intermediate_size, model.hidden_size, 2
    elif stage == "down":
        n, k, count = model.hidden_size, model.intermediate_size, 1
    else:
        raise ValueError(f"unsupported MoE GEMM stage: {stage}")
    coarse = (coarse_m, n, k)
    chunk = (chunk_m, n, k)
    return LogicalCase(
        case_id=(
            f"d{mesh.dies}_dispatch_gemm_"
            f"{model.model.lower().replace('-', '_')}_{stage}_s{seq_len}"
        ),
        operator_family="dispatch_gemm",
        stage=stage,
        model=model.model,
        D=mesh.dies,
        Px=mesh.rows,
        Py=mesh.columns,
        S=seq_len,
        logical_shape=coarse,
        runtime_shape=coarse,
        coarse_key=coarse,
        ring_key=None,
        rc_key=None,
        chunk_key=chunk,
        gemm_execution_count=count,
        valid_flops=2 * math.prod(coarse) * count,
        padded_flops=2 * math.prod(coarse) * count,
        padding={
            "rule": "exact_balanced_routed_tokens",
            "routed_tokens": routed_tokens,
            "coarse_divisor": mesh.dies,
            "chunk_divisor": mesh.dies * model.experts,
            "delta": [0, 0, 0],
        },
        hidden_size=model.hidden_size,
        intermediate_size=model.intermediate_size,
        experts=model.experts,
        topk=model.topk,
    )


def build_logical_cases() -> tuple[LogicalCase, ...]:
    cases: list[LogicalCase] = []
    for mesh in MESHES:
        for layer in DENSE_LAYERS:
            for seq_len in SEQUENCE_LENGTHS:
                cases.append(_dense_case(mesh, layer, seq_len))
        for model in MOE_MODELS:
            for stage in ("up_gate", "down"):
                for seq_len in SEQUENCE_LENGTHS:
                    cases.append(_moe_case(mesh, model, stage, seq_len))
    if len(cases) != 48 or len({case.case_id for case in cases}) != 48:
        raise AssertionError("Exp3.1 must contain exactly 48 unique logical cases")
    return tuple(cases)


def required_shape_references(
    cases: Iterable[LogicalCase] | None = None,
) -> tuple[GpuShapeReference, ...]:
    references: list[GpuShapeReference] = []
    for case in build_logical_cases() if cases is None else cases:
        if case.operator_family == "gemm_rs":
            candidates = (
                ("gemm_rs_coarse", case.coarse_key, "coarse"),
                ("gemm_rs_1d_ring_c_eq_d", case.ring_key, "1d_ring_c_eq_d"),
                ("gemm_rs_2d_row_column", case.rc_key, "2d_row_column"),
            )
        else:
            candidates = (
                ("dispatch_gemm_coarse", case.coarse_key, "coarse"),
                (
                    "dispatch_gemm_source_expert_chunk",
                    case.chunk_key,
                    "source_expert_chunk",
                ),
            )
        for group, shape, algorithm in candidates:
            if shape is None:
                raise AssertionError(f"{case.case_id} lacks {group} shape")
            references.append(
                GpuShapeReference(
                    group=group,
                    shape=shape,
                    case_id=case.case_id,
                    algorithm=algorithm,
                    gemm_execution_count=case.gemm_execution_count,
                )
            )
    counts = Counter(reference.group for reference in references)
    if counts != Counter({group: 24 for group in SHAPE_GROUPS}):
        raise AssertionError(f"unexpected GPU shape group counts: {dict(counts)}")
    return tuple(references)


def required_shape_groups(
    cases: Iterable[LogicalCase] | None = None,
) -> dict[str, tuple[GpuShapeReference, ...]]:
    grouped: dict[str, list[GpuShapeReference]] = {
        group: [] for group in SHAPE_GROUPS
    }
    for reference in required_shape_references(cases):
        grouped[reference.group].append(reference)
    return {group: tuple(items) for group, items in grouped.items()}


def _required_yaml(cases: Sequence[LogicalCase]) -> dict[str, object]:
    grouped = required_shape_groups(cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_name": "exp3_1_gpu_gemm_lut",
        "units": "ns",
        "shape_order": ["M", "N", "K"],
        "shape_semantics": "A[M,K] x B[K,N] -> C[M,N]",
        "expected_entries": 120,
        "gpu": {"name": None, "count": 1, "clock_policy": None},
        "software": {"driver": None, "cuda": None, "cublas_or_backend": None},
        "gemm": {
            "input_dtype": "bf16",
            "output_dtype": "bf16",
            "accumulation_dtype": "fp32",
            "transpose_a": False,
            "transpose_b": False,
        },
        "measurement": {
            "statistic": "p50",
            "warmup_iterations": None,
            "measured_iterations": None,
            "synchronization": "per_iteration",
            "operands_resident_on_device": True,
            "includes_host_to_device": False,
            "includes_device_to_host": False,
        },
        "experiment": {
            "die_counts": [mesh.dies for mesh in MESHES],
            "meshes": {
                str(mesh.dies): [mesh.rows, mesh.columns] for mesh in MESHES
            },
            "sequence_lengths": list(SEQUENCE_LENGTHS),
            "dense_runtime_shapes_include_semantic_padding": True,
            "moe_profiles": [asdict(model) for model in MOE_MODELS],
            "up_gate_execution_count": 2,
            "down_execution_count": 1,
        },
        "lookup": {
            group: [[list(reference.shape), None] for reference in grouped[group]]
            for group in SHAPE_GROUPS
        },
        "shape_references": [
            reference.to_dict() for reference in required_shape_references(cases)
        ],
    }


def build_artifacts() -> dict[str, object]:
    cases = build_logical_cases()
    references = required_shape_references(cases)
    reverse: dict[Shape, list[dict[str, object]]] = defaultdict(list)
    for reference in references:
        reverse[reference.shape].append(reference.to_dict())
    return {
        "required_gpu_shapes": _required_yaml(cases),
        "logical_cases": {
            "schema_version": SCHEMA_VERSION,
            "case_count": len(cases),
            "cases": [case.to_dict() for case in cases],
        },
        "shape_coverage_report": {
            "schema_version": SCHEMA_VERSION,
            "semantic_entry_count": len(references),
            "unique_shape_count": len(reverse),
            "group_counts": dict(Counter(ref.group for ref in references)),
            "reverse_references": [
                {"shape": list(shape), "references": refs}
                for shape, refs in sorted(reverse.items())
            ],
        },
    }


def emit_artifacts(output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = build_artifacts()
    paths = {
        "required_gpu_shapes": output / "required_gpu_shapes.yaml",
        "logical_cases": output / "logical_cases.json",
        "shape_coverage_report": output / "shape_coverage_report.json",
    }
    paths["required_gpu_shapes"].write_text(
        yaml.safe_dump(
            artifacts["required_gpu_shapes"],
            sort_keys=False,
            allow_unicode=True,
            width=1000,
        ),
        encoding="utf-8",
    )
    for name in ("logical_cases", "shape_coverage_report"):
        paths[name].write_text(
            json.dumps(artifacts[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return paths


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = emit_artifacts(args.emit_dir)
    print(
        json.dumps(
            {name: str(path) for name, path in paths.items()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
