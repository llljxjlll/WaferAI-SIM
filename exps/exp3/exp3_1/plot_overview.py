#!/usr/bin/env python3
"""Draw the single Exp3.1 per-shape relative-speedup overview as SVG.

The x-axis nesting is operator family -> mesh -> shape.  Each of the six
mesh groups contains its eight shapes sorted by evaluated padded FLOPs.  A
comparison series is connected only within one mesh group; it never implies a
line interpolation across a mesh or operator-family boundary.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


HERE = Path(__file__).resolve().parent
COMPARISONS = (
    ("native_full", "native_full", "#2563eb"),
    ("native_inter_only", "native_inter_only", "#059669"),
    ("gpu_inter", "gpu_inter", "#ea580c"),
)
FAMILIES = (
    ("gemm_rs", "GEMM+RS"),
    ("dispatch_gemm", "Dispatch+GEMM"),
)
MESHES = ((6, "2×3"), (9, "3×3"), (36, "6×6"))
MODEL_LABELS = {
    "LLaMA-2-7B": "LLaMA",
    "GPT-3-175B": "GPT-3",
    "Mixtral-8x7B": "Mix",
    "DeepSeek-V3": "DS",
}
STAGE_LABELS = {
    "o_proj": "O",
    "down_proj": "Down",
    "up_gate": "Up",
    "down": "Down",
}


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _shape_tooltip(point: Mapping[str, object]) -> str:
    logical = "×".join(str(value) for value in point["logical_shape"])
    runtime = "×".join(str(value) for value in point["runtime_shape"])
    values = ", ".join(
        f"{label}={float(point['speedups'][key]):.4f}×"
        for key, label, _ in COMPARISONS
    )
    return (
        f"{point['family_label']} | D={point['D']} ({point['mesh_label']}) | "
        f"{point['model']} {point['stage']} | S={point['S']} | "
        f"logical (M,N,K)={logical}; runtime (M,N,K)={runtime}; "
        f"padded FLOPs={int(point['padded_flops'])}; {values}"
    )


def load_groups(results: Path) -> dict[tuple[str, int], list[dict[str, object]]]:
    """Join comparison rows with shape metadata and order each mesh group."""
    document = json.loads(results.read_text(encoding="utf-8"))
    records = document.get("records")
    comparisons = document.get("comparisons")
    if not isinstance(records, list) or not isinstance(comparisons, list):
        raise ValueError("results JSON must contain records and comparisons lists")
    if len(comparisons) != 144:
        raise ValueError("results JSON must contain 144 paired comparisons")

    metadata: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("record must be an object")
        case_id = str(record["case_id"])
        if case_id not in metadata:
            metadata[case_id] = record

    by_case: dict[str, dict[str, Mapping[str, object]]] = {}
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise ValueError("comparison must be an object")
        case_id = str(comparison["case_id"])
        key = str(comparison["comparison"])
        by_case.setdefault(case_id, {})[key] = comparison

    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    expected_comparisons = {key for key, _, _ in COMPARISONS}
    family_labels = dict(FAMILIES)
    mesh_labels = dict(MESHES)
    for case_id, by_comparison in by_case.items():
        if set(by_comparison) != expected_comparisons:
            raise ValueError(f"{case_id} does not contain exactly the three comparisons")
        record = metadata.get(case_id)
        if record is None:
            raise ValueError(f"comparison {case_id} has no state-record metadata")
        representative = by_comparison["native_full"]
        family = str(representative["operator_family"])
        D = int(representative["D"])
        if family not in family_labels or D not in mesh_labels:
            raise ValueError(f"unsupported group {family}, D={D}")
        logical = record.get("logical_shape")
        runtime = record.get("runtime_shape")
        if not isinstance(logical, list) or not isinstance(runtime, list):
            raise ValueError(f"{case_id} lacks logical/runtime shape metadata")
        point = {
            "case_id": case_id,
            "family": family,
            "family_label": family_labels[family],
            "D": D,
            "mesh_label": mesh_labels[D],
            "model": str(representative["model_or_moe_config"]),
            "stage": str(representative["stage"]),
            "S": int(representative["S"]),
            "logical_shape": list(logical),
            "runtime_shape": list(runtime),
            "padded_flops": int(record["padded_flops"]),
            "valid_flops": int(record["valid_flops"]),
            "speedups": {
                key: _number(row["speedup"], f"{case_id}/{key} speedup")
                for key, row in by_comparison.items()
            },
        }
        groups.setdefault((family, D), []).append(point)

    for family, _ in FAMILIES:
        for D, _ in MESHES:
            points = groups.get((family, D))
            if points is None or len(points) != 8:
                actual = 0 if points is None else len(points)
                raise ValueError(f"{family}, D={D} must contain eight shapes, got {actual}")
            points.sort(key=lambda point: (
                int(point["padded_flops"]), int(point["valid_flops"]),
                str(point["model"]), str(point["stage"]), int(point["S"]),
            ))
    return groups


def draw(groups: Mapping[tuple[str, int], Sequence[Mapping[str, object]]], output: Path) -> None:
    """Render one wide SVG chart without a plotting-library dependency."""
    left, right = 70, 55
    top, plot_height = 125, 300
    slot_width, mesh_gap, family_gap = 15, 3, 16
    x_groups: list[tuple[str, int, list[float], float, float]] = []
    cursor = float(left)
    for family_index, (family, _) in enumerate(FAMILIES):
        for D, _ in MESHES:
            positions = [cursor + slot_width * (index + 0.5) for index in range(8)]
            start, end = cursor, cursor + slot_width * 8
            x_groups.append((family, D, positions, start, end))
            cursor = end + mesh_gap
        if family_index != len(FAMILIES) - 1:
            cursor += family_gap - mesh_gap
    width = int(cursor - mesh_gap + right)
    height = 470
    plot_bottom = top + plot_height
    all_values = [
        float(point["speedups"][comparison])
        for points in groups.values() for point in points
        for comparison, _, _ in COMPARISONS
    ]
    y_min = min(0.95, math.floor(min(all_values) * 20.0) / 20.0)
    y_max = max(1.05, math.ceil(max(all_values) * 20.0) / 20.0 + 0.05)
    if y_max <= y_min:
        y_max = y_min + 0.1

    def x_scale(value: float) -> float:
        return value

    def y_scale(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="29" text-anchor="middle" font-family="sans-serif" font-size="21" font-weight="bold">Exp3.1 relative speedup by operator, mesh, and shape</text>',
        '<text x="70" y="57" font-family="sans-serif" font-size="12">Shape # within each mesh: padded-FLOPs ascending. Lines do not cross mesh boundaries.</text>',
    ]
    legend_x = width - 590
    for index, (_, label, color) in enumerate(COMPARISONS):
        x = legend_x + index * 185
        parts.append(f'<line x1="{x}" y1="55" x2="{x + 27}" y2="55" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{x + 14}" cy="55" r="4" fill="{color}" stroke="white" stroke-width="1.3"/>')
        parts.append(f'<text x="{x + 34}" y="60" font-family="sans-serif" font-size="13">{label}</text>')

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5.0
        y = y_scale(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.2f}×</text>')
    if y_min <= 1.0 <= y_max:
        y = y_scale(1.0)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#6b7280" stroke-width="1.2" stroke-dasharray="5 4"/>')

    for family, D, positions, start, end in x_groups:
        points = groups[(family, D)]
        center = (start + end) / 2.0
        parts.append(f'<text x="{center:.2f}" y="101" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="bold">D={D} ({dict(MESHES)[D]})</text>')
        parts.append(f'<line x1="{end + mesh_gap / 2:.2f}" y1="{top}" x2="{end + mesh_gap / 2:.2f}" y2="{plot_bottom}" stroke="#cbd5e1" stroke-dasharray="3 5"/>')
        for comparison, _, color in COMPARISONS:
            polyline = " ".join(
                f"{x_scale(x):.2f},{y_scale(float(point['speedups'][comparison])):.2f}"
                for x, point in zip(positions, points)
            )
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
        for shape_rank, (x, point) in enumerate(zip(positions, points), start=1):
            tooltip = html.escape(_shape_tooltip(point))
            parts.append(f'<text x="{x:.2f}" y="{plot_bottom + 20}" text-anchor="middle" font-family="sans-serif" font-size="10">{shape_rank}</text>')
            for comparison, _, color in COMPARISONS:
                y = y_scale(float(point["speedups"][comparison]))
                parts.append(f'<g><title>shape #{shape_rank}: {tooltip}</title><circle cx="{x:.2f}" cy="{y:.2f}" r="4.0" fill="{color}" stroke="white" stroke-width="1.2"/></g>')

    for family, label in FAMILIES:
        family_groups = [item for item in x_groups if item[0] == family]
        start = family_groups[0][3]
        end = family_groups[-1][4]
        parts.append(f'<line x1="{start:.2f}" y1="114" x2="{end:.2f}" y2="114" stroke="#334155" stroke-width="1.3"/>')
        parts.append(f'<text x="{(start + end) / 2:.2f}" y="79" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold">{label}</text>')
    parts.extend((
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{plot_bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{plot_bottom}" x2="{width-right}" y2="{plot_bottom}" stroke="#111827"/>',
        f'<text x="23" y="{top + plot_height / 2:.2f}" transform="rotate(-90 23 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="14">relative speedup (baseline / optimized)</text>',
        '</svg>',
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results" / "exp3_1_results.json")
    parser.add_argument("--output", type=Path, default=HERE / "figures" / "speedup_overview_by_shape.svg")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    groups = load_groups(args.results)
    draw(groups, args.output)
    print(json.dumps({
        "output": str(args.output),
        "operator_families": len(FAMILIES),
        "mesh_groups": len(groups),
        "shapes_per_mesh_group": 8,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
