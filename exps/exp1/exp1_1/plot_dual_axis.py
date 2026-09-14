#!/usr/bin/env python3
"""Build eight hierarchical dual-axis exp1-1 charts from canonical results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import html
import json
import math
from pathlib import Path
from typing import Iterable


MESHES = ("1x4", "2x3", "3x3", "6x6")
OPERATORS = ("AG_GEMM", "GEMM_RS")
MODEL_ORDER = (
    "LLaMA-2-7B",
    "GPT-3-175B",
    "LLaMA-3-8B",
    "LLaMA-3.1-405B",
    "Mixtral-8x7B-single-expert",
    "DeepSeek-V3-single-routed-expert",
)
MODEL_LABELS = {
    "LLaMA-2-7B": "MHA-S / LLaMA-2-7B",
    "GPT-3-175B": "MHA-L / GPT-3-175B",
    "LLaMA-3-8B": "GQA-S / LLaMA-3-8B",
    "LLaMA-3.1-405B": "GQA-L / LLaMA-3.1-405B",
    "Mixtral-8x7B-single-expert": "MoE-S / Mixtral",
    "DeepSeek-V3-single-routed-expert": "MoE-L / DeepSeek-V3",
}
LAYER_ORDER = ("attention", "mlp")


@dataclass(frozen=True, slots=True)
class Point:
    case_id: str
    layer: str
    model: str
    seq_len: int
    x: float
    naive_tflops: float
    optimized_tflops: float
    naive_normalized: float
    optimized_normalized: float
    theoretical_speedup: float
    actual_speedup: float
    attainment_rate: float


@dataclass(frozen=True, slots=True)
class Span:
    label: str
    start_x: float
    end_x: float


@dataclass(frozen=True, slots=True)
class Chart:
    mesh: str
    operator: str
    points: tuple[Point, ...]
    model_spans: tuple[Span, ...]
    layer_spans: tuple[Span, ...]
    max_tflops: float


def _positive(row: dict[str, str], field: str) -> float:
    try:
        value = float(row.get(field, ""))
    except ValueError as exc:
        raise ValueError(f"{row.get('case_id', '?')}: invalid {field}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{row.get('case_id', '?')}: {field} must be positive")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "case_id", "mesh", "operator", "model", "layer", "seq_len",
            "logical_flops", "theory_time", "theory_speedup", "T00_cycles", "T11_cycles",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if len(rows) != 176:
        raise ValueError(f"expected 176 rows, got {len(rows)}")
    return rows


def _selected_rows(rows: Iterable[dict[str, str]], mesh: str, operator: str):
    selected = [
        row for row in rows
        if row["mesh"] == mesh and row["operator"].upper() == operator
    ]
    if len(selected) != 22:
        raise ValueError(f"{mesh}/{operator}: expected 22 rows, got {len(selected)}")
    return selected


def build_chart(rows: list[dict[str, str]], mesh: str, operator: str) -> Chart:
    selected = _selected_rows(rows, mesh, operator)
    by_key = {
        (row["layer"], row["model"], int(row["seq_len"])): row
        for row in selected
    }
    raw: list[dict[str, object]] = []
    model_spans: list[Span] = []
    layer_spans: list[Span] = []
    cursor = 0.0
    for layer_index, layer in enumerate(LAYER_ORDER):
        layer_start: float | None = None
        layer_end: float | None = None
        models = [
            model for model in MODEL_ORDER
            if any(key[0] == layer and key[1] == model for key in by_key)
        ]
        for model_index, model in enumerate(models):
            seqs = sorted(key[2] for key in by_key if key[:2] == (layer, model))
            if len(seqs) != 2:
                raise ValueError(f"{mesh}/{operator}/{layer}/{model}: expected 2 seq lengths")
            model_start = cursor
            for seq in seqs:
                row = by_key[(layer, model, seq)]
                flops = _positive(row, "logical_flops")
                t00 = _positive(row, "T00_cycles")
                t11 = _positive(row, "T11_cycles")
                theoretical_speedup = _positive(row, "theory_speedup")
                naive = flops / t00 / 1e3
                optimized = flops / t11 / 1e3
                actual_speedup = t00 / t11
                attainment_rate = actual_speedup / theoretical_speedup
                if actual_speedup > theoretical_speedup * (1 + 1e-6):
                    raise ValueError(f"{row['case_id']}: actual speedup exceeds theory")
                raw.append({
                    "row": row, "x": cursor, "naive": naive,
                    "optimized": optimized,
                    "theoretical_speedup": theoretical_speedup,
                    "actual_speedup": actual_speedup,
                    "attainment_rate": attainment_rate,
                })
                layer_start = cursor if layer_start is None else layer_start
                layer_end = cursor
                cursor += 1.0
            model_spans.append(Span(MODEL_LABELS[model], model_start, cursor - 1.0))
            if model_index != len(models) - 1:
                cursor += 0.65
        assert layer_start is not None and layer_end is not None
        layer_spans.append(Span("Attn" if layer == "attention" else "MLP",
                                layer_start, layer_end))
        if layer_index != len(LAYER_ORDER) - 1:
            cursor += 0.9

    max_tflops = max(
        float(item[name]) for item in raw for name in ("naive", "optimized")
    )
    points = tuple(
        Point(
            case_id=str(item["row"]["case_id"]),
            layer=str(item["row"]["layer"]),
            model=str(item["row"]["model"]),
            seq_len=int(item["row"]["seq_len"]),
            x=float(item["x"]),
            naive_tflops=float(item["naive"]),
            optimized_tflops=float(item["optimized"]),
            naive_normalized=float(item["naive"]) / max_tflops,
            optimized_normalized=float(item["optimized"]) / max_tflops,
            theoretical_speedup=float(item["theoretical_speedup"]),
            actual_speedup=float(item["actual_speedup"]),
            attainment_rate=float(item["attainment_rate"]),
        )
        for item in raw
    )
    return Chart(mesh, operator, points, tuple(model_spans), tuple(layer_spans), max_tflops)


def _nice_upper(value: float) -> float:
    target = max(1.0, value * 1.08)
    magnitude = 10 ** math.floor(math.log10(target))
    scaled = target / magnitude
    step = 1 if scaled <= 1 else 2 if scaled <= 2 else 5 if scaled <= 5 else 10
    return step * magnitude


def write_svg(chart: Chart, output: Path) -> None:
    width, height = 1500, 760
    left, right, top, bottom = 92, 100, 82, 218
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min = min(point.x for point in chart.points) - 0.65
    x_max = max(point.x for point in chart.points) + 0.65
    sx = lambda x: left + (x - x_min) / (x_max - x_min) * plot_w
    yn = lambda value: top + plot_h * (1.0 - value)
    yr = lambda value: top + plot_h * (1.0 - value)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" '
        f'font-size="21">{html.escape(chart.mesh)} — {html.escape(chart.operator)}</text>',
        f'<text x="{width/2}" y="56" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" fill="#555">bars normalized by chart max = '
        f'{chart.max_tflops:.3f} TFLOP/s; line shows actual/theoretical speedup ratio</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = yn(value)
        parts += [
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
            'stroke="#b8b8b8" stroke-width="1.15"/>',
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{value:.1f}</text>',
        ]
        parts.append(
            f'<text x="{width-right+10}" y="{y+4:.2f}" font-family="sans-serif" '
            f'font-size="11">{value:.0%}</text>'
        )
    bar_scale = plot_w / (x_max - x_min)
    bar_width = 0.74 * bar_scale
    seq_palettes = (
        ("#24557a", "#9ecae1"),  # shorter sequence: blue
        ("#28735a", "#a8dcc8"),  # longer sequence: green
    )
    for point_index, point in enumerate(chart.points):
        x = sx(point.x)
        opt_y = yn(point.optimized_normalized)
        naive_y = yn(point.naive_normalized)
        dark, light = seq_palettes[point_index % 2]
        # One equal-width segmented bar: dark baseline plus light optimization gain.
        parts.append(
            f'<rect data-role="optimized-gain" x="{x-bar_width/2:.2f}" '
            f'y="{opt_y:.2f}" width="{bar_width:.2f}" '
            f'height="{max(0.0, naive_y-opt_y):.2f}" fill="{light}"/>'
        )
        parts.append(
            f'<rect data-role="naive-baseline" x="{x-bar_width/2:.2f}" '
            f'y="{naive_y:.2f}" width="{bar_width:.2f}" '
            f'height="{top+plot_h-naive_y:.2f}" fill="{dark}"/>'
        )
        parts.append(
            f'<rect data-role="bar-outline" x="{x-bar_width/2:.2f}" '
            f'y="{opt_y:.2f}" width="{bar_width:.2f}" '
            f'height="{top+plot_h-opt_y:.2f}" fill="none" '
            f'stroke="#000" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top+plot_h+18}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="9.5">S={point.seq_len}</text>'
        )
    color = "#a23b72"
    for layer in LAYER_ORDER:
        layer_points = [point for point in chart.points if point.layer == layer]
        coords = " ".join(
            f'{sx(point.x):.2f},{yr(point.attainment_rate):.2f}'
            for point in layer_points
        )
        parts.append(
            f'<polyline data-layer="{layer}" points="{coords}" fill="none" '
            f'stroke="{color}" stroke-width="6.75" stroke-linejoin="round" '
            f'stroke-linecap="round"/>'
        )
    for point in chart.points:
        parts.append(
            f'<circle cx="{sx(point.x):.2f}" '
            f'cy="{yr(point.attainment_rate):.2f}" '
            f'r="8.68" fill="white" stroke="{color}" stroke-width="5.1"/>'
        )
    parts.append(
        f'<rect data-role="plot-frame" x="{left}" y="{top}" width="{plot_w}" '
        f'height="{plot_h}" fill="none" stroke="#111" stroke-width="1.5"/>'
    )
    for span in chart.model_spans:
        start, end = sx(span.start_x - 0.42), sx(span.end_x + 0.42)
        center = (start + end) / 2
        y = top + plot_h + 43
        family, model_name = span.label.split(" / ", 1)
        parts += [
            f'<line x1="{start:.2f}" y1="{y:.2f}" x2="{end:.2f}" y2="{y:.2f}" '
            f'stroke="#666" stroke-width="1.1"/>',
            f'<text x="{center:.2f}" y="{y+14:.2f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="9.5">'
            f'<tspan x="{center:.2f}">{html.escape(family)}</tspan>'
            f'<tspan x="{center:.2f}" dy="12">{html.escape(model_name)}</tspan>'
            f'</text>',
        ]
    for span in chart.layer_spans:
        start, end = sx(span.start_x - 0.52), sx(span.end_x + 0.52)
        center = (start + end) / 2
        y = top + plot_h + 91
        parts += [
            f'<line x1="{start:.2f}" y1="{y:.2f}" x2="{end:.2f}" y2="{y:.2f}" stroke="#333" stroke-width="2"/>',
            f'<text x="{center:.2f}" y="{y+21:.2f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" font-weight="bold">{span.label}</text>',
        ]
    legend_y = height - 45
    parts += [
        f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" transform="rotate(-90 22 {top+plot_h/2})">Normalized performance</text>',
        f'<text x="{width-20}" y="{top+plot_h/2}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" transform="rotate(90 {width-20} {top+plot_h/2})">Actual / theoretical speedup</text>',
        f'<rect x="{left}" y="{legend_y}" width="18" height="6" fill="#9ecae1"/>',
        f'<rect x="{left}" y="{legend_y+6}" width="18" height="7" fill="#24557a"/>',
        f'<rect x="{left}" y="{legend_y}" width="18" height="13" fill="none" stroke="#000" stroke-width="1.2"/>',
        f'<text x="{left+24}" y="{legend_y+11}" font-family="sans-serif" font-size="11">S=36864</text>',
        f'<rect x="{left+105}" y="{legend_y}" width="18" height="6" fill="#a8dcc8"/>',
        f'<rect x="{left+105}" y="{legend_y+6}" width="18" height="7" fill="#28735a"/>',
        f'<rect x="{left+105}" y="{legend_y}" width="18" height="13" fill="none" stroke="#000" stroke-width="1.2"/>',
        f'<text x="{left+129}" y="{legend_y+11}" font-family="sans-serif" font-size="11">S=147456</text>',
        f'<text x="{left+245}" y="{legend_y+11}" font-family="sans-serif" font-size="11" fill="#333">dark: naive T00 · light: gain to optimized T11</text>',
        f'<line x1="{left+570}" y1="{legend_y+7}" x2="{left+610}" y2="{legend_y+7}" stroke="#a23b72" stroke-width="6.75"/>',
        f'<circle cx="{left+590}" cy="{legend_y+7}" r="8.68" fill="white" stroke="#a23b72" stroke-width="5.1"/>',
        f'<text x="{left+620}" y="{legend_y+11}" font-family="sans-serif" font-size="11">Actual / theoretical speedup</text>',
        f'<rect data-role="figure-frame" x="1" y="1" width="{width-2}" height="{height-2}" fill="none" stroke="#111" stroke-width="1.4"/>',
        '</svg>',
    ]
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base / "results" / "results.csv")
    parser.add_argument("--output-dir", type=Path, default=base / "figures" / "dual_axis")
    parser.add_argument("--data-output", type=Path,
                        default=base / "results" / "dual_axis_data.json")
    args = parser.parse_args()
    rows = read_rows(args.input)
    charts = [build_chart(rows, mesh, operator) for mesh in MESHES for operator in OPERATORS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for chart in charts:
        write_svg(chart, args.output_dir / f"{chart.mesh}_{chart.operator.lower()}_dual_axis.svg")
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.data_output.write_text(json.dumps([
        {"mesh": chart.mesh, "operator": chart.operator,
         "max_tflops": chart.max_tflops,
         "points": [asdict(point) for point in chart.points],
         "model_spans": [asdict(span) for span in chart.model_spans],
         "layer_spans": [asdict(span) for span in chart.layer_spans]}
        for chart in charts
    ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"generated {len(charts)} dual-axis SVG charts in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
