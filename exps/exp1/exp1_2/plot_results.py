#!/usr/bin/env python3
"""Generate exp1-2 paired T00/T11 SVG charts for one hardware profile/mode."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path


OPERATORS = ("DISPATCH_GEMM", "GEMM_COMBINE")
PLACEMENTS = ("compact", "noncompact")
MODELS = ("Mixtral-8x7B", "DeepSeek-V3")


@dataclass(frozen=True, slots=True)
class Point:
    case_id: str
    placement: str
    model: str
    seq_len: int
    x: float
    naive_tflops: float
    optimized_tflops: float
    naive_normalized: float
    optimized_normalized: float
    theory_speedup: float
    actual_speedup: float
    attainment_rate: float
    status: str


@dataclass(frozen=True, slots=True)
class Span:
    label: str
    start_x: float
    end_x: float


@dataclass(frozen=True, slots=True)
class Chart:
    operator: str
    memory_mode: str
    tensor_profile: str
    estimate_source: str
    points: tuple[Point, ...]
    model_spans: tuple[Span, ...]
    placement_spans: tuple[Span, ...]
    max_tflops: float
    max_attainment: float


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("result file is empty")
    identities = {
        row.get("profile_case_id") or row.get("case_id", "") for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError("result file contains duplicate case identities")
    return rows


def _profile(row: dict[str, str]) -> str:
    return row.get("tensor_profile") or row.get("profile") or "legacy"


def build_chart(
    rows: list[dict[str, str]], operator: str, tensor_profile: str | None = None,
) -> Chart:
    selected = [row for row in rows if row["operator"] == operator]
    if tensor_profile is not None:
        selected = [row for row in selected if _profile(row) == tensor_profile]
    profiles = {_profile(row) for row in selected}
    if len(profiles) != 1:
        raise ValueError(
            f"{operator}: select exactly one tensor profile, got {sorted(profiles)}"
        )
    if len(selected) != 8:
        raise ValueError(f"{operator}: expected 8 rows, got {len(selected)}")
    profile = next(iter(profiles))
    memory_modes = {row.get("memory_mode", "architecture") for row in selected}
    if len(memory_modes) != 1:
        raise ValueError(f"{operator}/{profile}: mixed memory modes")
    sources = {row.get("estimate_source", "unknown") for row in selected}
    if len(sources) != 1:
        raise ValueError(f"{operator}/{profile}: mixed estimate sources")

    by_key = {
        (row["placement"], row["model"], int(row["seq_len"])): row
        for row in selected
    }
    raw: list[dict[str, object]] = []
    model_spans: list[Span] = []
    placement_spans: list[Span] = []
    cursor = 0.0
    for placement_index, placement in enumerate(PLACEMENTS):
        placement_start = cursor
        for model_index, model in enumerate(MODELS):
            model_start = cursor
            seqs = sorted(key[2] for key in by_key if key[:2] == (placement, model))
            if len(seqs) != 2:
                raise ValueError(
                    f"{operator}/{profile}/{placement}/{model}: expected two seq lengths"
                )
            for seq_len in seqs:
                row = by_key[(placement, model, seq_len)]
                theory = float(row["theory_speedup"])
                actual = float(row["actual_speedup"])
                raw.append({
                    "row": row,
                    "x": cursor,
                    "naive": float(row["naive_tflops"]),
                    "optimized": float(row["optimized_tflops"]),
                    "theory": theory,
                    "actual": actual,
                    "attainment": actual / theory if theory > 0 else 0.0,
                })
                cursor += 0.90
            model_spans.append(Span(model, model_start, cursor - 0.90))
            if model_index != len(MODELS) - 1:
                cursor += 0.30
        placement_spans.append(Span(
            "Compact / adjacent" if placement == "compact"
            else "Non-compact / corners",
            placement_start, cursor - 0.90,
        ))
        if placement_index != len(PLACEMENTS) - 1:
            cursor += 0.50

    maximum = max(
        float(item[name]) for item in raw for name in ("naive", "optimized")
    )
    points = tuple(Point(
        case_id=str(item["row"].get("profile_case_id") or item["row"]["case_id"]),
        placement=str(item["row"]["placement"]),
        model=str(item["row"]["model"]),
        seq_len=int(item["row"]["seq_len"]),
        x=float(item["x"]),
        naive_tflops=float(item["naive"]),
        optimized_tflops=float(item["optimized"]),
        naive_normalized=float(item["naive"]) / maximum,
        optimized_normalized=float(item["optimized"]) / maximum,
        theory_speedup=float(item["theory"]),
        actual_speedup=float(item["actual"]),
        attainment_rate=float(item["attainment"]),
        status=str(item["row"].get("status", "unknown")),
    ) for item in raw)
    return Chart(
        operator=operator,
        memory_mode=next(iter(memory_modes)),
        tensor_profile=profile,
        estimate_source=next(iter(sources)),
        points=points,
        model_spans=tuple(model_spans),
        placement_spans=tuple(placement_spans),
        max_tflops=maximum,
        max_attainment=max(point.attainment_rate for point in points),
    )


def write_svg(chart: Chart, output: Path) -> None:
    width, height = 980, 720
    left, right, top, bottom = 82, 94, 92, 205
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min = min(point.x for point in chart.points) - 0.55
    x_max = max(point.x for point in chart.points) + 0.55
    sx = lambda value: left + (value - x_min) / (x_max - x_min) * plot_w
    sy_perf = lambda value: top + plot_h * (1.0 - value / 1.10)
    right_max = max(1.10, chart.max_attainment * 1.08)
    sy_rate = lambda value: top + plot_h * (1.0 - value / right_max)
    title = (
        "Dispatch + GEMM" if chart.operator == "DISPATCH_GEMM"
        else "GEMM + Combine"
    )
    mode_label = (
        "HBM-free compute/communication ablation"
        if chart.memory_mode == "hbm_free_compute_comm"
        else "four-stack physical analytical replay"
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="31" text-anchor="middle" font-family="sans-serif" '
        f'font-size="21" font-weight="bold">{html.escape(title)} · '
        f'{html.escape(chart.tensor_profile)}</text>',
        f'<text x="{width/2}" y="55" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" fill="#444">{html.escape(mode_label)}; grouped T00/T11; '
        f'max={chart.max_tflops:.3f} TFLOP/s</text>',
        f'<text x="{width/2}" y="73" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10" fill="#666">source={html.escape(chart.estimate_source)}</text>',
    ]
    for value in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = sy_perf(value)
        parts += [
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
            f'stroke="#c5c5c5" stroke-width="1"/>',
            f'<text x="{left-9}" y="{y+4:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{value:.1f}</text>',
            f'<text x="{width-right+9}" y="{sy_rate(value)+4:.2f}" '
            f'font-family="sans-serif" font-size="11">{value:.0%}</text>',
        ]

    scale = plot_w / (x_max - x_min)
    bar_width = 0.31 * scale
    palettes = (("#633C8E", "#B99AD3"), ("#A65300", "#E7A74F"))
    for index, point in enumerate(chart.points):
        x = sx(point.x)
        baseline_x = x - bar_width
        optimized_x = x
        baseline_y = sy_perf(point.naive_normalized)
        optimized_y = sy_perf(point.optimized_normalized)
        baseline_color, optimized_color = palettes[index % 2]
        parts += [
            f'<rect data-role="T00-bar" x="{baseline_x:.2f}" y="{baseline_y:.2f}" '
            f'width="{bar_width:.2f}" height="{top+plot_h-baseline_y:.2f}" '
            f'fill="{baseline_color}" stroke="#111" stroke-width="0.8"/>',
            f'<rect data-role="T11-bar" x="{optimized_x:.2f}" y="{optimized_y:.2f}" '
            f'width="{bar_width:.2f}" height="{top+plot_h-optimized_y:.2f}" '
            f'fill="{optimized_color}" stroke="#111" stroke-width="0.8"/>',
            f'<rect data-role="bar-outline" x="{baseline_x:.2f}" y="{min(baseline_y, optimized_y):.2f}" '
            f'width="{2*bar_width:.2f}" height="{top+plot_h-min(baseline_y, optimized_y):.2f}" '
            f'fill="none" stroke="none"/>',
            f'<text x="{x:.2f}" y="{top+plot_h+18}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10">S={point.seq_len}</text>',
        ]

    line_color = "#007F86"
    for placement in PLACEMENTS:
        selected = [point for point in chart.points if point.placement == placement]
        coords = " ".join(
            f'{sx(point.x):.2f},{sy_rate(point.attainment_rate):.2f}'
            for point in selected
        )
        parts.append(
            f'<polyline data-placement="{placement}" points="{coords}" fill="none" '
            f'stroke="{line_color}" stroke-width="4.5" stroke-linejoin="round" '
            f'stroke-linecap="round"/>'
        )
    for point in chart.points:
        parts.append(
            f'<circle cx="{sx(point.x):.2f}" cy="{sy_rate(point.attainment_rate):.2f}" '
            f'r="5.5" fill="white" stroke="{line_color}" stroke-width="3"/>'
        )
    parts.append(
        f'<rect data-role="plot-frame" x="{left}" y="{top}" width="{plot_w}" '
        f'height="{plot_h}" fill="none" stroke="#111" stroke-width="1.5"/>'
    )
    for span in chart.model_spans:
        start, end = sx(span.start_x - 0.39), sx(span.end_x + 0.39)
        center = (start + end) / 2
        y = top + plot_h + 42
        parts += [
            f'<line x1="{start:.2f}" y1="{y:.2f}" x2="{end:.2f}" y2="{y:.2f}" '
            f'stroke="#666" stroke-width="1.1"/>',
            f'<text x="{center:.2f}" y="{y+16:.2f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{html.escape(span.label)}</text>',
        ]
    for span in chart.placement_spans:
        start, end = sx(span.start_x - 0.44), sx(span.end_x + 0.44)
        center = (start + end) / 2
        y = top + plot_h + 80
        parts += [
            f'<line x1="{start:.2f}" y1="{y:.2f}" x2="{end:.2f}" y2="{y:.2f}" '
            f'stroke="#333" stroke-width="2"/>',
            f'<text x="{center:.2f}" y="{y+20:.2f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" font-weight="bold">'
            f'{html.escape(span.label)}</text>',
        ]
    legend_y = height - 38
    parts += [
        f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" transform="rotate(-90 22 {top+plot_h/2})">Normalized performance</text>',
        f'<text x="{width-20}" y="{top+plot_h/2}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="13" '
        f'transform="rotate(90 {width-20} {top+plot_h/2})">Actual / theoretical speedup</text>',
        f'<rect x="{left}" y="{legend_y-13}" width="18" height="12" fill="#633C8E"/>',
        f'<text x="{left+25}" y="{legend_y-2}" font-family="sans-serif" font-size="11">T00</text>',
        f'<rect x="{left+70}" y="{legend_y-13}" width="18" height="12" fill="#B99AD3"/>',
        f'<text x="{left+95}" y="{legend_y-2}" font-family="sans-serif" font-size="11">T11</text>',
        f'<line x1="{left+150}" y1="{legend_y-7}" x2="{left+190}" y2="{legend_y-7}" '
        f'stroke="{line_color}" stroke-width="4.5"/>',
        f'<text x="{left+200}" y="{legend_y-2}" font-family="sans-serif" font-size="11">actual/theory</text>',
        f'<text x="{left+390}" y="{legend_y-2}" font-family="sans-serif" font-size="10" '
        f'fill="#666">Analytical extrapolation; negative speedup is retained</text>',
        f'<rect data-role="figure-frame" x="1" y="1" width="{width-2}" height="{height-2}" '
        f'fill="none" stroke="#111" stroke-width="1.4"/>',
        '</svg>',
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=base / "results" / "h128" / "architecture" / "results.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-data", type=Path)
    parser.add_argument("--profile")
    args = parser.parse_args()
    rows = read_rows(args.input)
    profiles = sorted({_profile(row) for row in rows})
    if args.profile is not None:
        profiles = [args.profile]
    if len(profiles) != 1:
        raise ValueError("pass --profile when the input contains multiple profiles")
    profile = profiles[0]
    charts = [build_chart(rows, operator, profile) for operator in OPERATORS]
    mode = charts[0].memory_mode
    output_dir = args.output_dir or base / "figures" / profile.lower() / mode
    for chart in charts:
        write_svg(chart, output_dir / f"{chart.operator.lower()}.svg")
    plot_data = args.plot_data or args.input.with_name("plot_data.json")
    plot_data.parent.mkdir(parents=True, exist_ok=True)
    plot_data.write_text(
        json.dumps([{
            "operator": chart.operator,
            "memory_mode": chart.memory_mode,
            "tensor_profile": chart.tensor_profile,
            "estimate_source": chart.estimate_source,
            "max_tflops": chart.max_tflops,
            "max_attainment": chart.max_attainment,
            "points": [asdict(point) for point in chart.points],
            "model_spans": [asdict(span) for span in chart.model_spans],
            "placement_spans": [asdict(span) for span in chart.placement_spans],
        } for chart in charts], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"generated {len(charts)} SVG charts for {profile}/{mode} in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
