#!/usr/bin/env python3
"""Render the five exp2-1 result figures as dependency-free SVG."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

from build_operator_amdahl import build_and_write as build_operator_amdahl
from build_prefill_operator_amdahl import (
    build_and_write as build_prefill_operator_amdahl,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _load(name: str) -> list[dict[str, Any]]:
    value = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name}: expected a non-empty JSON array")
    return value


def _e(value: object) -> str:
    return html.escape(str(value))


def _defs() -> str:
    return """<defs>
<pattern id="capacity-hatch" width="8" height="8" patternUnits="userSpaceOnUse">
  <path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="#111" stroke-width="1.2" opacity=".55"/>
</pattern>
<pattern id="mla-hatch" width="8" height="8" patternUnits="userSpaceOnUse">
  <path d="M0,0 L8,8 M8,0 L0,8" stroke="#111" stroke-width="1.15" opacity=".58"/>
</pattern>
<pattern id="mla-capacity-hatch" width="8" height="8" patternUnits="userSpaceOnUse">
  <path d="M-2,2 L2,-2 M0,8 L8,0 M0,0 L8,8 M8,0 L0,8" stroke="#111" stroke-width="1.1" opacity=".6"/>
</pattern>

</defs>"""


def _texture(row: dict[str, Any]) -> str | None:
    if (
        row.get("estimate_source") == "analytical_only_mla"
        and row.get("capacity_status") == "capacity_infeasible_projection"
    ):
        return "mla-capacity-hatch"
    if row.get("estimate_source") == "analytical_only_mla":
        return "mla-hatch"
    if row.get("capacity_status") == "capacity_infeasible_projection":
        return "capacity-hatch"
    return None


def _bar(
    parts: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    row: dict[str, Any],
    role: str,
) -> None:
    parts.append(
        f'<rect data-role="{role}" x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
        f'height="{max(0.0, height):.2f}" fill="{color}" stroke="#222" stroke-width="3"/>'
    )
    texture = _texture(row)
    if texture:
        parts.append(
            f'<rect data-role="{role}-texture" x="{x:.2f}" y="{y:.2f}" '
            f'width="{width:.2f}" height="{max(0.0, height):.2f}" '
            f'fill="url(#{texture})" stroke="none"/>'
        )


def _speed_band(
    parts: list[str],
    rows: list[dict[str, Any]],
    centers: list[float],
    sy,
) -> None:
    upper = " ".join(
        f"{x:.2f},{sy(float(row['uncertainty_high'])):.2f}"
        for x, row in zip(centers, rows)
    )
    lower = " ".join(
        f"{x:.2f},{sy(float(row['uncertainty_low'])):.2f}"
        for x, row in reversed(list(zip(centers, rows)))
    )
    parts.append(
        f'<polygon data-role="uncertainty-band" points="{upper} {lower}" '
        f'fill="#008C95" opacity=".13" stroke="none"/>'
    )
    coords = " ".join(
        f"{x:.2f},{sy(float(row['speedup'])):.2f}"
        for x, row in zip(centers, rows)
    )
    parts.append(
        f'<polyline data-role="speedup-line" points="{coords}" fill="none" '
        f'stroke="#007C83" stroke-width="4.5" stroke-linejoin="round"/>'
    )
    for x, row in zip(centers, rows):
        low = sy(float(row["uncertainty_low"]))
        high = sy(float(row["uncertainty_high"]))
        point = sy(float(row["speedup"]))
        parts.extend([
            f'<line x1="{x:.2f}" y1="{low:.2f}" x2="{x:.2f}" y2="{high:.2f}" '
            f'stroke="#007C83" stroke-width="1.2"/>',
            f'<circle data-role="speedup-point" cx="{x:.2f}" cy="{point:.2f}" r="6.3" fill="white" '
            f'stroke="#007C83" stroke-width="3.3"/>',
        ])


def _speed_points(
    parts: list[str],
    rows: list[dict[str, Any]],
    centers: list[float],
    sy,
    connect: bool = False,
    color: str = "#007C83",
    line_role: str = "speedup-line",
    point_role: str = "speedup-point",
) -> None:
    if connect:
        coords = " ".join(
            f'{x:.2f},{sy(float(row["speedup"])):.2f}'
            for x, row in zip(centers, rows)
        )
        parts.append(
            f'<polyline data-role="{line_role}" points="{coords}" fill="none" '
            f'stroke="{color}" stroke-width="10.125" stroke-linejoin="round" '
            f'stroke-linecap="round"/>'
        )
    for x, row in zip(centers, rows):
        point = sy(float(row["speedup"]))
        parts.append(
            f'<circle data-role="{point_role}" cx="{x:.2f}" cy="{point:.2f}" '
            f'r="{13.02 if connect else 7.5}" fill="white" stroke="{color}" '
            f'stroke-width="{7.65 if connect else 3.3}"/>'
        )


def _paired_chart(
    rows: list[dict[str, Any]],
    output: Path,
    title: str,
    base_key: str,
    overlap_key: str,
    condition_key: str,
    condition_prefix: str,
    third_key: str | None = None,
    overlap_label: str = "overlap",
    third_label: str = "full train",
    line_speed_key: str = "speedup",
    line_low_key: str = "uncertainty_low",
    line_high_key: str = "uncertainty_high",
    show_uncertainty: bool = True,
    connect_speedup: bool = False,
    speed_color: str = "#007C83",
    speed_axis_max: float | None = None,
    group_key: str | None = None,
    comparison_speed_key: str | None = None,
    comparison_color: str = "#4B3F99",
    comparison_label: str = "mean operator speedup",
    group_gap_units: float = 0.9,
    bars_touch: bool = False,
    bar_width_fraction: float | None = None,
    base_color: str = "#6676A8",
    overlap_color: str = "#E08B32",
    overlap_role: str = "overlap-bar",
    expected_cases: int = 12,
    align_speed_max_to_left_one: bool = False,
) -> None:
    if len(rows) != expected_cases:
        raise ValueError(
            f"{output.name}: expected {expected_cases} cases, got {len(rows)}"
        )
    width, height = 1600, 790
    left, right, top, bottom = 88, 105, 112, 190
    pw, ph = width - left - right, height - top - bottom
    performance_keys = (
        (base_key, overlap_key, third_key)
        if third_key is not None
        else (base_key, overlap_key)
    )
    line_rows = []
    for row in rows:
        line_row = {**row, "speedup": row[line_speed_key]}
        if show_uncertainty:
            line_row["uncertainty_low"] = row[line_low_key]
            line_row["uncertainty_high"] = row[line_high_key]
        line_rows.append(line_row)
    max_perf = max(float(row[key]) for row in rows for key in performance_keys)
    bar_axis_max = 1.1
    speed_ceiling_key = "uncertainty_high" if show_uncertainty else "speedup"
    max_speed = max(
        1.1, max(float(row[speed_ceiling_key]) for row in line_rows) * 1.03
    )
    sy_perf = lambda value: top + ph * (1.0 - value / max_perf / bar_axis_max)
    if comparison_speed_key is not None:
        max_speed = max(
            max_speed, max(float(row[comparison_speed_key]) for row in rows) * 1.03
        )
    if speed_axis_max is not None:
        if speed_axis_max < max_speed:
            raise ValueError(
                f"{output.name}: fixed speed axis {speed_axis_max} is below data {max_speed}"
            )
        max_speed = speed_axis_max
    speed_headroom = bar_axis_max if align_speed_max_to_left_one else 1.0
    sy_speed = lambda value: top + ph * (
        1.0 - value / max_speed / speed_headroom
    )
    if group_key is None:
        group_w = pw / len(rows)
        centers = [left + (index + 0.5) * group_w for index in range(len(rows))]
    else:
        group_indices: list[int] = []
        group_index = 0
        previous = rows[0][group_key]
        for row in rows:
            if row[group_key] != previous:
                group_index += 1
                previous = row[group_key]
            group_indices.append(group_index)
        group_w = pw / (len(rows) + group_gap_units * group_index)
        centers = [
            left + (index + 0.5 + group_gap_units * group_indices[index]) * group_w
            for index in range(len(rows))
        ]
    default_bar_fraction = 0.20 if third_key is not None else 0.28
    bw = group_w * (
        default_bar_fraction if bar_width_fraction is None else bar_width_fraction
    )

    if comparison_speed_key is not None:
        metric_description = "lines: system and mean operator speedup"
    else:
        metric_description = (
            "line: selected T_base/T_optimized"
            if show_uncertainty or connect_speedup
            else "points: selected T_base/T_optimized"
        )
    uncertainty_description = (
        "band = model uncertainty."
        if show_uncertainty
        else "speedup line omits model-uncertainty bands."
    )

    axis_description = (
        "left scale reserves 10% unlabeled headroom"
        if align_speed_max_to_left_one
        else f"axis maximum {bar_axis_max:.1f} (left)"
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _defs(),
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="sans-serif" '
        f'font-size="22" font-weight="bold">{_e(title)}</text>',
        f'<text x="{width/2}" y="58" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" fill="#444">Bars: normalized to figure maximum; {axis_description}; {metric_description} '
        f'(right). Analytical projections; heights are comparable only within this figure.</text>',
        f'<text x="{width/2}" y="78" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" fill="#8B1A1A">Hatched = capacity-infeasible projection; '
        f'cross-hatched = DeepSeek MLA analytical-only; {uncertainty_description}</text>',
    ]
    left_ticks = (
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        if align_speed_max_to_left_one
        else tuple(bar_axis_max * tick / 5 for tick in range(6))
    )
    for normalized_value in left_ticks:
        y = top + ph * (1 - normalized_value / bar_axis_max)
        tick_label = (
            f"{normalized_value:g}"
            if align_speed_max_to_left_one
            else f"{normalized_value:.2f}"
        )
        parts.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
            f'stroke="#B5B5B5"/>',
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="10">{tick_label}</text>',
        ])
    for tick in range(6):
        speed_value = max_speed * tick / 5
        speed_label = (
            f"{speed_value:g}"
            if align_speed_max_to_left_one
            else f"{speed_value:.2f}"
        )
        parts.append(
            f'<text x="{width-right+8}" y="{sy_speed(speed_value)+4:.2f}" '
            f'font-family="sans-serif" font-size="10">{speed_label}×</text>'
        )
    y1 = sy_speed(1.0)
    parts.append(
        f'<line data-role="one-x-line" x1="{left}" y1="{y1:.2f}" '
        f'x2="{width-right}" y2="{y1:.2f}" stroke="#A21D1D" '
        f'stroke-width="1.6" stroke-dasharray="7,5"/>'
    )
    for index, row in enumerate(rows):
        center = centers[index]
        base = float(row[base_key])
        opt = float(row[overlap_key])
        third = float(row[third_key]) if third_key is not None else None
        yb, yo = sy_perf(base), sy_perf(opt)
        if third is None:
            bar_gap = 0.0 if bars_touch else 1.0
            _bar(parts, center-bw-bar_gap, yb, bw, top+ph-yb, base_color, row, "base-bar")
            _bar(parts, center+bar_gap, yo, bw, top+ph-yo, overlap_color, row, overlap_role)
        else:
            _bar(parts, center-1.5*bw-1, yb, bw, top+ph-yb, "#6676A8", row, "base-bar")
            _bar(parts, center-.5*bw, yo, bw, top+ph-yo, "#E08B32", row, "forward-only-bar")
            yt = sy_perf(third)
            _bar(parts, center+.5*bw+1, yt, bw, top+ph-yt, "#3A9D70", row, "full-train-bar")
        parts.extend([
            f'<text x="{center:.2f}" y="{top+ph+20}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10">{_e(row["model"])}</text>',
            f'<text x="{center:.2f}" y="{top+ph+36}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10">{condition_prefix}'
            f'{_e(row[condition_key])}</text>',
        ])
    if show_uncertainty:
        _speed_band(parts, line_rows, centers, sy_speed)
    else:
        _speed_points(
            parts, line_rows, centers, sy_speed, connect_speedup, speed_color,
            "system-speedup-line" if comparison_speed_key else "speedup-line",
            "system-speedup-point" if comparison_speed_key else "speedup-point",
        )
        if comparison_speed_key is not None:
            comparison_rows = [
                {**row, "speedup": row[comparison_speed_key]} for row in rows
            ]
            _speed_points(
                parts,
                comparison_rows,
                centers,
                sy_speed,
                True,
                comparison_color,
                "operator-speedup-line",
                "operator-speedup-point",
            )
    legend_y = height - 52
    speed_x = left + (310 if third_key is not None else 190)
    parts.extend([
        f'<rect x="{left}" y="{legend_y-13}" width="18" height="13" fill="{base_color}"/>',
        f'<text x="{left+25}" y="{legend_y-2}" font-family="sans-serif" font-size="11">base</text>',
        f'<rect x="{left+80}" y="{legend_y-13}" width="18" height="13" fill="{overlap_color}"/>',
        f'<text x="{left+105}" y="{legend_y-2}" font-family="sans-serif" font-size="11">{_e(overlap_label)}</text>',
        *([
            f'<rect x="{left+190}" y="{legend_y-13}" width="18" height="13" fill="#3A9D70"/>',
            f'<text x="{left+215}" y="{legend_y-2}" font-family="sans-serif" font-size="11">{_e(third_label)}</text>',
        ] if third_key is not None else []),
        *([
            f'<line x1="{speed_x}" y1="{legend_y-7}" x2="{speed_x+40}" y2="{legend_y-7}" stroke="#007C83" stroke-width="4.5"/>',
            f'<text x="{speed_x+50}" y="{legend_y-2}" font-family="sans-serif" font-size="11">speedup ± uncertainty</text>',
        ] if show_uncertainty else ([
            f'<line x1="{speed_x}" y1="{legend_y-7}" x2="{speed_x+40}" y2="{legend_y-7}" stroke="{speed_color}" stroke-width="10.125" stroke-linecap="round"/>',
            f'<circle cx="{speed_x+20}" cy="{legend_y-7}" r="13.02" fill="white" stroke="{speed_color}" stroke-width="7.65"/>',
            f'<text x="{speed_x+50}" y="{legend_y-2}" font-family="sans-serif" font-size="11">system speedup</text>',
            *([
                f'<line x1="{speed_x+190}" y1="{legend_y-7}" x2="{speed_x+230}" y2="{legend_y-7}" stroke="{comparison_color}" stroke-width="10.125" stroke-linecap="round"/>',
                f'<circle cx="{speed_x+210}" cy="{legend_y-7}" r="13.02" fill="white" stroke="{comparison_color}" stroke-width="7.65"/>',
                f'<text x="{speed_x+240}" y="{legend_y-2}" font-family="sans-serif" font-size="11">{_e(comparison_label)}</text>',
            ] if comparison_speed_key is not None else []),
        ] if connect_speedup else [
            f'<circle cx="{speed_x+8}" cy="{legend_y-7}" r="7.5" fill="white" stroke="{speed_color}" stroke-width="3.3"/>',
            f'<text x="{speed_x+22}" y="{legend_y-2}" font-family="sans-serif" font-size="11">speedup</text>',
        ])),
        f'<text x="21" y="{top+ph/2}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" transform="rotate(-90 21 {top+ph/2})">normalized bar height</text>',
        f'<text x="{width-18}" y="{top+ph/2}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" transform="rotate(90 {width-18} {top+ph/2})">speedup</text>',
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" '
        f'stroke="#222" stroke-width="1.2"/>',
        '</svg>',
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _breakdown_chart(rows: list[dict[str, Any]], output: Path) -> None:
    if len(rows) != 12:
        raise ValueError(f"{output.name}: expected 12 cases, got {len(rows)}")
    width, height = 1600, 760
    left, right, top, bottom = 88, 105, 110, 175
    pw, ph = width-left-right, height-top-bottom
    max_total = max(float(row[key]) for row in rows for key in ("T_base_cycles", "T_overlap_cycles"))
    bar_axis_max = 1.1
    max_speed = 4.5
    sy = lambda value: top + ph*(1-value/max_total/bar_axis_max)
    sy_speed = lambda value: top + ph*(1-value/max_speed/bar_axis_max)
    group_indices: list[int] = []
    group_index = 0
    previous = rows[0]["model_id"]
    for row in rows:
        if row["model_id"] != previous:
            group_index += 1
            previous = row["model_id"]
        group_indices.append(group_index)
    group_w = pw / (len(rows) + 0.45 * group_index)
    centers = [
        left + (index + 0.5 + 0.45 * group_indices[index]) * group_w
        for index in range(len(rows))
    ]
    bw = group_w * .28
    phases = (
        ("prefill", "#4C78A8"),
        ("handoff", "#F58518"),
        ("handoff_wait", "#9C6ADE"),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _defs(),
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="sans-serif" '
        f'font-size="22" font-weight="bold">Inference TTFT: prefill / KV handoff / wait</text>',
        f'<text x="{width/2}" y="59" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" fill="#444">S={{2304, 36864}}; side-by-side stacked base/overlap bars; '
        f'bars normalized to figure maximum with 10% unlabeled headroom; right 4.5× aligns with left 1.</text>',
        f'<text x="{width/2}" y="79" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" fill="#8B1A1A">Texture conventions match the E2E figures; '
        f'capacity status is shown separately for every model/sequence case.</text>',
    ]
    for normalized_value in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y=top+ph*(1-normalized_value/bar_axis_max)
        parts.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#B5B5B5"/>',
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" '
            f'font-size="10">{normalized_value:g}</text>',
        ])
    for tick in range(6):
        speed_value=max_speed*tick/5
        parts.append(
            f'<text x="{width-right+8}" y="{sy_speed(speed_value)+4:.2f}" '
            f'font-family="sans-serif" font-size="10">{speed_value:g}×</text>'
        )
    y1=sy_speed(1.0)
    parts.append(
        f'<line data-role="one-x-line" x1="{left}" y1="{y1:.2f}" x2="{width-right}" '
        f'y2="{y1:.2f}" stroke="#A21D1D" stroke-width="1.6" stroke-dasharray="7,5"/>'
    )
    for i,row in enumerate(rows):
        center=centers[i]
        for variant, x in (("base",center-bw-1),("overlap",center+1)):
            cumulative=0.0
            total=float(row[f"T_{variant}_cycles"])
            for phase,color in phases:
                value=float(row[f"{phase}_cycles_{variant}"])
                y_top=sy(cumulative+value)
                y_bottom=sy(cumulative)
                _bar(parts,x,y_top,bw,y_bottom-y_top,color,row,f"{variant}-{phase}")
                cumulative += value
            if abs(cumulative-total) > max(1.0,total*1e-9):
                raise ValueError(f"{row['case_id']}: phase sum does not equal TTFT")
        parts.extend([
            f'<text x="{center:.2f}" y="{top+ph+21}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10">{_e(row["model"])}</text>',
            f'<text x="{center:.2f}" y="{top+ph+37}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="9">S={_e(row["seq_len"])}</text>',
            f'<text x="{center:.2f}" y="{top+ph+52}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="8">{_e(row["capacity_status"])}</text>',
        ])
    _speed_band(parts, rows, centers, sy_speed)
    legend_y=height-48
    lx=left
    for phase,color in phases:
        parts.extend([
            f'<rect x="{lx}" y="{legend_y-13}" width="18" height="13" fill="{color}"/>',
            f'<text x="{lx+24}" y="{legend_y-2}" font-family="sans-serif" font-size="11">{phase}</text>',
        ])
        lx += 145
    parts.extend([
        f'<line x1="{lx}" y1="{legend_y-7}" x2="{lx+40}" y2="{legend_y-7}" '
        f'stroke="#007C83" stroke-width="4.5"/>',
        f'<text x="{lx+48}" y="{legend_y-2}" font-family="sans-serif" font-size="11">speedup ± uncertainty</text>',
        f'<text x="21" y="{top+ph/2}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" transform="rotate(-90 21 {top+ph/2})">normalized bar height</text>',
        f'<text x="{width-18}" y="{top+ph/2}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" transform="rotate(90 {width-18} {top+ph/2})">speedup</text>',
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#222"/>',
        '</svg>',
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts)+"\n",encoding="utf-8")


def main() -> int:
    training=_load("training_e2e.json")
    operator_amdahl = build_operator_amdahl()
    operator_by_training_case = {
        row["training_case_id"]: row for row in operator_amdahl
    }
    training = [
        {**row, "operator_speedup_mean": operator_by_training_case[row["case_id"]]["operator_speedup_mean"]}
        for row in training
    ]
    decode=_load("inference_decode_e2e.json")
    prefill=_load("inference_prefill_pd_breakdown.json")
    prefill_operator_amdahl = build_prefill_operator_amdahl()
    operator_by_prefill_case = {
        row["prefill_case_id"]: row for row in prefill_operator_amdahl
    }
    prefill_main = [
        {
            **row,
            "prefill_speedup": (
                float(row["prefill_cycles_base"])
                / float(row["prefill_cycles_overlap"])
            ),
            "operator_speedup_mean": operator_by_prefill_case[row["case_id"]][
                "operator_speedup_mean"
            ],
        }
        for row in prefill
    ]
    request=_load("inference_request_e2e.json")
    _paired_chart(
        training,
        FIGURES/"training_e2e.svg",
        "Training full-step throughput",
        "training_tokens_per_s_base",
        "training_tokens_per_s_full_train",
        "seq_len",
        "S=",
        overlap_label="full-train",
        line_speed_key="speedup_full_train",
        show_uncertainty=False,
        connect_speedup=True,
        speed_color="#A23B72",
        comparison_speed_key="operator_speedup_mean",
        speed_axis_max=4.5,
        comparison_color="#4B3F99",
        comparison_label="mean operator speedup",
        group_key="model_id",
        group_gap_units=0.45,
        bars_touch=True,
        bar_width_fraction=0.42,
        base_color="#75635B",
        overlap_color="#D9822B",
        overlap_role="full-train-bar",
        align_speed_max_to_left_one=True,
    )
    _paired_chart(
        request,
        FIGURES/"inference_request_e2e.svg",
        "Combined request result: TTFT + 512 x local TPOT",
        "request_output_tokens_per_s_base",
        "request_output_tokens_per_s_overlap",
        "batch_size",
        "B=",
    )
    _paired_chart(
        decode,
        FIGURES/"inference_decode_e2e.svg",
        "Two-token local decode system throughput",
        "system_decode_tokens_per_s_base",
        "system_decode_tokens_per_s_overlap",
        "batch_size",
        "B=",
    )
    _paired_chart(
        prefill_main,
        FIGURES/"inference_prefill_e2e.svg",
        "Inference prefill latency",
        "prefill_cycles_base",
        "prefill_cycles_overlap",
        "seq_len",
        "S=",
        line_speed_key="prefill_speedup",
        show_uncertainty=False,
        connect_speedup=True,
        speed_color="#C23B22",
        speed_axis_max=4.5,
        comparison_speed_key="operator_speedup_mean",
        comparison_color="#2F4B7C",
        comparison_label="mean operator speedup",
        group_key="model_id",
        group_gap_units=0.45,
        bars_touch=True,
        bar_width_fraction=0.42,
        base_color="#6B5B95",
        overlap_color="#D9A441",
        expected_cases=12,
        align_speed_max_to_left_one=True,
    )
    _breakdown_chart(prefill,FIGURES/"inference_prefill_pd_breakdown.svg")
    plot_data={
        "schema_version":"exp2.plot_data.v7",
        "source_files":[
            "results/training_e2e.json",
            "results/training_operator_amdahl.json",
            "results/prefill_operator_amdahl.json",
            "results/inference_decode_e2e.json",
            "results/inference_request_e2e.json",
            "results/inference_prefill_pd_breakdown.json",
        ],
        "training_cases":len(training),
        "training_operator_amdahl_cases":len(operator_amdahl),
        "prefill_operator_amdahl_cases":len(prefill_operator_amdahl),
        "decode_cases":len(decode),
        "inference_request_cases":len(request),
        "prefill_pd_cases":len(prefill),
        "figures":[
            "figures/training_e2e.svg",
            "figures/inference_decode_e2e.svg",
            "figures/inference_prefill_e2e.svg",
            "figures/inference_prefill_pd_breakdown.svg",
            "figures/inference_request_e2e.svg",
        ],
    }
    (RESULTS/"plot_data.json").write_text(
        json.dumps(plot_data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("generated 5 SVG figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

