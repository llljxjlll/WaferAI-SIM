#!/usr/bin/env python3
"""Render an annotated Exp3.2 heatmap without third-party dependencies."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


HERE = Path(__file__).resolve().parent
M_VALUES = (256, 512, 1024, 2048, 4096, 8192, 16384)
K_VALUES = (65536, 16384, 4096, 1024, 256)
N = 12288


def _color(value: float, low: float, high: float) -> tuple[int, int, int]:
    """Blue-white-red palette matching the supplied heatmap style."""

    fraction = 0.5 if high <= low else max(0.0, min(1.0, (value - low) / (high - low)))
    blue, white, red = (69, 111, 170), (248, 246, 245), (198, 88, 111)
    if fraction <= 0.5:
        weight, first, second = fraction * 2, blue, white
    else:
        weight, first, second = (fraction - 0.5) * 2, white, red
    return tuple(round(first[index] * (1 - weight) + second[index] * weight) for index in range(3))


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _text_color(rgb: tuple[int, int, int]) -> str:
    luminance = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255
    return "#ffffff" if luminance < 0.48 else "#161616"


def _mn_label(m: int) -> str:
    value = m * N // 1024
    return str(value) if value < 10000 else f"{value // 1024}K"


def _k_label(k: int) -> str:
    value = k / 1024
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _pairs(results: Path) -> list[Mapping[str, object]]:
    data = json.loads(results.read_text(encoding="utf-8"))
    pairs = data.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 35 or any(not isinstance(row, Mapping) for row in pairs):
        raise ValueError("results must contain 35 object pairs")
    return pairs


def plot(results: Path, output_dir: Path, metric: str) -> Path:
    pairs = _pairs(results)
    grid = {(int(row["M"]), int(row["K"])): float(row[metric]) for row in pairs}
    expected = {(m, k) for m in M_VALUES for k in K_VALUES}
    if set(grid) != expected:
        raise ValueError("results do not cover the frozen 7x5 matrix")
    values = tuple(grid[key] for key in sorted(grid))
    low, high = min(values), max(values)
    if math.isclose(low, high):
        low, high = low - 0.01, high + 0.01

    width, height = 1120, 760
    left, top, cell_w, cell_h = 158, 74, 118, 98
    plot_w, plot_h = cell_w * len(M_VALUES), cell_h * len(K_VALUES)
    bar_x, bar_w = left + plot_w + 42, 34
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}</style>',
        f'<text x="{left + plot_w / 2:.1f}" y="36" text-anchor="middle" font-size="22" font-weight="bold">Exp3.2 intra-die ON/OFF stage speedup</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="58" text-anchor="middle" font-size="13" fill="#4b5563">fixed D=6, physical 2×3 Ring; analytical preflight model</text>',
    ]
    for row_index, k in enumerate(K_VALUES):
        for column_index, m in enumerate(M_VALUES):
            value = grid[(m, k)]
            rgb = _color(value, low, high)
            x, y = left + column_index * cell_w, top + row_index * cell_h
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{_hex(rgb)}" stroke="#ffffff" stroke-width="1.5"/>')
            parts.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 7:.1f}" text-anchor="middle" font-size="22" font-weight="bold" fill="{_text_color(rgb)}">{value:.2f}</text>')
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111827" stroke-width="2"/>')
    for index, m in enumerate(M_VALUES):
        x = left + (index + 0.5) * cell_w
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 29}" text-anchor="middle" font-size="15">{html.escape(_mn_label(m))}</text>')
    for index, k in enumerate(K_VALUES):
        y = top + (index + 0.5) * cell_h + 5
        parts.append(f'<text x="{left - 16}" y="{y:.1f}" text-anchor="end" font-size="16">{html.escape(_k_label(k))}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h + 70}" text-anchor="middle" font-size="19">M×N (×1024)</text>')
    parts.append(f'<text x="39" y="{top + plot_h / 2:.1f}" transform="rotate(-90 39 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="19">K (×1024)</text>')
    for index in range(160):
        fraction = index / 159
        rgb = _color(high - fraction * (high - low), low, high)
        y = top + fraction * plot_h
        parts.append(f'<rect x="{bar_x}" y="{y:.2f}" width="{bar_w}" height="{plot_h / 160 + 1:.2f}" fill="{_hex(rgb)}"/>')
    parts.append(f'<rect x="{bar_x}" y="{top}" width="{bar_w}" height="{plot_h}" fill="none" stroke="#111827" stroke-width="1.5"/>')
    for tick in range(5):
        fraction = tick / 4
        value, y = high - fraction * (high - low), top + fraction * plot_h
        parts.append(f'<line x1="{bar_x + bar_w}" y1="{y:.1f}" x2="{bar_x + bar_w + 7}" y2="{y:.1f}" stroke="#111827"/>')
        parts.append(f'<text x="{bar_x + bar_w + 14}" y="{y + 5:.1f}" font-size="15">{value:.2f}</text>')
    parts.append(f'<text x="{bar_x + bar_w / 2:.1f}" y="{top - 12}" text-anchor="middle" font-size="14">speedup</text>')
    parts.append('</svg>')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{metric}_heatmap.svg"
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results" / "exp3_2_results.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--metric", choices=("stage_speedup", "e2e_speedup", "attainment"), default="stage_speedup")
    args = parser.parse_args(argv)
    print(plot(args.results, args.output_dir, args.metric))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
