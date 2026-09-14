#!/usr/bin/env python3
"""Render the annotated Dispatch+GEMM Exp3.2 heatmap without dependencies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from plot_results import _color, _hex, _text_color

HERE = Path(__file__).resolve().parent
M_RANK_VALUES = (64, 256, 1024, 4096, 16384)
I_VALUES = (131072, 32768, 8192, 2048, 512)
H_PRIMARY = 7168


def _activation_label(m_rank: int) -> str:
    value = m_rank * H_PRIMARY // 1024
    return str(value) if value < 10_000 else f"{value // 1024}K"


def _i_label(intermediate: int) -> str:
    value = intermediate / 1024
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _main_pairs(results: Path) -> list[Mapping[str, object]]:
    data = json.loads(results.read_text(encoding="utf-8"))
    pairs = data.get("main_pairs")
    if not isinstance(pairs, list) or len(pairs) != 25 or any(not isinstance(row, Mapping) for row in pairs):
        raise ValueError("results must contain exactly 25 Dispatch main pairs")
    return pairs


def plot(results: Path, output_dir: Path, metric: str) -> Path:
    pairs = _main_pairs(results)
    grid = {(int(pair["M_rank"]), int(pair["I"])): float(pair[metric]) for pair in pairs}
    expected = {(m_rank, intermediate) for m_rank in M_RANK_VALUES for intermediate in I_VALUES}
    if set(grid) != expected:
        raise ValueError("results do not cover the frozen 5x5 Dispatch matrix")
    values = tuple(grid[key] for key in sorted(grid))
    low, high = min(values), max(values)
    if math.isclose(low, high):
        low, high = low - 0.01, high + 0.01

    width, height = 930, 700
    left, top, cell_w, cell_h = 155, 80, 124, 96
    plot_w, plot_h = cell_w * len(M_RANK_VALUES), cell_h * len(I_VALUES)
    bar_x, bar_w = left + plot_w + 40, 34
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}</style>',
        f'<text x="{left + plot_w / 2:.1f}" y="36" text-anchor="middle" font-size="22" font-weight="bold">Exp3.2 Dispatch+GEMM intra-die ON/OFF stage speedup</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="58" text-anchor="middle" font-size="13" fill="#4b5563">gate/up; H=7168; fixed D=6, physical 2×3 personalized-A2A Ring; analytical preflight model</text>',
    ]
    for row_index, intermediate in enumerate(I_VALUES):
        for column_index, m_rank in enumerate(M_RANK_VALUES):
            value = grid[(m_rank, intermediate)]
            rgb = _color(value, low, high)
            x, y = left + column_index * cell_w, top + row_index * cell_h
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{_hex(rgb)}" stroke="#ffffff" stroke-width="1.5"/>')
            parts.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 7:.1f}" text-anchor="middle" font-size="22" font-weight="bold" fill="{_text_color(rgb)}">{value:.2f}</text>')
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111827" stroke-width="2"/>')
    for index, m_rank in enumerate(M_RANK_VALUES):
        x = left + (index + 0.5) * cell_w
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 29}" text-anchor="middle" font-size="15">{_activation_label(m_rank)}</text>')
    for index, intermediate in enumerate(I_VALUES):
        y = top + (index + 0.5) * cell_h + 5
        parts.append(f'<text x="{left - 16}" y="{y:.1f}" text-anchor="end" font-size="16">{_i_label(intermediate)}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h + 70}" text-anchor="middle" font-size="19">M_rank×H (×1024)</text>')
    parts.append(f'<text x="39" y="{top + plot_h / 2:.1f}" transform="rotate(-90 39 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="19">I (×1024)</text>')
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
    parts.append(f'<text x="{bar_x + bar_w / 2:.1f}" y="{top - 12}" text-anchor="middle" font-size="14">{metric}</text>')
    parts.append("</svg>")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"dispatch_gemm_{metric}_heatmap.svg"
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results" / "dispatch_gemm" / "exp3_2_dispatch_gemm_results.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--metric", choices=("stage_speedup", "e2e_speedup", "attainment"), default="stage_speedup")
    args = parser.parse_args(argv)
    print(plot(args.results, args.output_dir, args.metric))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
