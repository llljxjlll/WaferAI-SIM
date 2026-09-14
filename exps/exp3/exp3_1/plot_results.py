#!/usr/bin/env python3
"""Create dependency-free SVG summaries from Exp3.1 paired comparisons."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


COMPARISONS = (
    "native_full", "native_inter_only", "gpu_inter",
)
COLORS = {
    "native_full": "#3b82f6", "native_inter_only": "#10b981",
    "gpu_inter": "#f97316",
}


def _svg(rows: Sequence[Mapping[str, object]], title: str, output: Path) -> None:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["D"]), str(row["comparison"]))].append(float(row["speedup"]))
    width, height = 820, 470
    left, top, plot_w, plot_h = 80, 65, 680, 315
    values = [value for items in grouped.values() for value in items]
    y_max = max(1.05, max(values, default=1.0) * 1.12)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = top + plot_h - plot_h * tick / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>')
    group_w, bar_w = plot_w / 3, 48
    for d_index, dies in enumerate((6, 9, 36)):
        center = left + group_w * (d_index + 0.5)
        for c_index, comparison in enumerate(COMPARISONS):
            avg = mean(grouped[(dies, comparison)])
            x = center + (c_index - (len(COMPARISONS) - 1) / 2) * (bar_w + 7) - bar_w / 2
            bar_h = plot_h * avg / y_max
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{COLORS[comparison]}"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">{avg:.3f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-family="sans-serif" font-size="13">D={dies}</text>')
    legend_y = height - 32
    for index, comparison in enumerate(COMPARISONS):
        x = 105 + index * 225
        parts.append(f'<rect x="{x}" y="{legend_y-12}" width="14" height="14" fill="{COLORS[comparison]}"/>')
        parts.append(f'<text x="{x+20}" y="{legend_y}" font-family="sans-serif" font-size="12">{comparison}</text>')
    parts.append('<text x="18" y="225" transform="rotate(-90 18 225)" text-anchor="middle" font-family="sans-serif" font-size="13">relative speedup (baseline/optimized)</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _slug(value: object) -> str:
    """Return a stable filesystem-safe label without adding dependencies."""
    text = str(value).lower()
    normalized = "".join(char if char.isascii() and char.isalnum() else "_" for char in text)
    return normalized.strip("_")


def _shape_key(row: Mapping[str, object]) -> tuple[str, str, str, int]:
    return (
        str(row["operator_family"]), str(row["model_or_moe_config"]),
        str(row["stage"]), int(row["S"]),
    )


def plot(results: Path, output_dir: Path) -> dict[str, object]:
    document = json.loads(results.read_text(encoding="utf-8"))
    comparisons = document.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 144:
        raise ValueError("results must contain 144 paired comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_files = []
    for family in ("gemm_rs", "dispatch_gemm"):
        for seq_len in (2304, 36864):
            rows = [r for r in comparisons if r["operator_family"] == family and int(r["S"]) == seq_len]
            path = output_dir / f"speedup_aggregate_{family}_s{seq_len}.svg"
            _svg(rows, f"Exp3.1 aggregate over 4 shapes: {family}, S={seq_len}", path)
            aggregate_files.append(str(path))
    summary_path = output_dir / "speedup_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["operator_family", "S", "D", "comparison", "mean_speedup", "min_speedup", "max_speedup", "case_count"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        buckets: dict[tuple[object, ...], list[float]] = defaultdict(list)
        for row in comparisons:
            key = (row["operator_family"], row["S"], row["D"], row["comparison"])
            buckets[key].append(float(row["speedup"]))
        for key in sorted(buckets, key=lambda x: (str(x[0]), int(x[1]), int(x[2]), str(x[3]))):
            values = buckets[key]
            writer.writerow(dict(zip(fields[:4], key)) | {
                "mean_speedup": mean(values), "min_speedup": min(values),
                "max_speedup": max(values), "case_count": len(values),
            })
    detail_path = output_dir / "speedup_by_shape.csv"
    detail_fields = [
        "case_id", "operator_family", "model_or_moe_config", "stage", "S", "D",
        "comparison", "baseline_state", "optimized_state", "speedup", "ideal_speedup", "attainment",
    ]
    with detail_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(
            comparisons,
            key=lambda row: (
                str(row["operator_family"]), str(row["model_or_moe_config"]),
                str(row["stage"]), int(row["S"]), int(row["D"]), str(row["comparison"]),
            ),
        ))

    by_shape: dict[tuple[str, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in comparisons:
        by_shape[_shape_key(row)].append(row)
    detail_files = []
    for (family, model, stage, seq_len), rows in sorted(by_shape.items()):
        if len(rows) != 9:
            raise ValueError(
                f"{family}/{model}/{stage}/S={seq_len} must contain 3 D x 3 comparisons"
            )
        filename = (
            f"speedup_by_shape_{family}_{_slug(model)}_{_slug(stage)}_s{seq_len}.svg"
        )
        path = output_dir / filename
        _svg(rows, f"Exp3.1 {family}: {model}, {stage}, S={seq_len}", path)
        detail_files.append(str(path))
    return {
        "figures": detail_files,
        "aggregate_figures": aggregate_files,
        "summary_csv": str(summary_path), "by_shape_csv": str(detail_path),
        "placeholder": document.get("status") == "placeholder_smoke_test",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(json.dumps(plot(args.results, args.output_dir), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
