#!/usr/bin/env python3
"""Plot exp1_1 results as one grouped-bar chart per mesh/operator."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


MESHES = ("1x4", "2x3", "3x3", "6x6")
OPERATORS = ("AG_GEMM", "GEMM_RS")
LEGACY_SERIES = (
    ("theory_time", "Theory", "#4c78a8"),
    ("swizzle_time", "Swizzle", "#f58518"),
    ("naive_time", "NAIVE", "#54a24b"),
)
T_SERIES = (
    ("T00_cycles", "T00 naive/naive", "#4c78a8"),
    ("T10_cycles", "T10 inter", "#f58518"),
    ("T01_cycles", "T01 intra", "#54a24b"),
    ("T11_cycles", "T11 combined", "#e45756"),
)
METRIC_SERIES = (
    ("inter_speedup_without_intra", "Inter only", "#f58518"),
    ("intra_speedup_without_inter", "Intra only", "#54a24b"),
    ("total_speedup", "Total", "#e45756"),
    ("synergy", "Synergy", "#72b7b2"),
    ("congestion_factor", "Congestion", "#b279a2"),
)
BAD_STATUSES = {"failed", "failure", "error", "unsupported"}


def positive_float(value: object) -> float | None:
    """Return a finite positive float, or None for an empty/failed value."""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def case_label(row: dict[str, str]) -> str:
    model = row.get("model", "?").strip() or "?"
    layer = row.get("layer", "?").strip() or "?"
    seq = row.get("seq_len", "?").strip() or "?"
    return f"{model}\n{layer}\nS={seq}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"mesh", "operator", "model", "layer", "seq_len"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
        if not {"logical_flops", "flops"}.intersection(reader.fieldnames or ()):
            raise ValueError("missing CSV column: logical_flops (or legacy flops)")
        return list(reader)


def result_series(rows: list[dict[str, str]]):
    return T_SERIES if any(row.get("T11_cycles") for row in rows) else LEGACY_SERIES


def values_for(row: dict[str, str], series=None) -> list[float | None]:
    series = series or (T_SERIES if row.get("T11_cycles") else LEGACY_SERIES)
    flops = positive_float(row.get("logical_flops") or row.get("flops"))
    if flops is None:
        return [None] * len(series)
    values: list[float | None] = []
    for field, _, _ in series:
        if field.endswith("_cycles"):
            cycles = positive_float(row.get(field))
            values.append(flops / cycles / 1e3 if cycles else None)
            continue
        branch = field.removesuffix("_time")
        if branch != "theory":
            status = row.get(f"{branch}_status", "").strip().lower()
            if status in BAD_STATUSES:
                values.append(None)
                continue
        elapsed = positive_float(row.get(field))
        values.append(flops / elapsed / 1e12 if elapsed else None)
    return values


def plot_matplotlib(rows: list[dict[str, str]], title: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [case_label(row) for row in rows]
    series = result_series(rows)
    values = [values_for(row, series) for row in rows]
    width = 0.8 / len(series)
    figure_width = max(12.0, len(rows) * 0.75)
    fig, ax = plt.subplots(figsize=(figure_width, 6.5))

    for series_index, (_, name, color) in enumerate(series):
        offset = (series_index - (len(series) - 1) / 2) * width
        xs = [index + offset for index in range(len(rows))]
        heights = [item[series_index] if item[series_index] is not None else math.nan for item in values]
        ax.bar(xs, heights, width=width, label=name, color=color)

    ax.set_title(title)
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_xticks(range(len(rows)), labels, rotation=55, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_metrics_matplotlib(
    rows: list[dict[str, str]], title: str, output: Path
) -> None:
    import matplotlib.pyplot as plt

    labels = [case_label(row) for row in rows]
    values = [
        [positive_float(row.get(field)) for field, _, _ in METRIC_SERIES]
        for row in rows
    ]
    width = 0.8 / len(METRIC_SERIES)
    fig, ax = plt.subplots(figsize=(max(12.0, len(rows) * 0.75), 6.5))
    for series_index, (_, name, color) in enumerate(METRIC_SERIES):
        offset = (series_index - (len(METRIC_SERIES) - 1) / 2) * width
        heights = [
            item[series_index] if item[series_index] is not None else math.nan
            for item in values
        ]
        ax.bar(
            [index + offset for index in range(len(rows))], heights,
            width=width, label=name, color=color,
        )
    ax.axhline(1.0, color="#333", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Ratio (higher speedup/synergy; congestion >= 1 is worse)")
    ax.set_xticks(range(len(rows)), labels, rotation=55, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_svg(
    rows: list[dict[str, str]], title: str, output: Path, *,
    series=None, metrics: bool = False,
) -> None:
    """Small dependency-free fallback used when matplotlib is unavailable."""
    series = series or result_series(rows)
    values = (
        [[positive_float(row.get(field)) for field, _, _ in series] for row in rows]
        if metrics else [values_for(row, series) for row in rows]
    )
    finite = [value for group in values for value in group if value is not None]
    y_max = max(finite, default=1.0)
    left, top, bottom = 80, 55, 155
    group_width = 72
    width = max(900, left + 30 + len(rows) * group_width)
    height = 620
    plot_height = height - top - bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" '
        f'font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - 20}" '
        f'y2="{top + plot_height}" stroke="#333"/>',
    ]
    for tick in range(6):
        y = top + plot_height * (1 - tick / 5)
        value = y_max * tick / 5
        parts.extend((
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - 20}" y2="{y:.1f}" '
            'stroke="#ddd"/>',
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="10">{value:.3g}</text>',
        ))
    bar_width = min(16, 54 / len(series))
    for index, (row, group) in enumerate(zip(rows, values)):
        center = left + 35 + index * group_width
        for series_index, value in enumerate(group):
            if value is None:
                continue
            bar_height = plot_height * value / y_max
            x = center + (series_index - (len(series) - 1) / 2) * bar_width
            y = top + plot_height - bar_height
            color = series[series_index][2]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 1}" '
                f'height="{bar_height:.1f}" fill="{color}"/>'
            )
        label = html.escape(case_label(row).replace("\n", " / "))
        parts.append(
            f'<text x="{center + 5}" y="{top + plot_height + 12}" '
            f'transform="rotate(55 {center + 5} {top + plot_height + 12})" '
            f'font-family="sans-serif" font-size="9">{label}</text>'
        )
    legend_x = width - 300
    for index, (_, name, color) in enumerate(series):
        x = legend_x + index * 95
        parts.extend((
            f'<rect x="{x}" y="36" width="13" height="13" fill="{color}"/>',
            f'<text x="{x + 18}" y="47" font-family="sans-serif" font-size="11">{name}</text>',
        ))
    parts.append(
        f'<text x="18" y="{top + plot_height / 2}" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + plot_height / 2})" '
        f'font-family="sans-serif" font-size="12">'
        f'{"Ratio" if metrics else "Performance (TFLOP/s)"}</text>'
    )
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base / "results" / "results.csv")
    parser.add_argument("--output-dir", type=Path, default=base / "figures")
    parser.add_argument("--format", choices=("svg", "png"), default="svg")
    args = parser.parse_args()

    rows = read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: F401
        use_matplotlib = True
    except ImportError:
        use_matplotlib = False

    generated: list[Path] = []
    for mesh in MESHES:
        for operator in OPERATORS:
            selected = [
                row for row in rows
                if row.get("mesh", "").strip().lower() == mesh
                and row.get("operator", "").strip().upper() == operator
            ]
            stem = f"{mesh}_{operator.lower()}"
            title = f"{mesh} — {operator}"
            if use_matplotlib:
                output = args.output_dir / f"{stem}.{args.format}"
                plot_matplotlib(selected, title, output)
            else:
                output = args.output_dir / f"{stem}.svg"
                plot_svg(selected, title, output)
            generated.append(output)
            if result_series(selected) == T_SERIES:
                metrics_output = args.output_dir / f"{stem}_metrics.{args.format if use_matplotlib else 'svg'}"
                if use_matplotlib:
                    plot_metrics_matplotlib(
                        selected, f"{title} — speedup, synergy, congestion",
                        metrics_output,
                    )
                else:
                    plot_svg(
                        selected, f"{title} — speedup, synergy, congestion",
                        metrics_output, series=METRIC_SERIES, metrics=True,
                    )
                generated.append(metrics_output)

    print(f"generated {len(generated)} figures in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
