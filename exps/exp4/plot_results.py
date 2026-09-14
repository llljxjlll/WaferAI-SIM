#!/usr/bin/env python3
"""Render the two Exp-4 primary figures as dependency-free SVG."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


COLORS = ("#9677b8", "#e8d8fc", "#fb9a99", "#fdbf6f", "#b2df8a", "#33a02c", "#e31a1c")
MODEL_LABELS = {
    "llama2_7b": "LLaMA2-7B", "gpt3_175b": "GPT3-175B",
    "llama3_8b": "LLaMA3-8B", "llama3_1_405b": "LLaMA3.1-405B",
    "mixtral_8x7b": "Mixtral-8x7B", "deepseek_v3": "DeepSeek-V3",
    "geomean_6_models": "6-model geomean",
}


def _svg(width: int, height: int, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1}'
            '.grid{stroke:#d1d5db;stroke-width:1;stroke-dasharray:3 4}.legend{font-size:12px}'
            '.tick{font-size:11px}.title{font-size:17px;font-weight:700}</style>\n'
            f'<rect width="{width}" height="{height}" fill="white"/>\n' +
            "\n".join(body) + "\n</svg>\n")


def _text(x: float, y: float, value: Any, css: str = "", anchor: str = "middle", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" text-anchor="{anchor}"{transform}>{html.escape(str(value))}</text>'



def _star_path(cx: float, cy: float, outer: float, inner: float) -> str:
    points = []
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = -math.pi / 2 + index * math.pi / 5
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return " ".join(("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
                    for index, (x, y) in enumerate(points)) + " Z"

def render_speedup(summary: dict[str, Any], output: str | Path) -> None:
    rows = summary.get("speedup_figure", {}).get("rows", [])
    width, height = 1280, 480
    body = [_text(width / 2, 26, "Software / hardware optimization speedup", "title")]
    kinds = (("training", "Training"), ("prefill", "Inference prefill"), ("decode", "Inference decode"))
    # Reddish purple, amber, and sky blue: harmonious and color-vision friendly.
    series = (("sw_opt_only", "SW only", "#CC79A7"),
              ("hw_opt_only", "HW only", "#E69F00"),
              ("sw_hw_opt", "SW + HW", "#56B4E9"))
    ymax = max([1.0] + [float(r[s[0]]) for r in rows for s in series if s[0] in r]) * 1.1
    top, bottom = 55, 395
    left, right = 65, 1260
    body += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
             f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>']
    for tick in range(5):
        value = ymax * tick / 4
        y = bottom - (bottom - top) * tick / 4
        body += [f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>',
                 _text(left - 7, y + 4, f"{value:.2g}×", "tick", "end")]

    # Keep the original outer bounds with a compact fixed gap between groups.
    group_gap = 6
    group_width = (1210 - 75 - 2 * group_gap) / 3
    group_ranges = tuple((75 + i * (group_width + group_gap),
                          75 + i * (group_width + group_gap) + group_width) for i in range(3))
    for i in range(2):
        divider_x = (group_ranges[i][1] + group_ranges[i + 1][0]) / 2
        body.append(f'<line x1="{divider_x:.1f}" y1="{top + 8}" x2="{divider_x:.1f}" y2="{bottom - 8}" '
                    'stroke="#9ca3af" stroke-width="1" stroke-dasharray="2 6"/>')
    for pi, (kind, label) in enumerate(kinds):
        group_left, group_right = group_ranges[pi]
        body.append(_text((group_left + group_right) / 2, 48, label))
        panel = [r for r in rows if r.get("workload_type") == kind]
        order = [m for m in MODEL_LABELS if m != "geomean_6_models" and any(r.get("model_id") == m for r in panel)]
        xs = {m: group_left + (group_right - group_left) * (i + .5) / max(1, len(order)) for i, m in enumerate(order)}
        for i, model in enumerate(order):
            body.append(_text(xs[model], bottom + 16 + (i % 2) * 13, MODEL_LABELS[model], "tick"))
        for field, slabel, color in series:
            pts = []
            for model in order:
                row = next(r for r in panel if r.get("model_id") == model)
                y = bottom - (bottom - top) * float(row[field]) / ymax
                pts.append((xs[model], y, bool(row.get("projection", {}).get(field))))
            if pts:
                body.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y,_ in pts)}" fill="none" stroke="{color}" stroke-width="4"/>')
                for x, y, _ in pts:
                    body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="white" '
                                f'stroke="{color}" stroke-width="4"/>')
    for i, (_, label, color) in enumerate(series):
        x = 465 + i * 130
        body += [f'<line x1="{x}" y1="455" x2="{x+20}" y2="455" stroke="{color}" stroke-width="4"/>',
                 f'<circle cx="{x+10}" cy="455" r="8" fill="white" stroke="{color}" stroke-width="4"/>',
                 _text(x + 25, 459, label, "legend", "start")]
    body.append(_text(20, (top + bottom) / 2, "Speedup vs H_exp2 + naive", "", "middle", -90))
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_svg(width, height, body), encoding="utf-8")


def _render_pareto_legacy(summary: dict[str, Any], output: str | Path) -> None:
    width, height = 1160, 570
    body = [_text(width / 2, 25, "Training–inference hardware Pareto front", "title")]
    groups = summary.get("groups", []) + summary.get("average_groups", [])
    if not groups:
        groups = summary.get("projection_groups", []) + summary.get("projection_average_groups", [])
        body.append(_text(width / 2, 44, "Capacity-infeasible performance projection (not a feasible Pareto)", "tick"))
    scatter = summary.get("evaluated_candidate_scatter", [])
    scatter_lookup = {(g["model_id"], g["software_state"]): g for g in scatter}
    model_order = ("llama2_7b", "gpt3_175b", "llama3_8b",
                   "llama3_1_405b", "mixtral_8x7b", "deepseek_v3")
    model_colors = {model: COLORS[i] for i, model in enumerate(model_order)}
    scatter_order = (*model_order, "geomean_6_models")
    for panel_i, state in enumerate(("naive", "sw_opt")):
        left, right = 70 + panel_i * 555, 535 + panel_i * 555
        top, bottom = 75, 460
        body += [_text((left + right) / 2, 65, state),
                 f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
                 f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>']
        for tick in range(6):
            v = tick / 5; x = left + (right-left)*v; y = bottom-(bottom-top)*v
            body += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>',
                     f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>',
                     _text(x, bottom + 16, f"{v:.1f}", "tick"), _text(left-7, y+4, f"{v:.1f}", "tick", "end")]
        # Draw structurally valid evaluated candidates; exact ties intentionally overlap.
        for model in scatter_order:
            group = scatter_lookup.get((model, state), {"points": []})
            color = COLORS[-1] if model == "geomean_6_models" else model_colors[model]
            for point in group["points"]:
                if not point["structurally_valid"]:
                    continue
                x = left + (right-left)*float(point["training_norm"])
                y = bottom - (bottom-top)*float(point["inference_norm"])
                title = html.escape(f"{MODEL_LABELS[model]} / {state} / {point['candidate_id']}")
                radius = 5.1 if model == "geomean_6_models" else 3.4
                opacity = .32 if model == "geomean_6_models" else .24
                body.append(f'<circle data-role="candidate" data-series="{model}" cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" fill-opacity="{opacity}"><title>{title}</title></circle>')
        chosen = [g for g in groups if g.get("software_state") == state]
        chosen.sort(key=lambda g: (g.get("model_id") == "geomean_6_models", g.get("model_id")))
        for group in chosen:
            mean = group.get("model_id") == "geomean_6_models"
            color = COLORS[-1] if mean else model_colors[group["model_id"]]
            unique_points = []
            seen = set()
            for point in group.get("pareto", []):
                key = (float(point["training_norm"]), float(point["inference_norm"]))
                if key not in seen:
                    seen.add(key); unique_points.append(point)
            coords = [(left + (right-left)*float(p["training_norm"]),
                       bottom - (bottom-top)*float(p["inference_norm"])) for p in unique_points]
            if coords:
                for x, y in coords:
                    body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{4 if mean else 3}" fill="white" stroke="{color}" stroke-width="{2 if mean else 1.5}"/>')
            if not group.get("points"):
                continue
            point_by_id = {point["candidate_id"]: point for point in group["points"]}
            for objective, field in (("training", "training_best_candidate"),
                                     ("inference", "inference_best_candidate")):
                candidate_id = group.get(field)
                point = point_by_id.get(candidate_id)
                if point is None:
                    continue
                x = left + (right-left)*float(point["training_norm"])
                y = bottom - (bottom-top)*float(point["inference_norm"])
                outer, inner = ((12.0, 5.2) if mean else (8.0, 3.5))
                title = html.escape(
                    f"{MODEL_LABELS[group['model_id']]} / {state} / {objective} best / {candidate_id}"
                )
                body.append(
                    f'<path data-role="best-point" data-series="{group["model_id"]}" '
                    f'data-objective="{objective}" data-outer-radius="{outer}" d="{_star_path(x, y, outer, inner)}" '
                    f'fill="{color}" stroke="white" stroke-width="1.4"><title>{title}</title></path>'
                )
        body += [_text((left+right)/2, 493, "Normalized training throughput"),
                 _text(left-45, (top+bottom)/2, "Normalized request throughput", "", "middle", -90)]
    for i, model in enumerate((*model_order, "geomean_6_models")):
        x = 45 + i * 155
        mean = model == "geomean_6_models"
        color = COLORS[-1] if mean else model_colors[model]
        radius = 5.1 if mean else 3.4
        body += [f'<circle cx="{x+radius}" cy="520" r="{radius}" fill="{color}" fill-opacity=".65"/>',
                 _text(x+2*radius+5, 524, MODEL_LABELS[model], "legend", "start")]
    body += [
        '<circle cx="125" cy="549" r="3.4" fill="#9677b8" fill-opacity=".5"/>',
        _text(135, 553, "candidate dots: model Ø6.8; geomean Ø10.2", "legend", "start"),
        '<circle cx="585" cy="549" r="4" fill="white" stroke="#9677b8" stroke-width="1.5"/>',
        _text(595, 553, "Pareto-optimal point", "legend", "start"),
        f'<path d="{_star_path(835, 549, 8, 3.5)}" fill="#9677b8" stroke="white" stroke-width="1.2"/>',
        _text(847, 553, "training/inference optimum", "legend", "start"),
    ]
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_svg(width, height, body), encoding="utf-8")

def render_pareto(summary: dict[str, Any], output: str | Path) -> None:
    """Plot both software states against the single H_exp2+naive baseline."""
    width, height = 1250, 830
    left, top, size = 90, 80, 650
    right, bottom = left + size, top + size
    models = ("llama2_7b", "gpt3_175b", "llama3_8b",
              "llama3_1_405b", "mixtral_8x7b", "deepseek_v3")
    series = (*models, "geomean_6_models")
    colors = {model: COLORS[i] for i, model in enumerate(models)}
    colors["geomean_6_models"] = COLORS[-1]
    states = ("naive", "sw_opt")
    auxiliary = {"naive": "#9677b8", "sw_opt": "#fb9a99"}

    body = [_text(width / 2, 27, "Training–inference hardware skyline", "title"),
            _text(width / 2, 48,
                  "Each axis is normalized by its maximum plotted value; "
                  "naive and sw_opt share the same scale",
                  "tick")]
    if not summary.get("groups"):
        body.append(_text(width / 2, 65,
                          "Capacity-infeasible projection; structurally invalid hardware excluded",
                          "tick"))

    scatter = summary.get("evaluated_candidate_scatter", [])
    lookup = {(g["model_id"], g["software_state"]): g for g in scatter}

    def skyline(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Descending-training skyline scan specified by the experiment."""
        ordered = sorted(
            (p for p in points if p.get("structurally_valid", True)),
            key=lambda p: (-float(p["training_norm"]),
                           -float(p["inference_norm"]), p["candidate_id"]),
        )
        front: list[dict[str, Any]] = []
        running_max_infer = -math.inf
        for point in ordered:
            inference = float(point["inference_norm"])
            if inference > running_max_infer:
                front.append(point)
                running_max_infer = inference
        return front

    # The fallback keeps the small plotting fixture useful without synthesizing dots.
    fallback = summary.get("groups", []) + summary.get("average_groups", [])
    fronts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for state in states:
        for model in series:
            points = lookup.get((model, state), {}).get("points", [])
            if not points:
                group = next((g for g in fallback
                              if g.get("model_id") == model
                              and g.get("software_state") == state), None)
                points = group.get("pareto", []) if group else []
            fronts[(model, state)] = skyline(points)

    valid = [p for group in scatter for p in group.get("points", [])
             if p.get("structurally_valid", True)]
    max_training = max([1.0] + [float(p["training_norm"]) for p in valid])
    max_inference = max([1.0] + [float(p["inference_norm"]) for p in valid])

    def nx(value: float) -> float:
        return float(value) / max_training

    def ny(value: float) -> float:
        return float(value) / max_inference

    def px(value: float) -> float:
        return left + float(value) * size

    def py(value: float) -> float:
        return bottom - float(value) * size

    def sx(value: float) -> float:
        return px(nx(value))

    def sy(value: float) -> float:
        return py(ny(value))

    body += [f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="#fdfcff"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
             f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>']
    for tick in range(6):
        value = tick / 5
        x, y = px(value), py(value)
        body += [f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" class="grid"/>',
                 f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" class="grid"/>',
                 _text(x, bottom + 17, f"{value:.1f}", "tick"),
                 _text(left - 8, y + 4, f"{value:.1f}", "tick", "end")]

    baseline_color = "#d8c9ed"
    body += [f'<line data-role="baseline-guide" x1="{sx(1):.2f}" y1="{top}" '
             f'x2="{sx(1):.2f}" y2="{bottom}" stroke="{baseline_color}" '
             'stroke-width="1.6" stroke-dasharray="6 5"/>',
             f'<line data-role="baseline-guide" x1="{left}" y1="{sy(1):.2f}" '
             f'x2="{right}" y2="{sy(1):.2f}" stroke="{baseline_color}" '
             'stroke-width="1.6" stroke-dasharray="6 5"/>',
             f'<circle data-role="baseline-point" cx="{sx(1):.2f}" cy="{sy(1):.2f}" '
             'r="6" fill="white" stroke="#765a9a" stroke-width="2"/>',
             _text(sx(1) + 9, sy(1) - 9,
                   f"H_exp2 + naive ({nx(1):.2f},{ny(1):.2f})",
                   "tick", "start")]

    # Candidate menu: naive is hollow, sw_opt filled; model remains encoded by color.
    for state in states:
        for model in series:
            color = colors[model]
            radius = 5.1 if model == "geomean_6_models" else 3.4
            for point in lookup.get((model, state), {}).get("points", []):
                if not point.get("structurally_valid", True):
                    continue
                x, y = sx(point["training_norm"]), sy(point["inference_norm"])
                title = html.escape(
                    f"{MODEL_LABELS[model]} / {state} / {point['candidate_id']} / "
                    f"T={float(point['training_norm']):.4f}, "
                    f"I={float(point['inference_norm']):.4f}"
                )
                if state == "naive":
                    style = (f'fill="white" fill-opacity=".72" stroke="{color}" '
                             'stroke-opacity=".38" stroke-width="1"')
                else:
                    style = f'fill="{color}" fill-opacity=".27" stroke="none"'
                body.append(
                    f'<circle data-role="candidate" data-series="{model}" data-state="{state}" '
                    f'cx="{x:.2f}" cy="{y:.2f}" r="{radius}" {style}>'
                    f'<title>{title}</title></circle>'
                )

    analysis: dict[str, dict[str, Any]] = {}
    # Auxiliary geometry is shown for the aggregate front only to avoid obscuring dots.
    for state in states:
        front = fronts[("geomean_6_models", state)]
        if not front:
            continue
        training = max(front, key=lambda p: (float(p["training_norm"]),
                                             float(p["inference_norm"])))
        inference = max(front, key=lambda p: (float(p["inference_norm"]),
                                              float(p["training_norm"])))
        balanced = max(front, key=lambda p: (float(p["training_norm"]) *
                                              float(p["inference_norm"]),
                                              p["candidate_id"]))
        tx, ty = nx(training["training_norm"]), ny(training["inference_norm"])
        ix, iy = nx(inference["training_norm"]), ny(inference["inference_norm"])
        bx, by = nx(balanced["training_norm"]), ny(balanced["inference_norm"])
        angle_t, angle_i = math.atan2(ty, tx), math.atan2(iy, ix)
        low, high = sorted((angle_t, angle_i))
        theta = math.degrees(high - low)
        split_gain = .5 * (tx / bx + iy / by) - 1
        color = auxiliary[state]
        dash = ' stroke-dasharray="8 5"' if state == "naive" else ""
        opacity = ".58" if state == "naive" else ".72"

        body += [f'<line data-role="angle-ray" data-state="{state}" '
                 f'x1="{left}" y1="{bottom}" x2="{px(tx):.2f}" y2="{py(ty):.2f}" '
                 f'stroke="{color}" stroke-width="1.8" stroke-opacity="{opacity}"{dash}/>',
                 f'<line data-role="angle-ray" data-state="{state}" '
                 f'x1="{left}" y1="{bottom}" x2="{px(ix):.2f}" y2="{py(iy):.2f}" '
                 f'stroke="{color}" stroke-width="1.8" stroke-opacity="{opacity}"{dash}/>']

        arc_radius = .34 if state == "naive" else .50
        ax1, ay1 = px(arc_radius * math.cos(low)), py(arc_radius * math.sin(low))
        ax2, ay2 = px(arc_radius * math.cos(high)), py(arc_radius * math.sin(high))
        body.append(
            f'<path data-role="angle-arc" data-state="{state}" '
            f'd="M {ax1:.2f},{ay1:.2f} A {arc_radius*size:.2f},{arc_radius*size:.2f} '
            f'0 0 0 {ax2:.2f},{ay2:.2f}" fill="none" stroke="{color}" '
            f'stroke-width="2"{dash}/>'
        )
        middle = (low + high) / 2
        label_radius = arc_radius + .10
        body.append(_text(px(label_radius * math.cos(middle)),
                          py(label_radius * math.sin(middle)) - 2,
                          f"θ={theta:.1f}°", "tick"))

        constant = bx * by
        xmin = max(constant, .02)
        xmax = 1.0
        curve = []
        for index in range(100):
            x_value = xmin + (xmax - xmin) * index / 99
            y_value = constant / x_value
            if 0 <= y_value <= 1.0:
                curve.append((px(x_value), py(y_value)))
        if curve:
            path = " ".join(("M" if index == 0 else "L") + f" {x:.2f},{y:.2f}"
                            for index, (x, y) in enumerate(curve))
            body.append(
                f'<path data-role="geomean-hyperbola" data-state="{state}" d="{path}" '
                f'fill="none" stroke="{color}" stroke-width="1.7" '
                f'stroke-opacity="{opacity}"{dash}/>'
            )
        analysis[state] = {"theta": theta, "split_gain": split_gain,
                           "balanced": balanced, "training": training,
                           "inference": inference, "corners": len(front)}

    # Strict H/V staircases: never imply a selectable intermediate hardware point.
    for state in states:
        for model in series:
            front = fronts[(model, state)]
            if not front:
                continue
            color = colors[model]
            mean = model == "geomean_6_models"
            coords = [(sx(p["training_norm"]), sy(p["inference_norm"])) for p in front]
            path = f"M {coords[0][0]:.2f},{coords[0][1]:.2f}"
            for x, y in coords[1:]:
                path += f" H {x:.2f} V {y:.2f}"
            dash = ' stroke-dasharray="8 5"' if state == "naive" else ""
            body.append(
                f'<path data-role="skyline" data-series="{model}" data-state="{state}" '
                f'd="{path}" fill="none" stroke="{color}" '
                f'stroke-width="{3.0 if mean else 1.5}" '
                f'stroke-opacity="{.96 if mean else .7}"{dash}/>'
            )
            for x, y in coords:
                body.append(
                    f'<circle data-role="skyline-corner" data-series="{model}" '
                    f'data-state="{state}" cx="{x:.2f}" cy="{y:.2f}" '
                    f'r="{4.2 if mean else 2.5}" fill="white" stroke="{color}" '
                    f'stroke-width="{2 if mean else 1.3}"/>'
                )

            training = max(front, key=lambda p: (float(p["training_norm"]),
                                                 float(p["inference_norm"])))
            inference = max(front, key=lambda p: (float(p["inference_norm"]),
                                                  float(p["training_norm"])))
            for objective, point in (("training", training), ("inference", inference)):
                x, y = sx(point["training_norm"]), sy(point["inference_norm"])
                outer, inner = ((12.0, 5.2) if mean else (8.0, 3.5))
                fill = "white" if state == "naive" else color
                stroke = color if state == "naive" else "white"
                title = html.escape(
                    f"{MODEL_LABELS[model]} / {state} / {objective} optimum / "
                    f"{point['candidate_id']}"
                )
                body.append(
                    f'<path data-role="best-point" data-series="{model}" data-state="{state}" '
                    f'data-objective="{objective}" data-outer-radius="{outer}" '
                    f'd="{_star_path(x, y, outer, inner)}" fill="{fill}" '
                    f'stroke="{stroke}" stroke-width="1.5"><title>{title}</title></path>'
                )

            if mean and state in analysis:
                item = analysis[state]
                balanced = item["balanced"]
                bx, by = sx(balanced["training_norm"]), sy(balanced["inference_norm"])
                color = auxiliary[state]
                body.append(
                    f'<path data-role="balanced-point" data-state="{state}" '
                    f'd="M {bx:.2f},{by-8:.2f} L {bx+8:.2f},{by:.2f} '
                    f'L {bx:.2f},{by+8:.2f} L {bx-8:.2f},{by:.2f} Z" '
                    f'fill="white" stroke="{color}" stroke-width="2.3"/>'
                )
                same_t = balanced["candidate_id"] == item["training"]["candidate_id"]
                same_i = balanced["candidate_id"] == item["inference"]["candidate_id"]
                label = "T*=I*=B" if same_t and same_i else (
                    "T*=B" if same_t else ("I*=B" if same_i else "B"))
                y_offset = 18 if state == "naive" else -11
                body.append(_text(bx + 10, by + y_offset, f"{label} ({state})",
                                  "tick", "start"))
                if not same_t:
                    point = item["training"]
                    label_x, label_y = sx(point["training_norm"]), sy(point["inference_norm"])
                    body.append(_text(label_x + 10, label_y + (18 if state == "naive" else -11),
                                      f"T* ({state})", "tick", "start"))
                if not same_i:
                    point = item["inference"]
                    label_x, label_y = sx(point["training_norm"]), sy(point["inference_norm"])
                    body.append(_text(label_x + 10, label_y + (18 if state == "naive" else -11),
                                      f"I* ({state})", "tick", "start"))

    body += [_text((left + right) / 2, bottom + 49,
                   f"Training throughput / plotted max ({max_training:.3f}× H_exp2 naive)"),
             _text(left - 60, (top + bottom) / 2,
                   f"Request throughput / plotted max ({max_inference:.3f}× H_exp2 naive)",
                   "", "middle", -90)]

    legend_x = 785
    body.append(_text(legend_x, 92,
                      "Model / skyline corners (naive, sw_opt)", "", "start"))
    for index, model in enumerate(series):
        y = 120 + index * 31
        color = colors[model]
        radius = 5.1 if model == "geomean_6_models" else 3.4
        body += [f'<circle cx="{legend_x+radius}" cy="{y}" r="{radius}" '
                 f'fill="{color}" fill-opacity=".65"/>',
                 _text(legend_x + 16, y + 4,
                       f"{MODEL_LABELS[model]}  "
                       f"({len(fronts[(model, 'naive')])}, "
                       f"{len(fronts[(model, 'sw_opt')])})",
                       "legend", "start")]

    state_y = 365
    body += [_text(legend_x, state_y, "Software state", "", "start"),
             f'<circle cx="{legend_x+7}" cy="{state_y+27}" r="5" fill="white" '
             'stroke="#9677b8" stroke-width="1.5"/>',
             f'<line x1="{legend_x+28}" y1="{state_y+27}" x2="{legend_x+65}" '
             f'y2="{state_y+27}" stroke="#9677b8" stroke-width="2" '
             'stroke-dasharray="8 5"/>',
             _text(legend_x + 76, state_y + 31, "naive: hollow / dashed",
                   "legend", "start"),
             f'<circle cx="{legend_x+7}" cy="{state_y+55}" r="5" '
             'fill="#fb9a99" fill-opacity=".55"/>',
             f'<line x1="{legend_x+28}" y1="{state_y+55}" x2="{legend_x+65}" '
             f'y2="{state_y+55}" stroke="#fb9a99" stroke-width="2"/>',
             _text(legend_x + 76, state_y + 59, "sw_opt: filled / solid",
                   "legend", "start")]

    metrics_y = 475
    body += [f'<rect x="{legend_x-12}" y="{metrics_y-28}" width="420" '
             'height="142" rx="8" fill="#faf7fd" stroke="#d8c9ed"/>',
             _text(legend_x, metrics_y - 6,
                   "6-model geomean diagnostics", "", "start")]
    for index, state in enumerate(states):
        item = analysis.get(state)
        if item:
            body.append(_text(
                legend_x, metrics_y + 23 + index * 27,
                f"{state}: k={item['corners']}   θ={item['theta']:.1f}°   "
                f"G_split={item['split_gain']:.3f}", "legend", "start"))
    body += [_text(legend_x, metrics_y + 86,
                   "T*: max training   I*: max inference", "legend", "start"),
             _text(legend_x, metrics_y + 104,
                   "B: max x·y; curved guide is x·y = x_B·y_B",
                   "legend", "start")]

    note_y = 635
    body += [_text(legend_x, note_y, "How to read", "", "start"),
             _text(legend_x, note_y + 23,
                   "• Each H/V corner is an available hardware choice.",
                   "legend", "start"),
             _text(legend_x, note_y + 43,
                   "• Ray angle θ measures training/inference spread.",
                   "legend", "start"),
             _text(legend_x, note_y + 63,
                   "• Hyperbola through B visualizes balance vs split.",
                   "legend", "start")]

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_svg(width, height, body), encoding="utf-8")



def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=root / "results/pareto_summary.json")
    ap.add_argument("--figures", default=root / "figures")
    args = ap.parse_args(argv)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    figures = Path(args.figures)
    render_speedup(summary, figures / "optimization_speedup.svg")
    render_pareto(summary, figures / "hardware_pareto.svg")
    print(f"wrote {figures / 'optimization_speedup.svg'} and {figures / 'hardware_pareto.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
