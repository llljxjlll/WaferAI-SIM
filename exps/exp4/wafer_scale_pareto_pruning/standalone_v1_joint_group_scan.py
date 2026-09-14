"""Standalone core-to-wafer scan and V1 Pareto-pruning pipeline.

This file intentionally depends only on the Python standard library. It
contains the complete analytical flow used by the final experiment:

1. enumerate and physically filter core microarchitecture configurations;
2. build N-by-N compute dies and enforce reticle limits;
3. place HBM, allocate clustered HBM/D2D ports, and check shoreline/bisection;
4. build compute-die/HBM rectangles and tile them in a 215 mm square window;
5. derive the five V1 objectives and diagnostic metrics;
6. compute Pareto fronts inside (router, N class, control-core count) groups;
7. add one minimum-network-distance sentinel per non-empty group, using the
   established deterministic tie-break rule;
8. export one CSV row per unique Pareto/sentinel configuration.

Default CSV columns are the nine independent parameters requested by the
design-space table plus DTE channel count. Use ``--include-derived`` to append
the five objectives, network distance, role flags, geometry, ports, and cost.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence


# =============================================================================
# Paths, units, and output schemas
# =============================================================================

HERE = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = HERE / "09-V1联合分组前沿与哨兵点.csv"

# Units used throughout this file:
#   area: mm^2, length: mm, bandwidth: GB/s, capacity: MiB or GB.
# Compute values are GFLOP/s unless a field explicitly contains "TFLOPs".

PARAMETER_CSV_FIELDS: tuple[str, ...] = (
    "n_ctrl", "r", "B", "K", "B_s", "N_PE", "N", "e_H", "m",
    "DTE_channel",
)

DERIVED_CSV_FIELDS: tuple[str, ...] = (
    # Five V1 objectives and the network-distance diagnostic.
    "wafer_compute_TFLOPs",
    "wafer_hbm_capacity_GB",
    "wafer_hbm_noc_GBs",
    "wafer_min_d2d_bisection_GBs",
    "core_sram_score",
    "network_hops_proxy",
    # Selection roles and grouping.
    "candidate_id",
    "is_group_pareto",
    "is_sentinel",
    "N_class",
    # Core, die, module, wafer, memory, and port diagnostics.
    "A_core_mm2",
    "comp_die_w_mm",
    "comp_die_h_mm",
    "comp_die_area_mm2",
    "module_w_mm",
    "module_h_mm",
    "module_area_mm2",
    "wafer_nx",
    "wafer_ny",
    "modules_per_wafer",
    "P_mat_die_TFLOPs",
    "HBM_capacity_per_module_GB",
    "HBM_NoC_observable_per_module_GBs",
    "D2D_edge_one_dir_GBs",
    "t_hbm_ports_per_hbm_edge",
    "d_d2d_ports_per_edge",
    "total_hbm_stacks",
    "hardware_cost_proxy",
)


# =============================================================================
# Independent design-space parameters
# =============================================================================

N_CTRL_SET = (1, 2)
ROUTER_SET = ("base", "broadcast", "reduce", "both")
B_SET = (64, 128, 192, 256, 384, 512)
K_SET = (1, 2, 3, 4)
B_S_SET = (256, 512, 768, 1024)
N_PE_SET = (1024, 4096, 8192, 12288, 16384)

N_ARRAY_SET = (4, 8, 16, 32)
HBM_EDGE_SET = (1, 2)
HBM_PER_EDGE_SCAN = (1, 2, 3, 4)

# Port placement is fixed rather than scanned.
PORT_PLACEMENT = "clustered"

# Stable field order for candidate IDs and CSV rows.
SEARCH_FIELDS = (
    "n_ctrl", "router", "B_link", "K_MiB", "B_S", "N_PE",
    "N", "e_HBM", "m_HBM_per_edge",
)


# =============================================================================
# Core analytical-model constants
# =============================================================================

G_7 = 0.08748
U_SC = 0.65
LAMBDA_7 = G_7 / U_SC
DELTA_IMPL = 0.10

B_CH = 128
W_B_MAX = 512
A_PE_7 = 4.7005e-5
A_LANE_7 = 0.004669
P_LANE = 1.9777
R_PV = 10
F_C = 1.0

# Router area multipliers for base, broadcast, reduce, and both primitives.
M_R = {
    "base": 1.0,
    "broadcast": 174.5 / 135,
    "reduce": 214.5 / 135,
    "both": 259.5 / 135,
}

# SRAM bank aspect-ratio calibration by capacity in MiB.
Q_S = {1: 2.001, 2: 2.152, 3: 2.404, 4: 2.609}


# =============================================================================
# Die, PHY, HBM, package, and wafer constants
# =============================================================================

RHO_GAP = 2 / 35
RETICLE_SHORT = 26.0
RETICLE_LONG = 33.0
RETICLE_AREA = RETICLE_SHORT * RETICLE_LONG

# Fixed edge strip is the maximum modeled HBM, D2D, and other-I/O PHY depth.
PHY_DEPTH = 1.540

HBM_BW = 819.2
HBM_ACTIVITY = 0.90
HBM_CAPACITY_GB = 16.0
HBM_SHORELINE = 8.50
D2D_SHORELINE = 0.55

HBM_WIDTH = 8.13
HBM_HEIGHT = 4.92
PACKAGE_GAP = 0.15

OTHER_PORT_RATIO = 0.20
D2D_PHY_PER_EDGE = 1
D2D_PHY_MODULE_DIR_BW = 512.0

WAFER_WIDTH = 215.0
WAFER_HEIGHT = 215.0


# =============================================================================
# Pareto metrics and numerical behavior
# =============================================================================

K_REF_MIB = 3.0
B_S_REF_GBS = 256.0
REL_TOL = 1e-9

BENEFIT_FIELDS_V1 = (
    "wafer_compute_TFLOPs",
    "wafer_hbm_capacity_GB",
    "wafer_hbm_noc_GBs",
    "wafer_min_d2d_bisection_GBs",
    "core_sram_score",
)


class ScanResult(NamedTuple):
    """Complete result required by CSV export and regression reporting."""

    feasible_records: list[dict[str, Any]]
    final_points: list[dict[str, Any]]
    pareto_ids: frozenset[int]
    sentinel_ids: frozenset[int]
    summary: dict[str, Any]


# =============================================================================
# Stage 1: core evaluation
# =============================================================================

def evaluate_core(
    params: tuple[int, str, int, int, int, int]
) -> tuple[str, dict[str, Any] | None]:
    """Evaluate one core configuration and return its status and record.

    Important checks include SRAM capacity/bandwidth, SRAM banking, compute
    area, compute-to-SRAM balance, communication-area share, and total area.
    """

    n_ctrl, router, b_gbs, k_mib, b_s_gbs, n_pe = params

    if b_s_gbs < b_gbs:
        return "SRAM_BW", None
    if k_mib < 1:
        return "SRAM_CAPACITY", None

    # Communication area: DTE, local interconnect, NI, link, and router.
    channel = math.ceil(b_gbs / B_CH)
    b_norm = b_gbs / 256
    a_dte = LAMBDA_7 * (0.72 + 0.15 * channel + 0.48 * b_norm)
    a_local = LAMBDA_7 * (0.60 + 1.20 * channel * b_norm)
    a_ni = LAMBDA_7 * (0.60 + 0.60 * b_norm)
    a_link = LAMBDA_7 * (0.60 * b_norm)
    a_router = LAMBDA_7 * (0.672 * b_norm * M_R[router])
    a_comm = a_dte + a_local + a_ni + a_link + a_router

    # Control, matrix PE, and vector-lane area.
    a_ctrl = 4.5 * n_ctrl * LAMBDA_7
    p_pe = 2 * n_pe * F_C
    n_vec = math.ceil(p_pe / (R_PV * P_LANE * F_C))
    p_vec = n_vec * P_LANE * F_C
    a_comp = n_pe * A_PE_7 + n_vec * A_LANE_7

    # SRAM capacity/port model and banking feasibility.
    w_s_port = math.ceil(8 * b_s_gbs)
    n_bank = max(16, math.ceil(w_s_port / W_B_MAX))
    if w_s_port / n_bank > W_B_MAX or Q_S[k_mib] > 3:
        return "SRAM_PHYSICAL", None
    a_sram = G_7 * (8 / 3) * k_mib + G_7 * b_s_gbs / 128

    # Core-level physical range checks.
    if not (0.5 <= a_comp <= 8.6):
        return "COMPUTE_RANGE", None
    if a_comp / a_sram > 6:
        return "COMPUTE_SRAM_RATIO", None

    a_core = (1 + DELTA_IMPL) * (a_comm + a_ctrl + a_comp + a_sram)
    if a_comm / a_core > 0.25:
        return "COMM_SHARE", None
    if a_comm > 4.0375:
        return "COMM_AREA", None
    if a_core > 13:
        return "CORE_AREA", None

    return "FEASIBLE", {
        "n_ctrl": n_ctrl,
        "router": router,
        "B_link": b_gbs,
        "K_MiB": k_mib,
        "B_S": b_s_gbs,
        "N_PE": n_pe,
        "N_vec": n_vec,
        "P_mat_core_GFLOPs": p_pe,
        "P_vec_core_GFLOPs": p_vec,
        "A_core": a_core,
        "core_side": math.sqrt(a_core),
    }


# =============================================================================
# Stage 2: compute die, HBM/ports, package rectangle, and wafer
# =============================================================================

def compute_die(core: dict[str, Any], n: int) -> dict[str, Any]:
    """Build the square core mesh plus a fixed-depth PHY strip."""

    factor = n + (n - 1) * RHO_GAP
    mesh_side = factor * core["core_side"]
    comp_die_w = mesh_side + PHY_DEPTH
    comp_die_h = mesh_side
    area = comp_die_w * comp_die_h
    reticle_ok = (
        min(comp_die_w, comp_die_h) <= RETICLE_SHORT
        and max(comp_die_w, comp_die_h) <= RETICLE_LONG
        and area <= RETICLE_AREA
    )
    return {
        "factor": factor,
        "mesh_side": mesh_side,
        "comp_die_w": comp_die_w,
        "comp_die_h": comp_die_h,
        "comp_die_area": area,
        "reticle_ok": reticle_ok,
    }


def max_hbm_per_edge(
    comp_die_w: float, n: int, b_link: float
) -> tuple[int, int, int, float]:
    """Compute shoreline and NoC-bisection limits on HBM stacks per edge."""

    # Shoreline rule: m*L_H + (m-1)*gap + L_D < compute-die width.
    ratio = (comp_die_w - D2D_SHORELINE + PACKAGE_GAP) / (
        HBM_SHORELINE + PACKAGE_GAP
    )
    m_io = max(0, math.ceil(ratio) - 1)

    # Bisection rule: N*B >= m*alpha_H*B_H.
    b_bisec = n * b_link
    m_bw = max(0, math.floor(b_bisec / (HBM_ACTIVITY * HBM_BW)))
    return m_io, m_bw, min(m_io, m_bw), b_bisec


def clustered_positions(n: int, count: int) -> list[int]:
    """Place ports symmetrically at the two ends of one die edge."""

    left = math.ceil(count / 2)
    right = math.floor(count / 2)
    return list(range(left)) + list(range(n - right, n))


def clustered_from_slots(slots: list[int], count: int) -> list[int]:
    """Select clustered D2D ports from slots not occupied by HBM ports."""

    left = math.ceil(count / 2)
    right = math.floor(count / 2)
    chosen = slots[:left] + (slots[-right:] if right else [])
    return sorted(chosen)


def attach_agents(
    n: int, b_link: float, e_hbm: int, m: int
) -> dict[str, Any] | None:
    """Allocate HBM and equal D2D ports while reserving other-agent capacity."""

    # HBM uses enough ports for target bandwidth unless N-1 caps the count.
    # The cap leaves at least one edge slot for D2D.
    t_raw = math.ceil(m * HBM_ACTIVITY * HBM_BW / b_link)
    t_hbm = min(t_raw, n - 1)

    p_edge = n
    p_total = 4 * n
    p_usable = math.floor((1 - OTHER_PORT_RATIO) * p_total)
    d_global = math.floor((p_usable - e_hbm * t_hbm) / 4)
    d_edge = p_edge - t_hbm
    d_d2d = min(d_global, d_edge)
    if d_d2d < 1:
        return None

    hbm_positions = clustered_positions(n, t_hbm)
    free_slots = [position for position in range(n) if position not in hbm_positions]
    d2d_hbm_edge = clustered_from_slots(free_slots, d_d2d)
    d2d_plain_edge = clustered_positions(n, d_d2d)
    assert set(hbm_positions).isdisjoint(d2d_hbm_edge)

    return {
        "t_hbm_raw": t_raw,
        "t_hbm": t_hbm,
        "d_d2d": d_d2d,
        "hbm_port_positions": hbm_positions,
        "d2d_positions_hbm_edge": d2d_hbm_edge,
        "d2d_positions_plain_edge": d2d_plain_edge,
    }


def composite_rectangle(
    comp_die: dict[str, Any], e_hbm: int, m: int
) -> dict[str, float]:
    """Build the bounding rectangle containing one compute die and its HBM."""

    hbm_row_w = m * HBM_WIDTH + (m - 1) * PACKAGE_GAP
    module_w = max(hbm_row_w, comp_die["comp_die_w"]) + 2 * PACKAGE_GAP

    if e_hbm == 1:
        module_h = comp_die["comp_die_h"] + HBM_HEIGHT + 3 * PACKAGE_GAP
    else:
        module_h = comp_die["comp_die_h"] + 2 * HBM_HEIGHT + 4 * PACKAGE_GAP

    return {
        "module_w": module_w,
        "module_h": module_h,
        "module_area": module_w * module_h,
    }


def throughput(
    core: dict[str, Any],
    n: int,
    b_link: float,
    e_hbm: int,
    m: int,
    ports: dict[str, Any],
) -> dict[str, float | int]:
    """Calculate compute, HBM, and D2D peak capabilities per module."""

    core_count = n * n
    p_mat = core_count * core["P_mat_core_GFLOPs"]
    p_vec = core_count * core["P_vec_core_GFLOPs"]

    hbm_peak = e_hbm * m * HBM_BW
    hbm_target = e_hbm * m * HBM_ACTIVITY * HBM_BW
    hbm_noc = e_hbm * min(
        m * HBM_ACTIVITY * HBM_BW,
        ports["t_hbm"] * b_link,
    )

    d2d_phy_edge = D2D_PHY_PER_EDGE * D2D_PHY_MODULE_DIR_BW
    d2d_edge_dir = min(ports["d_d2d"] * b_link, d2d_phy_edge)

    return {
        "core_count": core_count,
        "P_mat_TFLOPs": p_mat / 1000,
        "P_vec_TFLOPs": p_vec / 1000,
        "HBM_capacity_GB": e_hbm * m * HBM_CAPACITY_GB,
        "HBM_peak_GBs": hbm_peak,
        "HBM_target_GBs": hbm_target,
        "HBM_NoC_observable_GBs": hbm_noc,
        "D2D_edge_one_dir_GBs": d2d_edge_dir,
        "D2D_four_edge_one_dir_GBs": 4 * d2d_edge_dir,
    }


def wafer(
    module: dict[str, float], d2d_edge_one_dir_gbs: float
) -> dict[str, float | int | bool]:
    """Tile identical module rectangles in the fixed wafer research window."""

    nx = math.floor(WAFER_WIDTH / module["module_w"])
    ny = math.floor(WAFER_HEIGHT / module["module_h"])
    return {
        "wafer_nx": nx,
        "wafer_ny": ny,
        "modules_per_wafer": nx * ny,
        "wafer_mesh_links": max(0, nx - 1) * ny + max(0, ny - 1) * nx,
        "wafer_bisec_x_one_dir_GBs": ny * d2d_edge_one_dir_gbs,
        "wafer_bisec_y_one_dir_GBs": nx * d2d_edge_one_dir_gbs,
        "wafer_fit": nx >= 1 and ny >= 1,
    }


# =============================================================================
# Stage 3: complete physical scan and stable candidate IDs
# =============================================================================

def build_feasible_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Enumerate the full space and retain only physically feasible records."""

    raw_core_params = list(itertools.product(
        N_CTRL_SET, ROUTER_SET, B_SET, K_SET, B_S_SET, N_PE_SET,
    ))

    feasible_cores: list[dict[str, Any]] = []
    for params in raw_core_params:
        _, core = evaluate_core(params)
        if core is not None:
            feasible_cores.append(core)

    stage_counts = {
        "raw_core": len(raw_core_params),
        "feasible_core": len(feasible_cores),
        "reticle": 0,
        "hbm": 0,
        "ports": 0,
        "wafer": 0,
    }
    records: list[dict[str, Any]] = []

    for core in feasible_cores:
        for n in N_ARRAY_SET:
            comp = compute_die(core, n)
            for e_hbm in HBM_EDGE_SET:
                for m in HBM_PER_EDGE_SCAN:
                    if not comp["reticle_ok"]:
                        continue
                    stage_counts["reticle"] += 1

                    m_io, m_bw, m_max, b_bisec = max_hbm_per_edge(
                        comp["comp_die_w"], n, core["B_link"]
                    )
                    if m > m_max:
                        continue
                    stage_counts["hbm"] += 1

                    ports = attach_agents(n, core["B_link"], e_hbm, m)
                    if ports is None:
                        continue
                    stage_counts["ports"] += 1

                    module = composite_rectangle(comp, e_hbm, m)
                    perf = throughput(core, n, core["B_link"], e_hbm, m, ports)
                    wafer_result = wafer(module, perf["D2D_edge_one_dir_GBs"])
                    if not wafer_result["wafer_fit"]:
                        continue
                    stage_counts["wafer"] += 1

                    records.append({
                        **core,
                        "N": n,
                        "placement": PORT_PLACEMENT,
                        "e_HBM": e_hbm,
                        "m_HBM_per_edge": m,
                        "m_max_IO": m_io,
                        "m_max_BW": m_bw,
                        "m_max": m_max,
                        "B_bisec": b_bisec,
                        **comp,
                        **ports,
                        **module,
                        **perf,
                        **wafer_result,
                    })

    # IDs must remain stable because the final sentinel tie-break uses them.
    records.sort(key=lambda record: tuple(record[field] for field in SEARCH_FIELDS))
    for candidate_id, record in enumerate(records):
        record["candidate_id"] = candidate_id
    return records, stage_counts


# =============================================================================
# Stage 4: five V1 objectives and diagnostics
# =============================================================================

def enrich_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Add wafer objectives, SRAM score, cost proxy, and hop-distance proxy."""

    wafer_compute_areas = [
        record["modules_per_wafer"] * record["comp_die_area"]
        for record in records
    ]
    wafer_hbm_counts = [
        record["modules_per_wafer"]
        * record["e_HBM"]
        * record["m_HBM_per_edge"]
        for record in records
    ]

    # Freeze medians over the complete physical-feasible domain before pruning.
    area_ref = statistics.median(wafer_compute_areas)
    hbm_ref = statistics.median(wafer_hbm_counts)

    for record, wafer_area, wafer_hbm in zip(
        records, wafer_compute_areas, wafer_hbm_counts
    ):
        nx = record["wafer_nx"]
        ny = record["wafer_ny"]
        record["wafer_compute_TFLOPs"] = (
            record["modules_per_wafer"] * record["P_mat_TFLOPs"]
        )
        record["wafer_hbm_capacity_GB"] = (
            record["modules_per_wafer"] * record["HBM_capacity_GB"]
        )
        record["wafer_hbm_noc_GBs"] = (
            record["modules_per_wafer"] * record["HBM_NoC_observable_GBs"]
        )
        record["wafer_min_d2d_bisection_GBs"] = min(
            record["wafer_bisec_x_one_dir_GBs"],
            record["wafer_bisec_y_one_dir_GBs"],
        )
        record["core_sram_score"] = math.sqrt(
            (record["K_MiB"] / K_REF_MIB)
            * (record["B_S"] / B_S_REF_GBS)
        )

        record["normalized_compute_area"] = wafer_area / area_ref
        record["normalized_hbm_count"] = wafer_hbm / hbm_ref
        record["hardware_cost_proxy"] = (
            record["normalized_compute_area"]
            + record["normalized_hbm_count"]
        )

        # Mean Manhattan distance for the intra-die and inter-die 2D meshes.
        record["network_hops_proxy"] = (
            2 * (record["N"] ** 2 - 1) / (3 * record["N"])
            + (nx ** 2 - 1) / (3 * nx)
            + (ny ** 2 - 1) / (3 * ny)
        )

    return {
        "compute_area_ref_mm2": area_ref,
        "hbm_count_ref": hbm_ref,
        "K_ref_MiB": K_REF_MIB,
        "B_S_ref_GBs": B_S_REF_GBS,
        "c_A": 1,
        "c_H": 1,
        "c_P": 0,
    }


# =============================================================================
# Stage 5: exact group-wise Pareto skyline
# =============================================================================

def approximately_ge(a: float, b: float) -> bool:
    scale = max(1.0, abs(a), abs(b))
    return a >= b - REL_TOL * scale


def definitely_gt(a: float, b: float) -> bool:
    scale = max(1.0, abs(a), abs(b))
    return a > b + REL_TOL * scale


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return True if a weakly beats b everywhere and strictly beats it once."""

    return all(approximately_ge(a[field], b[field]) for field in BENEFIT_FIELDS_V1) and any(
        definitely_gt(a[field], b[field]) for field in BENEFIT_FIELDS_V1
    )


def pareto_front(points: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute an exact incremental skyline; equal objective vectors survive."""

    front: list[dict[str, Any]] = []
    for point in points:
        if any(dominates(other, point) for other in front):
            continue
        front = [other for other in front if not dominates(point, other)]
        front.append(point)
    return sorted(front, key=lambda record: record["candidate_id"])


def n_class(record: dict[str, Any]) -> str:
    return "N16" if record["N"] == 16 else "Nlt16"


def group_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return record["router"], n_class(record), record["n_ctrl"]


def choose_sentinel(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose one low-distance sentinel using the established tie-break rule."""

    return min(points, key=lambda record: (
        record["network_hops_proxy"],
        record["hardware_cost_proxy"],
        -record["wafer_min_d2d_bisection_GBs"],
        -record["wafer_compute_TFLOPs"],
        record["candidate_id"],
    ))


def select_points(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], frozenset[int], frozenset[int], list[dict[str, Any]]]:
    """Union each group Pareto front with its single selected sentinel."""

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)

    pareto_ids: set[int] = set()
    sentinel_ids: set[int] = set()
    final_by_id: dict[int, dict[str, Any]] = {}
    group_rows: list[dict[str, Any]] = []

    for key in sorted(groups):
        points = groups[key]
        front = pareto_front(points)
        front_ids = {point["candidate_id"] for point in front}
        sentinel = choose_sentinel(points)
        sentinel_id = sentinel["candidate_id"]

        pareto_ids.update(front_ids)
        sentinel_ids.add(sentinel_id)
        for point in front:
            final_by_id[point["candidate_id"]] = point
        final_by_id[sentinel_id] = sentinel

        group_rows.append({
            "router": key[0],
            "N_class": key[1],
            "n_ctrl": key[2],
            "feasible": len(points),
            "pareto": len(front),
            "new_sentinel": int(sentinel_id not in front_ids),
            "final": len(front_ids | {sentinel_id}),
        })

    final_points = sorted(final_by_id.values(), key=lambda point: point["candidate_id"])
    return final_points, frozenset(pareto_ids), frozenset(sentinel_ids), group_rows


# =============================================================================
# Stage 6: summary, representative points, and distributions
# =============================================================================

def sorted_counter(points: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(point[field]) for point in points)

    def sort_key(item: tuple[str, int]) -> tuple[int, float | str]:
        try:
            return 0, float(item[0])
        except ValueError:
            return 1, item[0]

    return dict(sorted(counts.items(), key=sort_key))


def binned(
    points: Sequence[dict[str, Any]],
    field: str,
    bins: Sequence[tuple[str, float, float]],
) -> dict[str, int]:
    return {
        label: sum(lower <= point[field] < upper for point in points)
        for label, lower, upper in bins
    }


def representative(point: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "candidate_id", "n_ctrl", "router", "B_link", "K_MiB", "B_S",
        "N_PE", "N", "e_HBM", "m_HBM_per_edge", "A_core",
        "comp_die_area", "module_area", "wafer_nx", "wafer_ny",
        "modules_per_wafer", "wafer_compute_TFLOPs",
        "wafer_hbm_capacity_GB", "wafer_hbm_noc_GBs",
        "wafer_min_d2d_bisection_GBs", "core_sram_score",
        "hardware_cost_proxy", "network_hops_proxy", "t_hbm", "d_d2d",
    )
    result = {field: point[field] for field in fields}
    result["total_hbm_stacks"] = point["e_HBM"] * point["m_HBM_per_edge"]
    result["DTE_channel"] = (point["B_link"] + B_CH - 1) // B_CH
    return result


def build_summary(
    records: list[dict[str, Any]],
    final_points: list[dict[str, Any]],
    pareto_ids: frozenset[int],
    sentinel_ids: frozenset[int],
    group_rows: list[dict[str, Any]],
    stage_counts: dict[str, int],
    normalization: dict[str, float],
) -> dict[str, Any]:
    """Build the same machine-readable diagnostics as the modular script."""

    maxima = {
        field: max(point[field] for point in final_points)
        for field in BENEFIT_FIELDS_V1
    }

    def balance(point: dict[str, Any]) -> tuple[float, float, int]:
        ratios = [point[field] / maxima[field] for field in BENEFIT_FIELDS_V1]
        return min(ratios), sum(ratios), -point["candidate_id"]

    representatives = {
        "balanced": representative(max(final_points, key=balance)),
        "maximum_compute": representative(max(
            final_points, key=lambda point: point["wafer_compute_TFLOPs"]
        )),
        "maximum_hbm_capacity": representative(max(
            final_points, key=lambda point: point["wafer_hbm_capacity_GB"]
        )),
        "maximum_hbm_bandwidth": representative(max(
            final_points, key=lambda point: point["wafer_hbm_noc_GBs"]
        )),
        "maximum_d2d_cut": representative(max(
            final_points, key=lambda point: point["wafer_min_d2d_bisection_GBs"]
        )),
        "maximum_sram_score": representative(max(
            final_points, key=lambda point: point["core_sram_score"]
        )),
        "minimum_cost": representative(min(
            final_points, key=lambda point: point["hardware_cost_proxy"]
        )),
    }

    derived_distribution = {
        "dte_channels": dict(sorted(Counter(
            (point["B_link"] + B_CH - 1) // B_CH for point in final_points
        ).items())),
        "total_hbm_stacks": dict(sorted(Counter(
            point["e_HBM"] * point["m_HBM_per_edge"] for point in final_points
        ).items())),
        "hbm_ports_per_hbm_edge": dict(sorted(Counter(
            point["t_hbm"] for point in final_points
        ).items())),
        "d2d_ports_per_edge": dict(sorted(Counter(
            point["d_d2d"] for point in final_points
        ).items())),
        "modules_per_wafer_bins": binned(final_points, "modules_per_wafer", (
            ("<50", 0, 50), ("50-99", 50, 100),
            ("100-199", 100, 200), (">=200", 200, float("inf")),
        )),
        "core_area_mm2_bins": binned(final_points, "A_core", (
            ("<4", 0, 4), ("4-7", 4, 7), ("7-10", 7, 10),
            ("10-13", 10, 13),
        )),
        "compute_die_area_mm2_bins": binned(final_points, "comp_die_area", (
            ("<100", 0, 100), ("100-300", 100, 300),
            ("300-600", 300, 600), ("600-858", 600, 858.000001),
        )),
        "module_area_mm2_bins": binned(final_points, "module_area", (
            ("<250", 0, 250), ("250-500", 250, 500),
            ("500-750", 500, 750), (">=750", 750, float("inf")),
        )),
        "hardware_cost_proxy_bins": binned(final_points, "hardware_cost_proxy", (
            ("<1.5", 0, 1.5), ("1.5-2", 1.5, 2),
            ("2-2.5", 2, 2.5), ("2.5-3", 2.5, 3),
            (">=3", 3, float("inf")),
        )),
        "network_hops_proxy_bins": binned(final_points, "network_hops_proxy", (
            ("<10", 0, 10), ("10-12", 10, 12),
            ("12-14", 12, 14), (">=14", 14, float("inf")),
        )),
    }

    objective_distribution: dict[str, dict[str, float | int]] = {}
    for field in BENEFIT_FIELDS_V1:
        values = [point[field] for point in final_points]
        objective_distribution[field] = {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
            "unique": len(set(values)),
        }

    return {
        "method": "V1 + one tie-broken sentinel per non-empty group",
        "group_key": ["router", "N16_or_Nlt16", "n_ctrl"],
        "control_core_is_objective": False,
        "benefit_fields": BENEFIT_FIELDS_V1,
        "normalization": normalization,
        "physical_scan_stages": stage_counts,
        "feasible_points": len(records),
        "nonempty_groups": len(group_rows),
        "pareto_points": len(pareto_ids),
        "sentinel_points": len(sentinel_ids),
        "new_sentinels": len(sentinel_ids - pareto_ids),
        "final_points": len(final_points),
        "final_ratio": len(final_points) / len(records),
        "group_results": group_rows,
        "feasible_distribution": {
            field: sorted_counter(records, field) for field in SEARCH_FIELDS
        },
        "final_distribution": {
            field: sorted_counter(final_points, field) for field in SEARCH_FIELDS
        },
        "derived_distribution": derived_distribution,
        "objective_distribution": objective_distribution,
        "representatives": representatives,
    }


def run_scan() -> ScanResult:
    """Execute the complete core-to-wafer scan and final pruning pipeline."""

    records, stage_counts = build_feasible_records()
    normalization = enrich_metrics(records)
    final_points, pareto_ids, sentinel_ids, group_rows = select_points(records)
    summary = build_summary(
        records, final_points, pareto_ids, sentinel_ids,
        group_rows, stage_counts, normalization,
    )
    return ScanResult(
        feasible_records=records,
        final_points=final_points,
        pareto_ids=pareto_ids,
        sentinel_ids=sentinel_ids,
        summary=summary,
    )


def analyze() -> dict[str, Any]:
    """Convenience API for programmatic use without writing output files."""

    return run_scan().summary


# =============================================================================
# Stage 7: CSV export
# =============================================================================

def csv_row(
    point: dict[str, Any], result: ScanResult, include_derived: bool
) -> dict[str, Any]:
    """Map internal fields to stable user-facing CSV column names."""

    candidate_id = point["candidate_id"]
    row: dict[str, Any] = {
        "n_ctrl": point["n_ctrl"],
        "r": point["router"],
        "B": point["B_link"],
        "K": point["K_MiB"],
        "B_s": point["B_S"],
        "N_PE": point["N_PE"],
        "N": point["N"],
        "e_H": point["e_HBM"],
        "m": point["m_HBM_per_edge"],
        "DTE_channel": (point["B_link"] + B_CH - 1) // B_CH,
    }
    if not include_derived:
        return row

    row.update({
        "wafer_compute_TFLOPs": point["wafer_compute_TFLOPs"],
        "wafer_hbm_capacity_GB": point["wafer_hbm_capacity_GB"],
        "wafer_hbm_noc_GBs": point["wafer_hbm_noc_GBs"],
        "wafer_min_d2d_bisection_GBs": point["wafer_min_d2d_bisection_GBs"],
        "core_sram_score": point["core_sram_score"],
        "network_hops_proxy": point["network_hops_proxy"],
        "candidate_id": candidate_id,
        "is_group_pareto": int(candidate_id in result.pareto_ids),
        "is_sentinel": int(candidate_id in result.sentinel_ids),
        "N_class": n_class(point),
        "A_core_mm2": point["A_core"],
        "comp_die_w_mm": point["comp_die_w"],
        "comp_die_h_mm": point["comp_die_h"],
        "comp_die_area_mm2": point["comp_die_area"],
        "module_w_mm": point["module_w"],
        "module_h_mm": point["module_h"],
        "module_area_mm2": point["module_area"],
        "wafer_nx": point["wafer_nx"],
        "wafer_ny": point["wafer_ny"],
        "modules_per_wafer": point["modules_per_wafer"],
        "P_mat_die_TFLOPs": point["P_mat_TFLOPs"],
        "HBM_capacity_per_module_GB": point["HBM_capacity_GB"],
        "HBM_NoC_observable_per_module_GBs": point["HBM_NoC_observable_GBs"],
        "D2D_edge_one_dir_GBs": point["D2D_edge_one_dir_GBs"],
        "t_hbm_ports_per_hbm_edge": point["t_hbm"],
        "d_d2d_ports_per_edge": point["d_d2d"],
        "total_hbm_stacks": point["e_HBM"] * point["m_HBM_per_edge"],
        "hardware_cost_proxy": point["hardware_cost_proxy"],
    })
    return row


def write_csv(result: ScanResult, output_path: Path, include_derived: bool) -> None:
    """Write one row per unique selected Pareto/sentinel configuration."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PARAMETER_CSV_FIELDS)
    if include_derived:
        fieldnames.extend(DERIVED_CSV_FIELDS)

    # UTF-8 BOM improves compatibility with spreadsheet applications.
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for point in result.final_points:
            writer.writerow(csv_row(point, result, include_derived))


# =============================================================================
# Command-line interface
# =============================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone core-to-wafer V1 Pareto scan and CSV exporter."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"CSV output path (default: {DEFAULT_CSV_PATH.name})",
    )
    parser.add_argument(
        "--include-derived",
        action="store_true",
        help="Append objectives, network distance, roles, and derived fields.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path for the complete machine-readable scan summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_scan()
    write_csv(result, args.output, args.include_derived)

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({
        "csv": str(args.output.resolve()),
        "include_derived": args.include_derived,
        "csv_rows": len(result.final_points),
        "feasible_points": result.summary["feasible_points"],
        "nonempty_groups": result.summary["nonempty_groups"],
        "pareto_points": result.summary["pareto_points"],
        "sentinel_points": result.summary["sentinel_points"],
        "new_sentinels": result.summary["new_sentinels"],
        "final_points": result.summary["final_points"],
        "final_ratio": result.summary["final_ratio"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
