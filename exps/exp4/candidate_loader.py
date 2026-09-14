"""Load and close the hardware semantics of the exp4 Pareto candidates.

The CSV contains only independent search variables.  This module reconstructs
the die/wafer fields with the formulas used by the frozen physical scan, while
all runtime rates are interpreted at the experiment's 500 MHz clock.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_CANDIDATE_CSV = (
    HERE / "wafer_scale_pareto_pruning" / "09-V1联合分组前沿与哨兵点.csv"
)

FREQUENCY_HZ = 500_000_000
DTE_CHANNEL_GBS = 128
DTE_CHANNEL_WIDTH_BITS = 2048
HBM_STACK_CAPACITY_GB = 16.0
HBM_STACK_PEAK_GBS = 819.2
HBM_ACTIVITY = 0.90
HBM_STACK_SUSTAINED_GBS = 737.28
D2D_PHY_EDGE_ONE_DIR_GBS = 512.0
PACKET_PAYLOAD_BYTES = 16

_ROUTERS = frozenset(("base", "broadcast", "reduce", "both"))
_FIELDS = ("n_ctrl", "r", "B", "K", "B_s", "N_PE", "N", "e_H", "m", "DTE_channel")

# Frozen geometry constants from standalone_v1_joint_group_scan.py.  They
# reproduce the provenance of the supplied CSV; they are not a new 500 MHz
# physical-area rescan.
_G7 = 0.08748
_LAMBDA7 = _G7 / 0.65
_DELTA_IMPL = 0.10
_A_PE7 = 4.7005e-5
_A_LANE7 = 0.004669
_P_LANE = 1.9777
_R_PV = 10
_M_R = {"base": 1.0, "broadcast": 174.5 / 135, "reduce": 214.5 / 135, "both": 259.5 / 135}
_RHO_GAP = 2 / 35
_PHY_DEPTH_MM = 1.540
_HBM_WIDTH_MM = 8.13
_HBM_HEIGHT_MM = 4.92
_PACKAGE_GAP_MM = 0.15
_WAFER_MM = 215.0


class CandidateFormatError(ValueError):
    """Raised when the candidate file itself is malformed or duplicated."""


@dataclass(frozen=True)
class PEOrganization:
    exu_x: int
    exu_y: int
    sa_count: int


@dataclass(frozen=True)
class CandidateHardware:
    candidate_id: str
    candidate_digest: str
    source_row: int
    source_csv_digest: str
    status: str
    semantic_issues: tuple[str, ...]

    n_ctrl: int
    router: str
    B_GBs: int
    K_MiB: int
    B_s_GBs: int
    N_PE: int
    N: int
    e_H: int
    m: int
    DTE_channel: int

    f_Hz: int
    cycle_ns: float
    exu_x: int
    exu_y: int
    sa_count: int
    P_core_FLOPs: int
    P_core_TFLOPs: float
    noc_link_width_bits: int
    noc_payload_per_cycle: int
    sram_read_GBs: int
    sram_write_GBs: int
    sram_read_width_bits: int
    sram_write_width_bits: int
    sram_bank_count: int
    dte_channel_width_bits: int
    dte_aggregate_width_bits: int

    t_hbm: int
    d_d2d: int
    HBM_stack_count: int
    HBM_stack_capacity_GB: float
    HBM_stack_peak_GBs: float
    HBM_stack_sustained_GBs: float
    HBM_NoC_observable_GBs: float
    D2D_edge_one_dir_GBs: float
    hbm_edges: tuple[str, ...]
    hbm_port_positions: tuple[int, ...]
    d2d_port_positions_hbm_edge: tuple[int, ...]
    d2d_port_positions_plain_edge: tuple[int, ...]

    core_area_mm2_geometry_provenance: float
    module_w_mm: float
    module_h_mm: float
    wafer_nx: int
    wafer_ny: int
    modules_per_wafer: int
    topology_status: str
    geometry_provenance: str

    def canonical_dict(self) -> dict[str, object]:
        return asdict(self)

    # Stable convenience names consumed by the experiment runner.
    @property
    def digest(self) -> str:
        return self.candidate_digest

    @property
    def K_bytes(self) -> int:
        return self.K_MiB * 1024 * 1024

    @property
    def compute_tflops_per_core(self) -> float:
        return self.P_core_TFLOPs

    @property
    def dte_channels(self) -> int:
        return self.DTE_channel

    @property
    def d2d_edge_GBs(self) -> float:
        return self.D2D_edge_one_dir_GBs

    @property
    def hbm_stacks_per_module(self) -> int:
        return self.HBM_stack_count

    def as_dict(self) -> dict[str, object]:
        result = self.canonical_dict()
        result.update({
            "digest": self.digest,
            "K_bytes": self.K_bytes,
            "compute_tflops_per_core": self.compute_tflops_per_core,
            "dte_channels": self.dte_channels,
            "d2d_edge_GBs": self.d2d_edge_GBs,
            "hbm_stacks_per_module": self.hbm_stacks_per_module,
        })
        return result


def pe_organization(n_pe: int) -> PEOrganization:
    """Return the frozen, shape-consistent PE organization."""

    if n_pe == 1024:
        return PEOrganization(32, 32, 1)
    if n_pe in (4096, 8192, 12288, 16384):
        return PEOrganization(64, 64, n_pe // 4096)
    raise CandidateFormatError(f"unsupported N_PE={n_pe}")


def _clustered_positions(n: int, count: int) -> tuple[int, ...]:
    left = math.ceil(count / 2)
    right = math.floor(count / 2)
    return tuple(range(left)) + tuple(range(n - right, n))


def _clustered_from_slots(slots: Iterable[int], count: int) -> tuple[int, ...]:
    slots = tuple(slots)
    left = math.ceil(count / 2)
    right = math.floor(count / 2)
    chosen = slots[:left] + (slots[-right:] if right else ())
    return tuple(sorted(chosen))


def _geometry(n_ctrl: int, router: str, b: int, k: int, b_s: int, n_pe: int, n: int,
              e_h: int, m: int) -> tuple[float, float, float, int, int]:
    """Reproduce geometry attached to the supplied, pre-rescan candidates."""

    channel = math.ceil(b / DTE_CHANNEL_GBS)
    b_norm = b / 256
    a_comm = _LAMBDA7 * (
        (0.72 + 0.15 * channel + 0.48 * b_norm)
        + (0.60 + 1.20 * channel * b_norm)
        + (0.60 + 0.60 * b_norm)
        + 0.60 * b_norm
        + 0.672 * b_norm * _M_R[router]
    )
    a_ctrl = 4.5 * n_ctrl * _LAMBDA7
    n_vec = math.ceil((2 * n_pe) / (_R_PV * _P_LANE))
    a_comp = n_pe * _A_PE7 + n_vec * _A_LANE7
    a_sram = _G7 * (8 / 3) * k + _G7 * b_s / 128
    a_core = (1 + _DELTA_IMPL) * (a_comm + a_ctrl + a_comp + a_sram)

    mesh_side = (n + (n - 1) * _RHO_GAP) * math.sqrt(a_core)
    comp_die_w = mesh_side + _PHY_DEPTH_MM
    comp_die_h = mesh_side
    hbm_row_w = m * _HBM_WIDTH_MM + (m - 1) * _PACKAGE_GAP_MM
    module_w = max(hbm_row_w, comp_die_w) + 2 * _PACKAGE_GAP_MM
    module_h = comp_die_h + e_h * _HBM_HEIGHT_MM + (e_h + 2) * _PACKAGE_GAP_MM
    nx = math.floor(_WAFER_MM / module_w)
    ny = math.floor(_WAFER_MM / module_h)
    return a_core, module_w, module_h, nx, ny


def _canonical_digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _build_candidate(raw: dict[str, str], source_row: int, source_digest: str) -> CandidateHardware:
    try:
        n_ctrl, b, k, b_s, n_pe, n, e_h, m, dte = (
            int(raw[name]) for name in ("n_ctrl", "B", "K", "B_s", "N_PE", "N", "e_H", "m", "DTE_channel")
        )
        router = raw["r"].strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateFormatError(f"invalid candidate at CSV row {source_row}: {exc}") from exc

    if router not in _ROUTERS:
        raise CandidateFormatError(f"unsupported router={router!r} at row {source_row}")
    if n_ctrl not in (1, 2) or e_h not in (1, 2) or min(b, k, b_s, n_pe, n, m, dte) <= 0:
        raise CandidateFormatError(f"out-of-domain candidate at CSV row {source_row}")
    org = pe_organization(n_pe)

    expected_dte = math.ceil(b / DTE_CHANNEL_GBS)
    issues: list[str] = []
    if dte != expected_dte:
        issues.append(f"DTE_channel={dte}, expected ceil(B/128)={expected_dte}")

    t_hbm = min(math.ceil(m * HBM_ACTIVITY * HBM_STACK_PEAK_GBS / b), n - 1)
    p_usable = math.floor(0.8 * 4 * n)
    d_d2d = min(math.floor((p_usable - e_h * t_hbm) / 4), n - t_hbm)
    if d_d2d < 1:
        issues.append(f"derived d_d2d={d_d2d} is infeasible")

    hbm_positions = _clustered_positions(n, t_hbm)
    free = tuple(i for i in range(n) if i not in hbm_positions)
    d2d_hbm = _clustered_from_slots(free, max(0, d_d2d))
    d2d_plain = _clustered_positions(n, max(0, d_d2d))
    if set(hbm_positions) & set(d2d_hbm):
        raise AssertionError("HBM and D2D clustered ports overlap")

    a_core, module_w, module_h, nx, ny = _geometry(
        n_ctrl, router, b, k, b_s, n_pe, n, e_h, m
    )
    independent = {
        "n_ctrl": n_ctrl, "router": router, "B_GBs": b, "K_MiB": k,
        "B_s_GBs": b_s, "N_PE": n_pe, "N": n, "e_H": e_h, "m": m,
        "DTE_channel": dte,
    }
    digest = _canonical_digest(independent)
    return CandidateHardware(
        candidate_id=f"cand-{digest[:16]}", candidate_digest=digest,
        source_row=source_row, source_csv_digest=source_digest,
        status="candidate_semantic_mismatch" if issues else "valid",
        semantic_issues=tuple(issues), **independent,
        f_Hz=FREQUENCY_HZ, cycle_ns=2.0,
        exu_x=org.exu_x, exu_y=org.exu_y, sa_count=org.sa_count,
        P_core_FLOPs=2 * n_pe * FREQUENCY_HZ,
        P_core_TFLOPs=2 * n_pe * FREQUENCY_HZ / 1e12,
        noc_link_width_bits=16 * b, noc_payload_per_cycle=b // 8,
        sram_read_GBs=b_s, sram_write_GBs=b_s,
        sram_read_width_bits=16 * b_s, sram_write_width_bits=16 * b_s,
        sram_bank_count=max(16, math.ceil(16 * b_s / 512)),
        dte_channel_width_bits=DTE_CHANNEL_WIDTH_BITS,
        dte_aggregate_width_bits=dte * DTE_CHANNEL_WIDTH_BITS,
        t_hbm=t_hbm, d_d2d=d_d2d, HBM_stack_count=e_h * m,
        HBM_stack_capacity_GB=HBM_STACK_CAPACITY_GB,
        HBM_stack_peak_GBs=HBM_STACK_PEAK_GBS,
        HBM_stack_sustained_GBs=HBM_STACK_SUSTAINED_GBS,
        HBM_NoC_observable_GBs=e_h * min(m * HBM_STACK_SUSTAINED_GBS, t_hbm * b),
        D2D_edge_one_dir_GBs=min(d_d2d * b, D2D_PHY_EDGE_ONE_DIR_GBS),
        hbm_edges=("north",) if e_h == 1 else ("north", "south"),
        hbm_port_positions=hbm_positions,
        d2d_port_positions_hbm_edge=d2d_hbm,
        d2d_port_positions_plain_edge=d2d_plain,
        core_area_mm2_geometry_provenance=a_core, module_w_mm=module_w, module_h_mm=module_h,
        wafer_nx=nx, wafer_ny=ny, modules_per_wafer=nx * ny,
        topology_status="valid" if nx * ny >= 36 else "topology_capacity_infeasible",
        geometry_provenance="supplied_csv_original_physical_scan_not_500MHz_area_rescan",
    )


def load_candidates(path: str | Path | None = None, *, expected_count: int | None = 383) -> list[CandidateHardware]:
    """Load candidates, reject duplicate rows, and derive all runtime fields."""

    csv_path = Path(path) if path is not None else DEFAULT_CANDIDATE_CSV
    source_bytes = csv_path.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _FIELDS:
            raise CandidateFormatError(f"unexpected CSV columns: {reader.fieldnames!r}")
        candidates = [
            _build_candidate(row, source_row=index, source_digest=source_digest)
            for index, row in enumerate(reader, start=2)
        ]
    if expected_count is not None and len(candidates) != expected_count:
        raise CandidateFormatError(f"expected {expected_count} candidates, found {len(candidates)}")
    digests = [candidate.candidate_digest for candidate in candidates]
    if len(set(digests)) != len(digests):
        raise CandidateFormatError("candidate CSV contains duplicate independent configurations")
    return candidates


__all__ = [
    "CandidateFormatError", "CandidateHardware", "DEFAULT_CANDIDATE_CSV",
    "DTE_CHANNEL_GBS", "DTE_CHANNEL_WIDTH_BITS", "FREQUENCY_HZ",
    "HBM_STACK_CAPACITY_GB", "HBM_STACK_PEAK_GBS", "PACKET_PAYLOAD_BYTES",
    "PEOrganization", "load_candidates", "pe_organization",
]
