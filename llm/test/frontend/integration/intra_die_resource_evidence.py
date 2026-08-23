"""Observable simulator resource evidence for intra-die comparisons."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping


SCHEMA_VERSION = "wafer_frontend.intra_die_resource_evidence/v1alpha1"
CYCLE_NS = 2
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_PRIMITIVE_RE = re.compile(
    r"Core (\d+) (start|end) compute primitive ([A-Za-z0-9_]+)\..*?"
    r"\|\s*(\d+) (ns|us)"
)
_D2D_RE = re.compile(r"\[D2D\].*?busy_cycles=(\d+) stall_cycles=(\d+)")
_MEMORY_RE = re.compile(r"\[PROGRAM_MEMORY\]\s+core=(\d+)\s+(.+)$")
_INTEGER_FIELD_RE = re.compile(r"([a-z0-9_]+)=(\d+)")


def _category(name: str) -> str:
    if name == "Matmul_f":
        return "compute"
    if name == "Lsu_mem":
        return "dte_lsu"
    return "issue_control"


def parse_resource_output(text: str) -> dict[str, object]:
    """Parse only explicit markers; reject unclosed or reordered intervals."""
    clean = _ANSI_RE.sub("", text)
    active: dict[int, tuple[str, int]] = {}
    busy: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for match in _PRIMITIVE_RE.finditer(clean):
        core = int(match.group(1)); phase = match.group(2)
        name = match.group(3)
        timestamp = int(match.group(4)) * (
            1000 if match.group(5) == "us" else 1
        )
        if phase == "start":
            if core in active:
                raise ValueError(f"overlapping primitive interval on core {core}")
            active[core] = (name, timestamp)
            continue
        started = active.pop(core, None)
        if started is None or started[0] != name or timestamp < started[1]:
            raise ValueError(f"unmatched primitive end on core {core}: {name}")
        category = _category(name)
        busy[core][category] += timestamp - started[1]
        counts[core][name] += 1
    if active:
        raise ValueError("simulator output contains unclosed primitive intervals")

    d2d_matches = tuple(_D2D_RE.finditer(clean))
    d2d = (
        {
            "busy_cycles": int(d2d_matches[-1].group(1)),
            "stall_cycles": int(d2d_matches[-1].group(2)),
        }
        if d2d_matches else {"busy_cycles": None, "stall_cycles": None}
    )
    memory: list[dict[str, object]] = []
    for match in _MEMORY_RE.finditer(clean):
        fields = {name: int(value) for name, value in _INTEGER_FIELD_RE.findall(match.group(2))}
        memory.append({"core": int(match.group(1)), **fields})
    per_core = []
    for core in sorted(set(busy) | {int(row["core"]) for row in memory}):
        category_ns = {
            category: busy[core].get(category, 0)
            for category in ("compute", "dte_lsu", "issue_control")
        }
        per_core.append({
            "core": core,
            "primitive_busy_ns": category_ns,
            "primitive_busy_cycles": {
                category: (duration + CYCLE_NS - 1) // CYCLE_NS
                for category, duration in category_ns.items()
            },
            "primitive_counts": dict(sorted(counts[core].items())),
        })
    return {
        "cycle_ns": CYCLE_NS,
        "per_core": per_core,
        "program_memory": sorted(memory, key=lambda row: int(row["core"])),
        "d2d": d2d,
        "overlap_cycles": None,
        "overlap_status": "unsupported:no_explicit_overlap_marker",
    }


def collect_resource_evidence(output_dir: Path, repeat: int) -> dict[str, object]:
    parsed: list[dict[str, object]] = []
    signatures: list[str] = []
    for index in range(repeat):
        path = output_dir / "run" / f"stdout.{index}.log"
        if not path.is_file():
            raise ValueError(f"missing simulator stdout for resource evidence: {path}")
        row = parse_resource_output(path.read_text(encoding="utf-8"))
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        parsed.append(row); signatures.append(hashlib.sha256(encoded).hexdigest())
    stable = len(set(signatures)) == 1
    if not stable:
        raise ValueError("resource evidence changed across simulator repeats")
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "producer_pass": "intra_die_performance_runner",
        "repeat": repeat,
        "repeat_signatures": signatures,
        "repeat_signature_stable": stable,
        "resources": parsed[0],
    }
    digest = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**semantic, "id": f"intra_die_resource_evidence_{digest[:16]}"}


def write_resource_evidence(output_dir: Path, repeat: int) -> dict[str, object]:
    evidence = collect_resource_evidence(output_dir, repeat)
    (output_dir / "intra_die_resource_evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


__all__ = ["collect_resource_evidence", "parse_resource_output", "write_resource_evidence"]
