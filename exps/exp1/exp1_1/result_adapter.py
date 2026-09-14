#!/usr/bin/env python3
"""Normalize and validate exp1-1 trace-calibrated result records.

The trace replay owns cycle estimation.  This module owns the stable public
JSON/CSV schema and derives every ratio from the four counterfactual cycles so
the two implementations cannot silently disagree.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


CYCLE_FIELDS = ("T00_cycles", "T10_cycles", "T01_cycles", "T11_cycles")
DERIVED_FIELDS = (
    "inter_speedup_without_intra",
    "inter_speedup_with_intra",
    "intra_speedup_without_inter",
    "intra_speedup_with_inter",
    "total_speedup",
    "synergy",
)
CONGESTION_FIELDS = (
    "normal_cycles",
    "congested_upper_bound_cycles",
    "congestion_factor",
)
SOURCE_FIELDS = (
    "T00_source", "T10_source", "T01_source", "T11_source",
    "simulation_signature", "schedule_source", "result_source",
)
OUTPUT_FIELDS = (
    "case_id", "mesh", "operator", "model", "layer", "seq_len",
    "logical_M", "logical_N", "logical_K",
    "runtime_M", "runtime_N", "runtime_K",
    "tile_M", "tile_N", "tile_K", "Tm", "Tn", "Tk",
    "logical_flops", "runtime_flops",
    *CYCLE_FIELDS, *DERIVED_FIELDS, *CONGESTION_FIELDS, *SOURCE_FIELDS,
    "status", "error",
)


def _positive(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return number


def _set_ratio(record: dict[str, object], field: str, expected: float) -> None:
    current = record.get(field)
    if current not in (None, ""):
        actual = _positive(current, field)
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                f"{field}={actual} disagrees with cycles-derived value {expected}"
            )
    record[field] = expected


def enrich_record(value: Mapping[str, object]) -> dict[str, object]:
    """Return one schema-complete record, deriving ratios fail-closed."""
    record = {str(key): item for key, item in value.items()}
    status = str(record.get("status", "ok")).strip().lower()
    successful = status in {"ok", "estimated_via_tiling_and_padding"}
    record["status"] = status
    record.setdefault("error", "")
    for field in OUTPUT_FIELDS:
        record.setdefault(field, None)

    cycles_present = all(record.get(field) not in (None, "") for field in CYCLE_FIELDS)
    if successful and not cycles_present:
        missing = [field for field in CYCLE_FIELDS if record.get(field) in (None, "")]
        raise ValueError(f"ok result is missing cycle fields: {', '.join(missing)}")
    if not cycles_present:
        return record

    t00, t10, t01, t11 = (
        _positive(record[field], field) for field in CYCLE_FIELDS
    )
    ratios = {
        "inter_speedup_without_intra": t00 / t10,
        "inter_speedup_with_intra": t01 / t11,
        "intra_speedup_without_inter": t00 / t01,
        "intra_speedup_with_inter": t10 / t11,
        "total_speedup": t00 / t11,
        "synergy": (t10 * t01) / (t00 * t11),
    }
    for field, expected in ratios.items():
        _set_ratio(record, field, expected)

    if record.get("normal_cycles") in (None, ""):
        record["normal_cycles"] = t11
    normal = _positive(record["normal_cycles"], "normal_cycles")
    congested_value = record.get("congested_upper_bound_cycles")
    if congested_value not in (None, ""):
        congested = _positive(congested_value, "congested_upper_bound_cycles")
        if record.get("congestion_factor") in (None, ""):
            record["congestion_factor"] = congested / normal
        else:
            _positive(record["congestion_factor"], "congestion_factor")
    elif successful:
        raise ValueError("ok result is missing congested_upper_bound_cycles")

    replay_source = record.get("T00_T10_T01_source")
    estimate_source = record.get("estimate_source")
    for field, default in (
        ("T00_source", replay_source or "cycle_accurate_trace_replay"),
        ("T10_source", replay_source or "cycle_accurate_trace_replay"),
        ("T01_source", replay_source or "cycle_accurate_trace_replay"),
        ("T11_source", "cycle_accurate_simulation"),
        ("result_source", estimate_source or "cycle_accurate_trace_calibrated_estimate"),
    ):
        if record.get(field) in (None, ""):
            record[field] = default
    return record


def enrich_records(
    records: Iterable[Mapping[str, object]], *, expected_cases: int | None = 176
) -> list[dict[str, object]]:
    enriched = [enrich_record(record) for record in records]
    if expected_cases is not None and len(enriched) != expected_cases:
        raise ValueError(f"expected {expected_cases} records, got {len(enriched)}")
    case_ids = [str(record.get("case_id", "")) for record in enriched]
    if any(not case_id for case_id in case_ids):
        raise ValueError("every record must have a non-empty case_id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case_id values must be unique")
    return enriched


def read_records(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("JSON input must be an array of objects")
        return value
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_records(records: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    extras = sorted({key for record in records for key in record} - set(OUTPUT_FIELDS))
    fields = [*OUTPUT_FIELDS, *extras]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record.get(key)) for key in fields})


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else value


def main() -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base / "results" / "trace_replay_results.json")
    parser.add_argument("--output-dir", type=Path, default=base / "results")
    parser.add_argument("--expected-cases", type=int, default=176)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    expected = None if args.allow_partial else args.expected_cases
    records = enrich_records(read_records(args.input), expected_cases=expected)
    write_records(records, args.output_dir)
    print(f"wrote {len(records)} schema-complete records to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
