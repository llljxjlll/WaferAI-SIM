#!/usr/bin/env python3
"""Program/legacy differential for the published P3 NPU compute chain."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote


EXPECTED = [
    {
        "opcode": "MATMUL",
        "prim": "Matmul_f",
        "params": {"B": 1, "T": 4, "C": 64, "OC": 64},
        "input_offset_bytes": 0,
        "data_offset_bytes": 4096,
        "output_offset_bytes": 8192,
        "input": "dram_label p3diff_input",
        "output": "p3diff_matmul_out",
        "cost": {"exu_ops": 0, "sfu_ops": 0, "vec_ops": 32768,
                 "compute_cycle_ns": 512},
    },
    {
        "opcode": "ATTENTION",
        "prim": "Attention_f",
        "params": {"B": 1, "T": 4, "C": 64, "NH": 4, "R": 3},
        "input_offset_bytes": 8192,
        "data_offset_bytes": 9216,
        "output_offset_bytes": 10240,
        "input": "p3diff_matmul_out",
        "output": "p3diff_attention_out",
        "cost": {"exu_ops": 4096, "sfu_ops": 64, "vec_ops": 128,
                 "compute_cycle_ns": 5},
    },
    {
        "opcode": "LAYERNORM",
        "prim": "Layernorm_f",
        "params": {"B": 1, "T": 4, "C": 64},
        "input_offset_bytes": 10240,
        "data_offset_bytes": 12288,
        "output_offset_bytes": 13312,
        "input": "p3diff_attention_out",
        "output": "p3diff_layernorm_out",
        "cost": {"exu_ops": 0, "sfu_ops": 4, "vec_ops": 2060,
                 "compute_cycle_ns": 32},
    },
    {
        "opcode": "GELU",
        "prim": "Gelu_f",
        "params": {"N": 256},
        "input_offset_bytes": 13312,
        "data_offset_bytes": 14336,
        "output_offset_bytes": 15360,
        "input": "p3diff_layernorm_out",
        "output": "p3diff_result",
        "cost": {"exu_ops": 0, "sfu_ops": 256, "vec_ops": 1024,
                 "compute_cycle_ns": 16},
    },
]

TYPE_TO_OPCODE = {
    "Matmul_f": "MATMUL",
    "Attention_f": "ATTENTION",
    "Layernorm_f": "LAYERNORM",
    "Gelu_f": "GELU",
}
PARAMETER_ORDER = {
    "MATMUL": ["B", "T", "C", "OC"],
    "ATTENTION": ["B", "T", "C", "NH", "R"],
    "LAYERNORM": ["B", "T", "C"],
    "GELU": ["N"],
}
LIFECYCLE_DETAIL = re.compile(
    r"^allocation=(\d+) address=(\d+) bytes=(\d+) label=(.*)$"
)
COMPLETION = re.compile(r"Core 0 end compute primitive ([A-Za-z0-9_]+)")


def fail(message: str) -> None:
    raise AssertionError(message)


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            fail(f"malformed fixture manifest token: {token}")
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def parse_fixture_manifest(stdout: str) -> tuple[list[dict[str, Any]],
                                                   list[dict[str, str]]]:
    computes: list[dict[str, Any]] = []
    binds: list[dict[str, str]] = []
    for line in stdout.splitlines():
        if line.startswith("P3_DIFF_BIND "):
            fields = parse_fields(line)
            binds.append({"input": unquote(fields["input"]),
                          "output": unquote(fields["output"])})
        elif line.startswith("P3_DIFF_COMPUTE "):
            fields = parse_fields(line)
            params = {}
            for item in fields["params"].split(","):
                name, value = item.split(":", 1)
                params[name] = int(value)
            computes.append({
                "opcode": fields["opcode"],
                "datatype": fields["datatype"],
                "params": params,
                "input_offset_bytes": int(fields["input_offset_bytes"]),
                "data_offset_bytes": int(fields["data_offset_bytes"]),
                "output_offset_bytes": int(fields["output_offset_bytes"]),
            })
    return computes, binds


def resolve(value: Any, variables: dict[str, Any]) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value in variables:
        resolved = variables[value]
        if isinstance(resolved, int):
            return resolved
    fail(f"legacy fixture value is not a direct integer/variable: {value!r}")
    return 0


def legacy_manifest(path: Path) -> tuple[list[dict[str, Any]],
                                         list[dict[str, str]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    variables = document["vars"]
    cores = document["chips"][0]["cores"]
    if len(cores) != 1 or cores[0]["id"] != 0:
        fail("legacy differential must contain exactly core 0")
    worklist = cores[0]["worklist"]
    if len(worklist) != 1:
        fail("legacy differential must contain exactly one worklist entry")

    computes: list[dict[str, Any]] = []
    labels: list[dict[str, str]] = []
    for prim in worklist[0]["prims"]:
        opcode = TYPE_TO_OPCODE.get(prim.get("type"))
        if opcode is None:
            fail(f"unexpected legacy primitive: {prim.get('type')}")
        addresses = prim["dram_address"]
        computes.append({
            "opcode": opcode,
            "datatype": "INT8",
            "params": {
                name: resolve(prim[name], variables)
                for name in PARAMETER_ORDER[opcode]
            },
            "input_offset_bytes": resolve(addresses["input"], variables),
            "data_offset_bytes": resolve(addresses["data"], variables),
            "output_offset_bytes": resolve(addresses["output"], variables),
        })
        labels.append({"input": prim["sram_address"]["indata"],
                       "output": prim["sram_address"]["outdata"]})
    return computes, labels


def expected_manifest() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    computes = [{
        "opcode": item["opcode"],
        "datatype": "INT8",
        "params": item["params"],
        "input_offset_bytes": item["input_offset_bytes"],
        "data_offset_bytes": item["data_offset_bytes"],
        "output_offset_bytes": item["output_offset_bytes"],
    } for item in EXPECTED]
    labels = [{"input": item["input"], "output": item["output"]}
              for item in EXPECTED]
    return computes, labels


def trace_threads(events: list[dict[str, Any]]) -> dict[tuple[int, int], str]:
    result: dict[tuple[int, int], str] = {}
    for event in events:
        if event.get("name") != "thread_name" or event.get("ph") != "M":
            continue
        result[(int(event["pid"]), int(event["tid"]))] = str(
            event.get("args", {}).get("name", ""))
    return result


def event_ns(event: dict[str, Any]) -> int:
    return round(float(event["ts"]) * 1000.0)


def primitive_spans(events: list[dict[str, Any]],
                    names: set[str]) -> dict[str, tuple[int, int]]:
    opened: dict[str, list[int]] = {}
    result: dict[str, tuple[int, int]] = {}
    for event in events:
        name = str(event.get("name", ""))
        phase = event.get("ph")
        if name not in names or phase not in {"B", "E"}:
            continue
        if phase == "B":
            opened.setdefault(name, []).append(event_ns(event))
        elif opened.get(name):
            if name in result:
                fail(f"primitive {name} executed more than once")
            result[name] = (opened[name].pop(0), event_ns(event))
    if set(result) != names or any(opened.values()):
        fail(f"incomplete primitive spans: {result}")
    return result


def ordered_bind_spans(events: list[dict[str, Any]]) -> list[tuple[int, int]]:
    opened: list[int] = []
    result: list[tuple[int, int]] = []
    for event in events:
        if event.get("name") != "Sram_bind_oneshot":
            continue
        if event.get("ph") == "B":
            opened.append(event_ns(event))
        elif event.get("ph") == "E" and opened:
            result.append((opened.pop(0), event_ns(event)))
    if opened:
        fail("unbalanced SRAM_BIND primitive spans")
    return result


def normalized_compute_spans(events: list[dict[str, Any]]) -> list[tuple[int, int]]:
    names = {item["prim"] for item in EXPECTED}
    spans = primitive_spans(events, names)
    ordered = [spans[item["prim"]] for item in EXPECTED]
    origin = ordered[0][0]
    return [(begin - origin, end - origin) for begin, end in ordered]


def compute_costs(events: list[dict[str, Any]]) -> list[dict[str, int]]:
    by_name: dict[str, dict[str, int]] = {}
    required = {"exu_ops", "sfu_ops", "vec_ops", "compute_cycle_ns"}
    expected_names = {item["prim"] for item in EXPECTED}
    for event in events:
        if event.get("ph") != "i":
            continue
        name = str(event.get("name", ""))
        if name not in expected_names:
            continue
        args = event.get("args", {})
        if not required.issubset(args):
            continue
        if name in by_name:
            fail(f"duplicate Compute_cost event for {name}")
        by_name[name] = {field: int(args[field]) for field in required}
    if set(by_name) != expected_names:
        fail(f"missing Compute_cost events: {by_name}")
    return [by_name[item["prim"]] for item in EXPECTED]


def lifecycle(events: list[dict[str, Any]]) -> tuple[list[tuple[Any, ...]],
                                                     set[str]]:
    threads = trace_threads(events)
    sequence: list[tuple[Any, ...]] = []
    active: set[str] = set()
    for event in events:
        phase = event.get("ph")
        if phase not in {"B", "E"}:
            continue
        thread = threads.get((int(event.get("pid", -1)),
                              int(event.get("tid", -1))), "")
        if not thread.startswith("SRAM_region_"):
            continue
        detail = LIFECYCLE_DETAIL.match(str(event.get("name", "")))
        if detail is None:
            fail(f"malformed SRAM lifecycle detail: {event.get('name')}")
        _, address, size_bytes, label = detail.groups()
        sequence.append((thread, phase, int(address), int(size_bytes), label))
        if phase == "E" and thread == "SRAM_region_alloc":
            active.add(label)
        elif phase == "E" and thread == "SRAM_region_free":
            active.discard(label)
        elif phase == "E" and thread == "SRAM_region_rename":
            active.add(label)
    return sequence, active


def run_sim(command: list[str], cwd: Path) -> tuple[str, list[dict[str, Any]]]:
    trace = cwd / "events.json"
    trace.unlink(missing_ok=True)
    proc = subprocess.run(command, cwd=cwd, text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=60)
    if proc.returncode != 0:
        print(proc.stdout)
        fail(f"npusim returned {proc.returncode}: {' '.join(command)}")
    if "[PROTO_WAIT]" in proc.stdout or "End DONE reception" not in proc.stdout:
        print(proc.stdout)
        fail("npusim did not complete the expected DONE path cleanly")
    if not trace.exists():
        fail(f"trace was not generated in {cwd}")
    return proc.stdout, json.loads(
        trace.read_text(encoding="utf-8"))["traceEvents"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", required=True, type=Path)
    parser.add_argument("--program-fixture", required=True, type=Path)
    parser.add_argument("--legacy-workload", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args()

    expected_compute, expected_labels = expected_manifest()
    legacy_compute, legacy_labels = legacy_manifest(args.legacy_workload)
    if legacy_compute != expected_compute or legacy_labels != expected_labels:
        fail("legacy JSON opcode/parameter/address/label manifest changed")

    source_root = args.legacy_workload.resolve().parents[3]
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p3-compute-diff-",
                                     dir=args.runtime_root) as temp_name:
        root = Path(temp_name)
        os.symlink(source_root / "font", root / "font",
                   target_is_directory=True)
        os.symlink(source_root / "DRAMSys", root / "DRAMSys",
                   target_is_directory=True)
        program_dir = root / "program"
        legacy_dir = root / "legacy"
        program_dir.mkdir()
        legacy_dir.mkdir()
        artifact = program_dir / "compute-chain.npup"

        generated = subprocess.run(
            [str(args.program_fixture.resolve()), str(artifact),
             "--p3-diff-chain"], cwd=program_dir, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        if generated.returncode != 0:
            print(generated.stdout)
            fail(f"program fixture returned {generated.returncode}")
        program_compute, program_binds = parse_fixture_manifest(
            generated.stdout)
        if program_compute != expected_compute or program_binds != expected_labels:
            print(generated.stdout)
            fail("encoded Program opcode/parameter/address/bind manifest changed")

        common = [
            "--hardware-config", str(args.hardware.resolve()),
            "--simulation-config", str(args.simulation.resolve()),
            "--mapping-config", str(args.mapping.resolve()),
            "--trace-window", "1000000",
        ]
        program_stdout, program_events = run_sim(
            [str(args.npusim.resolve()), "--program", str(artifact), *common],
            program_dir)
        legacy_stdout, legacy_events = run_sim(
            [str(args.npusim.resolve()), "--workload-config",
             str(args.legacy_workload.resolve()), *common], legacy_dir)

        expected_order = [item["prim"] for item in EXPECTED]
        program_order = [name for name in COMPLETION.findall(program_stdout)
                         if name in expected_order]
        legacy_order = [name for name in COMPLETION.findall(legacy_stdout)
                        if name in expected_order]
        if program_order != expected_order or legacy_order != expected_order:
            fail(f"completion order differs: program={program_order}, "
                 f"legacy={legacy_order}")

        expected_cost = [item["cost"] for item in EXPECTED]
        program_cost = compute_costs(program_events)
        legacy_cost = compute_costs(legacy_events)
        if program_cost != expected_cost or legacy_cost != expected_cost:
            fail(f"exact costs differ: expected={expected_cost}, "
                 f"program={program_cost}, legacy={legacy_cost}")

        # CONFIG duration is deliberately excluded by anchoring both traces at
        # the first compute B event. Program SRAM_BIND and legacy Set_addr are
        # equivalent per-compute label-setup work, so both remain in the spans.
        program_spans = normalized_compute_spans(program_events)
        legacy_spans = normalized_compute_spans(legacy_events)
        if program_spans != legacy_spans:
            fail(f"normalized compute spans differ: program={program_spans}, "
                 f"legacy={legacy_spans}")
        program_bind_spans = ordered_bind_spans(program_events)
        if (len(program_bind_spans) != len(EXPECTED) or
                ordered_bind_spans(legacy_events)):
            fail("one-shot bind execution count differs from 4/0")

        program_lifecycle, program_active = lifecycle(program_events)
        legacy_lifecycle, legacy_active = lifecycle(legacy_events)
        if program_lifecycle != legacy_lifecycle:
            fail("normalized SRAM label lifecycle differs between paths")
        expected_active = {
            "eternal_p3diff_matmul_w",
            "eternal_p3diff_matmul_b",
            "eternal_p3diff_layernorm_w",
            "eternal_p3diff_layernorm_b",
            "p3diff_result",
        }
        temporary = {
            "p3diff_input",
            "p3diff_matmul_out",
            "p3diff_attention_out",
            "p3diff_layernorm_out",
        }
        if program_active != expected_active or legacy_active != expected_active:
            fail(f"unexpected final labels: program={program_active}, "
                 f"legacy={legacy_active}")
        if program_active & temporary or legacy_active & temporary:
            fail("temporary chain labels remain live at DONE")

    print("[P3 COMPUTE DIFF] PASS: opcode/parameters/addresses and 4 binds match")
    print("[P3 COMPUTE DIFF] PASS: exact exu/sfu/vec/compute_cycle match")
    print("[P3 COMPUTE DIFF] PASS: CONFIG-normalized spans and order match")
    print("[P3 COMPUTE DIFF] PASS: lifecycle matches with no temporary residual")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, KeyError, ValueError, OSError,
            subprocess.TimeoutExpired) as error:
        print(f"[P3 COMPUTE DIFF] FAIL: {error}")
        sys.exit(1)
