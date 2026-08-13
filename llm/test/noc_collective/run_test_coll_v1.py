#!/usr/bin/env python3
"""NoC collective V1 planner + Tier0 end-to-end matrix."""
from __future__ import annotations
import json, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"
OPS = {
    "p2p": (2, 1), "scatter": (3, 2), "gather": (3, 2),
    "broadcast": (3, 2), "allgather": (3, 6), "alltoall": (3, 6),
}

def workload(op: str, n: int) -> dict:
    return {
        "vars": {"B": 1, "T": 1}, "pipeline": 1, "source": [],
        "chips": [{"chip_id": 0,
            "cores": [{"id": i, "loop": 1, "worklist": []} for i in range(n)],
            "collectives": [{"op": op, "collective_id": 100 + list(OPS).index(op),
                "group": list(range(n)), "root": 0, "count": n,
                "chunk_bits": 1024, "stride_bits": 1024, "terminal": True}]}]
    }

def run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT / "build", text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)

def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    oracle = run([sys.executable, str(HERE / "oracle.py")])
    tests.append(("independent Python oracle", oracle.returncode == 0 and "oracle self-test: PASS" in oracle.stdout, oracle.stdout.strip()))
    unit = run([str(NPUSIM), "--coll-v1-selftest"])
    tests.append(("planner/barrier self-test", unit.returncode == 0 and
                  "PASS (15 checks)" in unit.stdout, "15 checks"))
    with tempfile.TemporaryDirectory(prefix="coll_v1_") as td:
        for op, (n, expected_flows) in OPS.items():
            path = Path(td) / f"{op}.json"
            path.write_text(json.dumps(workload(op, n)))
            for backend in ("cycle", "beha"):
                result = run([str(NPUSIM), "--workload-config", str(path),
                    "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
                    "--simulation-config", f"../llm/test/noc_collective/simulation/v1_{backend}.json",
                    "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"])
                output = result.stdout
                flow_line = next((x for x in output.splitlines() if "[FLOW_DONE]" in x), "")
                flows = flow_line.count("@")
                cycle_match = re.search(r"Catch test finished.*?\|.*?(\d+) ns", output)
                passed = (result.returncode == 0 and "Catch test finished" in output and
                          "router_residual=0" in output and "data_balanced=1 ctrl_balanced=1" in output and
                          flows == expected_flows)
                detail = f"flows={flows}/{expected_flows}, cycle={cycle_match.group(1) if cycle_match else '?'}"
                tests.append((f"{op}/{backend}", passed, detail))
    mixed_path = HERE / "workload" / "v1_mixed_unicast.json"
    for backend in ("cycle", "beha"):
        result = run([str(NPUSIM), "--workload-config", str(mixed_path),
            "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
            "--simulation-config", f"../llm/test/noc_collective/simulation/v1_{backend}.json",
            "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"])
        line = next((x for x in result.stdout.splitlines() if "[FLOW_DONE]" in x), "")
        tags = [int(x) for x in re.findall(r"\d+:(\d+):\d+@", line)]
        isolated = len(tags) == 2 and any(tag < 0x8000 for tag in tags) and any(tag >= 0x8000 for tag in tags)
        tests.append((f"mixed collective+unicast/{backend}", result.returncode == 0 and isolated and "router_residual=0" in result.stdout and "data_balanced=1 ctrl_balanced=1" in result.stdout, f"tags={tags}"))

    with tempfile.TemporaryDirectory(prefix="coll_v1_epoch_") as td:
        repeated = workload("p2p", 2)
        first = repeated["chips"][0]["collectives"][0]
        first["terminal"] = False; first["epoch"] = 0
        second = json.loads(json.dumps(first)); second["epoch"] = 1; second["terminal"] = True
        repeated["chips"][0]["collectives"].append(second)
        work_path = Path(td) / "epochs.json"; work_path.write_text(json.dumps(repeated))
        for backend in ("cycle", "beha"):
            result = run([str(NPUSIM), "--workload-config", str(work_path),
                "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
                "--simulation-config", f"../llm/test/noc_collective/simulation/v1_{backend}.json",
                "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"])
            line = next((x for x in result.stdout.splitlines() if "[FLOW_DONE]" in x), "")
            tests.append((f"consecutive epochs/{backend}", result.returncode == 0 and line.count("@") == 2 and "router_residual=0" in result.stdout, f"flows={line.count(chr(64))}/2"))

    with tempfile.TemporaryDirectory(prefix="coll_v1_negative_") as td:
        base_sim = json.loads((HERE / "simulation" / "v1_cycle.json").read_text())
        negative = []
        disabled = json.loads(json.dumps(base_sim)); disabled["noc"]["collective"]["enabled"] = False
        negative.append(("disabled gate", disabled, workload("broadcast", 2)))
        invalid_tier = json.loads(json.dumps(base_sim)); invalid_tier["noc"]["collective"]["tier"] = 3
        negative.append(("invalid tier", invalid_tier, workload("broadcast", 2)))
        no_dte = json.loads(json.dumps(base_sim)); no_dte["dte"]["use_beha_dte"] = False
        negative.append(("DTE required", no_dte, workload("broadcast", 2)))
        bad_group = workload("broadcast", 2); bad_group["chips"][0]["collectives"][0]["group"] = [1, 0]
        negative.append(("unsorted group", base_sim, bad_group))
        duplicate = workload("p2p", 2); copy = json.loads(json.dumps(duplicate["chips"][0]["collectives"][0])); copy["terminal"] = False; duplicate["chips"][0]["collectives"].append(copy)
        negative.append(("duplicate CollectiveKey", base_sim, duplicate))
        reserved = json.loads((HERE / "workload" / "v1_mixed_unicast.json").read_text()); reserved["chips"][0]["cores"][2]["worklist"][0]["cast"][0]["tag"] = 0x8000
        negative.append(("regular tag in collective namespace", base_sim, reserved))
        for index, (name, sim, work) in enumerate(negative):
            sim_path = Path(td) / f"sim_{index}.json"; sim_path.write_text(json.dumps(sim))
            work_path = Path(td) / f"work_{index}.json"; work_path.write_text(json.dumps(work))
            result = run([str(NPUSIM), "--workload-config", str(work_path),
                "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
                "--simulation-config", str(sim_path),
                "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"])
            tests.append((name, result.returncode != 0, f"exit={result.returncode}"))

    for name, ok, detail in tests:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    passed = sum(ok for _, ok, _ in tests)
    print(f"NoC collective V1: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1

if __name__ == "__main__": raise SystemExit(main())
