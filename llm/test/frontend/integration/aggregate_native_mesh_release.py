"""Fail-closed 100-shape release aggregation over audited native matrix shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import run_dense_native_mesh_matrix as dense
from . import run_moe_full_model_native_mesh_matrix as moe


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def aggregate(family: str, roots: tuple[Path, ...]) -> dict:
    if family not in ("dense", "moe"):
        raise ValueError("family must be dense or moe")
    if not roots:
        raise ValueError("at least one physical matrix shard is required")
    driver = dense if family == "dense" else moe
    expected = set(driver.RELEASE_SHAPES)
    completed: dict[str, dict] = {}
    count = None
    shards: set[int] = set()
    contract = None
    for raw in roots:
        root = raw.resolve()
        binding_path = root / "matrix_binding.json"
        binding = _read(binding_path)
        driver_path = Path(driver.__file__).resolve()
        runner_name = ("run_dense_sequence_runtime_canary.py" if family == "dense"
                       else "run_moe_full_model_sequence_runtime_canary.py")
        runner_path = driver_path.parent / runner_name
        if (binding.get("driver_sha256") != hashlib.sha256(driver_path.read_bytes()).hexdigest()
                or binding.get("runner_sha256") != hashlib.sha256(runner_path.read_bytes()).hexdigest()):
            raise ValueError("release shard driver or native workload runner source drifted")
        if family == "dense" and binding.get("dram_config_sha256") != hashlib.sha256(
                (driver_path.parents[4] / "DRAMSys/configs/hbm2-example.json").read_bytes()
        ).hexdigest():
            raise ValueError("release shard frozen DRAM configuration drifted")
        receipt = _read(root / "matrix_receipt.json")
        if receipt.get("binding_sha256") != hashlib.sha256(binding_path.read_bytes()).hexdigest():
            raise ValueError(f"shard binding bytes drifted: {root}")
        index, size = binding.get("shard_index"), binding.get("shard_count")
        if type(index) is not int or type(size) is not int or size < 1 or index not in range(size):
            raise ValueError(f"invalid shard index/count: {root}")
        if count is None:
            count = size
        if size != count or index in shards:
            raise ValueError("duplicate or inconsistent release shard")
        shards.add(index)
        shapes = driver.select_shapes(driver.RELEASE_SHAPES, shard_index=index, shard_count=size)
        if (tuple(binding.get("shapes", ())) != shapes or
            tuple(receipt.get("shapes", ())) != shapes or
            tuple(receipt.get("completed_shapes", ())) != shapes or
            receipt.get("status") != "verified"):
            raise ValueError(f"shard lacks its exact complete 100-shape partition: {root}")
        stable = {key: value for key, value in binding.items()
                  if key not in ("shapes", "shard_index", "shard_count")}
        if contract is None:
            contract = stable
        elif stable != contract:
            raise ValueError("different runner, native tools, simulation or source across shards")
        for shape in shapes:
            if shape in completed:
                raise ValueError(f"duplicate shape: {shape}")
            driver.audit_cached_case(root / shape, shape)
            completed[shape] = {"shard_index": index, "case": f"{root / shape / 'case_evidence.json'}"}
    if shards != set(range(count)) or set(completed) != expected:
        raise ValueError(f"release matrix incomplete: {len(completed)}/100 shapes, {len(shards)}/{count} shards")
    return {"status": "verified", "family": family, "physical_shapes": len(completed),
            "independent_native_executions": 2 * len(completed),
            "shard_count": count, "binding": contract,
            "cases": {shape: completed[shape] for shape in driver.RELEASE_SHAPES}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("dense", "moe"), required=True)
    parser.add_argument("--shard-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("release receipt output already exists")
    report = aggregate(args.family, tuple(args.shard_roots))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.family} resident release PASS 100 shapes, 200 native executions", flush=True)


if __name__ == "__main__":
    main()
