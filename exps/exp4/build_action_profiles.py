#!/usr/bin/env python3
"""Build compressed FLOP/byte/dependency profiles from the frozen exp2 DAGs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
EXP2 = ROOT.parent / "exp2" / "exp2_1"
if str(EXP2) not in sys.path:
    sys.path.insert(0, str(EXP2))

from e2e_replay import build_inference_replay, build_training_replay  # noqa: E402
from model_manifests import load_model_manifests  # noqa: E402


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _die(resource: str) -> int:
    found = re.search(r"die(\d+)", resource)
    if not found:
        raise ValueError(f"resource has no die: {resource}")
    return int(found.group(1))


def _route_endpoints(resources: Iterable[str]) -> tuple[int, int] | None:
    links = [r for r in resources if r.startswith("d2d.")]
    if not links:
        return None
    first = re.fullmatch(r"d2d\.die(\d+)\.to\.die(\d+)", links[0])
    last = re.fullmatch(r"d2d\.die(\d+)\.to\.die(\d+)", links[-1])
    if not first or not last:
        raise ValueError("invalid D2D resource")
    return int(first.group(1)), int(last.group(2))


def _descriptor(action: Any) -> dict[str, Any]:
    resources = tuple(action.resource_set)
    endpoint = _route_endpoints(resources)
    tensors = [r for r in resources if r.startswith("tensor.")]
    vectors = [r for r in resources if r.startswith("vector.")]
    hbm = any(r.startswith("hbm.") for r in resources)
    return {
        "operator": action.operator, "bytes": float(action.bytes),
        "runtime_work": float(action.runtime_work),
        "tensor_count": len(tensors), "vector_count": len(vectors),
        "has_hbm": hbm,
        "d2d_source": endpoint[0] if endpoint else None,
        "d2d_destination": endpoint[1] if endpoint else None,
        "reference_hops": sum(r.startswith("d2d.") for r in resources),
        "reference_duration_cycles": float(action.duration_cycles),
    }


def _critical_path(actions: tuple[Any, ...]) -> list[dict[str, Any]]:
    by_id = {action.action_id: action for action in actions}
    ids = set(by_id)
    finish: dict[str, float] = {}
    previous: dict[str, str | None] = {}
    for action in actions:
        deps = [dep for dep in action.deps if dep in ids]
        predecessor = max(deps, key=lambda dep: finish[dep]) if deps else None
        finish[action.action_id] = (finish[predecessor] if predecessor else 0.0) + action.duration_cycles
        previous[action.action_id] = predecessor
    tail = max(finish, key=finish.get)
    path = []
    while tail is not None:
        action = by_id[tail]
        path.append(_descriptor(action))
        tail = previous[tail]
    path.reverse()
    return path


def _add(mapping: dict[str, float], key: str, value: float) -> None:
    mapping[key] = mapping.get(key, 0.0) + value


def _demands(actions: tuple[Any, ...]) -> dict[str, Any]:
    tensor: dict[str, float] = {}; vector: dict[str, float] = {}
    sram_read: dict[str, float] = {}; sram_write: dict[str, float] = {}
    noc: dict[str, float] = {}; dte: dict[str, float] = {}; reducer: dict[str, float] = {}
    control: dict[str, float] = {}; hbm_ref_bytes: dict[str, float] = {}
    hbm_ref_count: dict[str, float] = {}; hbm_local_bytes: dict[str, float] = {}
    hbm_local_count: dict[str, float] = {}; flows: dict[tuple[int, int, str], list[float]] = {}
    for action in actions:
        resources = tuple(action.resource_set)
        tensors = [r for r in resources if r.startswith("tensor.")]
        vectors = [r for r in resources if r.startswith("vector.")]
        for resource in tensors:
            _add(tensor, str(_die(resource)), action.runtime_work / len(tensors))
        for resource in vectors:
            _add(vector, str(_die(resource)), action.runtime_work / len(vectors))
        for resource in resources:
            if resource.startswith("sram") and "read" in resource:
                _add(sram_read, str(_die(resource)), action.bytes)
            elif resource.startswith("sram") and "write" in resource:
                _add(sram_write, str(_die(resource)), action.bytes)
            elif resource.startswith("noc."):
                _add(noc, resource, action.bytes)
            elif resource.startswith("dte."):
                _add(dte, str(_die(resource)), action.bytes)
            elif resource.startswith("reducer."):
                _add(reducer, str(_die(resource)), action.bytes)
            elif resource.startswith("control."):
                _add(control, str(_die(resource)), 1.0)
        stacks = [r for r in resources if re.fullmatch(r"hbm\.stack\d+", r)]
        if stacks:
            stack = stacks[0].split("stack", 1)[1]
            _add(hbm_ref_bytes, stack, action.bytes); _add(hbm_ref_count, stack, 1.0)
        ingress = next((r for r in resources if r.startswith("hbm.ingress.")), None)
        append = next((r for r in resources if r.startswith("hbm.append.")), None)
        if ingress or append:
            target = str(_die(ingress or append))
            _add(hbm_local_bytes, target, action.bytes); _add(hbm_local_count, target, 1.0)
        endpoint = _route_endpoints(resources)
        if endpoint:
            kind = "hbm" if ingress else "fabric"
            key = (endpoint[0], endpoint[1], kind)
            item = flows.setdefault(key, [0.0, 0.0])
            item[0] += action.bytes; item[1] += 1.0
    return {
        "tensor_flops": tensor, "vector_flops": vector,
        "sram_read_bytes": sram_read, "sram_write_bytes": sram_write,
        "noc_bytes": noc, "dte_bytes": dte, "reducer_bytes": reducer,
        "control_actions": control, "hbm_ref_stack_bytes": hbm_ref_bytes,
        "hbm_ref_stack_actions": hbm_ref_count, "hbm_local_die_bytes": hbm_local_bytes,
        "hbm_local_die_actions": hbm_local_count,
        "d2d_flows": [{"source": a, "destination": b, "kind": kind,
                       "bytes": value[0], "actions": value[1]}
                      for (a, b, kind), value in sorted(flows.items())],
    }


def _profile(case_id: str, kind: str, base: tuple[Any, ...], opt: tuple[Any, ...],
             source: dict[str, Any], opt_field: str) -> dict[str, Any]:
    if {a.invariant_tuple() for a in base} != {a.invariant_tuple() for a in opt}:
        raise AssertionError(f"same-work failure: {case_id}")
    value = {
        "case_id": case_id, "kind": kind, "action_count": len(base),
        "demands": _demands(base),
        "naive_dependency_path": _critical_path(base),
        "sw_opt_dependency_path": _critical_path(opt),
        "source_naive_cycles": float(source["T_base_cycles"]),
        "source_sw_opt_cycles": float(source[opt_field]),
        "source_result_digest": source["result_digest"],
    }
    value["profile_digest"] = _digest(value)
    return value


def build() -> dict[str, Any]:
    training = json.loads((EXP2 / "results/training_e2e.json").read_text())
    prefill = json.loads((EXP2 / "results/inference_prefill_pd_breakdown.json").read_text())
    decode = json.loads((EXP2 / "results/inference_decode_e2e.json").read_text())
    sources = {row["case_id"]: row for row in training + prefill + decode}
    profiles = []
    for model_id, manifest in load_model_manifests().items():
        for seq in (2304, 36864):
            pair = build_training_replay(manifest, seq)
            profiles.append(_profile(f"train__{model_id}__s{seq}", "training",
                                     pair.base_actions, pair.full_train_actions,
                                     sources[f"train__{model_id}__s{seq}"],
                                     "T_full_train_overlap_cycles"))
            inference = build_inference_replay(manifest, 64, prefill_seq=seq, kv_length=36864)
            phases = {"prefill", "handoff", "handoff_wait"}
            base = tuple(a for a in inference.base_actions if a.phase in phases)
            opt = tuple(a for a in inference.overlap_actions if a.phase in phases)
            case = f"prefill_pd__{model_id}__s{seq}"
            profiles.append(_profile(case, "prefill", base, opt, sources[case], "T_overlap_cycles"))
        for batch in (64, 512):
            pair = build_inference_replay(manifest, batch, prefill_seq=2304, kv_length=36864)
            base = tuple(a for a in pair.base_actions if a.phase == "decode")
            opt = tuple(a for a in pair.overlap_actions if a.phase == "decode")
            case = f"decode__{model_id}__b{batch}"
            profiles.append(_profile(case, "decode", base, opt, sources[case], "T_overlap_cycles"))
    document = {"schema_version": "exp4.action_profiles.v1", "profile_count": len(profiles),
                "profiles": sorted(profiles, key=lambda row: row["case_id"])}
    document["document_digest"] = _digest(document)
    return document


def main() -> int:
    output = ROOT / "inputs/action_profiles.json"; output.parent.mkdir(parents=True, exist_ok=True)
    document = build()
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {document['profile_count']} action-derived workload profiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
