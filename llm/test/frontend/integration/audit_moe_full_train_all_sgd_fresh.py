"""Independent read-only reopen audit of two EP1 all-SGD partial Fresh runs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


_TOOLS = {"finalizer": "npusim_program_finalizer",
          "resolver": "npusim_program_io_selftest", "npusim": "npusim"}
_OPCODES = {
    1: 17, 12: 4, 14: 10, 16: 5, 26: 2, 27: 2, 28: 1,
    30: 1, 31: 1, 32: 19, 34: 2, 35: 1, 36: 5, 37: 13,
    38: 13, 39: 2, 40: 2, 41: 5, 42: 2, 43: 2, 44: 4,
    67: 2, 128: 48, 129: 19, 130: 4, 132: 113, 134: 159,
    137: 159, 192: 4,
}
_FILES = {"hardware.json", "mapping.spec", "receipt.json",
          "all_sgd_partial_sequence.npusim.log"}
for _step in (0, 1):
    _FILES.update({f"step{_step}.linked.json", f"step{_step}.npup",
                   f"step{_step}.program_io.json",
                   f"step{_step}.finalizer.json",
                   f"step{_step}.finalizer.log",
                   f"step{_step}.resolver.log",
                   f"step{_step}.npusim.log"})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def _source_map(freeze: Path, mapping: dict[str, str], label: str) -> None:
    _require(bool(mapping) and all(
        (freeze / name).is_file() and _sha(freeze / name) == expected
        for name, expected in mapping.items()), f"{label}: source byte drift")


def _fragment_records(fragment: dict) -> list[dict]:
    return [record for stream in fragment["core_streams"]
            for record in stream["records"]]


def _physical_coverage(manifest: dict, expected_refs: list[str],
                       step: int, fresh: int) -> dict[str, tuple]:
    label = f"fresh{fresh} step{step}"
    fragments = manifest["fragments"]
    _require(manifest["producer_pass"] == "manifest_linker" and
             len(fragments) == 160 and len(manifest["core_streams"]) == 1,
             f"{label}: physical shape drift")
    opcodes = Counter(record["opcode"] for fragment in fragments
                      for record in _fragment_records(fragment))
    _require(dict(opcodes) == _OPCODES and sum(opcodes.values()) == 621,
             f"{label}: 621-record opcode closure drift")
    state_abis: dict[str, dict] = {}
    for fragment in fragments:
        for state in fragment["state_abi"]:
            prior = state_abis.setdefault(state["id"], state)
            _require(prior == state, f"{label}: conflicting StateABI")
    trainable = {abi["id"]: abi for abi in state_abis.values()
                 if abi["kind"] == "trainable_parameter"}
    route = {abi["state_ref"] for abi in state_abis.values()
             if abi["kind"] == "moe_static_route"}
    _require(len(trainable) == 19 and len(route) == 2 and
             sorted(abi["state_ref"] for abi in trainable.values()) ==
             expected_refs and
             sum(abi["size_bytes"] for abi in trainable.values()) == 952 and
             all(abi["access"] == "read_write" and abi["dtype"] == "fp16"
                 and abi["die_id"] == 0 for abi in trainable.values()),
             f"{label}: 19-state ABI closure drift")
    by_id = {item["id"]: item for item in fragments}
    state_binding_by_fragment: dict[str, list[tuple[int, dict]]] = {}
    for binding in manifest["state_operand_bindings"]:
        fragment = by_id.get(binding["fragment_id"])
        _require(fragment is not None, f"{label}: unknown state fragment")
        stream = next((item for item in fragment["core_streams"]
                       if item["logical_core"] == binding["logical_core"]), None)
        _require(stream is not None and
                 binding["fragment_record_index"] < len(stream["records"]),
                 f"{label}: unknown state record")
        opcode = stream["records"][binding["fragment_record_index"]]["opcode"]
        if opcode in (128, 129):
            _require(binding["operand_id"] == 6 and
                     binding["state_abi_id"] in state_abis,
                     f"{label}: unbound HBM address")
            if opcode == 129:
                _require(binding["state_abi_id"] in trainable,
                         f"{label}: STORE targets non-trainable StateABI")
            if binding["state_abi_id"] in trainable:
                state_binding_by_fragment.setdefault(fragment["id"], []).append(
                    (opcode, trainable[binding["state_abi_id"]]))
    source_values: dict[str, list[dict]] = {}
    for fragment in fragments:
        if 32 in {r["opcode"] for r in _fragment_records(fragment)}:
            continue
        for buffer in fragment["buffer_abi"]:
            source_values.setdefault(buffer["value_id"], []).append(fragment)
    coverage = {}
    for fragment in fragments:
        if 32 not in {r["opcode"] for r in _fragment_records(fragment)}:
            continue
        buffers = fragment["buffer_abi"]
        gradients = [b for b in buffers if b["dtype"] == "fp32"]
        updates = [b for b in buffers if
                   b["value_id"].startswith("sgd_update::") and
                   b["value_id"].endswith(".updated_weight")]
        stagings = [b for b in buffers if
                    b["value_id"].startswith("state_staging_value_")]
        _require(len(gradients) == len(updates) == len(stagings) == 1 and
                 updates[0]["dtype"] == stagings[0]["dtype"] == "fp16",
                 f"{label}: SGD lacks exact weight/FP32 gradient/staging")
        gradient_id = gradients[0]["value_id"]
        producers = [item for item in source_values.get(gradient_id, ())
                     if {r["opcode"] for r in _fragment_records(item)} &
                        {34, 35, 36, 37}]
        _require(len(producers) == 1,
                 f"{label}: SGD gradient has no unique physical WGRAD producer")
        staging = stagings[0]["value_id"]
        loads, stores = [], []
        for candidate in fragments:
            if not any(b["value_id"] == staging for b in
                       candidate["buffer_abi"]):
                continue
            for opcode, abi in state_binding_by_fragment.get(candidate["id"], ()):
                if opcode == 128:
                    loads.append(abi)
                else:
                    stores.append(abi)
        _require(len(loads) == len(stores) == 1 and
                 loads[0]["id"] == stores[0]["id"],
                 f"{label}: SGD staging lacks unique LOAD/STORE StateABI")
        state = stores[0]
        _require(state["state_ref"] not in coverage,
                 f"{label}: duplicate parameter SGD/STORE")
        coverage[state["state_ref"]] = (
            gradient_id, updates[0]["value_id"], state["id"],
            state["address"], state["size_bytes"],
        )
    _require(sorted(coverage) == expected_refs,
             f"{label}: WGRAD→SGD→STORE missed a trainable state")
    return coverage


def _sequence(content: str, fresh: int) -> None:
    states = re.findall(
        r"\[MOE_ALL_SGD_PARTIAL_STATE\] version=([012]) bytes=952 "
        r"digest=([0-9a-f]{64}) content_changed=0 functional=0 "
        r"full_training=0 pass=1", content)
    steps = re.findall(
        r"\[MOE_ALL_SGD_PARTIAL_SEQUENCE_STEP\] index=([01]) "
        r"input_version=[01] output_version=[12] trainable_states=19 "
        r"route_states=2 records=621 sgd=19 store=19 "
        r"state_digest_before=([0-9a-f]{64}) "
        r"state_digest_after=([0-9a-f]{64}) "
        r"full_training=0 functional=0 pass=1", content)
    input_digest = re.findall(
        r"\[MOE_ALL_SGD_PARTIAL_INPUT\] index=1 "
        r"prior_store_completed=1 same_hbm_state=1 "
        r"digest=([0-9a-f]{64}) pass=1", content)
    _require(tuple(version for version, _ in states) == ("0", "1", "2")
             and tuple(step for step, _, _ in steps) == ("0", "1")
             and len(input_digest) == 1
             and steps[0][1:] == (states[0][1], states[1][1])
             and steps[1][1:] == (states[1][1], states[2][1])
             and input_digest[0] == states[1][1]
             and content.count("[TRAIN_SGD]") == 38
             and all(content.count(f"[DENSE_SEQUENCE_PROGRAM_IO] index={step} "
                                   "probes=1 pass=1") == 1 for step in (0, 1))
             and content.count("[DENSE_SEQUENCE_DRAIN] segments=2 one_shot=1") == 1
             and "lsu_hbm_read_bytes=5184 lsu_hbm_write_bytes=1904" in content
             and "[CREDIT] data_balanced=1 ctrl_balanced=1" in content
             and "[DRAIN] d2d_link_residual=0" in content,
             f"fresh{fresh}: shared-HBM two-step version/drain drift")


def audit(freeze: Path, tools: Path, roots: tuple[Path, Path]) -> dict:
    freeze, tools = freeze.resolve(), tools.resolve()
    roots = tuple(path.resolve() for path in roots)
    _require(len(set(roots)) == 2, "Fresh roots are not independent")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=freeze, text=True).strip()
    _require(not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=freeze, text=True).strip(), "source freeze is dirty")
    hashes = {name: _sha(tools / executable)
              for name, executable in _TOOLS.items()}
    bindings, linked_ids, files = [], [], []
    for fresh, root in enumerate(roots):
        _require({path.name for path in root.iterdir() if path.is_file()} == _FILES,
                 f"fresh{fresh}: exact 18 artifact set drift")
        receipt = json.loads((root / "receipt.json").read_text())
        _require(receipt["status"] == "all_parameter_sgd_physical_partial" and
                 receipt["full_training_gate"] == "closed" and
                 receipt["source_commit"] == commit and
                 receipt["source_tree_clean_at_entry"] and
                 receipt["runner_cwd"] == str(freeze) and
                 receipt["native_runtime_cwd"] == str(tools) and
                 receipt["native_tool_sha256"] == hashes and
                 len(receipt["steps"]) == 2,
                 f"fresh{fresh}: source/tool/runtime binding drift")
        for label in ("finalizer", "resolver", "npusim"):
            _require(receipt[f"{label}_sha256"] == hashes[label],
                     f"fresh{fresh}: {label} changed")
        source_maps = tuple(receipt[key] for key in (
            "source_file_sha256", "imported_python_sha256_at_entry",
            "imported_python_sha256_at_exit"))
        for index, mapping in enumerate(source_maps):
            _source_map(freeze, mapping, f"fresh{fresh} source map {index}")
        _require(all(source_maps[2].get(name) == value
                     for name, value in source_maps[1].items()),
                 f"fresh{fresh}: imported module changed during run")
        _require(_sha(root / "hardware.json") == receipt["hardware_sha256"] and
                 _sha(freeze / "llm/test/program/p5_behavioral_simulation.json") ==
                 receipt["simulation_sha256"] and
                 (root / "mapping.spec").read_text() == "0:0\n",
                 f"fresh{fresh}: hardware/simulation/mapping drift")
        sequence = root / "all_sgd_partial_sequence.npusim.log"
        _require(_sha(sequence) == receipt["all_sgd_partial_sequence_log_sha256"] and
                 receipt["router_sgd_partial_sequence_log_sha256"] is None and
                 receipt["layer1_sgd_partial_sequence_log_sha256"] is None,
                 f"fresh{fresh}: wrong sequence receipt")
        _sequence(sequence.read_text(), fresh)
        coverage_by_step, ids = [], []
        for step, witness in enumerate(receipt["steps"]):
            prefix = f"step{step}"
            manifest_path = root / f"{prefix}.linked.json"
            artifact = root / f"{prefix}.npup"
            io_path = root / f"{prefix}.program_io.json"
            native = root / f"{prefix}.npusim.log"
            _require(witness["step"] == step and witness["leaves"] == 160 and
                     witness["records"] == 621 and witness["sgd_records"] == 19 and
                     witness["all_sgd_write_bytes"] == 952 and
                     witness["numeric_gradient_witness"] is False and
                     _sha(artifact) == witness["artifact_sha256"] and
                     _sha(io_path) == witness["program_io_sha256"] and
                     _sha(native) == witness["npusim_log_sha256"],
                     f"fresh{fresh} step{step}: executable receipt drift")
            manifest = json.loads(manifest_path.read_text())
            _require(manifest["id"] == witness["linked"],
                     f"fresh{fresh} step{step}: linked ID drift")
            coverage = _physical_coverage(manifest,
                                          witness["all_sgd_state_refs"],
                                          step, fresh)
            coverage_by_step.append(coverage)
            ids.append(manifest["id"])
            io = json.loads(io_path.read_text())
            _require(len(io["initializations"]) == 171 and
                     len(io["output_probes"]) == 1 and
                     io["output_probes"][0]["target"]["value_id"] == "T0.loss"
                     and "ProgramIo resolved id=" in
                     (root / f"{prefix}.resolver.log").read_text() and
                     (root / f"{prefix}.finalizer.json").is_file() and
                     (root / f"{prefix}.finalizer.log").is_file(),
                     f"fresh{fresh} step{step}: resolver/finalizer drift")
            content = native.read_text()
            _require(content.count("[TRAIN_SGD]") == 19 and
                     content.count("[PROGRAM_IO] phase=verify") == 1 and
                     content.count("[PROGRAM_MEMORY] core=0") == 1 and
                     "lsu_hbm_read_bytes=2592 lsu_hbm_write_bytes=952" in content
                     and "[CREDIT] data_balanced=1 ctrl_balanced=1" in content
                     and "[DRAIN] d2d_link_residual=0" in content,
                     f"fresh{fresh} step{step}: native execution/drain drift")
        _require(ids[0] != ids[1] and
                 coverage_by_step[0] == coverage_by_step[1],
                 f"fresh{fresh}: step0 STORE→step1 LOAD state mapping drift")
        bindings.append(source_maps)
        linked_ids.append(ids)
        files.append({path.name: _sha(path) for path in root.iterdir()
                      if path.is_file()})
    _require(bindings[0] == bindings[1] and linked_ids[0] == linked_ids[1] and
             files[0].keys() == files[1].keys(),
             "Fresh source/import/linked IDs or artifact names diverged")
    return {"status": "pass_timing_partial_only", "source_commit": commit,
            "full_training_gate": "closed", "numeric_gradient_witness": False,
            "fresh_roots": [str(root) for root in roots],
            "native_tool_sha256": hashes, "file_count_per_fresh": [18, 18],
            "source_file_count": len(bindings[0][0]),
            "imported_python_at_entry_count": len(bindings[0][1]),
            "imported_python_at_exit_count": len(bindings[0][2]),
            "native_invocations": 5, "native_executions": 6,
            "states_with_wgrad_sgd_store_next_load": 19,
            "records_per_step": 621, "state_bytes_per_step": 952,
            "linked_ids_by_step": linked_ids[0],
            "file_sha256_by_fresh": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, nargs=2, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.freeze, args.tools, tuple(args.fresh))
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key != "file_sha256_by_fresh"}, sort_keys=True))


if __name__ == "__main__":
    main()
