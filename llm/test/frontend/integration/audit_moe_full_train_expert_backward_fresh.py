"""Read-only exact reopen audit for two EP1 expert-reverse partial Fresh runs."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


TOOLS = {
    "finalizer": "npusim_program_finalizer",
    "resolver": "npusim_program_io_selftest",
    "npusim": "npusim",
}
# Frozen public record-ISA numeric tags; the auditor stays independent of
# Python modules in the current mutable worktree.
EXPECTED_OPCODES = {
    31: 1,  # CROSS_ENTROPY_BACKWARD
    37: 5,  # GEMM_WEIGHT_WGRAD_TIMING
    38: 4,  # GEMM_DX_TIMING; router dX adds one
    34: 1,  # SWIGLU_BACKWARD_TIMING
    14: 4,  # RESIDUAL; MoE dX merge/backbone each add one
    36: 1,  # NORM_GAMMA_WGRAD_TIMING; backbone adds one
    41: 1,  # RMSNORM_BACKWARD_TIMING; backbone adds one
    67: 1,  # LOCAL_REDUCE
    40: 1,  # MOE_SCORE_WEIGHT_BACKWARD
    32: 0,  # SGD_UPDATE; layer1 parameter profile adds four
    129: 0, # LSU_STORE; layer1 parameter profile adds four
}
PROFILES = {
    "expert_backward": dict(status="expert_backward_physical_partial",
                            leaves=61, records=293, dx=4, hbm_read=1240,
                            initializations=98, probes=9, residual=4, norm=1,
                            sgd=0, hbm_write=0,
                            expert_probes=4, router_probes=0, merged_probes=0,
                            backbone_probes=0),
    "router_dx": dict(status="router_dx_physical_partial",
                      leaves=63, records=300, dx=5, hbm_read=1248,
                      initializations=100, probes=10, residual=4, norm=1,
                      sgd=0, hbm_write=0,
                      expert_probes=4, router_probes=1, merged_probes=0,
                      backbone_probes=0),
    "input_gradient": dict(status="input_gradient_physical_partial",
                           leaves=64, records=304, dx=5, hbm_read=1248,
                           initializations=101, probes=9, residual=5, norm=1,
                           sgd=0, hbm_write=0,
                           expert_probes=3, router_probes=0, merged_probes=1,
                           backbone_probes=0),
    "layer1_backbone": dict(status="layer1_backbone_physical_partial",
                            leaves=67, records=316, dx=5, hbm_read=1248,
                            initializations=104, probes=9, residual=6, norm=2,
                            sgd=0, hbm_write=0,
                            expert_probes=3, router_probes=0, merged_probes=0,
                            backbone_probes=2),
    "layer1_parameter_sgd": dict(
        status="layer1_parameter_sgd_physical_partial",
        leaves=79, records=340, dx=5, hbm_read=1448,
        initializations=108, probes=5, residual=6, norm=2,
        sgd=4, hbm_write=200,
        expert_probes=0, router_probes=0, merged_probes=0,
        backbone_probes=2,
    ),
}
PROFILES["layer1_sgd_sequence"] = dict(
    PROFILES["layer1_parameter_sgd"], sequence=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit(freeze: Path, roots: tuple[Path, Path], tools: Path,
          profile_name: str = "expert_backward") -> dict:
    profile = PROFILES[profile_name]
    freeze, tools = freeze.resolve(), tools.resolve()
    roots = tuple(path.resolve() for path in roots)
    require(len(set(roots)) == 2, "Fresh directories must be distinct")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=freeze, text=True).strip()
    require(not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=freeze, text=True).strip(), "source freeze is no longer clean")
    frozen_tools = {key: digest(tools / filename)
                    for key, filename in TOOLS.items()}
    bindings = []
    case_files = []
    linked_ids = []
    for fresh_index, root in enumerate(roots):
        receipt_path = root / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        require(receipt["source_commit"] == commit and
                receipt["source_tree_clean_at_entry"],
                f"fresh{fresh_index} source commit/clean binding changed")
        require(receipt["runner_cwd"] == str(freeze) and
                receipt["native_runtime_cwd"] == str(tools),
                f"fresh{fresh_index} runtime cwd changed")
        require(receipt["native_tool_sha256"] == frozen_tools and
                receipt["finalizer_sha256"] == frozen_tools["finalizer"] and
                receipt["resolver_sha256"] == frozen_tools["resolver"] and
                receipt["npusim_sha256"] == frozen_tools["npusim"],
                f"fresh{fresh_index} native tool bytes changed")
        source_maps = (
            receipt["source_file_sha256"],
            receipt["imported_python_sha256_at_entry"],
            receipt["imported_python_sha256_at_exit"],
        )
        for source_map in source_maps:
            require(source_map and all(
                digest(freeze / name) == expected
                for name, expected in source_map.items()),
                f"fresh{fresh_index} imported or explicit source bytes drifted")
        require(all(receipt["imported_python_sha256_at_exit"].get(name) == expected
                    for name, expected in receipt["imported_python_sha256_at_entry"].items()),
                f"fresh{fresh_index} Python source changed during run")
        require(receipt["status"] == profile["status"] and
                receipt["full_training_gate"] == "closed" and
                len(receipt["steps"]) == 2 and
                receipt["router_sgd_partial_sequence_log_sha256"] is None,
                f"fresh{fresh_index} stage status drifted")
        require(digest(root / "hardware.json") == receipt["hardware_sha256"] and
                digest(freeze / "llm/test/program/p5_behavioral_simulation.json") ==
                receipt["simulation_sha256"],
                f"fresh{fresh_index} native config changed")
        ids = []
        for step, witness in enumerate(receipt["steps"]):
            prefix = f"step{step}"
            linked = root / f"{prefix}.linked.json"
            artifact = root / f"{prefix}.npup"
            io_path = root / f"{prefix}.program_io.json"
            native_log = root / f"{prefix}.npusim.log"
            require(witness["step"] == step and
                    digest(artifact) == witness["artifact_sha256"] and
                    digest(io_path) == witness["program_io_sha256"] and
                    digest(native_log) == witness["npusim_log_sha256"],
                    f"fresh{fresh_index} step{step} artifact receipt drifted")
            manifest = json.loads(linked.read_text())
            records = [record for fragment in manifest["fragments"]
                       for stream in fragment["core_streams"]
                       for record in stream["records"]]
            opcodes = Counter(record["opcode"] for record in records)
            require(manifest["id"] == witness["linked"] and
                    len(records) == witness["records"] == profile["records"] and
                    all(opcodes[name] == (profile["dx"] if name == 38 else
                                          profile["residual"] if name == 14 else
                                          profile["norm"] if name in (36, 41) else
                                          profile["sgd"] if name in (32, 129) else
                                          count)
                        for name, count in EXPECTED_OPCODES.items()) and
                    witness["leaves"] == profile["leaves"],
                    f"fresh{fresh_index} step{step} linked opcode closure drifted")
            fragments = {item["id"]: item for item in manifest["fragments"]}
            states = {abi["id"]: abi for fragment in manifest["fragments"]
                      for abi in fragment["state_abi"]}
            stores = []
            for binding in manifest["state_operand_bindings"]:
                fragment = fragments[binding["fragment_id"]]
                stream = next(item for item in fragment["core_streams"]
                              if item["logical_core"] == binding["logical_core"])
                record = stream["records"][binding["fragment_record_index"]]
                if record["opcode"] == 129:
                    require(binding["operand_id"] == 6 and
                            binding["state_abi_id"] in states,
                            f"fresh{fresh_index} step{step} STORE state binding drifted")
                    stores.append(states[binding["state_abi_id"]])
            require(len(stores) == profile["sgd"] and
                    len({abi["state_ref"] for abi in stores}) == profile["sgd"] and
                    all(abi["access"] == "read_write" and
                        abi["kind"] == "trainable_parameter" and
                        abi["dtype"] == "fp16" and abi["die_id"] == 0
                        for abi in stores) and
                    sum(abi["size_bytes"] for abi in stores) ==
                    profile["hbm_write"] and
                    sorted(abi["state_ref"] for abi in stores) ==
                    witness.get("layer1_sgd_state_refs", []) and
                    witness.get("layer1_sgd_write_bytes", 0) ==
                    profile["hbm_write"],
                    f"fresh{fresh_index} step{step} physical SGD state closure drifted")
            io = json.loads(io_path.read_text())
            blobs = {item["id"]: base64.b64decode(item["bytes_base64"])
                     for item in io["blobs"]}
            expert = [item for item in io["output_probes"]
                      if item["target"]["value_id"].startswith(
                          "backward::T0.layer1.moe.expert0.")]
            nonzero = sum(byte != 0 for item in expert
                          for byte in blobs[item["blob_ref"]])
            router = [item for item in io["output_probes"]
                      if item["target"]["value_id"] ==
                      "backward::T0.layer1.moe.router.input.gradient"]
            merged = [item for item in io["output_probes"]
                      if item["target"]["value_id"] ==
                      "backward::T0.layer1.moe.input_sum.norm2_gradient"]
            backbone = [item for item in io["output_probes"]
                        if item["target"]["value_id"] ==
                           "backward::T0.layer1.residual1.merge.input_gradient"
                        or item["target"]["value_id"].startswith(
                            "backward::T0.layer1.norm2::")]
            require(len(expert) == witness["expert_gradient_probes"] ==
                    profile["expert_probes"] and
                    len(router) == profile["router_probes"] and
                    len(merged) == profile["merged_probes"] and
                    len(backbone) == profile["backbone_probes"] and
                    all(not any(blobs[item["blob_ref"]])
                            for item in (*router, *merged, *backbone)) and
                    len(io["initializations"]) == profile["initializations"] and
                    len(io["output_probes"]) == profile["probes"] and
                    nonzero == witness["expert_gradient_nonzero_expected_bytes"] == 0 and
                    witness["numeric_gradient_witness"] is False and
                    witness["sgd_records"] == profile["sgd"] and
                    witness["gate_hbm_write_bytes"] == 0,
                    f"fresh{fresh_index} step{step} timing-only gradient gate changed")
            log = native_log.read_text()
            require(log.count("[PROGRAM_IO] phase=verify") == 1 and
                    log.count("[PROGRAM_MEMORY] core=0") == 1 and
                    f"lsu_hbm_read_bytes={profile['hbm_read']} lsu_hbm_write_bytes={profile['hbm_write']}" in log and
                    log.count("[TRAIN_SGD]") == profile["sgd"] and
                    "[CREDIT] data_balanced=1 ctrl_balanced=1" in log and
                    "[DRAIN] d2d_link_residual=0" in log,
                    f"fresh{fresh_index} step{step} native drain/IO failed")
            require((root / f"{prefix}.finalizer.json").is_file() and
                    (root / f"{prefix}.finalizer.log").is_file() and
                    "ProgramIo resolved id=" in
                    (root / f"{prefix}.resolver.log").read_text(),
                    f"fresh{fresh_index} step{step} finalizer/resolver missing")
            ids.append(manifest["id"])
        if profile.get("sequence", False):
            sequence_log = root / "layer1_sgd_partial_sequence.npusim.log"
            require(sequence_log.is_file() and
                    digest(sequence_log) ==
                    receipt.get("layer1_sgd_partial_sequence_log_sha256"),
                    f"fresh{fresh_index} partial two-step sequence log drifted")
            content = sequence_log.read_text()
            states = re.findall(
                r"\[MOE_LAYER1_SGD_PARTIAL_STATE\] version=([012]) "
                r"bytes=952 digest=([0-9a-f]{64}) content_changed=0 "
                r"functional=0 full_training=0 pass=1", content)
            steps = re.findall(
                r"\[MOE_LAYER1_SGD_PARTIAL_SEQUENCE_STEP\] "
                r"index=([01]) input_version=[01] output_version=[12] "
                r"trainable_states=19 route_states=2 records=340 "
                r"sgd=4 store=4 state_digest_before=([0-9a-f]{64}) "
                r"state_digest_after=([0-9a-f]{64}) "
                r"full_training=0 functional=0 pass=1", content)
            input_digest = re.findall(
                r"\[MOE_LAYER1_SGD_PARTIAL_INPUT\] index=1 "
                r"prior_store_completed=1 same_hbm_state=1 "
                r"digest=([0-9a-f]{64}) pass=1", content)
            require(tuple(version for version, _digest in states) ==
                    ("0", "1", "2") and
                    tuple(index for index, _before, _after in steps) ==
                    ("0", "1") and len(input_digest) == 1 and
                    steps[0][1:] == (states[0][1], states[1][1]) and
                    steps[1][1:] == (states[1][1], states[2][1]) and
                    input_digest[0] == states[1][1] and
                    content.count("[TRAIN_SGD]") == 8 and
                    content.count("[DENSE_SEQUENCE_PROGRAM_IO] index=0 probes=5 pass=1") == 1 and
                    content.count("[DENSE_SEQUENCE_PROGRAM_IO] index=1 probes=5 pass=1") == 1 and
                    content.count("[DENSE_SEQUENCE_DRAIN] segments=2 one_shot=1") == 1 and
                    "lsu_hbm_read_bytes=2896 lsu_hbm_write_bytes=400" in content and
                    "[CREDIT] data_balanced=1 ctrl_balanced=1" in content and
                    "[DRAIN] d2d_link_residual=0" in content,
                    f"fresh{fresh_index} partial two-step HBM version chain drifted")
        else:
            require(receipt.get("layer1_sgd_partial_sequence_log_sha256") is None,
                    f"fresh{fresh_index} unexpected partial sequence claim")
        bindings.append(tuple(source_maps))
        linked_ids.append(ids)
        case_files.append({path.name: digest(path) for path in sorted(root.iterdir())
                           if path.is_file()})
        require(len(case_files[-1]) == (18 if profile.get("sequence", False) else 17),
                f"fresh{fresh_index} artifact set size drifted")
    require(bindings[0] == bindings[1] and linked_ids[0] == linked_ids[1],
            "independent Fresh source/import/linked IDs diverged")
    require(set(case_files[0]) == set(case_files[1]),
            "independent Fresh artifact sets differ")
    return {
        "status": "pass_partial_only", "profile": profile_name,
        "full_training_gate": "closed",
        "numeric_gradient_witness": False,
        "source_commit": commit, "native_tool_sha256": frozen_tools,
        "source_file_count": len(bindings[0][0]),
        "imported_python_at_entry_count": len(bindings[0][1]),
        "imported_python_at_exit_count": len(bindings[0][2]),
        "fresh_roots": [str(path) for path in roots],
        "native_runtime_cwd": str(tools),
        "native_invocations": 5 if profile.get("sequence", False) else 4,
        "native_executions": 6 if profile.get("sequence", False) else 4,
        "linked_ids_by_step": linked_ids[0],
        "file_count_per_fresh": [len(files) for files in case_files],
        "file_sha256_by_fresh": case_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True, nargs=2)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES),
                        default="expert_backward")
    args = parser.parse_args()
    report = audit(args.freeze, tuple(args.fresh), args.tools, args.profile)
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key != "file_sha256_by_fresh"}, sort_keys=True))


if __name__ == "__main__":
    main()
