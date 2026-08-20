"""Self-contained Stage 4 prefill/decode integration cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path

from llm.frontend.wafer_frontend.passes import (
    build_deterministic_timing_state_overrides,
    build_stage4_fused_ir0,
    build_stage4_global_action,
    build_stage4_pd_plan,
    build_stage4_separated_ir0,
    build_timing_program_io,
    hbm_address_spaces_from_data,
    link_stage4,
    lower_stage4,
    partition_stage4,
    physical_fabric_from_data,
    place_stage4_carrier,
    plan_stage4,
    project_stage4,
    schedule_stage4,
    validate_identity_mapping_text,
)
from llm.frontend.wafer_frontend.policies.registry import production_registry
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest,
)
from llm.frontend.wafer_frontend.schema.common import ProfileKey
from llm.frontend.wafer_frontend.schema.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
)
from llm.frontend.wafer_frontend.schema.ir0 import IR0
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanningContext,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    IntraDieSchedulingContext,
    Stage4GlobalAction,
    Stage4ProjectToIR2Context,
)
from llm.frontend.wafer_frontend.schema.n6 import (
    Stage4LinkedProgram,
    Stage4LoweredProgram,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.policy import RegistryKind
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.serde import from_data
from llm.frontend.wafer_frontend.schema.s1_naive_evidence import (
    S1NaivePolicyEvidence,
)
from llm.frontend.wafer_frontend.schema.stage3_profile import (
    KvPageSpan,
    Stage3StaticProfile,
    StaticRequestShape,
)
from llm.frontend.wafer_frontend.schema.stage4_pd import Stage4PdPlan


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "llm/test/sram/hardware_numa.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_SRAM_REGION_BYTES = 16 * 1024 * 1024
_PDR_SRAM_HIGH_WATER_BYTES = {
    "double_a": 14656,
    "comm": 3584,
}
_PDR_MIN_REGION_BYTES = 64


class Stage4PdCaseKind(str, Enum):
    FUSED = "fused"
    PDS = "pds"
    PDR = "pdr"


@dataclass(frozen=True, slots=True)
class Stage4RuntimeHardwareInputs:
    source_hardware_path: Path
    source_mapping_path: Path
    hardware_json: str
    mapping_text: str


@dataclass(frozen=True, slots=True)
class Stage4PdCase:
    kind: Stage4PdCaseKind
    spec: ExperimentSpec
    pd_plan: Stage4PdPlan
    logical_graph: IR0
    graph: IR1
    global_carrier: Stage4GlobalAction
    lowered: Stage4LoweredProgram
    profile: Stage4LinkedProgram
    manifest: LinkedProgramManifest
    program_io: ProgramIoContract
    planning_context: InterDiePlanningContext
    scheduling_context: IntraDieSchedulingContext
    policy: S1NaivePolicyEvidence
    runtime_hardware_inputs: Stage4RuntimeHardwareInputs
    state_seed_refs: tuple[str, ...]
    state_expected_refs: tuple[str, ...]


def _profile(*, prefill: bool) -> Stage3StaticProfile:
    request = StaticRequestShape(
        request_ref="request_0",
        prefill_tokens=8 if prefill else 0,
        decode_tokens=0 if prefill else 1,
        context_tokens=8 if prefill else 9,
        kv_span=KvPageSpan(
            page_start=0,
            page_count=1,
            page_size_tokens=16,
        ),
    )
    return Stage3StaticProfile.create(
        key=ProfileKey(
            prefill_tokens=request.prefill_tokens,
            decode_tokens=request.decode_tokens,
            num_seqs=1,
            context_sum=request.context_tokens,
            context_max=request.context_tokens,
            kv_pages=1,
            expert_load=None,
        ),
        requests=(request,),
    )


def _profile_data(profile: Stage3StaticProfile) -> dict[str, object]:
    return {
        "prefill_tokens": profile.key.prefill_tokens,
        "decode_tokens": profile.key.decode_tokens,
        "num_seqs": profile.key.num_seqs,
        "context_sum": profile.key.context_sum,
        "context_max": profile.key.context_max,
        "kv_pages": profile.key.kv_pages,
        "expert_load": None,
    }


def _spec(kind: Stage4PdCaseKind) -> ExperimentSpec:
    prefill = _profile(prefill=True)
    decode = _profile(prefill=False)
    if kind is Stage4PdCaseKind.FUSED:
        instances = (
            {
                "id": "F0",
                "role": "both",
                "tp": 1,
                "sp": False,
                "replicas": 1,
                "dp": 1,
                "pp": 1,
                "ep": 1,
            },
        )
        prefill_instance_ref = decode_instance_ref = "F0"
        placement = {"strategy": "compact", "groups": []}
    else:
        prefill_tp = 2 if kind is Stage4PdCaseKind.PDR else 1
        decode_die = 2 if kind is Stage4PdCaseKind.PDR else 1
        instances = (
            {
                "id": "P0",
                "role": "prefill",
                "tp": prefill_tp,
                "sp": prefill_tp > 1,
                "replicas": 1,
                "dp": 1,
                "pp": 1,
                "ep": 1,
            },
            {
                "id": "D0",
                "role": "decode",
                "tp": 1,
                "sp": False,
                "replicas": 1,
                "dp": 1,
                "pp": 1,
                "ep": 1,
            },
        )
        prefill_instance_ref = "P0"
        decode_instance_ref = "D0"
        placement = {
            "strategy": "explicit",
            "groups": [
                {
                    "instance_id": "D0",
                    "mesh_ref": "D0.mesh.tp",
                    "die_ids": [decode_die],
                },
                {
                    "instance_id": "P0",
                    "mesh_ref": "P0.mesh.tp",
                    "die_ids": list(range(prefill_tp)),
                },
            ],
        }
    raw = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "model": {
            "source": "analytic",
            "arch": "llama",
            "V": 32,
            "H": 16,
            "I": 32,
            "NH": 4,
            "KVH": 4,
            "DH": 4,
            "rotary_dim": 4,
            "L": 2,
            "dtype": "fp16",
            "tie_word_embeddings": False,
            "rms_norm_epsilon": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128,
            "moe": None,
        },
        "hardware": {"ref": "llm/test/sram/hardware_numa.json"},
        "workload": {
            "mode": "infer",
            "infer": {
                "source": "pd_static",
                "output": "logits",
                "pd_static": {
                    "prefill_profile": _profile_data(prefill),
                    "decode_profile": _profile_data(decode),
                    "prefill_instance_ref": prefill_instance_ref,
                    "decode_instance_ref": decode_instance_ref,
                },
            },
        },
        "parallel": {"instances": list(instances)},
        "placement": placement,
        "policy": {
            "partition": "gemm_coll",
            "inter_die": "naive",
            "intra_die": "naive",
        },
        "backend": {
            "execution": "unified_stream",
            "ordinary_lowering": "json_coarse",
            "fused_lowering": "isa_region",
            "standalone_collective_lowering": "strict_actions",
            "reduction_contract": {
                "accumulate": "fp32",
                "rounding": "rne",
                "validation": "timing",
            },
            "transport": "strict",
            "static_link": True,
            "dynamic_region_dispatch": False,
        },
    }
    return from_data(ExperimentSpec, raw, path="stage4_pd.spec")


def _runtime_inputs(kind: Stage4PdCaseKind) -> Stage4RuntimeHardwareInputs:
    raw = json.loads(_HARDWARE.read_text(encoding="utf-8"))
    if kind is Stage4PdCaseKind.PDR:
        raw["die"]["x"] = 3
        stack = dict(raw["memory_system"]["hbm_stacks"][-1])
        stack.update(
            {
                "stack_id": 2,
                "compute_die_id": 2,
            }
        )
        raw["memory_system"]["hbm_stacks"].append(stack)
        home = dict(raw["memory_system"]["address_policy"]["home_ranges"][-1])
        home.update(
            {
                "die_id": 2,
                "base": 2 * home["size_bytes"],
            }
        )
        raw["memory_system"]["address_policy"]["home_ranges"].append(home)
    regions = raw["memory"]["sram"]["regions"]
    if kind is Stage4PdCaseKind.PDR:
        cursor = 0
        for region in regions:
            size_bytes = _PDR_SRAM_HIGH_WATER_BYTES.get(
                region["name"], _PDR_MIN_REGION_BYTES
            )
            region["base_bytes"] = cursor
            region["size_bytes"] = size_bytes
            region["allocator"] = "block"
            cursor += size_bytes
        if cursor > 65536:
            raise RuntimeError("PDR SRAM layout exceeds uint16 wire address")
        raw["memory"]["sram_size"] = cursor
        raw["memory"]["sram"]["capacity_bytes"] = cursor
    else:
        raw["memory"]["sram_size"] = len(regions) * _SRAM_REGION_BYTES
        raw["memory"]["sram"]["capacity_bytes"] = (
            len(regions) * _SRAM_REGION_BYTES
        )
        for index, region in enumerate(regions):
            region["base_bytes"] = index * _SRAM_REGION_BYTES
            region["size_bytes"] = _SRAM_REGION_BYTES
            region["allocator"] = "block"
    hardware_json = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    mapping_text = _MAPPING.read_text(encoding="utf-8")
    fabric = physical_fabric_from_data(
        json.loads(hardware_json),
        path="stage4_pd.hardware",
    )
    validate_identity_mapping_text(
        mapping_text,
        total_cores=sum(len(die.cores) for die in fabric.dies),
        path="stage4_pd.mapping",
    )
    return Stage4RuntimeHardwareInputs(
        source_hardware_path=_HARDWARE,
        source_mapping_path=_MAPPING,
        hardware_json=hardware_json,
        mapping_text=mapping_text,
    )


def build_stage4_pd_case(kind: Stage4PdCaseKind) -> Stage4PdCase:
    """Build one formal Stage 4 PD-F/PDS/PDR case through ProgramIo."""

    if type(kind) is not Stage4PdCaseKind:
        raise TypeError("kind must be a Stage4PdCaseKind")
    spec = _spec(kind)
    prefill = _profile(prefill=True)
    decode = _profile(prefill=False)
    pd_plan = build_stage4_pd_plan(
        spec,
        prefill_profile=prefill,
        decode_profile=decode,
    )
    logical_graph = (
        build_stage4_fused_ir0(spec, pd_plan)
        if kind is Stage4PdCaseKind.FUSED
        else build_stage4_separated_ir0(spec, pd_plan)
    )
    runtime_inputs = _runtime_inputs(kind)
    hardware = json.loads(runtime_inputs.hardware_json)
    placement_context = PlacementContext.create(
        producer_pass="stage4_pd_cases",
        fabric=physical_fabric_from_data(
            hardware,
            path="stage4_pd.hardware",
        ),
        placement=spec.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(
            hardware,
            path="stage4_pd.hardware",
        ),
    )
    placed = place_stage4_carrier(
        logical_graph,
        placement_context,
        pd_plan,
    )
    partition_context = FusionPartitionContext.create(
        producer_pass="stage4_pd_cases"
    )
    partitioned = partition_stage4(placed, partition_context)
    registry = production_registry()
    inter_die_policy = registry.instantiate(
        RegistryKind.INTER_DIE,
        "naive",
    ).selection
    intra_die_policy = registry.instantiate(
        RegistryKind.INTRA_DIE,
        "naive",
    ).selection
    standalone_policy = registry.instantiate(
        RegistryKind.STANDALONE_COLLECTIVE,
        "direct_all_gather",
    ).selection
    planning_context = InterDiePlanningContext.create(
        producer_pass="stage4_pd_cases",
        fused_policy=inter_die_policy,
        standalone_policy=standalone_policy,
    )
    planned = plan_stage4(partitioned, planning_context)
    projection_context = Stage4ProjectToIR2Context.create(
        producer_pass="stage4_pd_cases"
    )
    projected = project_stage4(planned, projection_context)
    scheduling_context = IntraDieSchedulingContext.create(
        producer_pass="stage4_pd_cases",
        policy=intra_die_policy,
    )
    scheduled = schedule_stage4(projected, scheduling_context)
    policy = S1NaivePolicyEvidence(
        selections=(
            inter_die_policy,
            standalone_policy,
            intra_die_policy,
        ),
        planning_context_id=planning_context.id,
        scheduling_context_id=scheduling_context.id,
    )
    policy.validate_against(planning_context, scheduling_context)
    global_carrier = build_stage4_global_action(scheduled)
    lowered = lower_stage4(global_carrier)
    profile = link_stage4(lowered)
    state_seeds, state_expected = (
        build_deterministic_timing_state_overrides(profile)
    )
    topology = (
        "tp2-to-tp1" if kind is Stage4PdCaseKind.PDR else "tp1"
    )
    placeholder_sha256 = sha256(
        f"stage4-pd:{kind.value}:{topology}".encode("ascii")
    ).hexdigest()
    program_io = build_timing_program_io(
        profile,
        placeholder_sha256,
        state_seed_overrides=state_seeds,
        state_expected_overrides=state_expected,
    )
    program_io.validate_against(profile.manifest)
    return Stage4PdCase(
        kind=kind,
        spec=spec,
        pd_plan=pd_plan,
        logical_graph=logical_graph,
        graph=profile.lowering_context.ir1,
        global_carrier=global_carrier,
        lowered=lowered,
        profile=profile,
        manifest=profile.manifest,
        program_io=program_io,
        planning_context=planning_context,
        scheduling_context=scheduling_context,
        policy=policy,
        runtime_hardware_inputs=runtime_inputs,
        state_seed_refs=tuple(sorted(state_seeds)),
        state_expected_refs=tuple(sorted(state_expected)),
    )


__all__ = [
    "Stage4PdCase",
    "Stage4PdCaseKind",
    "Stage4RuntimeHardwareInputs",
    "build_stage4_pd_case",
]
