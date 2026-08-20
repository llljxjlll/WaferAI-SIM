"""Self-contained Stage2 Dense-forward integration cases.

The builder starts from a typed ExperimentSpec and exercises only production
passes through manifest linking and ProgramIo.  It deliberately does not
import unit-test fixtures or construct backend fragments by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from llm.frontend.wafer_frontend.passes import (
    build_deterministic_timing_state_overrides,
    build_global_bundle,
    build_ir0,
    build_stage2_dense_forward_oracle,
    build_timing_program_io,
    hbm_address_spaces_from_data,
    link_bundle,
    logical_expand,
    lower_bundle,
    partition_bundle,
    physical_fabric_from_data,
    place_bundle,
    plan_bundle,
    project_bundle,
    schedule_bundle,
    validate_identity_mapping_text,
)
from llm.frontend.wafer_frontend.policies.registry import production_registry
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
    InferOutput,
)
from llm.frontend.wafer_frontend.schema.global_action import GlobalActionDAG
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.logical import IR0Template
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanningContext,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    GlobalActionProfile,
    IntraDieSchedulingContext,
    ProjectToIR2Context,
)
from llm.frontend.wafer_frontend.schema.n6 import (
    LinkedProgramBundle,
    LinkedProgramProfile,
    LoweredProgramBundle,
    LoweredProgramProfile,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.policy import RegistryKind
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.serde import from_data
from llm.frontend.wafer_frontend.schema.stage2_dense_forward_oracle import (
    Stage2DenseForwardOracle,
)


_ROOT = Path(__file__).resolve().parents[4]
_TP12_HARDWARE = _ROOT / "llm/test/sram/hardware_numa.json"
_TP4_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x2.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_SRAM_CAPACITY_BYTES = 64 * 1024
_SRAM_REGIONS = (
    (
        "double_a",
        0,
        32 * 1024,
        False,
        ("compute", "dte", "lsu"),
    ),
    (
        "double_b",
        32 * 1024,
        8 * 1024,
        False,
        ("compute", "dte", "lsu"),
    ),
    (
        "input",
        40 * 1024,
        8 * 1024,
        True,
        ("compute", "dte", "lsu", "legacy"),
    ),
    (
        "intermediate",
        48 * 1024,
        8 * 1024,
        True,
        ("compute", "dte", "lsu", "legacy"),
    ),
    (
        "comm",
        56 * 1024,
        8 * 1024,
        False,
        ("compute", "dte", "lsu", "noc_rx"),
    ),
)


@dataclass(frozen=True, slots=True)
class Stage2RuntimeHardwareInputs:
    """Exact text inputs suitable for a later production runtime invocation."""

    source_hardware_path: Path
    source_mapping_path: Path
    hardware_json: str
    mapping_text: str


@dataclass(frozen=True, slots=True)
class Stage2DenseForwardCase:
    spec: ExperimentSpec
    template: IR0Template
    oracle: Stage2DenseForwardOracle
    graph: IR1
    global_profile: GlobalActionProfile
    global_dag: GlobalActionDAG
    planning_context: InterDiePlanningContext
    scheduling_context: IntraDieSchedulingContext
    lowered_bundle: LoweredProgramBundle
    lowered: LoweredProgramProfile
    linked_bundle: LinkedProgramBundle
    profile: LinkedProgramProfile
    manifest: LinkedProgramManifest
    program_io: ProgramIoContract
    runtime_hardware_inputs: Stage2RuntimeHardwareInputs
    state_seed_refs: tuple[str, ...]
    state_expected_refs: tuple[str, ...]


def _spec(tp_degree: int, output: InferOutput) -> ExperimentSpec:
    hardware_ref = (
        "notes/frontend/examples/hardware_2x2.json"
        if tp_degree == 4
        else "llm/test/sram/hardware_numa.json"
    )
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
        "hardware": {"ref": hardware_ref},
        "workload": {
            "mode": "infer",
            "infer": {
                "source": "static_profile",
                "output": output.value,
                "profile": {
                    "prefill_tokens": 8,
                    "decode_tokens": 0,
                    "num_seqs": 1,
                    "context_sum": 8,
                    "context_max": 8,
                    "kv_pages": 1,
                    "expert_load": None,
                },
            },
        },
        "parallel": {
            "instances": [
                {
                    "id": "P0",
                    "role": "prefill",
                    "tp": tp_degree,
                    "sp": tp_degree > 1,
                    "replicas": 1,
                    "dp": 1,
                    "pp": 1,
                    "ep": 1,
                }
            ]
        },
        "placement": {"strategy": "compact", "groups": []},
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
    return from_data(ExperimentSpec, raw, path="stage2_dense_forward.spec")


def _runtime_inputs(tp_degree: int) -> Stage2RuntimeHardwareInputs:
    hardware_path = _TP4_HARDWARE if tp_degree == 4 else _TP12_HARDWARE
    raw = json.loads(hardware_path.read_text(encoding="utf-8"))
    memory = raw["memory"]
    sram = memory["sram"]
    memory["sram_size"] = _SRAM_CAPACITY_BYTES
    sram["capacity_bytes"] = _SRAM_CAPACITY_BYTES
    sram["regions"] = [
        {
            "name": name,
            "base_bytes": base,
            "size_bytes": size,
            "allocator": "block",
            "spillable": spillable,
            "access": list(access),
        }
        for name, base, size, spillable, access in _SRAM_REGIONS
    ]
    mapping_text = _MAPPING.read_text(encoding="utf-8")
    hardware_json = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    fabric = physical_fabric_from_data(
        json.loads(hardware_json),
        path="stage2_dense_forward.hardware",
    )
    validate_identity_mapping_text(
        mapping_text,
        total_cores=sum(len(die.cores) for die in fabric.dies),
        path="stage2_dense_forward.mapping",
    )
    return Stage2RuntimeHardwareInputs(
        source_hardware_path=hardware_path,
        source_mapping_path=_MAPPING,
        hardware_json=hardware_json,
        mapping_text=mapping_text,
    )


def build_stage2_dense_forward_case(
    tp_degree: int,
    output: InferOutput = InferOutput.LOGITS,
) -> Stage2DenseForwardCase:
    """Build one production-pass Dense forward case through ProgramIo."""

    if tp_degree not in (1, 2, 4):
        raise ValueError("tp_degree must be one of 1, 2, or 4")
    if type(output) is not InferOutput:
        raise TypeError("output must be an InferOutput")
    spec = _spec(tp_degree, output)
    template = build_ir0(spec)
    expanded = logical_expand(template)
    profile_key = template.profiles[0].key
    oracle = build_stage2_dense_forward_oracle(
        template,
        profile_key,
        tp_degree=tp_degree,
    )
    oracle.validate_against_ir0(template, expanded.entries[0].graph)

    runtime_inputs = _runtime_inputs(tp_degree)
    hardware = json.loads(runtime_inputs.hardware_json)
    fabric = physical_fabric_from_data(
        hardware,
        path="stage2_dense_forward.hardware",
    )
    placement_context = PlacementContext.create(
        producer_pass="stage2_dense_forward_cases",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(
            hardware,
            path="stage2_dense_forward.hardware",
        ),
    )
    placed = place_bundle(expanded, placement_context)
    oracle.validate_against_ir1(template, expanded.entries[0].graph, placed.entries[0].graph)
    partitioned = partition_bundle(
        placed,
        FusionPartitionContext.create(
            producer_pass="stage2_dense_forward_cases"
        ),
    )

    registry = production_registry()
    planning_context = InterDiePlanningContext.create(
        producer_pass="stage2_dense_forward_cases",
        fused_policy=registry.instantiate(
            RegistryKind.INTER_DIE,
            "naive",
        ).selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE,
            "direct_all_gather",
        ).selection,
    )
    planned = plan_bundle(
        partitioned,
        planning_context,
    )
    projected = project_bundle(
        planned,
        ProjectToIR2Context.create(
            producer_pass="stage2_dense_forward_cases",
            state_transfers=(),
        ),
    )
    scheduling_context = IntraDieSchedulingContext.create(
        producer_pass="stage2_dense_forward_cases",
        policy=registry.instantiate(
            RegistryKind.INTRA_DIE,
            "naive",
        ).selection,
    )
    scheduled = schedule_bundle(
        projected,
        scheduling_context,
    )
    global_bundle = build_global_bundle(scheduled)
    lowered_bundle = lower_bundle(global_bundle)
    linked_bundle = link_bundle(lowered_bundle)

    global_profile = global_bundle.entries[0]
    lowered = lowered_bundle.entries[0]
    profile = linked_bundle.entries[0]
    state_seeds, state_expected = (
        build_deterministic_timing_state_overrides(profile)
    )
    artifact_sha256 = sha256(
        f"stage2-dense-forward:{tp_degree}:{output.value}".encode("ascii")
    ).hexdigest()
    program_io = build_timing_program_io(
        profile,
        artifact_sha256,
        state_seed_overrides=state_seeds,
        state_expected_overrides=state_expected,
    )
    program_io.validate_against(profile.manifest)
    return Stage2DenseForwardCase(
        spec=spec,
        template=template,
        oracle=oracle,
        graph=profile.lowering_context.ir1,
        global_profile=global_profile,
        global_dag=global_profile.global_dag,
        planning_context=planning_context,
        scheduling_context=scheduling_context,
        lowered_bundle=lowered_bundle,
        lowered=lowered,
        linked_bundle=linked_bundle,
        profile=profile,
        manifest=profile.manifest,
        program_io=program_io,
        runtime_hardware_inputs=runtime_inputs,
        state_seed_refs=tuple(sorted(state_seeds)),
        state_expected_refs=tuple(sorted(state_expected)),
    )


__all__ = [
    "Stage2DenseForwardCase",
    "Stage2RuntimeHardwareInputs",
    "build_stage2_dense_forward_case",
]
