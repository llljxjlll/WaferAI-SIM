"""Self-contained Stage 3 pure-decode integration cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_deterministic_timing_state_overrides,
    build_global_bundle,
    build_ir0,
    build_stage3_dense_inference_oracle,
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
from llm.frontend.wafer_frontend.schema.common import ProfileKey
from llm.frontend.wafer_frontend.schema.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
)
from llm.frontend.wafer_frontend.schema.global_action import GlobalActionDAG
from llm.frontend.wafer_frontend.schema.ir0 import AttentionWorkload, OpKind
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
    LinkedProgramProfile,
    LoweredProgramProfile,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.policy import RegistryKind
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.serde import from_data
from llm.frontend.wafer_frontend.schema.s1_naive_evidence import (
    S1NaivePolicyEvidence,
)
from llm.frontend.wafer_frontend.schema.stage3_dense_inference_oracle import (
    Stage3DenseInferenceOracle,
)
from llm.frontend.wafer_frontend.schema.stage3_profile import (
    KvPageSpan,
    Stage3ProfileMode,
    Stage3StaticProfile,
    StaticRequestShape,
)


_ROOT = Path(__file__).resolve().parents[4]
_TP12_HARDWARE = _ROOT / "llm/test/sram/hardware_numa.json"
_TP4_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x2.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_SRAM_CAPACITY_BYTES = 64 * 1024
_SRAM_REGIONS = (
    ("double_a", 0, 48 * 1024, False, ("compute", "dte", "lsu")),
    ("double_b", 48 * 1024, 4 * 1024, False, ("compute", "dte", "lsu")),
    (
        "input",
        52 * 1024,
        4 * 1024,
        True,
        ("compute", "dte", "lsu", "legacy"),
    ),
    (
        "intermediate",
        56 * 1024,
        4 * 1024,
        True,
        ("compute", "dte", "lsu", "legacy"),
    ),
    (
        "comm",
        60 * 1024,
        4 * 1024,
        False,
        ("compute", "dte", "lsu", "noc_rx"),
    ),
)
_CONTEXTS = (4, 8, 12, 16, 20, 24, 28, 32)


class Stage3StaticCaseKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class Stage3RuntimeHardwareInputs:
    source_hardware_path: Path
    source_mapping_path: Path
    hardware_json: str
    mapping_text: str


@dataclass(frozen=True, slots=True)
class Stage3DecodeCase:
    case_kind: Stage3StaticCaseKind
    spec: ExperimentSpec
    template: IR0Template
    static_profile: Stage3StaticProfile
    oracle: Stage3DenseInferenceOracle
    policy: S1NaivePolicyEvidence
    planning_context: InterDiePlanningContext
    scheduling_context: IntraDieSchedulingContext
    graph: IR1
    global_profile: GlobalActionProfile
    global_dag: GlobalActionDAG
    lowered: LoweredProgramProfile
    profile: LinkedProgramProfile
    manifest: LinkedProgramManifest
    program_io: ProgramIoContract
    runtime_hardware_inputs: Stage3RuntimeHardwareInputs
    state_seed_refs: tuple[str, ...]
    state_expected_refs: tuple[str, ...]


def _request_shapes(
    case_kind: Stage3StaticCaseKind,
) -> tuple[tuple[str, int, int, int], ...]:
    if case_kind is Stage3StaticCaseKind.PREFILL:
        return (("prefill_0", 8, 0, 8),)
    if case_kind is Stage3StaticCaseKind.DECODE:
        return tuple(
            (f"decode_{index}", 0, 1, context_tokens)
            for index, context_tokens in enumerate(_CONTEXTS)
        )
    return (
        ("prefill_0", 1, 0, 1),
        ("prefill_1", 3, 0, 3),
        ("decode_0", 0, 1, 4),
        ("decode_1", 0, 1, 8),
        ("decode_2", 0, 1, 12),
        ("decode_3", 0, 1, 16),
    )


def _static_profile(case_kind: Stage3StaticCaseKind) -> Stage3StaticProfile:
    requests: list[StaticRequestShape] = []
    page_start = 0
    for request_ref, prefill_tokens, decode_tokens, context_tokens in sorted(
        _request_shapes(case_kind), key=lambda item: item[0]
    ):
        page_count = (context_tokens + 15) // 16
        requests.append(
            StaticRequestShape(
                request_ref=request_ref,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                context_tokens=context_tokens,
                kv_span=KvPageSpan(
                    page_start=page_start,
                    page_count=page_count,
                    page_size_tokens=16,
                ),
            )
        )
        page_start += page_count
    request_tuple = tuple(requests)
    return Stage3StaticProfile.create(
        key=ProfileKey(
            prefill_tokens=sum(item.prefill_tokens for item in request_tuple),
            decode_tokens=sum(item.decode_tokens for item in request_tuple),
            num_seqs=len(request_tuple),
            context_sum=sum(item.context_tokens for item in request_tuple),
            context_max=max(item.context_tokens for item in request_tuple),
            kv_pages=sum(item.kv_span.page_count for item in request_tuple),
            expert_load=None,
        ),
        requests=request_tuple,
    )


def _spec(
    tp_degree: int,
    static_profile: Stage3StaticProfile,
) -> ExperimentSpec:
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
                "output": "logits",
                "profile": {
                    "prefill_tokens": static_profile.key.prefill_tokens,
                    "decode_tokens": static_profile.key.decode_tokens,
                    "num_seqs": static_profile.key.num_seqs,
                    "context_sum": static_profile.key.context_sum,
                    "context_max": static_profile.key.context_max,
                    "kv_pages": static_profile.key.kv_pages,
                    "expert_load": None,
                },
            },
        },
        "parallel": {
            "instances": [
                {
                    "id": "D0",
                    "role": (
                        "both"
                        if static_profile.mode is Stage3ProfileMode.MIXED
                        else static_profile.mode.value
                    ),
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
    return from_data(ExperimentSpec, raw, path="stage3_decode.spec")


def _runtime_inputs(tp_degree: int) -> Stage3RuntimeHardwareInputs:
    hardware_path = _TP4_HARDWARE if tp_degree == 4 else _TP12_HARDWARE
    raw = json.loads(hardware_path.read_text(encoding="utf-8"))
    sram = raw["memory"]["sram"]
    raw["memory"]["sram_size"] = _SRAM_CAPACITY_BYTES
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
        json.loads(hardware_json), path="stage3_decode.hardware"
    )
    validate_identity_mapping_text(
        mapping_text,
        total_cores=sum(len(die.cores) for die in fabric.dies),
        path="stage3_decode.mapping",
    )
    return Stage3RuntimeHardwareInputs(
        source_hardware_path=hardware_path,
        source_mapping_path=_MAPPING,
        hardware_json=hardware_json,
        mapping_text=mapping_text,
    )


def _validate_graph(
    graph: IR1,
    oracle: Stage3DenseInferenceOracle,
) -> None:
    attention = tuple(
        node.workload for node in graph.nodes if node.kind is OpKind.ATTENTION
    )
    if not attention or any(type(item) is not AttentionWorkload for item in attention):
        raise SchemaError("attention workloads are not exact", path="stage3_decode.graph")
    if sum(item.query_key_pairs for item in attention) != (
        oracle.logical_work.attention.query_key_pairs
    ):
        raise SchemaError("attention pair closure differs", path="stage3_decode.graph")
    if sum(item.logical_kv_read_bytes for item in attention) != (
        oracle.kv.logical_read_bytes
    ):
        raise SchemaError("KV read closure differs", path="stage3_decode.graph")
    if sum(item.logical_kv_write_bytes for item in attention) != (
        oracle.kv.logical_write_bytes
    ):
        raise SchemaError("KV write closure differs", path="stage3_decode.graph")


def build_stage3_static_case(
    case_kind: Stage3StaticCaseKind,
    tp_degree: int,
) -> Stage3DecodeCase:
    """Build one exact static-profile case through production ProgramIo."""

    if type(case_kind) is not Stage3StaticCaseKind:
        raise TypeError("case_kind must be a Stage3StaticCaseKind")
    if tp_degree not in (1, 2, 4):
        raise ValueError("tp_degree must be one of 1, 2, or 4")
    static_profile = _static_profile(case_kind)
    spec = _spec(tp_degree, static_profile)
    template = build_ir0(spec, exact_profiles=(static_profile,))
    oracle = build_stage3_dense_inference_oracle(
        template, static_profile, tp_degree=tp_degree
    )
    expanded = logical_expand(template)
    runtime_inputs = _runtime_inputs(tp_degree)
    hardware = json.loads(runtime_inputs.hardware_json)
    fabric = physical_fabric_from_data(hardware, path="stage3_decode.hardware")
    placement_context = PlacementContext.create(
        producer_pass="stage3_decode_cases",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(
            hardware, path="stage3_decode.hardware"
        ),
    )
    placed = place_bundle(expanded, placement_context)
    graph = placed.entries[0].graph
    _validate_graph(graph, oracle)
    state_manifest = graph.persistent_state_manifest
    if state_manifest is None:
        raise SchemaError("state manifest is required", path="stage3_decode.graph")
    parameter_bytes = sum(
        item.tensor_bytes
        for item in state_manifest.declarations
        if item.identity.kind is StateKind.PARAMETER
    )
    if parameter_bytes != oracle.parameters.placed_bytes:
        raise SchemaError("parameter byte closure differs", path="stage3_decode.graph")

    partitioned = partition_bundle(
        placed,
        FusionPartitionContext.create(producer_pass="stage3_decode_cases"),
    )
    registry = production_registry()
    inter_die = registry.instantiate(RegistryKind.INTER_DIE, "naive")
    standalone = registry.instantiate(
        RegistryKind.STANDALONE_COLLECTIVE,
        "direct_all_gather",
    )
    intra_die = registry.instantiate(RegistryKind.INTRA_DIE, "naive")
    planning_context = InterDiePlanningContext.create(
        producer_pass="stage3_decode_cases",
        fused_policy=inter_die.selection,
        standalone_policy=standalone.selection,
    )
    planned = plan_bundle(
        partitioned,
        planning_context,
    )
    projected = project_bundle(
        planned,
        ProjectToIR2Context.create(
            producer_pass="stage3_decode_cases", state_transfers=()
        ),
    )
    scheduling_context = IntraDieSchedulingContext.create(
        producer_pass="stage3_decode_cases",
        policy=intra_die.selection,
    )
    scheduled = schedule_bundle(
        projected,
        scheduling_context,
    )
    policy = S1NaivePolicyEvidence(
        selections=(
            inter_die.selection,
            standalone.selection,
            intra_die.selection,
        ),
        planning_context_id=planning_context.id,
        scheduling_context_id=scheduling_context.id,
    )
    policy.validate_against(planning_context, scheduling_context)
    global_bundle = build_global_bundle(scheduled)
    lowered_bundle = lower_bundle(global_bundle)
    linked_bundle = link_bundle(lowered_bundle)
    global_profile = global_bundle.entries[0]
    lowered = lowered_bundle.entries[0]
    profile = linked_bundle.entries[0]
    state_seeds, state_expected = build_deterministic_timing_state_overrides(
        profile
    )
    artifact_sha256 = sha256(
        f"stage3-static:{case_kind.value}:{tp_degree}:{static_profile.id}".encode(
            "ascii"
        )
    ).hexdigest()
    program_io = build_timing_program_io(
        profile,
        artifact_sha256,
        state_seed_overrides=state_seeds,
        state_expected_overrides=state_expected,
    )
    program_io.validate_against(profile.manifest)
    return Stage3DecodeCase(
        case_kind=case_kind,
        spec=spec,
        template=template,
        static_profile=static_profile,
        oracle=oracle,
        policy=policy,
        planning_context=planning_context,
        scheduling_context=scheduling_context,
        graph=graph,
        global_profile=global_profile,
        global_dag=global_profile.global_dag,
        lowered=lowered,
        profile=profile,
        manifest=profile.manifest,
        program_io=program_io,
        runtime_hardware_inputs=runtime_inputs,
        state_seed_refs=tuple(sorted(state_seeds)),
        state_expected_refs=tuple(sorted(state_expected)),
    )


def build_stage3_decode_case(tp_degree: int) -> Stage3DecodeCase:
    """Build the checked pure-decode case through production ProgramIo."""

    return build_stage3_static_case(Stage3StaticCaseKind.DECODE, tp_degree)


__all__ = [
    "Stage3DecodeCase",
    "Stage3RuntimeHardwareInputs",
    "Stage3StaticCaseKind",
    "build_stage3_decode_case",
    "build_stage3_static_case",
]
