"""Self-contained source case for the isolated S3-Lite static MoE model.

The source builder preserves the Dense host/reference inputs.  The execution
builder then calls only production S3-Lite passes through the exact N6
lowering intent; it deliberately stops before command fragments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_moe import (
    build_lite_moe_oracle,
)
from llm.frontend.wafer_frontend.passes.lite_moe_execution import (
    build_lite_moe_global,
    project_lite_moe,
    schedule_lite_moe,
    validate_lite_moe_global,
    validate_lite_moe_projection,
    validate_lite_moe_schedule,
)
from llm.frontend.wafer_frontend.passes.lite_moe_graph import (
    LiteMoeIR0Validator,
    build_lite_moe_ir0_adapter,
)
from llm.frontend.wafer_frontend.passes.lite_moe_n4 import (
    build_lite_moe_n4,
    place_lite_moe_adapter,
    validate_lite_moe_n4,
    validate_lite_moe_placement,
)
from llm.frontend.wafer_frontend.passes.lite_moe_n6 import (
    build_lite_moe_n6_intent,
    validate_lite_moe_n6_intent,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
    validate_identity_mapping_text,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
    InferOutput,
    InferSource,
    InstanceRole,
    WorkloadMode,
)
from llm.frontend.wafer_frontend.schema.lite_moe import (
    S3_LITE_STATIC_MOE_CASE_ID,
    LiteMoeOracle,
    LiteMoeSpec,
    LiteMoeStaticTrace,
    LiteMoeTraceAssignment,
)
from llm.frontend.wafer_frontend.schema.lite_moe_execution import (
    LiteMoeGlobalDag,
    LiteMoeProjection,
    LiteMoeScheduled,
)
from llm.frontend.wafer_frontend.schema.lite_moe_graph import LiteMoeIR0Adapter
from llm.frontend.wafer_frontend.schema.lite_moe_n4 import (
    LiteMoeN4IR1,
    LiteMoePlacedIR1,
)
from llm.frontend.wafer_frontend.schema.lite_moe_n6 import LiteMoeN6Intent
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanningContext,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.policy import RegistryKind
from llm.frontend.wafer_frontend.schema.serde import from_data
from llm.frontend.wafer_frontend.policies.registry import production_registry


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x1.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"


@dataclass(frozen=True, slots=True)
class LiteMoeSourceCase:
    """Typed source inputs for the S3-Lite static-route preview."""

    spec: ExperimentSpec
    moe_spec: LiteMoeSpec
    oracle: LiteMoeOracle
    hardware_json: str
    mapping_text: str

    def validate(self, path: str = "lite_moe_source_case") -> None:
        if type(self.spec) is not ExperimentSpec:
            raise SchemaError("must be an ExperimentSpec", path=f"{path}.spec")
        if type(self.moe_spec) is not LiteMoeSpec:
            raise SchemaError(
                "must be a LiteMoeSpec", path=f"{path}.moe_spec"
            )
        if type(self.oracle) is not LiteMoeOracle:
            raise SchemaError(
                "must be a LiteMoeOracle", path=f"{path}.oracle"
            )
        self.spec.validate(f"{path}.spec")
        self.moe_spec.validate(f"{path}.moe_spec")
        self.oracle.validate_against(self.moe_spec, f"{path}.oracle")

        model = self.spec.model
        expected_model = {
            "V": 32,
            "H": 16,
            "I": 32,
            "NH": 4,
            "KVH": 4,
            "DH": 4,
            "rotary_dim": 4,
            "L": 1,
            "max_position_embeddings": 128,
        }
        for field_name, expected in expected_model.items():
            if getattr(model, field_name) != expected:
                raise SchemaError(
                    f"must equal the S3-Lite tiny value {expected}",
                    path=f"{path}.spec.model.{field_name}",
                )
        if model.moe is not None:
            raise SchemaError(
                "base ExperimentSpec is only a Dense host/reference; "
                "LiteMoeSpec is the sole EP/expert source of truth",
                path=f"{path}.spec.model.moe",
            )
        if (
            model.H != self.moe_spec.hidden_size
            or model.I != self.moe_spec.intermediate_size
        ):
            raise SchemaError(
                "base model H/I must equal the LiteMoeSpec H/I",
                path=f"{path}.moe_spec",
            )

        workload = self.spec.workload
        if workload.mode is not WorkloadMode.INFER or workload.infer is None:
            raise SchemaError(
                "must be an infer workload", path=f"{path}.spec.workload"
            )
        infer = workload.infer
        profile = infer.profile
        if (
            infer.source is not InferSource.STATIC_PROFILE
            or infer.output is not InferOutput.LOGITS
            or profile is None
            or (
                profile.prefill_tokens,
                profile.decode_tokens,
                profile.num_seqs,
                profile.context_sum,
                profile.context_max,
                profile.kv_pages,
                profile.expert_load,
            )
            != (8, 0, 1, 8, 8, 1, None)
        ):
            raise SchemaError(
                "must use the exact T8 pure-prefill Dense reference profile",
                path=f"{path}.spec.workload.infer",
            )
        instances = self.spec.parallel.instances
        if (
            len(instances) != 1
            or instances[0].role is not InstanceRole.PREFILL
            or (
                instances[0].tp,
                instances[0].sp,
                instances[0].replicas,
                instances[0].dp,
                instances[0].pp,
                instances[0].ep,
            )
            != (1, False, 1, 1, 1, 1)
        ):
            raise SchemaError(
                "base Dense host/reference must remain TP1/EP1; "
                "consumers must read EP/expert topology only from LiteMoeSpec",
                path=f"{path}.spec.parallel.instances",
            )
        if self.moe_spec.case_id != S3_LITE_STATIC_MOE_CASE_ID:
            raise SchemaError(
                "unexpected S3-Lite case identity",
                path=f"{path}.moe_spec.case_id",
            )

        if type(self.hardware_json) is not str:
            raise SchemaError(
                "must preserve hardware JSON text", path=f"{path}.hardware_json"
            )
        if type(self.mapping_text) is not str:
            raise SchemaError(
                "must preserve mapping text", path=f"{path}.mapping_text"
            )
        try:
            hardware = json.loads(self.hardware_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise SchemaError(str(error), path=f"{path}.hardware_json") from error
        fabric = physical_fabric_from_data(
            hardware, path=f"{path}.hardware"
        )
        if (
            fabric.die_grid != (2, 1)
            or tuple(die.id for die in fabric.dies) != (0, 1)
            or any(not die.cores for die in fabric.dies)
        ):
            raise SchemaError(
                "hardware must provide exactly two usable dies",
                path=f"{path}.hardware",
            )
        if {
            (link.source_die, link.destination_die) for link in fabric.links
        } != {(0, 1), (1, 0)}:
            raise SchemaError(
                "hardware must provide exact bidirectional one-hop die links",
                path=f"{path}.hardware",
            )
        address_spaces = hbm_address_spaces_from_data(
            hardware, path=f"{path}.hardware"
        )
        if tuple(space.die_id for space in address_spaces) != (0, 1):
            raise SchemaError(
                "hardware must provide one HBM address space per die",
                path=f"{path}.hardware",
            )
        validate_identity_mapping_text(
            self.mapping_text,
            total_cores=sum(len(die.cores) for die in fabric.dies),
            path=f"{path}.mapping_text",
        )


@dataclass(frozen=True, slots=True)
class LiteMoeExecutionCase:
    """Self-contained production carrier chain through exact N6 intent."""

    source: LiteMoeSourceCase
    adapter: LiteMoeIR0Adapter
    placement_context: PlacementContext
    partition_context: FusionPartitionContext
    planning_context: InterDiePlanningContext
    placed: LiteMoePlacedIR1
    n4: LiteMoeN4IR1
    projection: LiteMoeProjection
    schedule: LiteMoeScheduled
    global_dag: LiteMoeGlobalDag
    n6_intent: LiteMoeN6Intent

    def validate(self, path: str = "lite_moe_execution_case") -> None:
        self.source.validate(f"{path}.source")
        self.placement_context.validate(f"{path}.placement_context")
        self.partition_context.validate(f"{path}.partition_context")
        self.planning_context.validate(f"{path}.planning_context")
        LiteMoeIR0Validator.validate(
            self.adapter,
            self.source.spec,
            self.source.moe_spec,
            self.source.oracle,
            f"{path}.adapter",
        )
        validate_lite_moe_placement(
            self.placed, self.adapter, self.placement_context
        )
        validate_lite_moe_n4(
            self.n4,
            self.placed,
            self.partition_context,
            self.planning_context,
        )
        validate_lite_moe_projection(self.projection, self.n4)
        validate_lite_moe_schedule(self.schedule, self.projection, self.n4)
        validate_lite_moe_global(
            self.global_dag, self.schedule, self.projection, self.n4
        )
        validate_lite_moe_n6_intent(
            self.n6_intent,
            self.global_dag,
            self.schedule,
            self.projection,
            self.n4,
        )
        if (
            len(self.adapter.graph.nodes),
            len(self.global_dag.actions),
            len(self.n6_intent.buffer_abis),
            len(self.n6_intent.state_loads),
            len(self.n6_intent.compute_units),
            len(self.n6_intent.dte_units),
        ) != (40, 80, 64, 24, 32, 8):
            raise SchemaError(
                "execution carrier counts do not match the frozen S3-Lite case",
                path=path,
            )
        if (
            self.placed.source_adapter_id != self.adapter.id
            or self.n4.source_placed_id != self.placed.id
            or self.projection.source_n4_id != self.n4.id
            or self.schedule.source_projection_id != self.projection.id
            or self.global_dag.source_schedule_id != self.schedule.id
            or self.n6_intent.source_global_id != self.global_dag.id
        ):
            raise SchemaError(
                "source-to-N6 provenance chain is not exact", path=path
            )


def _base_dense_spec() -> ExperimentSpec:
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
            "L": 1,
            "dtype": "fp16",
            "tie_word_embeddings": False,
            "rms_norm_epsilon": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128,
            "moe": None,
        },
        "hardware": {"ref": "notes/frontend/examples/hardware_2x1.json"},
        "workload": {
            "mode": "infer",
            "infer": {
                "source": "static_profile",
                "output": "logits",
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
                    "tp": 1,
                    "sp": False,
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
    return from_data(ExperimentSpec, raw, path="lite_moe_source_case.spec")


def _balanced_trace() -> LiteMoeStaticTrace:
    return LiteMoeStaticTrace.create(
        token_count=8,
        assignments=tuple(
            LiteMoeTraceAssignment(
                token_index=token_index,
                expert_index=(token_index // 2) % 4,
                slot_index=token_index % 2,
            )
            for token_index in range(8)
        ),
        expert_histogram=(2, 2, 2, 2),
    )


def build_lite_moe_source_case() -> LiteMoeSourceCase:
    """Build deterministic typed S3-Lite inputs without constructing IR0."""

    moe_spec = LiteMoeSpec.create(
        hidden_size=16,
        intermediate_size=32,
        capacity_per_expert=2,
        trace=_balanced_trace(),
    )
    result = LiteMoeSourceCase(
        spec=_base_dense_spec(),
        moe_spec=moe_spec,
        oracle=build_lite_moe_oracle(moe_spec),
        hardware_json=_HARDWARE.read_text(encoding="utf-8"),
        mapping_text=_MAPPING.read_text(encoding="utf-8"),
    )
    result.validate()
    return result


def build_lite_moe_execution_case() -> LiteMoeExecutionCase:
    """Run the production S3-Lite carrier chain through N6 intent."""

    source = build_lite_moe_source_case()
    adapter = build_lite_moe_ir0_adapter(
        source.spec, source.moe_spec, source.oracle
    )
    hardware = json.loads(source.hardware_json)
    placement_context = PlacementContext.create(
        producer_pass="lite_moe_integration",
        fabric=physical_fabric_from_data(hardware),
        placement=source.spec.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(hardware),
    )
    registry = production_registry()
    partition_context = FusionPartitionContext.create(
        producer_pass="lite_moe_integration"
    )
    planning_context = InterDiePlanningContext.create(
        producer_pass="lite_moe_integration",
        fused_policy=registry.instantiate(
            RegistryKind.INTER_DIE, "naive"
        ).selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE,
            "direct_all_gather",
        ).selection,
    )
    placed = place_lite_moe_adapter(
        adapter,
        source.spec,
        source.moe_spec,
        source.oracle,
        placement_context,
    )
    n4 = build_lite_moe_n4(placed, partition_context, planning_context)
    projection = project_lite_moe(n4)
    schedule = schedule_lite_moe(projection, n4)
    global_dag = build_lite_moe_global(schedule, projection, n4)
    n6_intent = build_lite_moe_n6_intent(
        global_dag, schedule, projection, n4
    )
    result = LiteMoeExecutionCase(
        source,
        adapter,
        placement_context,
        partition_context,
        planning_context,
        placed,
        n4,
        projection,
        schedule,
        global_dag,
        n6_intent,
    )
    result.validate()
    return result


__all__ = [
    "LiteMoeExecutionCase",
    "LiteMoeSourceCase",
    "build_lite_moe_execution_case",
    "build_lite_moe_source_case",
]
