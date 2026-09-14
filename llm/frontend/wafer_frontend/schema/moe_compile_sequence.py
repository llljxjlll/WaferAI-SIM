"""Compile-only P3 MoE block sequence bound to production linked manifests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .artifact_manifest import LinkedProgramManifest, ManifestInputKind
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .e2e_workload_graph import E2EOperationKind, E2EStateKind
from .flexible_moe import (
    FlexibleMoeExecutablePlan,
    FlexibleMoeMode,
    FlexibleMoeSpec,
    MoeRectActionKind,
    MoeRectStateRole,
)
from .memory_plan import MemoryPlanExecution
from .serde import canonical_digest
from .workload_materialization import (
    WorkloadMaterializationManifest,
    WorkloadMaterializationStatus,
)
from .workload_run import (
    WorkloadFamily,
    WorkloadMemoryMode,
)


MOE_COMPILE_SEQUENCE_SCHEMA_VERSION = (
    "wafer_frontend.moe_compile_sequence/v1alpha1"
)
MOE_COMPILE_UNIT_SCHEMA_VERSION = "wafer_frontend.moe_compile_unit/v1alpha1"
MOE_OPERATION_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.moe_operation_binding/v1alpha1"
)
MOE_PARAMETER_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.moe_parameter_binding/v1alpha1"
)
MOE_PARAMETER_READ_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.moe_parameter_read_binding/v1alpha1"
)


class MoeCompileCoverage(str, Enum):
    MOE_BLOCKS_ONLY = "moe_blocks_only"


class MoeCompileStatus(str, Enum):
    INDEPENDENT_PRODUCTION_LINKED_BLOCKS = (
        "independent_production_linked_blocks"
    )


class MoeCompileRuntimeStatus(str, Enum):
    RUNTIME_NOT_MATERIALIZED = "runtime_not_materialized"


class MoeParameterLowering(str, Enum):
    ROUTER_REPLICATED_GATE = "router_replicated_gate"
    EXPERT_FUSED_GATE_UP_DOWN = "expert_fused_gate_up_down"


@dataclass(frozen=True, slots=True)
class MoeParameterReadBinding:
    id: str
    step: int
    layer: int
    expert: int | None
    parameter_refs: tuple[str, ...]
    parameter_state_refs: tuple[str, ...]
    production_parameter_state_refs: tuple[str, ...]
    production_load_action_refs: tuple[str, ...]
    lowering: MoeParameterLowering

    @classmethod
    def create(cls, **semantic: object) -> "MoeParameterReadBinding":
        result = cls(
            id=stable_artifact_id(
                "moe_parameter_read_binding",
                semantic,
                schema_version=MOE_PARAMETER_READ_BINDING_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "moe_parameter_read_binding") -> None:
        validate_uint64(self.step, f"{path}.step")
        validate_uint64(self.layer, f"{path}.layer")
        if self.expert is not None:
            validate_uint64(self.expert, f"{path}.expert")
        arity = len(self.parameter_refs)
        if arity == 0 or len(self.parameter_state_refs) != arity:
            raise SchemaError("must bind every P3 parameter state", path=path)
        for name in (
            "parameter_refs",
            "parameter_state_refs",
            "production_parameter_state_refs",
            "production_load_action_refs",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or len(set(values)) != len(values):
                raise SchemaError("must contain unique refs", path=f"{path}.{name}")
            for index, value in enumerate(values):
                validate_nonempty(value, f"{path}.{name}[{index}]")
        if type(self.lowering) is not MoeParameterLowering:
            raise SchemaError("must use a typed lowering", path=f"{path}.lowering")
        if self.lowering is MoeParameterLowering.ROUTER_REPLICATED_GATE:
            if self.expert is not None or arity != 1:
                raise SchemaError("router read must bind one shared parameter", path=path)
        elif self.expert is None or arity != 3:
            raise SchemaError("expert read must fuse gate/up/down", path=path)
        expected = stable_artifact_id(
            "moe_parameter_read_binding",
            self._semantic(),
            schema_version=MOE_PARAMETER_READ_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable parameter read id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeOperationBinding:
    id: str
    router_operation_ref: str
    route_freeze_operation_ref: str
    dispatch_operation_ref: str
    expert_forward_operation_refs: tuple[str, ...]
    combine_operation_ref: str
    grad_dispatch_operation_ref: str | None
    expert_backward_operation_refs: tuple[str, ...]
    dx_combine_operation_ref: str | None

    @classmethod
    def create(cls, **semantic: object) -> "MoeOperationBinding":
        result = cls(
            id=stable_artifact_id(
                "moe_operation_binding",
                semantic,
                schema_version=MOE_OPERATION_BINDING_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "moe_operation_binding") -> None:
        for name in (
            "router_operation_ref",
            "route_freeze_operation_ref",
            "dispatch_operation_ref",
            "combine_operation_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if not self.expert_forward_operation_refs:
            raise SchemaError(
                "must bind expert forward operations",
                path=f"{path}.expert_forward_operation_refs",
            )
        if len(set(self.expert_forward_operation_refs)) != len(
            self.expert_forward_operation_refs
        ):
            raise SchemaError(
                "contains duplicate operation refs",
                path=f"{path}.expert_forward_operation_refs",
            )
        for index, ref in enumerate(self.expert_forward_operation_refs):
            validate_nonempty(ref, f"{path}.expert_forward_operation_refs[{index}]")
        training_fields = (
            self.grad_dispatch_operation_ref,
            self.dx_combine_operation_ref,
        )
        if any(item is None for item in training_fields):
            if any(item is not None for item in training_fields) or self.expert_backward_operation_refs:
                raise SchemaError("training bindings must be all present or absent", path=path)
        else:
            for name in ("grad_dispatch_operation_ref", "dx_combine_operation_ref"):
                validate_nonempty(getattr(self, name), f"{path}.{name}")
            if len(self.expert_backward_operation_refs) != len(
                self.expert_forward_operation_refs
            ):
                raise SchemaError(
                    "forward/backward expert arity differs",
                    path=f"{path}.expert_backward_operation_refs",
                )
        expected = stable_artifact_id(
            "moe_operation_binding",
            self._semantic(),
            schema_version=MOE_OPERATION_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable operation binding id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeParameterBinding:
    id: str
    step: int
    layer: int
    expert: int | None
    parameter_refs: tuple[str, ...]
    input_parameter_state_refs: tuple[str, ...]
    raw_gradient_state_refs: tuple[str, ...]
    synced_gradient_state_refs: tuple[str, ...]
    output_parameter_state_refs: tuple[str, ...]
    gradient_operation_refs: tuple[str, ...]
    sync_operation_refs: tuple[str, ...]
    sgd_operation_refs: tuple[str, ...]
    store_operation_refs: tuple[str, ...]
    production_parameter_state_refs: tuple[str, ...]
    production_wgrad_action_refs: tuple[str, ...]
    production_sync_action_refs: tuple[str, ...]
    production_sgd_action_refs: tuple[str, ...]
    production_store_action_refs: tuple[str, ...]
    lowering: MoeParameterLowering

    @classmethod
    def create(cls, **semantic: object) -> "MoeParameterBinding":
        result = cls(
            id=stable_artifact_id(
                "moe_parameter_binding",
                semantic,
                schema_version=MOE_PARAMETER_BINDING_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "moe_parameter_binding") -> None:
        validate_uint64(self.step, f"{path}.step")
        validate_uint64(self.layer, f"{path}.layer")
        if self.expert is not None:
            validate_uint64(self.expert, f"{path}.expert")
        arity = len(self.parameter_refs)
        if arity == 0:
            raise SchemaError("must bind parameters", path=f"{path}.parameter_refs")
        for name in (
            "input_parameter_state_refs",
            "raw_gradient_state_refs",
            "synced_gradient_state_refs",
            "output_parameter_state_refs",
            "gradient_operation_refs",
            "sync_operation_refs",
            "sgd_operation_refs",
            "store_operation_refs",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != arity:
                raise SchemaError(
                    "must contain one item per P3 parameter",
                    path=f"{path}.{name}",
                )
        for name in (
            "parameter_refs",
            "input_parameter_state_refs",
            "raw_gradient_state_refs",
            "synced_gradient_state_refs",
            "output_parameter_state_refs",
            "gradient_operation_refs",
            "sync_operation_refs",
            "sgd_operation_refs",
            "store_operation_refs",
            "production_parameter_state_refs",
            "production_wgrad_action_refs",
            "production_sync_action_refs",
            "production_sgd_action_refs",
            "production_store_action_refs",
        ):
            values = getattr(self, name)
            if type(values) is not tuple:
                raise SchemaError("must be a tuple", path=f"{path}.{name}")
            if len(set(values)) != len(values):
                raise SchemaError("contains duplicate refs", path=f"{path}.{name}")
            for index, value in enumerate(values):
                validate_nonempty(value, f"{path}.{name}[{index}]")
        if type(self.lowering) is not MoeParameterLowering:
            raise SchemaError("must use a typed lowering", path=f"{path}.lowering")
        if self.lowering is MoeParameterLowering.ROUTER_REPLICATED_GATE:
            if self.expert is not None or arity != 1:
                raise SchemaError("router binding must contain one shared parameter", path=path)
        elif self.expert is None or arity != 3:
            raise SchemaError("expert binding must fuse gate/up/down", path=path)
        expected = stable_artifact_id(
            "moe_parameter_binding",
            self._semantic(),
            schema_version=MOE_PARAMETER_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable parameter binding id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeCompileUnit:
    id: str
    phase: str
    step: int
    layer: int
    route_trace_ref: str
    route_trace_digest: str
    operation_binding: MoeOperationBinding
    parameter_reads: tuple[MoeParameterReadBinding, ...]
    parameter_bindings: tuple[MoeParameterBinding, ...]
    source_rank_policy: str
    spec: FlexibleMoeSpec
    plan: FlexibleMoeExecutablePlan
    linked_manifest: LinkedProgramManifest
    linked_manifest_digest: str
    lower_link_verified: bool
    runtime_verified: bool

    @classmethod
    def create(cls, **semantic: object) -> "MoeCompileUnit":
        result = cls(
            id=stable_artifact_id(
                "moe_compile_unit",
                semantic,
                schema_version=MOE_COMPILE_UNIT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "moe_compile_unit") -> None:
        validate_nonempty(self.phase, f"{path}.phase")
        validate_uint64(self.step, f"{path}.step")
        validate_uint64(self.layer, f"{path}.layer")
        validate_nonempty(self.route_trace_ref, f"{path}.route_trace_ref")
        validate_nonempty(self.route_trace_digest, f"{path}.route_trace_digest")
        if self.source_rank_policy != "token_index_mod_ep":
            raise SchemaError("unsupported source-rank policy", path=f"{path}.source_rank_policy")
        if type(self.operation_binding) is not MoeOperationBinding:
            raise SchemaError("must carry a typed operation binding", path=f"{path}.operation_binding")
        self.operation_binding.validate(f"{path}.operation_binding")
        for index, binding in enumerate(self.parameter_reads):
            if type(binding) is not MoeParameterReadBinding:
                raise SchemaError("must carry typed parameter reads", path=f"{path}.parameter_reads[{index}]")
            binding.validate(f"{path}.parameter_reads[{index}]")
            if (binding.step, binding.layer) != (self.step, self.layer):
                raise SchemaError("parameter read belongs to another unit", path=f"{path}.parameter_reads[{index}]")
        for index, binding in enumerate(self.parameter_bindings):
            if type(binding) is not MoeParameterBinding:
                raise SchemaError("must carry typed parameter bindings", path=f"{path}.parameter_bindings[{index}]")
            binding.validate(f"{path}.parameter_bindings[{index}]")
            if (binding.step, binding.layer) != (self.step, self.layer):
                raise SchemaError("parameter binding belongs to another unit", path=f"{path}.parameter_bindings[{index}]")
        self.spec.validate(f"{path}.spec")
        self.plan.validate_against(self.spec, f"{path}.plan")
        self.linked_manifest.validate(f"{path}.linked_manifest")
        if self.plan.source_spec_id != self.spec.id:
            raise SchemaError("plan/spec lineage differs", path=path)
        if (
            self.linked_manifest.source_ir1_id != self.spec.id
            or self.linked_manifest.source_global_dag_id != self.plan.id
        ):
            raise SchemaError("linked manifest lineage differs", path=f"{path}.linked_manifest")
        plan_inputs = tuple(
            item for item in self.linked_manifest.input_digests
            if item.kind is ManifestInputKind.FLEXIBLE_MOE_PLAN
        )
        if (
            len(plan_inputs) != 1
            or plan_inputs[0].artifact_id != self.plan.id
            or plan_inputs[0].digest != canonical_digest(self.plan)
        ):
            raise SchemaError("linked manifest plan digest is not exact", path=f"{path}.linked_manifest")
        if self.linked_manifest_digest != canonical_digest(self.linked_manifest):
            raise SchemaError("linked manifest digest mismatch", path=f"{path}.linked_manifest_digest")
        if self.lower_link_verified is not True or self.runtime_verified is not False:
            raise SchemaError("compile unit overclaims runtime evidence", path=path)
        expected = stable_artifact_id(
            "moe_compile_unit",
            self._semantic(),
            schema_version=MOE_COMPILE_UNIT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable compile unit id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeCompileSequence:
    schema_version: str
    producer_pass: str
    id: str
    materialization: WorkloadMaterializationManifest
    materialization_digest: str
    logical_graph_digest: str
    units: tuple[MoeCompileUnit, ...]
    coverage: MoeCompileCoverage
    compile_status: MoeCompileStatus
    runtime_status: MoeCompileRuntimeStatus

    @classmethod
    def create(
        cls,
        *,
        materialization: WorkloadMaterializationManifest,
        units: tuple[MoeCompileUnit, ...],
    ) -> "MoeCompileSequence":
        semantic = {
            "materialization": materialization,
            "materialization_digest": materialization.digest,
            "logical_graph_digest": materialization.logical_graph_digest,
            "units": units,
            "coverage": MoeCompileCoverage.MOE_BLOCKS_ONLY,
            "compile_status": MoeCompileStatus.INDEPENDENT_PRODUCTION_LINKED_BLOCKS,
            "runtime_status": MoeCompileRuntimeStatus.RUNTIME_NOT_MATERIALIZED,
        }
        result = cls(
            schema_version=MOE_COMPILE_SEQUENCE_SCHEMA_VERSION,
            producer_pass="compile_moe_sequence",
            id=stable_artifact_id(
                "moe_compile_sequence",
                semantic,
                schema_version=MOE_COMPILE_SEQUENCE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "moe_compile_sequence") -> None:
        if self.schema_version != MOE_COMPILE_SEQUENCE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "compile_moe_sequence":
            raise SchemaError("unexpected producer", path=f"{path}.producer_pass")
        self.materialization.validate(f"{path}.materialization")
        request = self.materialization.request
        if (
            self.materialization.status is not WorkloadMaterializationStatus.PARTIAL
            or request.family not in (
                WorkloadFamily.MOE_INFERENCE,
                WorkloadFamily.MOE_TRAINING,
            )
            or request.memory.mode is not WorkloadMemoryMode.RESIDENT_HBM
            or self.materialization.memory_plan.execution
            is not MemoryPlanExecution.NO_EXTERNAL_TRANSPORT_REQUIRED
        ):
            raise SchemaError("requires a resident partial MoE materialization", path=f"{path}.materialization")
        if self.materialization_digest != self.materialization.digest:
            raise SchemaError("materialization digest mismatch", path=f"{path}.materialization_digest")
        if self.logical_graph_digest != self.materialization.logical_graph_digest:
            raise SchemaError("logical graph digest mismatch", path=f"{path}.logical_graph_digest")
        if (
            self.coverage is not MoeCompileCoverage.MOE_BLOCKS_ONLY
            or self.compile_status
            is not MoeCompileStatus.INDEPENDENT_PRODUCTION_LINKED_BLOCKS
            or self.runtime_status
            is not MoeCompileRuntimeStatus.RUNTIME_NOT_MATERIALIZED
        ):
            raise SchemaError("compile/runtime coverage overclaim", path=path)
        training = request.family is WorkloadFamily.MOE_TRAINING
        step_count = (
            request.steps.training.step_count
            if training and request.steps.training is not None
            else request.steps.inference.decode_steps + 1
            if request.steps.inference is not None
            else 0
        )
        expected_keys = tuple(
            (step, layer)
            for step in range(step_count)
            for layer in range(request.model.num_layers)
        )
        if tuple((unit.step, unit.layer) for unit in self.units) != expected_keys:
            raise SchemaError("units must cover every step/layer in canonical order", path=f"{path}.units")
        graph = self.materialization.logical_graph
        operations = {item.id: item for item in graph.operations}
        graph_states = {item.id: item for item in graph.state_versions}
        graph_values = {item.id: item for item in graph.tensor_values}
        traces = {item.id: item for item in graph.route_traces}
        previous_outputs: dict[tuple[int, str], str] = {}
        for index, unit in enumerate(self.units):
            unit.validate(f"{path}.units[{index}]")
            expected_mode = FlexibleMoeMode.TRAIN if training else FlexibleMoeMode.INFERENCE
            if unit.spec.mode is not expected_mode:
                raise SchemaError("production mode differs from family", path=f"{path}.units[{index}]")
            if (
                unit.spec.mesh.rows,
                unit.spec.mesh.columns,
                unit.spec.hidden_size,
                unit.spec.intermediate_size,
                unit.spec.expert_count,
                unit.spec.expert_parallel_degree,
                unit.spec.top_k,
            ) != (
                request.mesh.rows,
                request.mesh.columns,
                request.model.hidden_size,
                request.model.intermediate_size,
                request.model.num_experts,
                request.parallel.ep,
                request.model.experts_per_token,
            ):
                raise SchemaError("production spec differs from P3 request", path=f"{path}.units[{index}]")
            trace = traces.get(unit.route_trace_ref)
            if trace is None or canonical_digest(trace) != unit.route_trace_digest:
                raise SchemaError("route trace digest closure differs", path=f"{path}.units[{index}]")
            if (trace.phase, trace.step, trace.layer) != (unit.phase, unit.step, unit.layer):
                raise SchemaError("route trace belongs to another unit", path=f"{path}.units[{index}]")
            self._validate_trace(unit, trace, path)
            self._validate_operations(unit, trace, operations, training, path)
            self._validate_parameter_reads(
                unit, operations, graph_states, graph_values, path
            )
            self._validate_parameter_bindings(
                unit,
                operations,
                graph_states,
                graph_values,
                previous_outputs,
                training,
                path,
            )
        expected_routes = {trace.id for trace in graph.route_traces}
        if {unit.route_trace_ref for unit in self.units} != expected_routes:
            raise SchemaError("sequence must cover every P3 route trace", path=f"{path}.units")
        expected = stable_artifact_id(
            "moe_compile_sequence",
            self._semantic(),
            schema_version=MOE_COMPILE_SEQUENCE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable sequence id", path=f"{path}.id")

    @staticmethod
    def _validate_trace(unit, trace, path: str) -> None:
        assignments = unit.spec.trace.assignments
        if (
            unit.spec.trace.token_count != trace.token_count
            or unit.spec.trace.expert_histogram != trace.expert_token_counts
            or tuple(item.token_index for item in assignments)
            != tuple(range(trace.token_count))
            or tuple(item.source_rank for item in assignments)
            != tuple(index % unit.spec.expert_parallel_degree for index in range(trace.token_count))
            or tuple(item.expert_index for item in assignments) != trace.expert_by_token
            or tuple(item.expert_home_rank for item in assignments) != trace.expert_by_token
        ):
            raise SchemaError("production trace differs from frozen P3 route", path=f"{path}.units")
        slots = [0] * unit.spec.expert_count
        expected_slots = []
        for expert in trace.expert_by_token:
            expected_slots.append(slots[expert])
            slots[expert] += 1
        if tuple(item.slot_index for item in assignments) != tuple(expected_slots):
            raise SchemaError("production expert slots differ from P3 route", path=f"{path}.units")

    @staticmethod
    def _validate_operations(
        unit, trace, operations, training: bool, path: str
    ) -> None:
        binding = unit.operation_binding
        expected = (
            (binding.router_operation_ref, E2EOperationKind.ROUTER, None),
            (binding.route_freeze_operation_ref, E2EOperationKind.ROUTE_FREEZE, None),
            (binding.dispatch_operation_ref, E2EOperationKind.DISPATCH, None),
            *((ref, E2EOperationKind.EXPERT_FORWARD, expert)
              for expert, ref in enumerate(binding.expert_forward_operation_refs)),
            (binding.combine_operation_ref, E2EOperationKind.COMBINE, None),
        )
        if training:
            expected += (
                (binding.grad_dispatch_operation_ref, E2EOperationKind.GRAD_DISPATCH, None),
                *((ref, E2EOperationKind.EXPERT_BACKWARD, expert)
                  for expert, ref in enumerate(binding.expert_backward_operation_refs)),
                (binding.dx_combine_operation_ref, E2EOperationKind.DX_COMBINE, None),
            )
        for ref, kind, expert in expected:
            operation = operations.get(ref)
            if operation is None or (
                operation.kind,
                operation.step,
                operation.layer,
                operation.expert,
            ) != (kind, unit.step, unit.layer, expert):
                raise SchemaError("P3 MoE operation binding drifted", path=f"{path}.units")
        selected = [operations[ref] for ref, _, _ in expected]
        if tuple(item.sequence_index for item in selected) != tuple(
            sorted(item.sequence_index for item in selected)
        ):
            raise SchemaError("P3 MoE operation order drifted", path=f"{path}.units")
        router, freeze, dispatch = selected[:3]
        experts = selected[3:3 + len(binding.expert_forward_operation_refs)]
        combine = selected[3 + len(experts)]
        if (
            len(router.writes) != 1
            or freeze.reads != router.writes
            or tuple(freeze.output_value_refs) != trace.route_value_refs
            or dispatch.reads != freeze.writes
            or tuple(dispatch.output_value_refs) != trace.dispatch_value_refs
            or tuple(
                value
                for operation in experts
                for value in operation.input_value_refs
                if value in trace.dispatch_value_refs
            ) != trace.dispatch_value_refs
            or tuple(
                value
                for operation in experts
                for value in operation.output_value_refs
            ) != trace.expert_output_value_refs
            or tuple(combine.input_value_refs) != trace.expert_output_value_refs
            or tuple(combine.output_value_refs) != trace.combine_value_refs
        ):
            raise SchemaError("P3 frozen route value lineage drifted", path=f"{path}.units")

    @staticmethod
    def _validate_parameter_bindings(
        unit,
        operations,
        graph_states,
        graph_values,
        previous_outputs,
        training: bool,
        path: str,
    ) -> None:
        if not training:
            if unit.parameter_bindings:
                raise SchemaError("inference unit cannot claim training state", path=f"{path}.units")
            return
        expected_names = {
            f"layer.{unit.layer}.router.weight",
            *(
                f"layer.{unit.layer}.expert.{expert}.{projection}.weight"
                for expert in range(unit.spec.expert_count)
                for projection in ("gate", "up", "down")
            ),
        }
        observed_names = {
            name for binding in unit.parameter_bindings for name in binding.parameter_refs
        }
        if observed_names != expected_names:
            raise SchemaError("training unit must bind all router/expert parameters", path=f"{path}.units")
        plan_actions = {item.id: item for item in unit.plan.actions}
        plan_states = {item.id: item for item in unit.plan.state_bindings}
        for binding in unit.parameter_bindings:
            for offset, parameter in enumerate(binding.parameter_refs):
                refs = (
                    (binding.gradient_operation_refs[offset],
                     E2EOperationKind.ROUTER_GRADIENT if binding.expert is None else E2EOperationKind.EXPERT_GRADIENT),
                    (binding.sync_operation_refs[offset], E2EOperationKind.GRADIENT_SYNC),
                    (binding.sgd_operation_refs[offset], E2EOperationKind.SGD_UPDATE),
                    (binding.store_operation_refs[offset], E2EOperationKind.PARAMETER_STORE),
                )
                selected = []
                for ref, kind in refs:
                    operation = operations.get(ref)
                    if operation is None or (
                        operation.kind,
                        operation.step,
                        operation.layer,
                        operation.expert,
                        operation.parameter_ref,
                    ) != (
                        kind,
                        binding.step,
                        binding.layer,
                        binding.expert,
                        parameter,
                    ):
                        raise SchemaError("P3 parameter operation binding drifted", path=f"{path}.units")
                    selected.append(operation)
                gradient, sync, sgd, store = selected
                expected_denominator = (
                    unit.spec.expert_parallel_degree
                    if binding.expert is None
                    else 1
                )
                if sync.normalization_denominator != expected_denominator:
                    raise SchemaError("P3 gradient sync group differs from production lowering", path=f"{path}.units")
                expected_edges = (
                    (gradient.writes, (binding.raw_gradient_state_refs[offset],)),
                    (sync.reads, (binding.raw_gradient_state_refs[offset],)),
                    (sync.writes, (binding.synced_gradient_state_refs[offset],)),
                    (sgd.reads, (
                        binding.input_parameter_state_refs[offset],
                        binding.synced_gradient_state_refs[offset],
                    )),
                    (sgd.writes, (binding.output_parameter_state_refs[offset],)),
                    (store.reads, (binding.output_parameter_state_refs[offset],)),
                )
                if any(actual != expected for actual, expected in expected_edges):
                    raise SchemaError("P3 parameter state lineage drifted", path=f"{path}.units")
                input_state = graph_states.get(binding.input_parameter_state_refs[offset])
                output_state = graph_states.get(binding.output_parameter_state_refs[offset])
                if input_state is None or output_state is None or (
                    input_state.kind,
                    output_state.kind,
                    input_state.logical_name,
                    output_state.logical_name,
                    input_state.version,
                    output_state.version,
                ) != (
                    E2EStateKind.PARAMETER,
                    E2EStateKind.PARAMETER,
                    parameter,
                    parameter,
                    binding.step,
                    binding.step + 1,
                ):
                    raise SchemaError("P3 parameter version binding drifted", path=f"{path}.units")
                key = (binding.layer, parameter)
                if binding.step and previous_outputs.get(key) != input_state.id:
                    raise SchemaError("parameter version is not continuous across steps", path=f"{path}.units")
                previous_outputs[key] = output_state.id
            expected_role = (
                MoeRectStateRole.GATE_PARAMETER
                if binding.expert is None
                else MoeRectStateRole.EXPERT_PARAMETER
            )
            production_states = tuple(
                sorted(
                    (item for item in plan_states.values()
                     if item.role is expected_role
                     and (binding.expert is None or item.expert_index == binding.expert)),
                    key=lambda item: item.owner_rank,
                )
            )
            if tuple(item.id for item in production_states) != binding.production_parameter_state_refs:
                raise SchemaError("production parameter state binding drifted", path=f"{path}.units")
            logical_bytes = sum(
                next(
                    value.size_bytes
                    for value in graph_values.values()
                    if value.state_ref == state_ref
                )
                for state_ref in binding.input_parameter_state_refs
            )
            if binding.expert is None:
                expected_bytes = {logical_bytes}
            else:
                expected_bytes = {logical_bytes}
            if (
                not production_states
                or {item.size_bytes for item in production_states}
                != expected_bytes
            ):
                raise SchemaError("production parameter bytes differ from P3 tensors", path=f"{path}.units")
            owner_ranks = {item.owner_rank for item in production_states}
            action_specs = (
                ("production_wgrad_action_refs",
                 MoeRectActionKind.GATE_WGRAD if binding.expert is None else MoeRectActionKind.EXPERT_WGRAD),
                ("production_sync_action_refs",
                 MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE if binding.expert is None else None),
                ("production_sgd_action_refs",
                 MoeRectActionKind.GATE_SGD if binding.expert is None else MoeRectActionKind.EXPERT_SGD),
                ("production_store_action_refs", MoeRectActionKind.STATE_STORE),
            )
            for field, kind in action_specs:
                expected_refs = tuple(sorted(
                    action.id for action in plan_actions.values()
                    if kind is not None
                    and action.kind is kind
                    and action.rank in owner_ranks
                    and (
                        field != "production_store_action_refs"
                        or any(state.id in action.state_refs for state in production_states)
                    )
                ))
                if getattr(binding, field) != expected_refs:
                    raise SchemaError("production action binding drifted", path=f"{path}.units")

    @staticmethod
    def _validate_parameter_reads(
        unit, operations, graph_states, graph_values, path: str
    ) -> None:
        expected_names = {
            f"layer.{unit.layer}.router.weight",
            *(
                f"layer.{unit.layer}.expert.{expert}.{projection}.weight"
                for expert in range(unit.spec.expert_count)
                for projection in ("gate", "up", "down")
            ),
        }
        observed_names = {
            name for binding in unit.parameter_reads for name in binding.parameter_refs
        }
        if observed_names != expected_names:
            raise SchemaError("unit must bind every P3 MoE weight read", path=f"{path}.units")
        plan_states = {item.id: item for item in unit.plan.state_bindings}
        plan_actions = {item.id: item for item in unit.plan.actions}
        for binding in unit.parameter_reads:
            operation_ref = (
                unit.operation_binding.router_operation_ref
                if binding.expert is None
                else unit.operation_binding.expert_forward_operation_refs[binding.expert]
            )
            operation = operations[operation_ref]
            parameter_reads = tuple(
                state_ref
                for state_ref in operation.reads
                if graph_states[state_ref].kind is E2EStateKind.PARAMETER
            )
            if parameter_reads != binding.parameter_state_refs:
                raise SchemaError("P3 MoE weight read lineage drifted", path=f"{path}.units")
            states = tuple(graph_states[ref] for ref in binding.parameter_state_refs)
            expected_version = (
                binding.step
                if unit.spec.mode is FlexibleMoeMode.TRAIN
                else 0
            )
            if (
                tuple(item.logical_name for item in states) != binding.parameter_refs
                or {item.version for item in states} != {expected_version}
            ):
                raise SchemaError("P3 MoE weight version drifted", path=f"{path}.units")
            role = (
                MoeRectStateRole.GATE_PARAMETER
                if binding.expert is None
                else MoeRectStateRole.EXPERT_PARAMETER
            )
            production_states = tuple(sorted(
                (
                    item for item in plan_states.values()
                    if item.role is role
                    and (binding.expert is None or item.expert_index == binding.expert)
                ),
                key=lambda item: item.owner_rank,
            ))
            if tuple(item.id for item in production_states) != binding.production_parameter_state_refs:
                raise SchemaError("production weight state binding drifted", path=f"{path}.units")
            logical_bytes = sum(
                next(
                    value.size_bytes
                    for value in graph_values.values()
                    if value.state_ref == state_ref
                )
                for state_ref in binding.parameter_state_refs
            )
            if (
                not production_states
                or {item.size_bytes for item in production_states} != {logical_bytes}
            ):
                raise SchemaError("production weight bytes differ from P3", path=f"{path}.units")
            owner_ranks = {item.owner_rank for item in production_states}
            expected_loads = tuple(sorted(
                action.id
                for action in plan_actions.values()
                if action.kind is MoeRectActionKind.STATE_LOAD
                and action.rank in owner_ranks
                and any(state.id in action.state_refs for state in production_states)
            ))
            if binding.production_load_action_refs != expected_loads:
                raise SchemaError("production weight load binding drifted", path=f"{path}.units")


__all__ = [
    "MOE_COMPILE_SEQUENCE_SCHEMA_VERSION",
    "MoeCompileCoverage",
    "MoeCompileRuntimeStatus",
    "MoeCompileSequence",
    "MoeCompileStatus",
    "MoeCompileUnit",
    "MoeOperationBinding",
    "MoeParameterBinding",
    "MoeParameterLowering",
    "MoeParameterReadBinding",
]
