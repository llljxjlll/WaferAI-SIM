"""Current-epoch timing evidence shared by the S1-N naive cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .artifact_manifest import LinkedProgramManifest, RecordOpcode, RegionManifest
from .common import stable_artifact_id, validate_nonempty
from .global_action import GlobalActionDAG
from .ir1 import IR1
from .logical import IR0Template
from .n6 import LoweredProgramProfile
from .policy import PolicySelection, RegistryKind
from .program_io import (
    ProgramHbmTarget,
    ProgramIoContract,
    ProgramSramTarget,
)
from .serde import canonical_digest
from .stage2_dense_forward_evidence import (
    Stage2DenseForwardArtifactEvidence,
    Stage2DenseForwardCompileEvidence,
    Stage2DenseForwardControlEvidence,
    Stage2DenseForwardD2DEvidence,
    Stage2DenseForwardMemoryEvidence,
    Stage2DenseForwardProbeEvidence,
    Stage2DenseForwardRepeatEvidence,
    Stage2DenseForwardSidecarEvidence,
    Stage2DenseForwardToolEvidence,
)
from .stage2_dense_forward_oracle import Stage2DenseForwardOracle


S1_N_BASELINE_EPOCH = "s1-n-v1"
S1_N_PREFILL_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.s1_n_prefill_runtime_report/v1alpha1"
)
S1_N_PREFILL_MARKER_SCHEMA_VERSION = (
    "wafer_frontend.s1_n_prefill_runtime_markers/v1"
)


class S1NaivePrefillCase(str, Enum):
    F_P1 = "F-P1"
    F_P2 = "F-P2"


_EXPECTED_POLICIES = (
    (RegistryKind.INTER_DIE, "naive"),
    (RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"),
    (RegistryKind.INTRA_DIE, "naive"),
)


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(
            "must be a canonical lowercase SHA-256 digest", path=path
        )


@dataclass(frozen=True, slots=True)
class S1NaivePolicyEvidence:
    """Exact active registry selections and their consuming contexts."""

    selections: tuple[PolicySelection, ...]
    planning_context_id: str
    scheduling_context_id: str

    def validate(self, path: str = "s1_naive_policy_evidence") -> None:
        if type(self.selections) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.selections"
            )
        actual: list[tuple[RegistryKind, str]] = []
        ids: list[str] = []
        for index, selection in enumerate(self.selections):
            item_path = f"{path}.selections[{index}]"
            if type(selection) is not PolicySelection:
                raise SchemaError("must be a PolicySelection", path=item_path)
            selection.validate(item_path)
            actual.append((selection.kind, selection.name))
            ids.append(selection.id)
        if tuple(actual) != _EXPECTED_POLICIES or len(ids) != len(set(ids)):
            raise SchemaError(
                "must contain the exact canonical production policy selections",
                path=f"{path}.selections",
            )
        validate_nonempty(
            self.planning_context_id, f"{path}.planning_context_id"
        )
        validate_nonempty(
            self.scheduling_context_id, f"{path}.scheduling_context_id"
        )

    def validate_against(
        self,
        planning_context: object,
        scheduling_context: object,
        path: str = "s1_naive_policy_evidence",
    ) -> None:
        from .n4 import InterDiePlanningContext
        from .n5 import IntraDieSchedulingContext

        self.validate(path)
        if type(planning_context) is not InterDiePlanningContext:
            raise SchemaError(
                "must be an InterDiePlanningContext",
                path=f"{path}.planning_context",
            )
        if type(scheduling_context) is not IntraDieSchedulingContext:
            raise SchemaError(
                "must be an IntraDieSchedulingContext",
                path=f"{path}.scheduling_context",
            )
        planning_context.validate(f"{path}.planning_context")
        scheduling_context.validate(f"{path}.scheduling_context")
        expected = (
            planning_context.fused_policy,
            planning_context.standalone_policy,
            scheduling_context.policy,
        )
        if (
            self.selections != expected
            or self.planning_context_id != planning_context.id
            or self.scheduling_context_id != scheduling_context.id
        ):
            raise SchemaError(
                "does not match the supplied production policy contexts",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class S1NaivePrefillRuntimeReport:
    """Current-epoch identity envelope over typed F-P1/F-P2 evidence."""

    schema_version: str
    producer_pass: str
    id: str
    baseline_epoch: str
    case: S1NaivePrefillCase
    tp_degree: int
    oracle_id: str
    oracle_digest: str
    policy: S1NaivePolicyEvidence
    compile: Stage2DenseForwardCompileEvidence
    tools: Stage2DenseForwardToolEvidence
    hardware_digest: str
    simulation_digest: str
    mapping_digest: str
    artifact: Stage2DenseForwardArtifactEvidence
    sidecar: Stage2DenseForwardSidecarEvidence
    memory: tuple[Stage2DenseForwardMemoryEvidence, ...]
    probes: tuple[Stage2DenseForwardProbeEvidence, ...]
    control: Stage2DenseForwardControlEvidence
    d2d: Stage2DenseForwardD2DEvidence
    marker_schema_version: str
    makespan_cycles: int
    repeats: tuple[Stage2DenseForwardRepeatEvidence, ...]
    analytic_observed_exact: bool
    actual_sha_exact: bool
    repeat_exact: bool
    timing_execution: bool
    compute_functional: bool
    model_functional: bool

    @classmethod
    def create(cls, **semantic_key: object) -> "S1NaivePrefillRuntimeReport":
        result = cls(
            schema_version=S1_N_PREFILL_RUNTIME_REPORT_SCHEMA_VERSION,
            producer_pass="s1_n_prefill_runtime",
            id=stable_artifact_id(
                "s1_n_prefill_runtime_report",
                semantic_key,
                schema_version=S1_N_PREFILL_RUNTIME_REPORT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "s1_n_prefill_runtime_report") -> None:
        if self.schema_version != S1_N_PREFILL_RUNTIME_REPORT_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "s1_n_prefill_runtime":
            raise SchemaError(
                "must be 's1_n_prefill_runtime'", path=f"{path}.producer_pass"
            )
        if self.baseline_epoch != S1_N_BASELINE_EPOCH:
            raise SchemaError(
                f"must be {S1_N_BASELINE_EPOCH!r}",
                path=f"{path}.baseline_epoch",
            )
        if type(self.case) is not S1NaivePrefillCase:
            raise SchemaError(
                "must be a S1NaivePrefillCase", path=f"{path}.case"
            )
        expected_tp = {
            S1NaivePrefillCase.F_P1: 1,
            S1NaivePrefillCase.F_P2: 2,
        }[self.case]
        if type(self.tp_degree) is not int or self.tp_degree != expected_tp:
            raise SchemaError(
                f"must be exactly {expected_tp}", path=f"{path}.tp_degree"
            )
        for field_name in (
            "oracle_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        for field_name in (
            "oracle_digest",
            "hardware_digest",
            "simulation_digest",
            "mapping_digest",
        ):
            _digest(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.policy) is not S1NaivePolicyEvidence:
            raise SchemaError(
                "must be a S1NaivePolicyEvidence", path=f"{path}.policy"
            )
        self.policy.validate(f"{path}.policy")
        for field_name, expected_type in (
            ("compile", Stage2DenseForwardCompileEvidence),
            ("tools", Stage2DenseForwardToolEvidence),
            ("artifact", Stage2DenseForwardArtifactEvidence),
            ("sidecar", Stage2DenseForwardSidecarEvidence),
            ("control", Stage2DenseForwardControlEvidence),
            ("d2d", Stage2DenseForwardD2DEvidence),
        ):
            value = getattr(self, field_name)
            if type(value) is not expected_type:
                raise SchemaError(
                    f"must be a {expected_type.__name__}",
                    path=f"{path}.{field_name}",
                )
            value.validate(f"{path}.{field_name}")
        self._validate_runtime_tuples(path)
        if self.marker_schema_version != S1_N_PREFILL_MARKER_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported marker schema version",
                path=f"{path}.marker_schema_version",
            )
        if type(self.makespan_cycles) is not int or self.makespan_cycles <= 0:
            raise SchemaError(
                "must be a positive integer", path=f"{path}.makespan_cycles"
            )
        for field_name in (
            "analytic_observed_exact",
            "actual_sha_exact",
            "repeat_exact",
            "timing_execution",
            "compute_functional",
            "model_functional",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise SchemaError("must be a bool", path=f"{path}.{field_name}")
        if (
            not self.analytic_observed_exact
            or not self.actual_sha_exact
            or not self.repeat_exact
            or not self.timing_execution
            or self.compute_functional
            or self.model_functional
        ):
            raise SchemaError("proof is exact timing/accounting only", path=path)
        expected_id = stable_artifact_id(
            "s1_n_prefill_runtime_report",
            self._semantic_key(),
            schema_version=S1_N_PREFILL_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def _validate_runtime_tuples(self, path: str) -> None:
        typed = (
            ("memory", self.memory, Stage2DenseForwardMemoryEvidence,
             lambda item: item.runtime_core_id),
            ("probes", self.probes, Stage2DenseForwardProbeEvidence,
             lambda item: item.probe_id),
            ("repeats", self.repeats, Stage2DenseForwardRepeatEvidence,
             lambda item: item.run_index),
        )
        for name, values, expected_type, key in typed:
            if type(values) is not tuple:
                raise SchemaError(
                    "must be an immutable tuple", path=f"{path}.{name}"
                )
            keys = []
            for index, item in enumerate(values):
                if type(item) is not expected_type:
                    raise SchemaError(
                        f"must be a {expected_type.__name__}",
                        path=f"{path}.{name}[{index}]",
                    )
                item.validate(f"{path}.{name}[{index}]")
                keys.append(key(item))
            if keys != sorted(set(keys)):
                raise SchemaError(
                    "must be unique and canonical", path=f"{path}.{name}"
                )
        if len(self.memory) != self.tp_degree or len(self.probes) != self.tp_degree:
            raise SchemaError(
                "must cover one memory core and terminal probe per rank", path=path
            )
        if len(self.repeats) != 2:
            raise SchemaError(
                "must contain exactly two repeats", path=f"{path}.repeats"
            )
        digests = (
            canonical_digest(self.memory),
            canonical_digest(self.probes),
            canonical_digest(self.control),
            canonical_digest(self.d2d),
        )
        marker_digest = self.repeats[0].marker_digest
        for index, repeat in enumerate(self.repeats):
            if (
                repeat.run_index != index
                or repeat.makespan_cycles != self.makespan_cycles
                or (
                    repeat.memory_digest,
                    repeat.probe_digest,
                    repeat.control_digest,
                    repeat.d2d_digest,
                ) != digests
                or repeat.marker_digest != marker_digest
            ):
                raise SchemaError(
                    "repeat does not reproduce canonical runtime evidence",
                    path=f"{path}.repeats[{index}]",
                )

    def validate_against(
        self,
        *,
        oracle: Stage2DenseForwardOracle,
        template: IR0Template,
        graph: IR1,
        global_dag: GlobalActionDAG,
        lowered: LoweredProgramProfile,
        manifest: LinkedProgramManifest,
        program_io: ProgramIoContract,
        planning_context: object,
        scheduling_context: object,
        path: str = "s1_n_prefill_runtime_report",
    ) -> None:
        self.validate(path)
        if type(oracle) is not Stage2DenseForwardOracle:
            raise SchemaError(
                "must be a Stage2DenseForwardOracle", path=f"{path}.oracle"
            )
        oracle.validate_against_template(template)
        if type(graph) is not IR1 or type(global_dag) is not GlobalActionDAG:
            raise SchemaError("must receive exact IR1/GlobalAction inputs", path=path)
        graph.validate(f"{path}.graph")
        global_dag.validate(f"{path}.global_dag")
        if type(lowered) is not LoweredProgramProfile:
            raise SchemaError(
                "must be a LoweredProgramProfile", path=f"{path}.lowered"
            )
        lowered.validate(f"{path}.lowered")
        context = lowered.lowering_context
        manifest.validate_against(
            context.ir1,
            context.fusion_plans,
            context.standalone_plans,
            context.projection,
            context.schedule_set,
            context.global_dag,
            lowered.fragments,
        )
        program_io.validate_against(manifest)
        self.policy.validate_against(planning_context, scheduling_context,
                                     f"{path}.policy")
        if (
            self.oracle_id != oracle.id
            or self.oracle_digest != canonical_digest(oracle)
            or self.tp_degree != oracle.tp_degree
            or self.compile.template_id != template.id
            or self.compile.template_digest != canonical_digest(template)
            or self.compile.ir1_id != graph.id
            or self.compile.ir1_digest != canonical_digest(graph)
            or self.compile.global_dag_id != global_dag.id
            or self.compile.global_dag_digest != canonical_digest(global_dag)
            or self.compile.lowered_id != lowered.id
            or self.compile.lowered_digest != canonical_digest(lowered)
        ):
            raise SchemaError("compile/oracle provenance differs", path=path)
        leaves = tuple(
            item.fragment if type(item) is RegionManifest else item
            for item in manifest.fragments
        )
        records = tuple(
            record
            for leaf in leaves
            for stream in leaf.core_streams
            for record in stream.records
        )
        relocations = sum(
            len(stream.address_relocations) + len(stream.runtime_relocations)
            for leaf in leaves
            for stream in leaf.core_streams
        )
        opcode_counts = {
            item.opcode: item.count for item in self.artifact.opcode_counts
        }
        observed_opcodes = {
            opcode: sum(record.opcode is opcode for record in records)
            for opcode in RecordOpcode
            if any(record.opcode is opcode for record in records)
        }
        if (
            self.artifact.linked_manifest_id != manifest.id
            or self.artifact.linked_manifest_digest != canonical_digest(manifest)
            or self.artifact.action_count != len(global_dag.actions)
            or self.artifact.leaf_fragment_count != len(leaves)
            or self.artifact.record_count != len(records)
            or self.artifact.address_binding_count
            != len(manifest.address_operand_bindings)
            or self.artifact.relocation_count != relocations
            or opcode_counts != observed_opcodes
        ):
            raise SchemaError("artifact structure differs", path=f"{path}.artifact")
        hbm_init = sum(
            type(item.target) is ProgramHbmTarget
            for item in program_io.initializations
        )
        sram_init = sum(
            type(item.target) is ProgramSramTarget
            for item in program_io.initializations
        )
        hbm_probe = sum(
            type(item.target) is ProgramHbmTarget
            for item in program_io.output_probes
        )
        sram_probe = sum(
            type(item.target) is ProgramSramTarget
            for item in program_io.output_probes
        )
        if (
            self.sidecar.contract_id != program_io.id
            or self.sidecar.contract_digest != canonical_digest(program_io)
            or self.sidecar.mode is not program_io.mode
            or (
                self.sidecar.hbm_initialization_count,
                self.sidecar.sram_initialization_count,
                self.sidecar.hbm_probe_count,
                self.sidecar.sram_probe_count,
            ) != (hbm_init, sram_init, hbm_probe, sram_probe)
            or program_io.program_artifact_sha256
            != self.artifact.program_artifact_sha256
        ):
            raise SchemaError("ProgramIo/artifact closure differs", path=path)
        hbm_read = sum(item.lsu_hbm_read_bytes for item in self.memory)
        hbm_write = sum(item.lsu_hbm_write_bytes for item in self.memory)
        collective_bytes = (
            oracle.collectives.all_gather.group_payload_bytes_total
            + oracle.collectives.reduce_scatter.group_payload_bytes_total
        )
        if (
            hbm_read != oracle.parameters.placed_bytes
            or hbm_write != oracle.kv.logical_write_bytes
            or self.d2d.logical_bytes != collective_bytes
            or hbm_init != oracle.graph.parameter_declaration_count
            or opcode_counts.get(RecordOpcode.LSU_LOAD, 0)
            != oracle.graph.parameter_declaration_count
            or opcode_counts.get(RecordOpcode.LSU_STORE, 0)
            != oracle.graph.kv_declaration_count
        ):
            raise SchemaError(
                "analytic and observed traffic differ", path=path
            )


__all__ = [
    "S1_N_BASELINE_EPOCH",
    "S1_N_PREFILL_MARKER_SCHEMA_VERSION",
    "S1_N_PREFILL_RUNTIME_REPORT_SCHEMA_VERSION",
    "S1NaivePolicyEvidence",
    "S1NaivePrefillCase",
    "S1NaivePrefillRuntimeReport",
]
