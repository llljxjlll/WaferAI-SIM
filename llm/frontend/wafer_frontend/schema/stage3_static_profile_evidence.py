"""Strict runtime evidence for Stage 3 static Dense inference profiles."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import RecordOpcode
from .capability import CapabilityStatus
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .experiment import InferOutput
from .program_io import ProgramIoMode, ProgramIoTargetKind
from .serde import canonical_digest
from .s1_naive_evidence import S1_N_BASELINE_EPOCH, S1NaivePolicyEvidence
from .stage2_dense_forward_evidence import (
    Stage2DenseForwardArtifactEvidence,
    Stage2DenseForwardCompileEvidence,
    Stage2DenseForwardControlEvidence,
    Stage2DenseForwardCoreCount,
    Stage2DenseForwardD2DEvidence,
    Stage2DenseForwardMemoryEvidence,
    Stage2DenseForwardOpcodeCount,
    Stage2DenseForwardProbeEvidence,
    Stage2DenseForwardRepeatEvidence,
    Stage2DenseForwardSidecarEvidence,
    Stage2DenseForwardToolEvidence,
)
from .stage3_dense_inference_oracle import Stage3DenseInferenceOracle
from .stage3_profile import Stage3ProfileMode


STAGE3_STATIC_PROFILE_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.stage3_static_profile_runtime_report/v1alpha2"
)
STAGE3_STATIC_PROFILE_MARKER_SCHEMA_VERSION = (
    "wafer_frontend.stage3_static_profile_runtime_markers/v1"
)
STAGE3_STATIC_PROFILE_BASELINE_EPOCH = S1_N_BASELINE_EPOCH


def _validate_digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(
            "must be a canonical lowercase SHA-256 digest", path=path
        )


def _validate_bool(value: bool, path: str) -> None:
    if type(value) is not bool:
        raise SchemaError("must be a bool", path=path)


def _opcode_counts(**values: int) -> tuple[Stage2DenseForwardOpcodeCount, ...]:
    return tuple(
        Stage2DenseForwardOpcodeCount(RecordOpcode[name], count)
        for name, count in sorted(
            values.items(), key=lambda item: int(RecordOpcode[item[0]])
        )
    )


_ZERO_D2D = Stage2DenseForwardD2DEvidence(0, 0, 0, 0, 0, 0, 0, ())
_CASE_GOLDENS = {
    Stage3ProfileMode.PREFILL: {
        "artifact": (
            22258,
            44,
            44,
            159,
            278,
            297,
            "4b1204fa698553e202d97a77dc3fdb4f60f38051b1a82a4dca84ec4981d76777",
        ),
        "opcodes": _opcode_counts(
            ATTENTION_EXACT=2,
            EMBEDDING_LOOKUP=1,
            LSU_LOAD=15,
            LSU_STORE=4,
            MATMUL=9,
            RESIDUAL=4,
            RMSNORM=5,
            ROPE_QK_EXACT=2,
            SRAM_ALLOC_AT=45,
            SRAM_BIND=25,
            SRAM_FREE=45,
            SWIGLU=2,
        ),
        "sidecar": (15, 45, 0, 1),
        "memory": (
            Stage2DenseForwardMemoryEvidence(
                0, 19, 19, 12448, 1024, 1024, 12448, 0, 0
            ),
        ),
        "attention": (2, 8, 8, 8, 36, 0, 512),
        "makespan": 5805,
        "marker_digest": (
            "f8fa4a291d691110f634368275564ad99e8f62e606b9e59bfbe8993e4f7ed106"
        ),
    },
    Stage3ProfileMode.MIXED: {
        "artifact": (
            32714,
            80,
            80,
            235,
            374,
            429,
            "60af99215ab929d5c107a5f09aaebdfde6a9bed53f1ccdac317ac9ff6a7ad22e",
        ),
        "opcodes": _opcode_counts(
            ATTENTION_EXACT=2,
            EMBEDDING_LOOKUP=1,
            LSU_LOAD=31,
            LSU_STORE=24,
            MATMUL=9,
            RESIDUAL=4,
            RMSNORM=5,
            ROPE_QK_EXACT=2,
            SRAM_ALLOC_AT=65,
            SRAM_BIND=25,
            SRAM_FREE=65,
            SWIGLU=2,
        ),
        "sidecar": (31, 65, 16, 1),
        "memory": (
            Stage2DenseForwardMemoryEvidence(
                0, 55, 55, 17568, 1024, 1024, 17568, 0, 0
            ),
        ),
        "attention": (2, 8, 44, 16, 47, 2560, 512),
        "makespan": 8807,
        "marker_digest": (
            "8ab5b8d1a1efb23a899278d309d27c3ec9caa63b16f9ba64f528a165dfad060f"
        ),
    },
    Stage3ProfileMode.DECODE: {
        "artifact": (
            37818,
            104,
            104,
            275,
            422,
            501,
            "8cd6f01aada63d8091b7ceb9c70726c7287010d63a5b8c2362a905cf51cc5543",
        ),
        "opcodes": _opcode_counts(
            ATTENTION_EXACT=2,
            EMBEDDING_LOOKUP=1,
            LSU_LOAD=47,
            LSU_STORE=32,
            MATMUL=9,
            RESIDUAL=4,
            RMSNORM=5,
            ROPE_QK_EXACT=2,
            SRAM_ALLOC_AT=73,
            SRAM_BIND=25,
            SRAM_FREE=73,
            SWIGLU=2,
        ),
        "sidecar": (47, 73, 32, 1),
        "memory": (
            Stage2DenseForwardMemoryEvidence(
                0, 79, 79, 30880, 1024, 1024, 30880, 0, 0
            ),
        ),
        "attention": (2, 8, 144, 32, 144, 9216, 512),
        "makespan": 13081,
        "marker_digest": (
            "00d6dde839fcc3a3fd1a4b24b42808ec8b325fba97f743746519f38a2eef2678"
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Stage3StaticAttentionEvidence:
    record_count: int
    query_tokens_per_record: int
    context_sum_per_record: int
    context_max_per_record: int
    query_key_pairs_per_record: int
    rank_kv_read_bytes_per_record: int
    rank_kv_write_bytes_per_record: int

    def validate(self, path: str = "stage3_static_attention_evidence") -> None:
        for field_name in self.__dataclass_fields__:
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if (
            self.record_count == 0
            or self.query_tokens_per_record == 0
            or self.context_sum_per_record == 0
            or self.context_max_per_record == 0
            or self.query_key_pairs_per_record == 0
            or self.rank_kv_write_bytes_per_record == 0
        ):
            raise SchemaError(
                "profile Attention evidence must be non-zero except KV read",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class Stage3StaticProfileRuntimeReport:
    schema_version: str
    producer_pass: str
    id: str
    baseline_epoch: str
    profile_mode: Stage3ProfileMode
    tp_degree: int
    infer_output: InferOutput
    capability_status: CapabilityStatus
    static_profile_id: str
    static_profile_digest: str
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
    attention: Stage3StaticAttentionEvidence
    memory: tuple[Stage2DenseForwardMemoryEvidence, ...]
    probes: tuple[Stage2DenseForwardProbeEvidence, ...]
    control: Stage2DenseForwardControlEvidence
    d2d: Stage2DenseForwardD2DEvidence
    marker_schema_version: str
    repeat_count: int
    makespan_cycles: int
    repeats: tuple[Stage2DenseForwardRepeatEvidence, ...]
    timing_execution: bool
    dense_forward_structure_exact: bool
    static_request_shape_exact: bool
    analytic_work_exact: bool
    program_io_boundary_exact: bool
    traffic_accounting_exact: bool
    compute_functional: bool
    model_functional: bool

    @classmethod
    def create(cls, **semantic_key: object) -> "Stage3StaticProfileRuntimeReport":
        result = cls(
            schema_version=STAGE3_STATIC_PROFILE_RUNTIME_REPORT_SCHEMA_VERSION,
            producer_pass="stage3_static_profile_runtime",
            id=stable_artifact_id(
                "stage3_static_profile_runtime_report",
                semantic_key,
                schema_version=(
                    STAGE3_STATIC_PROFILE_RUNTIME_REPORT_SCHEMA_VERSION
                ),
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

    def validate(
        self, path: str = "stage3_static_profile_runtime_report"
    ) -> None:
        if (
            self.schema_version
            != STAGE3_STATIC_PROFILE_RUNTIME_REPORT_SCHEMA_VERSION
        ):
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "stage3_static_profile_runtime":
            raise SchemaError(
                "must be 'stage3_static_profile_runtime'",
                path=f"{path}.producer_pass",
            )
        if self.baseline_epoch != STAGE3_STATIC_PROFILE_BASELINE_EPOCH:
            raise SchemaError(
                f"must be {STAGE3_STATIC_PROFILE_BASELINE_EPOCH!r}",
                path=f"{path}.baseline_epoch",
            )
        if type(self.profile_mode) is not Stage3ProfileMode:
            raise SchemaError(
                "must be a Stage3ProfileMode", path=f"{path}.profile_mode"
            )
        golden = _CASE_GOLDENS[self.profile_mode]
        if self.tp_degree != 1:
            raise SchemaError(
                "reviewed static-profile evidence requires TP1",
                path=f"{path}.tp_degree",
            )
        if self.infer_output is not InferOutput.LOGITS:
            raise SchemaError(
                "reviewed evidence covers LOGITS only",
                path=f"{path}.infer_output",
            )
        if self.capability_status is not CapabilityStatus.E2E_TIMING:
            raise SchemaError(
                "static-profile evidence is timing-only",
                path=f"{path}.capability_status",
            )
        for field_name in ("static_profile_id", "oracle_id"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        for field_name in (
            "static_profile_digest",
            "oracle_digest",
            "hardware_digest",
            "simulation_digest",
            "mapping_digest",
        ):
            _validate_digest(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.policy) is not S1NaivePolicyEvidence:
            raise SchemaError(
                "must be an S1NaivePolicyEvidence",
                path=f"{path}.policy",
            )
        self.policy.validate(f"{path}.policy")
        from .n4 import InterDiePlanningContext
        from .n5 import IntraDieSchedulingContext

        planning_context = InterDiePlanningContext.create(
            producer_pass="stage3_static_profile_evidence",
            fused_policy=self.policy.selections[0],
            standalone_policy=self.policy.selections[1],
        )
        scheduling_context = IntraDieSchedulingContext.create(
            producer_pass="stage3_static_profile_evidence",
            policy=self.policy.selections[2],
        )
        self.policy.validate_against(
            planning_context,
            scheduling_context,
            f"{path}.policy",
        )
        for field_name, expected_type in (
            ("compile", Stage2DenseForwardCompileEvidence),
            ("tools", Stage2DenseForwardToolEvidence),
            ("artifact", Stage2DenseForwardArtifactEvidence),
            ("sidecar", Stage2DenseForwardSidecarEvidence),
            ("attention", Stage3StaticAttentionEvidence),
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
        if self.memory != golden["memory"]:
            raise SchemaError(
                "memory evidence disagrees with the frozen profile",
                path=f"{path}.memory",
            )
        for index, item in enumerate(self.memory):
            if type(item) is not Stage2DenseForwardMemoryEvidence:
                raise SchemaError(
                    "must be a Stage2DenseForwardMemoryEvidence",
                    path=f"{path}.memory[{index}]",
                )
            item.validate(f"{path}.memory[{index}]")
        expected_probe_types = {
            Stage3ProfileMode.PREFILL: (0, 1),
            Stage3ProfileMode.MIXED: (16, 1),
            Stage3ProfileMode.DECODE: (32, 1),
        }[self.profile_mode]
        hbm_probes = 0
        sram_probes = 0
        probe_ids: list[str] = []
        for index, probe in enumerate(self.probes):
            if type(probe) is not Stage2DenseForwardProbeEvidence:
                raise SchemaError(
                    "must be a Stage2DenseForwardProbeEvidence",
                    path=f"{path}.probes[{index}]",
                )
            probe.validate(f"{path}.probes[{index}]")
            probe_ids.append(probe.probe_id)
            if probe.target_kind is ProgramIoTargetKind.HBM:
                hbm_probes += 1
            elif probe.target_kind is ProgramIoTargetKind.SRAM:
                sram_probes += 1
        if (
            probe_ids != sorted(set(probe_ids))
            or (hbm_probes, sram_probes) != expected_probe_types
        ):
            raise SchemaError(
                "probe evidence disagrees with the frozen profile",
                path=f"{path}.probes",
            )
        artifact_values = (
            self.artifact.artifact_size_bytes,
            self.artifact.action_count,
            self.artifact.leaf_fragment_count,
            self.artifact.record_count,
            self.artifact.address_binding_count,
            self.artifact.relocation_count,
            self.artifact.program_artifact_sha256,
        )
        if (
            artifact_values != golden["artifact"]
            or self.artifact.opcode_counts != golden["opcodes"]
        ):
            raise SchemaError(
                "artifact evidence disagrees with the frozen profile",
                path=f"{path}.artifact",
            )
        sidecar_values = (
            self.sidecar.hbm_initialization_count,
            self.sidecar.sram_initialization_count,
            self.sidecar.hbm_probe_count,
            self.sidecar.sram_probe_count,
        )
        if (
            self.sidecar.mode is not ProgramIoMode.TIMING
            or sidecar_values != golden["sidecar"]
        ):
            raise SchemaError(
                "sidecar evidence disagrees with the frozen profile",
                path=f"{path}.sidecar",
            )
        attention_values = tuple(
            getattr(self.attention, name)
            for name in self.attention.__dataclass_fields__
        )
        if attention_values != golden["attention"]:
            raise SchemaError(
                "Attention evidence disagrees with the frozen profile",
                path=f"{path}.attention",
            )
        if self.d2d != _ZERO_D2D:
            raise SchemaError(
                "reviewed TP1 profiles require exact zero D2D",
                path=f"{path}.d2d",
            )
        expected_ack = (Stage2DenseForwardCoreCount(0, 2),)
        expected_done = (Stage2DenseForwardCoreCount(0, 1),)
        if (
            self.control.ack_counts != expected_ack
            or self.control.done_counts != expected_done
        ):
            raise SchemaError(
                "control evidence disagrees with TP1",
                path=f"{path}.control",
            )
        if (
            self.marker_schema_version
            != STAGE3_STATIC_PROFILE_MARKER_SCHEMA_VERSION
        ):
            raise SchemaError(
                "unsupported marker schema version",
                path=f"{path}.marker_schema_version",
            )
        if (
            self.repeat_count != 2
            or len(self.repeats) != 2
            or self.makespan_cycles != golden["makespan"]
        ):
            raise SchemaError(
                "reviewed evidence requires two exact repeats",
                path=f"{path}.repeats",
            )
        memory_digest = canonical_digest(self.memory)
        probe_digest = canonical_digest(self.probes)
        control_digest = canonical_digest(self.control)
        d2d_digest = canonical_digest(self.d2d)
        for index, repeat in enumerate(self.repeats):
            if type(repeat) is not Stage2DenseForwardRepeatEvidence:
                raise SchemaError(
                    "must be a Stage2DenseForwardRepeatEvidence",
                    path=f"{path}.repeats[{index}]",
                )
            repeat.validate(f"{path}.repeats[{index}]")
            if (
                repeat.run_index != index
                or repeat.makespan_cycles != self.makespan_cycles
                or repeat.marker_digest != golden["marker_digest"]
                or repeat.memory_digest != memory_digest
                or repeat.probe_digest != probe_digest
                or repeat.control_digest != control_digest
                or repeat.d2d_digest != d2d_digest
            ):
                raise SchemaError(
                    "repeat does not reproduce canonical profile evidence",
                    path=f"{path}.repeats[{index}]",
                )
        for field_name in (
            "timing_execution",
            "dense_forward_structure_exact",
            "static_request_shape_exact",
            "analytic_work_exact",
            "program_io_boundary_exact",
            "traffic_accounting_exact",
            "compute_functional",
            "model_functional",
        ):
            _validate_bool(getattr(self, field_name), f"{path}.{field_name}")
        if (
            not self.timing_execution
            or not self.dense_forward_structure_exact
            or not self.static_request_shape_exact
            or not self.analytic_work_exact
            or not self.program_io_boundary_exact
            or not self.traffic_accounting_exact
            or self.compute_functional
            or self.model_functional
        ):
            raise SchemaError(
                "proof is exact static-profile timing/accounting only", path=path
            )
        expected_id = stable_artifact_id(
            "stage3_static_profile_runtime_report",
            self._semantic_key(),
            schema_version=STAGE3_STATIC_PROFILE_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        oracle: Stage3DenseInferenceOracle,
        path: str = "stage3_static_profile_runtime_report",
    ) -> None:
        self.validate(path)
        if type(oracle) is not Stage3DenseInferenceOracle:
            raise SchemaError(
                "must be a Stage3DenseInferenceOracle",
                path=f"{path}.oracle",
            )
        oracle.validate("stage3_dense_inference_oracle")
        profile = oracle.static_profile
        if (
            self.profile_mode is not profile.mode
            or self.static_profile_id != profile.id
            or self.static_profile_digest != canonical_digest(profile)
            or self.oracle_id != oracle.id
            or self.oracle_digest != canonical_digest(oracle)
            or self.tp_degree != oracle.tp_degree
            or self.infer_output is not oracle.infer_output
            or self.compile.template_id != oracle.source_template_id
        ):
            raise SchemaError(
                "does not identify the supplied oracle/profile", path=path
            )
        hbm_read = sum(item.lsu_hbm_read_bytes for item in self.memory)
        hbm_write = sum(item.lsu_hbm_write_bytes for item in self.memory)
        if (
            hbm_read
            != oracle.parameters.placed_bytes + oracle.kv.logical_read_bytes
            or hbm_write != oracle.kv.logical_write_bytes
        ):
            raise SchemaError(
                "HBM traffic disagrees with parameter/KV oracle",
                path=f"{path}.memory",
            )
        expected_d2d = (
            oracle.collectives.all_gather.group_payload_bytes_total
            + oracle.collectives.reduce_scatter.group_payload_bytes_total
        )
        if self.d2d.logical_bytes != expected_d2d:
            raise SchemaError(
                "D2D bytes disagree with collective oracle",
                path=f"{path}.d2d.logical_bytes",
            )
        request_count = len(profile.requests)
        if oracle.graph.kv_declaration_count % request_count:
            raise SchemaError(
                "KV declarations do not partition requests", path=f"{path}.oracle"
            )
        states_per_request = oracle.graph.kv_declaration_count // request_count
        read_state_count = states_per_request * sum(
            item.decode_tokens > 0 for item in profile.requests
        )
        opcode_counts = {
            item.opcode: item.count for item in self.artifact.opcode_counts
        }
        if (
            opcode_counts.get(RecordOpcode.LSU_LOAD, 0)
            != oracle.graph.parameter_declaration_count + read_state_count
            or opcode_counts.get(RecordOpcode.LSU_STORE, 0)
            != oracle.graph.kv_declaration_count
            or self.sidecar.hbm_initialization_count
            != oracle.graph.parameter_declaration_count + read_state_count
            or self.sidecar.hbm_probe_count != read_state_count
        ):
            raise SchemaError(
                "runtime state counts disagree with profile oracle", path=path
            )
        if (
            self.attention.query_tokens_per_record
            != profile.key.prefill_tokens + profile.key.decode_tokens
            or self.attention.context_sum_per_record != profile.key.context_sum
            or self.attention.context_max_per_record != profile.key.context_max
            or self.attention.record_count == 0
            or self.attention.query_key_pairs_per_record
            * self.attention.record_count
            != oracle.logical_work.attention.query_key_pairs
            or self.attention.rank_kv_read_bytes_per_record
            * self.attention.record_count
            != oracle.kv.rank_read_bytes
            or self.attention.rank_kv_write_bytes_per_record
            * self.attention.record_count
            != oracle.kv.rank_write_bytes
        ):
            raise SchemaError(
                "Attention evidence disagrees with profile oracle",
                path=f"{path}.attention",
            )


__all__ = [
    "STAGE3_STATIC_PROFILE_BASELINE_EPOCH",
    "STAGE3_STATIC_PROFILE_MARKER_SCHEMA_VERSION",
    "STAGE3_STATIC_PROFILE_RUNTIME_REPORT_SCHEMA_VERSION",
    "Stage3StaticAttentionEvidence",
    "Stage3StaticProfileRuntimeReport",
]
