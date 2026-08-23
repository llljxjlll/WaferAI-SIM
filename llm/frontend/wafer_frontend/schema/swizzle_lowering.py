"""Independent timing-only W9 lowering and linked-manifest carriers.

These carriers intentionally stop before ``CommandFragment``.  The production
Swizzle projection has exact rank/Die tasks and buffers, but no intra-Die core
schedule, address allocation, or runtime token/FSM bindings.  Reusing the
existing opcode enum is lossless; fabricating relocatable operands is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .artifact_manifest import RecordOpcode
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import FusionPattern
from .serde import canonical_digest
from .swizzle import SwizzleActionKind, SwizzleAlgorithm
from .swizzle_ir2 import (
    SwizzleIr2ConsumerContract,
    SwizzleIr2Projection,
    SwizzleIr2Task,
)
from .swizzle_plan import SwizzleBoundAction, SwizzleFusionPlan


SWIZZLE_LOWERED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_lowered_program/v1alpha1"
)
SWIZZLE_OPCODE_RECORD_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_opcode_record/v1alpha1"
)
SWIZZLE_LINKED_MANIFEST_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_linked_manifest/v1alpha1"
)


class SwizzleManifestInputKind(str, Enum):
    CANDIDATE = "candidate"
    DECISION = "decision"
    FUSION_PLAN = "fusion_plan"
    LOWERED_PROGRAM = "lowered_program"
    PROJECTION = "projection"


class SwizzleFinalizerContract(str, Enum):
    REQUIRES_EXACT_CORE_ADDRESS_ABI_V1 = "requires_exact_core_address_abi/v1"


@dataclass(frozen=True, slots=True)
class SwizzleManifestInputDigest:
    kind: SwizzleManifestInputKind
    artifact_id: str
    schema_version: str
    digest: str

    def validate(self, path: str) -> None:
        if type(self.kind) is not SwizzleManifestInputKind:
            raise SchemaError("must be a SwizzleManifestInputKind", path=f"{path}.kind")
        validate_nonempty(self.artifact_id, f"{path}.artifact_id")
        validate_nonempty(self.schema_version, f"{path}.schema_version")
        if (
            type(self.digest) is not str
            or len(self.digest) != 64
            or any(character not in "0123456789abcdef" for character in self.digest)
        ):
            raise SchemaError("must be a lowercase SHA-256 digest", path=f"{path}.digest")


def _bound_action_index(plan: SwizzleFusionPlan) -> dict[str, SwizzleBoundAction]:
    return {
        action.source_action.id: action
        for program in plan.rank_programs
        for action in program.actions
    }


def validate_swizzle_plan_projection(
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
    path: str = "swizzle_lowering_input",
) -> None:
    """Prove that a production W8 projection is the exact quotient of W7."""

    if type(plan) is not SwizzleFusionPlan:
        raise SchemaError("must be a SwizzleFusionPlan", path=f"{path}.plan")
    if type(projection) is not SwizzleIr2Projection:
        raise SchemaError("must be a SwizzleIr2Projection", path=f"{path}.projection")
    plan.validate(f"{path}.plan")
    projection.validate(f"{path}.projection")
    projection.require_consumer(SwizzleIr2ConsumerContract.STRICT_SWIZZLE_TIMING_V1)
    if (
        projection.source_ir1_id,
        projection.source_decision_ref,
        projection.source_candidate_ref,
        projection.fused_op_id,
        projection.group_ref,
        projection.pattern,
        projection.algorithm,
        projection.split_axis,
        projection.chunk_count,
        projection.unroll_degree,
    ) != (
        plan.source_ir1_id,
        plan.decision.id,
        plan.candidate.id,
        plan.fused_op_id,
        plan.group_ref,
        plan.pattern,
        plan.algorithm,
        plan.candidate.split_axis,
        plan.candidate.chunk_count,
        plan.candidate.unroll_degree,
    ):
        raise SchemaError("plan/projection provenance is not exact", path=path)

    actions = _bound_action_index(plan)
    tasks = {
        task.source_action_ref: task
        for dag in projection.rank_dags
        for task in dag.tasks
    }
    if set(actions) != set(tasks):
        raise SchemaError("every plan action must map to one projection task", path=path)
    task_id_by_action = {
        task.source_action_ref: task.id for task in tasks.values()
    }
    for action_ref, bound in actions.items():
        source = bound.source_action
        task = tasks[action_ref]
        if (
            task.rank,
            task.kind,
            task.phase,
            task.member_ref,
            task.chunk_index,
            task.deps,
            task.peer_rank,
            task.route_ref,
            task.expected_route,
            task.logical_bytes,
            task.flops,
        ) != (
            source.rank,
            source.kind,
            source.phase,
            bound.member_ref,
            source.chunk_index,
            tuple(task_id_by_action[ref] for ref in source.deps),
            source.peer_rank,
            source.route_ref,
            bound.expected_route,
            source.logical_bytes,
            source.flops,
        ):
            raise SchemaError(
                "projection task drifts from its bound plan action",
                path=f"{path}.projection.task.{task.id}",
            )


def expected_swizzle_opcodes(
    action: SwizzleBoundAction,
    task: SwizzleIr2Task,
) -> tuple[RecordOpcode, ...]:
    """Map one exact action to existing opcodes without constructing operands."""

    if action.source_action.id != task.source_action_ref:
        raise SchemaError("task/action ref mismatch", path="swizzle_opcode_record")
    mapping = {
        SwizzleActionKind.COMP: (RecordOpcode.MATMUL,),
        SwizzleActionKind.SEND: (RecordOpcode.DTE_SEND,),
        SwizzleActionKind.RECV: (RecordOpcode.DTE_RECV,),
        SwizzleActionKind.WAIT: (RecordOpcode.DTE_WAIT,),
        SwizzleActionKind.REDUCE: (RecordOpcode.LOCAL_REDUCE,),
        SwizzleActionKind.LOCAL_COPY: (
            RecordOpcode.DTE_ISSUE,
            RecordOpcode.DTE_WAIT,
        ),
    }
    if task.kind is not SwizzleActionKind.BARRIER:
        return mapping[task.kind]
    barrier = action.sync.barrier
    if (
        barrier is None
        or len(barrier.participant_ranks) < 2
        or task.rank not in barrier.participant_ranks
    ):
        raise SchemaError(
            "BARRIER requires an exact multi-rank sync contract",
            path="swizzle_opcode_record.barrier",
        )
    if task.rank == barrier.participant_ranks[0]:
        peers = len(barrier.participant_ranks) - 1
        return (RecordOpcode.EVENT_WAIT,) * peers + (RecordOpcode.EVENT_SET,) * peers
    return (RecordOpcode.EVENT_SET, RecordOpcode.EVENT_WAIT)


@dataclass(frozen=True, slots=True)
class SwizzleOpcodeRecord:
    schema_version: str
    id: str
    task: SwizzleIr2Task
    opcodes: tuple[RecordOpcode, ...]

    @classmethod
    def create(
        cls,
        *,
        task: SwizzleIr2Task,
        opcodes: tuple[RecordOpcode, ...],
    ) -> "SwizzleOpcodeRecord":
        semantic = {"task": task, "opcodes": opcodes}
        result = cls(
            schema_version=SWIZZLE_OPCODE_RECORD_SCHEMA_VERSION,
            id=stable_artifact_id(
                "swizzle_opcode_record",
                semantic,
                schema_version=SWIZZLE_OPCODE_RECORD_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {"task": self.task, "opcodes": self.opcodes}

    def validate(self, path: str = "swizzle_opcode_record") -> None:
        if self.schema_version != SWIZZLE_OPCODE_RECORD_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.task.validate(f"{path}.task")
        if not self.opcodes or any(type(opcode) is not RecordOpcode for opcode in self.opcodes):
            raise SchemaError("must contain typed existing opcodes", path=f"{path}.opcodes")
        expected = stable_artifact_id(
            "swizzle_opcode_record",
            self._semantic_key(),
            schema_version=SWIZZLE_OPCODE_RECORD_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleLoweredRankStream:
    rank: int
    die_id: int
    records: tuple[SwizzleOpcodeRecord, ...]

    def validate(self, path: str) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        validate_uint64(self.die_id, f"{path}.die_id")
        if not self.records:
            raise SchemaError("must contain opcode records", path=f"{path}.records")
        for index, record in enumerate(self.records):
            record.validate(f"{path}.records[{index}]")
            if record.task.rank != self.rank or record.task.die_id != self.die_id:
                raise SchemaError("record placement disagrees with stream", path=f"{path}.records[{index}]")


@dataclass(frozen=True, slots=True)
class SwizzleLoweredProgram:
    schema_version: str
    producer_pass: str
    id: str
    source_plan_ref: str
    source_projection_ref: str
    source_ir1_id: str
    source_decision_ref: str
    source_candidate_ref: str
    pattern: FusionPattern
    algorithm: SwizzleAlgorithm
    rank_streams: tuple[SwizzleLoweredRankStream, ...]
    timing_execution: bool
    functional_execution: bool

    @classmethod
    def create(cls, *, producer_pass: str, **semantic: object) -> "SwizzleLoweredProgram":
        result = cls(
            schema_version=SWIZZLE_LOWERED_PROGRAM_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "swizzle_lowered_program",
                semantic,
                schema_version=SWIZZLE_LOWERED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_plan_ref",
                "source_projection_ref",
                "source_ir1_id",
                "source_decision_ref",
                "source_candidate_ref",
                "pattern",
                "algorithm",
                "rank_streams",
                "timing_execution",
                "functional_execution",
            )
        }

    def validate(self, path: str = "swizzle_lowered_program") -> None:
        if self.schema_version != SWIZZLE_LOWERED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in (
            "producer_pass",
            "source_plan_ref",
            "source_projection_ref",
            "source_ir1_id",
            "source_decision_ref",
            "source_candidate_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.pattern) is not FusionPattern or type(self.algorithm) is not SwizzleAlgorithm:
            raise SchemaError("must carry typed pattern/algorithm", path=path)
        if self.timing_execution is not True or self.functional_execution is not False:
            raise SchemaError("W9 v1 is timing-only", path=path)
        if tuple(stream.rank for stream in self.rank_streams) != tuple(range(len(self.rank_streams))):
            raise SchemaError("rank streams must be dense canonical ranks", path=f"{path}.rank_streams")
        task_refs: list[str] = []
        for index, stream in enumerate(self.rank_streams):
            stream.validate(f"{path}.rank_streams[{index}]")
            task_refs.extend(record.task.id for record in stream.records)
        if len(task_refs) != len(set(task_refs)):
            raise SchemaError("tasks must lower exactly once", path=f"{path}.rank_streams")
        expected = stable_artifact_id(
            "swizzle_lowered_program",
            self._semantic_key(),
            schema_version=SWIZZLE_LOWERED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(
        self,
        plan: SwizzleFusionPlan,
        projection: SwizzleIr2Projection,
        path: str = "swizzle_lowered_program",
    ) -> None:
        self.validate(path)
        validate_swizzle_plan_projection(plan, projection, f"{path}.inputs")
        if (
            self.source_plan_ref,
            self.source_projection_ref,
            self.source_ir1_id,
            self.source_decision_ref,
            self.source_candidate_ref,
            self.pattern,
            self.algorithm,
        ) != (
            plan.id,
            projection.id,
            plan.source_ir1_id,
            plan.decision.id,
            plan.candidate.id,
            plan.pattern,
            plan.algorithm,
        ):
            raise SchemaError("lowered provenance is not exact", path=path)
        actions = _bound_action_index(plan)
        if tuple((stream.rank, stream.die_id) for stream in self.rank_streams) != tuple(
            (dag.rank, dag.die_id) for dag in projection.rank_dags
        ):
            raise SchemaError("lowered rank/Die streams drift from projection", path=f"{path}.rank_streams")
        for stream, dag in zip(self.rank_streams, projection.rank_dags, strict=True):
            if tuple(record.task for record in stream.records) != dag.tasks:
                raise SchemaError("lowered task order drifts from projection", path=f"{path}.rank_streams")
            for record in stream.records:
                expected_opcodes = expected_swizzle_opcodes(
                    actions[record.task.source_action_ref], record.task
                )
                if record.opcodes != expected_opcodes:
                    raise SchemaError("opcode sequence drifts from exact action kind", path=f"{path}.rank_streams")


def expected_swizzle_manifest_digests(
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
    lowered: SwizzleLoweredProgram,
) -> tuple[SwizzleManifestInputDigest, ...]:
    values = (
        SwizzleManifestInputDigest(
            SwizzleManifestInputKind.FUSION_PLAN,
            plan.id,
            plan.schema_version,
            canonical_digest(plan),
        ),
        SwizzleManifestInputDigest(
            SwizzleManifestInputKind.DECISION,
            plan.decision.id,
            plan.decision.schema_version,
            canonical_digest(plan.decision),
        ),
        SwizzleManifestInputDigest(
            SwizzleManifestInputKind.CANDIDATE,
            plan.candidate.id,
            plan.candidate.schema_version,
            canonical_digest(plan.candidate),
        ),
        SwizzleManifestInputDigest(
            SwizzleManifestInputKind.PROJECTION,
            projection.id,
            projection.schema_version,
            canonical_digest(projection),
        ),
        SwizzleManifestInputDigest(
            SwizzleManifestInputKind.LOWERED_PROGRAM,
            lowered.id,
            lowered.schema_version,
            canonical_digest(lowered),
        ),
    )
    return tuple(sorted(values, key=lambda item: (item.kind.value, item.artifact_id)))


@dataclass(frozen=True, slots=True)
class SwizzleLinkedManifest:
    schema_version: str
    producer_pass: str
    id: str
    source_plan_ref: str
    source_projection_ref: str
    source_lowered_ref: str
    source_ir1_id: str
    source_decision_ref: str
    source_candidate_ref: str
    pattern: FusionPattern
    algorithm: SwizzleAlgorithm
    input_digests: tuple[SwizzleManifestInputDigest, ...]
    rank_streams: tuple[SwizzleLoweredRankStream, ...]
    finalizer_contract: SwizzleFinalizerContract
    timing_execution: bool
    functional_execution: bool
    finalizer_gate_reason: str

    @classmethod
    def create(cls, *, producer_pass: str, **semantic: object) -> "SwizzleLinkedManifest":
        result = cls(
            schema_version=SWIZZLE_LINKED_MANIFEST_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "swizzle_linked_manifest",
                semantic,
                schema_version=SWIZZLE_LINKED_MANIFEST_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_plan_ref",
                "source_projection_ref",
                "source_lowered_ref",
                "source_ir1_id",
                "source_decision_ref",
                "source_candidate_ref",
                "pattern",
                "algorithm",
                "input_digests",
                "rank_streams",
                "finalizer_contract",
                "timing_execution",
                "functional_execution",
                "finalizer_gate_reason",
            )
        }

    def validate(self, path: str = "swizzle_linked_manifest") -> None:
        if self.schema_version != SWIZZLE_LINKED_MANIFEST_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in (
            "producer_pass",
            "source_plan_ref",
            "source_projection_ref",
            "source_lowered_ref",
            "source_ir1_id",
            "source_decision_ref",
            "source_candidate_ref",
            "finalizer_gate_reason",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.pattern) is not FusionPattern or type(self.algorithm) is not SwizzleAlgorithm:
            raise SchemaError("must carry typed pattern/algorithm", path=path)
        if self.finalizer_contract is not SwizzleFinalizerContract.REQUIRES_EXACT_CORE_ADDRESS_ABI_V1:
            raise SchemaError("unsupported finalizer contract", path=f"{path}.finalizer_contract")
        if self.timing_execution is not True or self.functional_execution is not False:
            raise SchemaError("W9 v1 is timing-only", path=path)
        keys = tuple((item.kind.value, item.artifact_id) for item in self.input_digests)
        if keys != tuple(sorted(set(keys))):
            raise SchemaError("input digests must be unique and canonical", path=f"{path}.input_digests")
        for index, digest in enumerate(self.input_digests):
            digest.validate(f"{path}.input_digests[{index}]")
        if tuple(stream.rank for stream in self.rank_streams) != tuple(range(len(self.rank_streams))):
            raise SchemaError("rank streams must be dense canonical ranks", path=f"{path}.rank_streams")
        for index, stream in enumerate(self.rank_streams):
            stream.validate(f"{path}.rank_streams[{index}]")
        expected = stable_artifact_id(
            "swizzle_linked_manifest",
            self._semantic_key(),
            schema_version=SWIZZLE_LINKED_MANIFEST_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(
        self,
        plan: SwizzleFusionPlan,
        projection: SwizzleIr2Projection,
        lowered: SwizzleLoweredProgram,
        path: str = "swizzle_linked_manifest",
    ) -> None:
        self.validate(path)
        lowered.validate_against(plan, projection, f"{path}.lowered")
        if (
            self.source_plan_ref,
            self.source_projection_ref,
            self.source_lowered_ref,
            self.source_ir1_id,
            self.source_decision_ref,
            self.source_candidate_ref,
            self.pattern,
            self.algorithm,
        ) != (
            plan.id,
            projection.id,
            lowered.id,
            plan.source_ir1_id,
            plan.decision.id,
            plan.candidate.id,
            plan.pattern,
            plan.algorithm,
        ):
            raise SchemaError("linked provenance is not exact", path=path)
        if self.input_digests != expected_swizzle_manifest_digests(plan, projection, lowered):
            raise SchemaError(
                "manifest digests must exactly preserve Plan/Decision/Candidate/Projection/Lowered inputs",
                path=f"{path}.input_digests",
            )
        if self.rank_streams != lowered.rank_streams:
            raise SchemaError("linker must preserve lowered streams byte-exact", path=f"{path}.rank_streams")


__all__ = [name for name in globals() if name.startswith("Swizzle") or name.startswith("SWIZZLE_")] + [
    "expected_swizzle_manifest_digests",
    "expected_swizzle_opcodes",
    "validate_swizzle_plan_projection",
]
