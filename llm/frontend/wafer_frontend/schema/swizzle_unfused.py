"""Typed, non-fused comparison DAGs for the W10 naive branch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_dependency_dag, validate_nonempty, validate_uint64
from .ir0 import FusionPattern
from .ir1 import IR1
from .swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
    SwizzleCandidate,
    SwizzleProblem,
)
from .swizzle_plan import SwizzleValueUse


UNFUSED_COMPARISON_PLAN_SCHEMA_VERSION = "wafer_frontend.unfused_comparison_plan/v1alpha1"
UNFUSED_COMPARISON_ACTION_SCHEMA_VERSION = "wafer_frontend.unfused_comparison_action/v1alpha1"
UNFUSED_COMPARISON_PROJECTION_SCHEMA_VERSION = "wafer_frontend.unfused_comparison_projection/v1alpha1"
UNFUSED_COMPARISON_FLOW_SCHEMA_VERSION = "wafer_frontend.unfused_comparison_flow/v1alpha1"


class UnfusedComparisonStage(str, Enum):
    GEMM = "gemm"
    COLLECTIVE = "collective"
    REDUCTION = "reduction"
    REPLICATION = "replication"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class UnfusedComparisonAction:
    schema_version: str
    id: str
    rank: int
    kind: SwizzleActionKind
    stage: UnfusedComparisonStage
    member_ref: str
    deps: tuple[str, ...]
    read_value_refs: tuple[str, ...]
    write_value_refs: tuple[str, ...]
    peer_rank: int | None
    route_ref: str | None
    die_path: tuple[int, ...]
    logical_bytes: int
    flops: int

    @classmethod
    def create(cls, **semantic: object) -> "UnfusedComparisonAction":
        result = cls(
            schema_version=UNFUSED_COMPARISON_ACTION_SCHEMA_VERSION,
            id=stable_artifact_id(
                "unfused_comparison_action",
                semantic,
                schema_version=UNFUSED_COMPARISON_ACTION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate(self, path: str = "unfused_comparison_action") -> None:
        if self.schema_version != UNFUSED_COMPARISON_ACTION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_uint64(self.rank, f"{path}.rank")
        if type(self.kind) is not SwizzleActionKind:
            raise SchemaError("must be a SwizzleActionKind", path=f"{path}.kind")
        if type(self.stage) is not UnfusedComparisonStage:
            raise SchemaError("must be an UnfusedComparisonStage", path=f"{path}.stage")
        validate_nonempty(self.member_ref, f"{path}.member_ref")
        for name in ("deps", "read_value_refs", "write_value_refs"):
            refs = getattr(self, name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate refs", path=f"{path}.{name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{name}[{index}]")
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        validate_uint64(self.flops, f"{path}.flops")
        transport = self.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV)
        if transport:
            if self.peer_rank is None or self.route_ref is None or len(self.die_path) < 2 or self.logical_bytes == 0 or self.flops:
                raise SchemaError("transport requires exact peer/route/payload", path=path)
        elif self.peer_rank is not None or self.route_ref is not None or self.die_path:
            raise SchemaError("non-transport cannot carry route fields", path=path)
        if self.kind is SwizzleActionKind.COMP:
            if self.stage is not UnfusedComparisonStage.GEMM or self.flops == 0 or self.logical_bytes:
                raise SchemaError("GEMM action requires exact FLOPs", path=path)
        elif self.flops:
            raise SchemaError("only GEMM carries FLOPs", path=f"{path}.flops")
        if self.kind is SwizzleActionKind.LOCAL_COPY and (
            self.stage is not UnfusedComparisonStage.REDUCTION
            or self.logical_bytes == 0
            or len(self.read_value_refs) != 1
            or len(self.write_value_refs) != 1
        ):
            raise SchemaError(
                "packing copy requires one typed source/destination and positive bytes",
                path=path,
            )
        expected = stable_artifact_id(
            "unfused_comparison_action",
            self._semantic_key(),
            schema_version=UNFUSED_COMPARISON_ACTION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class UnfusedComparisonRankProgram:
    rank: int
    actions: tuple[UnfusedComparisonAction, ...]

    def validate(self, path: str) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        if not self.actions or any(action.rank != self.rank for action in self.actions):
            raise SchemaError("rank program requires rank-local actions", path=f"{path}.actions")
        for index, action in enumerate(self.actions):
            action.validate(f"{path}.actions[{index}]")


@dataclass(frozen=True, slots=True)
class UnfusedComparisonPlan:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    problem: SwizzleProblem
    baseline: SwizzleCandidate
    pattern: FusionPattern
    rank_programs: tuple[UnfusedComparisonRankProgram, ...]

    @classmethod
    def create(cls, **semantic: object) -> "UnfusedComparisonPlan":
        result = cls(
            schema_version=UNFUSED_COMPARISON_PLAN_SCHEMA_VERSION,
            producer_pass="unfused_comparison_planner",
            id=stable_artifact_id(
                "unfused_comparison_plan",
                semantic,
                schema_version=UNFUSED_COMPARISON_PLAN_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in ("source_ir1_id", "problem", "baseline", "pattern", "rank_programs")
        }

    def validate(self, path: str = "unfused_comparison_plan") -> None:
        if self.schema_version != UNFUSED_COMPARISON_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "unfused_comparison_planner":
            raise SchemaError("requires exact producer", path=f"{path}.producer_pass")
        validate_nonempty(self.source_ir1_id, f"{path}.source_ir1_id")
        self.problem.validate(f"{path}.problem")
        self.baseline.validate(f"{path}.baseline")
        if (
            self.baseline.algorithm is not SwizzleAlgorithm.UNFUSED
            or self.baseline.problem_ref != self.problem.id
            or self.source_ir1_id != self.problem.source_ir1_id
            or self.pattern is not self.problem.pattern
        ):
            raise SchemaError("must bind the exact problem and UNFUSED baseline", path=path)
        ranks = self.problem.collective.participant_ranks
        placement_ranks = tuple(item.rank for item in self.problem.group.placements)
        if (
            tuple(item.rank for item in self.rank_programs) != ranks
            or placement_ranks != ranks
        ):
            raise SchemaError(
                "comparison programs must exactly cover canonical participant ranks",
                path=f"{path}.rank_programs",
            )
        peers = len(ranks) - 1
        if self.pattern is FusionPattern.AG_GEMM:
            expected_kinds = (
                (
                    SwizzleActionKind.SEND,
                    SwizzleActionKind.RECV,
                    SwizzleActionKind.WAIT,
                )
                * peers
                + (SwizzleActionKind.COMP,)
            )
        elif self.pattern is FusionPattern.GEMM_RS:
            expected_kinds = (
                (
                    SwizzleActionKind.COMP,
                    SwizzleActionKind.SEND,
                    SwizzleActionKind.LOCAL_COPY,
                    SwizzleActionKind.RECV,
                    SwizzleActionKind.WAIT,
                    SwizzleActionKind.REDUCE,
                )
                if len(ranks) == 2
                else (
                    SwizzleActionKind.COMP,
                    SwizzleActionKind.LOCAL_COPY,
                )
                + (
                    SwizzleActionKind.SEND,
                    SwizzleActionKind.RECV,
                    SwizzleActionKind.WAIT,
                    SwizzleActionKind.REDUCE,
                )
                * peers
            )
        else:
            if len(ranks) != 2:
                raise SchemaError(
                    "multi-rank UNFUSED comparison supports AG and RS only",
                    path=f"{path}.pattern",
                )
            expected_kinds = (
                SwizzleActionKind.COMP, SwizzleActionKind.SEND,
                SwizzleActionKind.LOCAL_COPY,
                SwizzleActionKind.RECV, SwizzleActionKind.WAIT,
                SwizzleActionKind.REDUCE, SwizzleActionKind.SEND,
                SwizzleActionKind.RECV, SwizzleActionKind.WAIT,
                SwizzleActionKind.BARRIER,
            )
        actions = ()
        for index, program in enumerate(self.rank_programs):
            program.validate(f"{path}.rank_programs[{index}]")
            if tuple(action.kind for action in program.actions) != expected_kinds:
                raise SchemaError("rank action sequence is not the exact naive baseline", path=f"{path}.rank_programs[{index}]")
            actions += program.actions
        validate_dependency_dag(actions, f"{path}.rank_programs.actions")
        routes = {
            (route.source_rank, route.destination_rank): route
            for route in self.problem.group.routes
        }
        for action in actions:
            if action.kind not in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
                continue
            endpoints = (
                (action.rank, action.peer_rank)
                if action.kind is SwizzleActionKind.SEND
                else (action.peer_rank, action.rank)
            )
            route = routes.get(endpoints)
            if (
                route is None
                or action.route_ref != route.id
                or action.die_path != route.die_path
            ):
                raise SchemaError(
                    "transport action must bind its exact directed route",
                    path=f"{path}.rank_programs.actions",
                )
        payload = (
            self.problem.collective.rank_input_bytes
            if self.pattern is FusionPattern.AG_GEMM
            else self.problem.collective.rank_output_bytes
        )
        expected_transport_bytes = len(ranks) * peers * payload
        if any(
            sum(
                action.logical_bytes
                for action in actions
                if action.kind is kind
            ) != expected_transport_bytes
            for kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV)
        ):
            raise SchemaError(
                "transport work bytes do not close over ordered rank pairs",
                path=f"{path}.rank_programs.actions",
            )
        if sum(action.flops for action in actions) != self.problem.gemm.flops:
            raise SchemaError(
                "GEMM work does not exactly cover participant ranks",
                path=f"{path}.rank_programs.actions",
            )
        expected = stable_artifact_id(
            "unfused_comparison_plan",
            self._semantic_key(),
            schema_version=UNFUSED_COMPARISON_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(self, ir1: IR1, path: str = "unfused_comparison_plan") -> None:
        self.validate(path)
        ir1.validate(f"{path}.ir1")
        if ir1.id != self.source_ir1_id:
            raise SchemaError("plan references a different IR1", path=f"{path}.source_ir1_id")
        from ..passes.project_unfused_comparison import build_unfused_comparison_plan

        expected = build_unfused_comparison_plan(ir1, self.problem, self.baseline)
        if self != expected:
            raise SchemaError("plan is not the exact deterministic UNFUSED baseline", path=path)


@dataclass(frozen=True, slots=True)
class UnfusedComparisonOperand:
    task_ref: str
    ordinal: int
    use: SwizzleValueUse
    value_ref: str
    source_tensor_ref: str
    shape: tuple[int, ...]
    tensor_offset: tuple[int, ...]
    layout: str
    dtype: DType
    byte_extent: int
    storage_ref: str
    storage_bytes: int
    byte_offset: int

    def validate(self, path: str) -> None:
        for name in ("task_ref", "value_ref", "source_tensor_ref", "layout", "storage_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        validate_uint64(self.ordinal, f"{path}.ordinal")
        if type(self.use) is not SwizzleValueUse or type(self.dtype) is not DType:
            raise SchemaError("requires typed use/dtype", path=path)
        if not self.shape or any(type(item) is not int or item <= 0 for item in self.shape):
            raise SchemaError("shape must be positive", path=f"{path}.shape")
        if (
            type(self.tensor_offset) is not tuple
            or len(self.tensor_offset) != len(self.shape)
        ):
            raise SchemaError(
                "tensor offset must match typed rank",
                path=f"{path}.tensor_offset",
            )
        for index, offset in enumerate(self.tensor_offset):
            validate_uint64(offset, f"{path}.tensor_offset[{index}]")
        validate_uint64(self.byte_extent, f"{path}.byte_extent")
        validate_uint64(self.storage_bytes, f"{path}.storage_bytes")
        validate_uint64(self.byte_offset, f"{path}.byte_offset")
        dtype_bytes = 2 if self.dtype is DType.FP16 else 4
        elements = 1
        for extent in self.shape:
            elements *= extent
        if self.byte_extent != elements * dtype_bytes:
            raise SchemaError("byte extent disagrees with typed shape", path=f"{path}.byte_extent")
        if self.storage_bytes == 0 or self.byte_offset > self.storage_bytes or self.byte_extent > self.storage_bytes - self.byte_offset:
            raise SchemaError("typed view exceeds its explicit storage", path=f"{path}.storage_bytes")


@dataclass(frozen=True, slots=True)
class UnfusedComparisonFlow:
    schema_version: str
    id: str
    route_ref: str
    source_rank: int
    destination_rank: int
    die_path: tuple[int, ...]
    send_task_ref: str
    recv_task_ref: str
    logical_bytes: int

    @classmethod
    def create(cls, **semantic: object) -> "UnfusedComparisonFlow":
        result = cls(
            schema_version=UNFUSED_COMPARISON_FLOW_SCHEMA_VERSION,
            id=stable_artifact_id(
                "unfused_comparison_flow",
                semantic,
                schema_version=UNFUSED_COMPARISON_FLOW_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate("unfused_comparison_flow")
        return result

    def validate(self, path: str) -> None:
        if self.schema_version != UNFUSED_COMPARISON_FLOW_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in ("route_ref", "send_task_ref", "recv_task_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if self.source_rank == self.destination_rank or len(self.die_path) < 2:
            raise SchemaError("flow requires distinct exact endpoints", path=path)
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        if self.logical_bytes == 0:
            raise SchemaError("flow payload must be positive", path=f"{path}.logical_bytes")
        semantic = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }
        expected = stable_artifact_id(
            "unfused_comparison_flow",
            semantic,
            schema_version=UNFUSED_COMPARISON_FLOW_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class UnfusedComparisonRankProjection:
    rank: int
    die_id: int
    task_refs: tuple[str, ...]
    terminal_task_ref: str

    def validate(self, path: str) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        validate_uint64(self.die_id, f"{path}.die_id")
        if not self.task_refs or len(set(self.task_refs)) != len(self.task_refs):
            raise SchemaError("task refs must be unique and nonempty", path=f"{path}.task_refs")
        validate_nonempty(self.terminal_task_ref, f"{path}.terminal_task_ref")
        if self.terminal_task_ref != self.task_refs[-1]:
            raise SchemaError("terminal must be the last rank task", path=f"{path}.terminal_task_ref")


@dataclass(frozen=True, slots=True)
class UnfusedComparisonProjection:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_plan_ref: str
    problem_ref: str
    baseline_ref: str
    pattern: FusionPattern
    ranks: tuple[UnfusedComparisonRankProjection, ...]
    operands: tuple[UnfusedComparisonOperand, ...]
    flows: tuple[UnfusedComparisonFlow, ...]

    @classmethod
    def create(cls, **semantic: object) -> "UnfusedComparisonProjection":
        result = cls(
            schema_version=UNFUSED_COMPARISON_PROJECTION_SCHEMA_VERSION,
            producer_pass="unfused_comparison_projector",
            id=stable_artifact_id(
                "unfused_comparison_projection",
                semantic,
                schema_version=UNFUSED_COMPARISON_PROJECTION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_ir1_id", "source_plan_ref", "problem_ref",
                "baseline_ref", "pattern", "ranks", "operands", "flows",
            )
        }

    def validate(self, path: str = "unfused_comparison_projection") -> None:
        if self.schema_version != UNFUSED_COMPARISON_PROJECTION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "unfused_comparison_projector":
            raise SchemaError("requires exact producer", path=f"{path}.producer_pass")
        for name in ("source_ir1_id", "source_plan_ref", "problem_ref", "baseline_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        projected_ranks = tuple(item.rank for item in self.ranks)
        if (
            len(projected_ranks) < 2
            or projected_ranks != tuple(range(len(projected_ranks)))
        ):
            raise SchemaError(
                "projection requires dense canonical participant ranks",
                path=f"{path}.ranks",
            )
        for index, rank in enumerate(self.ranks):
            rank.validate(f"{path}.ranks[{index}]")
        for index, operand in enumerate(self.operands):
            operand.validate(f"{path}.operands[{index}]")
        for index, flow in enumerate(self.flows):
            flow.validate(f"{path}.flows[{index}]")
        keys = tuple((item.task_ref, item.ordinal) for item in self.operands)
        if len(keys) != len(set(keys)):
            raise SchemaError("contains duplicate task operands", path=f"{path}.operands")
        expected = stable_artifact_id(
            "unfused_comparison_projection",
            self._semantic_key(),
            schema_version=UNFUSED_COMPARISON_PROJECTION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(self, ir1: IR1, plan: UnfusedComparisonPlan, path: str = "unfused_comparison_projection") -> None:
        self.validate(path)
        plan.validate_against(ir1, f"{path}.plan")
        if (
            self.source_ir1_id, self.source_plan_ref, self.problem_ref,
            self.baseline_ref, self.pattern,
        ) != (
            ir1.id, plan.id, plan.problem.id, plan.baseline.id, plan.pattern,
        ):
            raise SchemaError("projection provenance is not exact", path=path)
        if tuple(item.rank for item in self.ranks) != plan.problem.collective.participant_ranks:
            raise SchemaError(
                "projection ranks drift from collective participants",
                path=f"{path}.ranks",
            )
        actions = {action.id: action for program in plan.rank_programs for action in program.actions}
        if {ref for rank in self.ranks for ref in rank.task_refs} != set(actions):
            raise SchemaError("rank projections do not exactly cover plan actions", path=f"{path}.ranks")
        flow_pairs = {(item.send_task_ref, item.recv_task_ref) for item in self.flows}
        expected_pairs = {
            (action.id, recv.id)
            for action in actions.values()
            if action.kind is SwizzleActionKind.SEND
            for recv in actions.values()
            if recv.kind is SwizzleActionKind.RECV
            and recv.stage is action.stage
            and recv.rank == action.peer_rank
            and recv.peer_rank == action.rank
        }
        if flow_pairs != expected_pairs:
            raise SchemaError("flows do not exactly cover SEND/RECV pairs", path=f"{path}.flows")
        from ..passes.project_unfused_comparison import project_unfused_comparison

        expected = project_unfused_comparison(ir1, plan)
        if self != expected:
            raise SchemaError("projection is not the exact deterministic UNFUSED lowering", path=path)


__all__ = [name for name in globals() if name.startswith("Unfused") or name.startswith("UNFUSED_")]
