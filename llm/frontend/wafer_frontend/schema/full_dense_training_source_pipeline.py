"""Exact source task graph for two-step full-model Dense training.

This module expands the validated forward-only ``FlexibleDenseTrainPlan``
into the operation inventory that a production IR1/projection/schedule/global
DAG must materialize.  It deliberately carries no physical-success bit: the
lowering gate remains the authority for native records, BufferABI closures,
StateABI versions and runtime evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id
from .dense_backbone_reverse_source_admission import (
    DenseReverseSourceFamily,
    DenseTwoStepBackboneSourceAdmission,
    build_dense_two_step_backbone_source_admission,
)
from .flexible_dense_train import FlexibleDenseTrainPlan
from .full_dense_gradient_requirements import (
    DenseFullTrainRequirements,
    build_dense_full_train_requirements,
)
from .serde import canonical_digest


FULL_DENSE_TRAINING_SOURCE_PIPELINE_SCHEMA_VERSION = (
    "wafer_frontend.full_dense_training_source_pipeline/v1alpha1"
)


class DenseFullTrainingTaskKind(str, Enum):
    PARAMETER_LOAD = "parameter_load"
    FORWARD = "forward"
    LOSS_SEED = "loss_seed"
    CE_BACKWARD = "ce_backward"
    BACKBONE_BACKWARD = "backbone_backward"
    PARAMETER_WGRAD = "parameter_wgrad"
    GRADIENT_SYNC = "gradient_sync"
    SGD_UPDATE = "sgd_update"
    PARAMETER_STORE = "parameter_store"


@dataclass(frozen=True, slots=True)
class DenseFullTrainingSourceTask:
    id: str
    step: int
    rank: int
    ordinal: int
    kind: DenseFullTrainingTaskKind
    operation_ref: str
    source_forward_refs: tuple[str, ...]
    parameter_state_ref: str | None
    reverse_family: DenseReverseSourceFamily | None
    read_version: int | None
    write_version: int | None
    depends_on: tuple[str, ...]

    def validate(self, path: str) -> None:
        if not self.id or not self.operation_ref:
            raise SchemaError("source task identity is empty", path=path)
        if (type(self.step) is not int or self.step not in (0, 1)
                or type(self.rank) is not int or self.rank < 0
                or type(self.ordinal) is not int or self.ordinal < 0):
            raise SchemaError("source task coordinate is invalid", path=path)
        if type(self.kind) is not DenseFullTrainingTaskKind:
            raise SchemaError("source task kind is untyped", path=path)
        if (not self.source_forward_refs
                or len(set(self.source_forward_refs)) !=
                len(self.source_forward_refs)):
            raise SchemaError("source task needs unique forward provenance", path=path)
        reverse = self.kind is DenseFullTrainingTaskKind.BACKBONE_BACKWARD
        if reverse != (self.reverse_family is not None):
            raise SchemaError("reverse family is exact for backbone leaves", path=path)
        parameter = self.kind in (
            DenseFullTrainingTaskKind.PARAMETER_LOAD,
            DenseFullTrainingTaskKind.PARAMETER_WGRAD,
            DenseFullTrainingTaskKind.GRADIENT_SYNC,
            DenseFullTrainingTaskKind.SGD_UPDATE,
            DenseFullTrainingTaskKind.PARAMETER_STORE,
        )
        if parameter != (self.parameter_state_ref is not None):
            raise SchemaError("parameter identity is exact for state tasks", path=path)
        versioned = self.kind in (
            DenseFullTrainingTaskKind.PARAMETER_LOAD,
            DenseFullTrainingTaskKind.SGD_UPDATE,
            DenseFullTrainingTaskKind.PARAMETER_STORE,
        )
        if versioned != (self.read_version is not None
                          and self.write_version is not None):
            raise SchemaError("state versions are exact for load/update/store", path=path)
        if versioned and not (self.read_version == self.step
                              and self.write_version == self.step + 1):
            raise SchemaError("source task state version differs from SGD step", path=path)
        if self.depends_on != tuple(sorted(set(self.depends_on))):
            raise SchemaError("source dependencies must be canonical", path=path)


@dataclass(frozen=True, slots=True)
class DenseFullTrainingSourceIR:
    schema_version: str
    id: str
    source_plan_id: str
    source_plan_digest: str
    source_forward_graph_digest: str
    requirements_digest: str
    reverse_admission_digest: str
    mesh_shape: tuple[int, int]
    steps: int
    tasks: tuple[DenseFullTrainingSourceTask, ...]
    state_version_edges: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, **semantic: object) -> "DenseFullTrainingSourceIR":
        result = cls(
            FULL_DENSE_TRAINING_SOURCE_PIPELINE_SCHEMA_VERSION,
            stable_artifact_id(
                "full_dense_training_source_ir", semantic,
                schema_version=FULL_DENSE_TRAINING_SOURCE_PIPELINE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "full_dense_training_source_ir") -> None:
        if self.schema_version != FULL_DENSE_TRAINING_SOURCE_PIPELINE_SCHEMA_VERSION:
            raise SchemaError("unsupported source IR schema", path=path)
        if (not self.source_plan_id or len(self.source_plan_digest) != 64
                or len(self.source_forward_graph_digest) != 64
                or len(self.requirements_digest) != 64
                or len(self.reverse_admission_digest) != 64):
            raise SchemaError("source IR lineage digest is invalid", path=path)
        rows, columns = self.mesh_shape
        if (type(rows) is not int or type(columns) is not int
                or rows < 1 or columns < 1 or self.steps != 2 or not self.tasks):
            raise SchemaError("source IR mesh/step domain is invalid", path=path)
        ids = tuple(task.id for task in self.tasks)
        if len(set(ids)) != len(ids):
            raise SchemaError("source task ids must be unique", path=path)
        by_id = {task.id: task for task in self.tasks}
        by_rank_step: dict[tuple[int, int], list[DenseFullTrainingSourceTask]] = {}
        for index, task in enumerate(self.tasks):
            task.validate(f"{path}.tasks[{index}]")
            if task.rank >= rows * columns:
                raise SchemaError("source task escapes mesh", path=f"{path}.tasks[{index}]")
            by_rank_step.setdefault((task.step, task.rank), []).append(task)
            if any(dependency not in by_id for dependency in task.depends_on):
                raise SchemaError("source dependency is dangling", path=path)
        for key, tasks in by_rank_step.items():
            if tuple(task.ordinal for task in tasks) != tuple(range(len(tasks))):
                raise SchemaError("rank/step task order is not contiguous", path=f"{path}.{key}")
        if set(by_rank_step) != {
            (step, rank) for step in range(2) for rank in range(rows * columns)
        }:
            raise SchemaError("every rank must run both SGD steps", path=path)
        if self.state_version_edges != tuple(sorted(set(self.state_version_edges))):
            raise SchemaError("state version edges must be canonical", path=path)
        for source, target in self.state_version_edges:
            if (source not in by_id or target not in by_id
                    or by_id[source].kind is not DenseFullTrainingTaskKind.PARAMETER_STORE
                    or by_id[target].kind is not DenseFullTrainingTaskKind.PARAMETER_LOAD
                    or by_id[source].step != 0 or by_id[target].step != 1
                    or by_id[source].rank != by_id[target].rank
                    or by_id[source].parameter_state_ref !=
                    by_id[target].parameter_state_ref
                    or source not in by_id[target].depends_on):
                raise SchemaError("state version edge is not exact STORE0 to LOAD1", path=path)
        semantic = {
            name: getattr(self, name) for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }
        expected = stable_artifact_id(
            "full_dense_training_source_ir", semantic,
            schema_version=FULL_DENSE_TRAINING_SOURCE_PIPELINE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("source IR id is unstable", path=f"{path}.id")

    def validate_against(self, plan: FlexibleDenseTrainPlan) -> None:
        if self != build_full_dense_training_source_ir(plan):
            raise SchemaError("source IR drifted from production Dense plan",
                              path="full_dense_training_source_ir")


def _task_id(*, plan_id: str, step: int, rank: int, ordinal: int,
             kind: DenseFullTrainingTaskKind, operation_ref: str) -> str:
    return stable_artifact_id(
        "full_dense_training_source_task",
        {"plan": plan_id, "step": step, "rank": rank, "ordinal": ordinal,
         "kind": kind.value, "operation": operation_ref},
        schema_version=FULL_DENSE_TRAINING_SOURCE_PIPELINE_SCHEMA_VERSION,
    )


def build_full_dense_training_source_ir(
    plan: FlexibleDenseTrainPlan,
) -> DenseFullTrainingSourceIR:
    """Expand every forward/backward/WGRAD/sync/SGD/state task for two steps."""
    if type(plan) is not FlexibleDenseTrainPlan:
        raise SchemaError("requires production FlexibleDenseTrainPlan", path="plan")
    plan.validate()
    requirements = build_dense_full_train_requirements(plan, steps=2)
    admission = build_dense_two_step_backbone_source_admission(plan)
    admission.validate_against(plan)
    leaves = {leaf.backward_ref: leaf for leaf in admission.leaves}
    templates = {template.state_ref: template
                 for template in plan.parameter_templates}
    paths = {(path.step, path.rank, path.parameter_state_ref): path
             for path in requirements.paths}
    tasks: list[DenseFullTrainingSourceTask] = []
    state_edges: list[tuple[str, str]] = []
    stores: dict[tuple[int, str], str] = {}

    for step in range(2):
        for rank in range(plan.spec.mesh.rank_count):
            local_states = tuple(sorted(
                template.state_ref for template in plan.parameter_templates
                if rank in template.owner_ranks
            ))
            ordinal = 0
            previous: str | None = None

            def append(
                kind: DenseFullTrainingTaskKind, operation_ref: str,
                source_forward_refs: tuple[str, ...], *,
                state_ref: str | None = None,
                family: DenseReverseSourceFamily | None = None,
                versioned: bool = False,
                extra_dependencies: tuple[str, ...] = (),
            ) -> str:
                nonlocal ordinal, previous
                identifier = _task_id(
                    plan_id=plan.id, step=step, rank=rank, ordinal=ordinal,
                    kind=kind, operation_ref=operation_ref,
                )
                dependencies = set(extra_dependencies)
                if previous is not None:
                    dependencies.add(previous)
                task = DenseFullTrainingSourceTask(
                    id=identifier, step=step, rank=rank, ordinal=ordinal,
                    kind=kind, operation_ref=operation_ref,
                    source_forward_refs=source_forward_refs,
                    parameter_state_ref=state_ref, reverse_family=family,
                    read_version=step if versioned else None,
                    write_version=step + 1 if versioned else None,
                    depends_on=tuple(sorted(dependencies)),
                )
                tasks.append(task)
                previous = identifier
                ordinal += 1
                return identifier

            for state_ref in local_states:
                dependency = (() if step == 0 else (stores[(rank, state_ref)],))
                load = append(
                    DenseFullTrainingTaskKind.PARAMETER_LOAD,
                    f"load::{state_ref}::r{rank}::step{step}",
                    templates[state_ref].forward_consumer_refs,
                    state_ref=state_ref, versioned=True,
                    extra_dependencies=dependency,
                )
                if step == 1:
                    state_edges.append((stores[(rank, state_ref)], load))
            for forward_ref in requirements.required_forward_refs:
                append(DenseFullTrainingTaskKind.FORWARD, forward_ref,
                       (forward_ref,))
            append(DenseFullTrainingTaskKind.LOSS_SEED,
                   f"seed::{requirements.loss_gradient_seed_ref}::r{rank}::step{step}",
                   (requirements.forward_ce_ref,))
            append(DenseFullTrainingTaskKind.CE_BACKWARD,
                   requirements.backward_ce_ref,
                   (requirements.forward_ce_ref,))
            for reverse_ref in requirements.required_backbone_backward_refs:
                leaf = leaves[reverse_ref]
                append(DenseFullTrainingTaskKind.BACKBONE_BACKWARD,
                       reverse_ref, (leaf.forward_ref,), family=leaf.family)
            for state_ref in local_states:
                path = paths[(step, rank, state_ref)]
                append(DenseFullTrainingTaskKind.PARAMETER_WGRAD,
                       path.named_wgrad_op_ref, path.forward_op_refs,
                       state_ref=state_ref)
            for state_ref in local_states:
                path = paths[(step, rank, state_ref)]
                append(DenseFullTrainingTaskKind.GRADIENT_SYNC,
                       path.named_sync_op_ref, path.forward_op_refs,
                       state_ref=state_ref)
            for state_ref in local_states:
                path = paths[(step, rank, state_ref)]
                append(DenseFullTrainingTaskKind.SGD_UPDATE,
                       path.named_optimizer_op_ref, path.forward_op_refs,
                       state_ref=state_ref, versioned=True)
            for state_ref in local_states:
                path = paths[(step, rank, state_ref)]
                store = append(DenseFullTrainingTaskKind.PARAMETER_STORE,
                               path.named_store_op_ref, path.forward_op_refs,
                               state_ref=state_ref, versioned=True)
                if step == 0:
                    stores[(rank, state_ref)] = store

    result = DenseFullTrainingSourceIR.create(
        source_plan_id=plan.id,
        source_plan_digest=canonical_digest(plan),
        source_forward_graph_digest=canonical_digest(plan.forward_graph),
        requirements_digest=canonical_digest(requirements),
        reverse_admission_digest=canonical_digest(admission),
        mesh_shape=(plan.spec.mesh.rows, plan.spec.mesh.columns),
        steps=2,
        tasks=tuple(tasks),
        state_version_edges=tuple(sorted(state_edges)),
    )
    _validate_exact_coverage(result, plan, requirements, admission)
    return result


def _validate_exact_coverage(
    source: DenseFullTrainingSourceIR,
    plan: FlexibleDenseTrainPlan,
    requirements: DenseFullTrainRequirements,
    admission: DenseTwoStepBackboneSourceAdmission,
) -> None:
    """Independent set/count checks prevent labels from implying coverage."""
    source.validate()
    required_forward = set(requirements.required_forward_refs)
    required_reverse = set(requirements.required_backbone_backward_refs)
    families = {leaf.backward_ref: leaf.family for leaf in admission.leaves}
    for step in range(2):
        for rank in range(plan.spec.mesh.rank_count):
            current = tuple(task for task in source.tasks
                            if (task.step, task.rank) == (step, rank))
            by_kind: dict[DenseFullTrainingTaskKind, list[DenseFullTrainingSourceTask]] = {}
            for task in current:
                by_kind.setdefault(task.kind, []).append(task)
            local_paths = tuple(path for path in requirements.paths
                                if (path.step, path.rank) == (step, rank))
            if ({task.operation_ref for task in by_kind.get(
                    DenseFullTrainingTaskKind.FORWARD, [])} != required_forward
                    or {task.operation_ref for task in by_kind.get(
                        DenseFullTrainingTaskKind.BACKBONE_BACKWARD, [])} !=
                    required_reverse
                    or {task.operation_ref: task.reverse_family for task in
                        by_kind.get(DenseFullTrainingTaskKind.BACKBONE_BACKWARD, [])}
                    != families
                    or [task.operation_ref for task in by_kind.get(
                        DenseFullTrainingTaskKind.CE_BACKWARD, [])] !=
                    [requirements.backward_ce_ref]
                    or len(by_kind.get(DenseFullTrainingTaskKind.LOSS_SEED, [])) != 1):
                raise SchemaError("full forward/loss/backbone source coverage differs",
                                  path=f"source.step{step}.rank{rank}")
            expected_by_kind = {
                DenseFullTrainingTaskKind.PARAMETER_LOAD:
                    {f"load::{path.parameter_state_ref}::r{rank}::step{step}"
                     for path in local_paths},
                DenseFullTrainingTaskKind.PARAMETER_WGRAD:
                    {path.named_wgrad_op_ref for path in local_paths},
                DenseFullTrainingTaskKind.GRADIENT_SYNC:
                    {path.named_sync_op_ref for path in local_paths},
                DenseFullTrainingTaskKind.SGD_UPDATE:
                    {path.named_optimizer_op_ref for path in local_paths},
                DenseFullTrainingTaskKind.PARAMETER_STORE:
                    {path.named_store_op_ref for path in local_paths},
            }
            for kind, expected in expected_by_kind.items():
                if {task.operation_ref for task in by_kind.get(kind, [])} != expected:
                    raise SchemaError("per-parameter source task coverage differs",
                                      path=f"source.step{step}.rank{rank}.{kind.value}")
    expected_versions = {
        (rank, template.state_ref)
        for template in plan.parameter_templates for rank in template.owner_ranks
    }
    if len(source.state_version_edges) != len(expected_versions):
        raise SchemaError("STORE0 to LOAD1 state version coverage differs",
                          path="source.state_version_edges")


__all__ = [
    "FULL_DENSE_TRAINING_SOURCE_PIPELINE_SCHEMA_VERSION",
    "DenseFullTrainingSourceIR",
    "DenseFullTrainingSourceTask",
    "DenseFullTrainingTaskKind",
    "build_full_dense_training_source_ir",
]
