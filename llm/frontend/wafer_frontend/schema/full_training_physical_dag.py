"""Production physical dependency receipt for one training timeline.

This artifact signs the original executable action bytes, on-core sequencing,
and the precise SEND→RECV and state-version edges supplied by source plans.
Its digest may be used as the GLOBAL_ACTION_DAG input of an actual linked
program only after every full-model forward/loss/backward action is covered.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

from ..errors import SchemaError
from .artifact_manifest import CommandFragment, LinkedCoreStream, RecordOpcode
from .common import stable_artifact_id
from .global_action import LogicalCoreRef


FULL_TRAINING_PHYSICAL_DAG_SCHEMA_VERSION = (
    "wafer_frontend.full_training_physical_dag/v1alpha1"
)


def full_training_physical_dag_source_id(
    source_artifact_ids: tuple[str, ...],
) -> str:
    """Choose the DAG id from trusted sources before cloning any records.

    The content SHA-256 remains the canonical_digest of the later validated
    full physical DAG; assigning a content-derived artifact id before cloning
    would create an impossible identity/reference hash cycle.
    """
    if (not source_artifact_ids or source_artifact_ids !=
            tuple(sorted(set(source_artifact_ids)))):
        raise SchemaError("source DAG anchors must be nonempty, canonical and exact",
                          path="source_artifact_ids")
    return stable_artifact_id(
        "full_training_physical_dag",
        {"source_artifact_ids": source_artifact_ids},
        schema_version=FULL_TRAINING_PHYSICAL_DAG_SCHEMA_VERSION,
    )


@dataclass(frozen=True, slots=True)
class PhysicalTrainingAction:
    id: str
    logical_core: LogicalCoreRef
    source_action_ref: str
    operation_ref: str
    phase: str
    step: int
    layer: int | None
    executable_records: tuple[tuple[str, int, RecordOpcode], ...]
    depends_on: tuple[str, ...]

    def validate(self, path: str) -> None:
        self.logical_core.validate(f"{path}.logical_core")
        if not self.id or not self.source_action_ref or not self.operation_ref:
            raise SchemaError("physical action requires source and operation identity",
                              path=path)
        if self.phase not in ("forward", "loss", "backward", "optimizer"):
            raise SchemaError("physical TRAIN phase is unknown", path=path)
        if self.step not in (0, 1) or self.layer not in (None, 0, 1):
            raise SchemaError("physical TRAIN needs exact step/layer namespace", path=path)
        if not self.executable_records or not all(
            isinstance(opcode, RecordOpcode) and fragment and type(index) is int
            and index >= 0 for fragment, index, opcode in self.executable_records
        ):
            raise SchemaError("action must own true nonempty physical records", path=path)
        if len(set(self.executable_records)) != len(self.executable_records):
            raise SchemaError("action repeats one physical carrier record", path=path)
        if tuple(sorted(set(self.depends_on))) != self.depends_on or self.id in self.depends_on:
            raise SchemaError("action dependencies must be canonical and non-self", path=path)


@dataclass(frozen=True, slots=True)
class FullTrainingPhysicalDAG:
    schema_version: str
    id: str
    source_artifact_ids: tuple[str, ...]
    actions: tuple[PhysicalTrainingAction, ...]
    transport_edges: tuple[tuple[str, str], ...]
    state_version_edges: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, *, source_artifact_ids, actions, transport_edges=(),
               state_version_edges=()) -> "FullTrainingPhysicalDAG":
        payload = dict(source_artifact_ids=tuple(source_artifact_ids),
                       actions=tuple(actions),
                       transport_edges=tuple(transport_edges),
                       state_version_edges=tuple(state_version_edges))
        artifact = cls(FULL_TRAINING_PHYSICAL_DAG_SCHEMA_VERSION,
                       full_training_physical_dag_source_id(payload["source_artifact_ids"]),
                       **payload)
        artifact.validate()
        return artifact

    def validate(self, path: str = "full_training_physical_dag") -> None:
        if self.schema_version != FULL_TRAINING_PHYSICAL_DAG_SCHEMA_VERSION:
            raise SchemaError("physical DAG schema is unknown", path=f"{path}.schema_version")
        if (not self.source_artifact_ids
                or self.source_artifact_ids != tuple(sorted(set(self.source_artifact_ids)))):
            raise SchemaError("physical source artifacts must be exact and sorted", path=path)
        ids = [action.id for action in self.actions]
        if not ids or ids != sorted(set(ids)):
            raise SchemaError("physical DAG actions must be unique/canonical", path=path)
        by_id = {action.id: action for action in self.actions}
        records = set()
        for index, action in enumerate(self.actions):
            action.validate(f"{path}.actions[{index}]")
            for fragment_id, record_index, opcode in action.executable_records:
                key = (action.logical_core, fragment_id, record_index)
                if key in records:
                    raise SchemaError("two actions claim the same physical record", path=path)
                records.add(key)
            if any(dep not in by_id for dep in action.depends_on):
                raise SchemaError("physical dependency lacks its producer", path=path)
        for name, edges in (("transport_edges", self.transport_edges),
                            ("state_version_edges", self.state_version_edges)):
            if edges != tuple(sorted(set(edges))) or any(
                source not in by_id or target not in by_id
                or source not in by_id[target].depends_on
                for source, target in edges
            ):
                raise SchemaError("version/transport edge must be a witnessed dependency",
                                  path=f"{path}.{name}")
        if any(by_id[src].logical_core == by_id[dst].logical_core
               for src, dst in self.transport_edges):
            raise SchemaError("transport edge cannot remain on one physical core",
                              path=f"{path}.transport_edges")
        if any(
            not any(opcode is RecordOpcode.DTE_SEND for _, _, opcode
                    in by_id[src].executable_records)
            or not any(opcode is RecordOpcode.DTE_RECV for _, _, opcode
                       in by_id[dst].executable_records)
            for src, dst in self.transport_edges
        ):
            raise SchemaError("transport edge must connect true DTE SEND to RECV",
                              path=f"{path}.transport_edges")
        dependents = defaultdict(set)
        degree = {}
        for action in self.actions:
            degree[action.id] = len(action.depends_on)
            for predecessor in action.depends_on:
                dependents[predecessor].add(action.id)
        ready = sorted(action for action, count in degree.items() if not count)
        visited = 0
        while ready:
            current = ready.pop(0)
            visited += 1
            for successor in sorted(dependents[current]):
                degree[successor] -= 1
                if degree[successor] == 0:
                    ready.append(successor)
            ready.sort()
        if visited != len(by_id):
            raise SchemaError("real physical TRAIN timeline has a dependency cycle",
                              path=f"{path}.actions")
        expected = full_training_physical_dag_source_id(self.source_artifact_ids)
        if self.id != expected:
            raise SchemaError("physical DAG source identity differs from trusted artifacts",
                              path=f"{path}.id")

    def validate_against(
        self, fragments: tuple[CommandFragment, ...],
        streams: tuple[LinkedCoreStream, ...],
        *,
        required_operation_ids: tuple[str, ...],
    ) -> None:
        """Only an actual full-model operation inventory may sign the DAG."""
        self.validate()
        if (not required_operation_ids
                or tuple(sorted(set(required_operation_ids))) != required_operation_ids):
            raise SchemaError("global TRAIN coverage inventory is empty/noncanonical",
                              path="required_operation_ids")
        if not set(required_operation_ids).issubset(
            {action.operation_ref for action in self.actions}
        ):
            raise SchemaError("full training physical DAG lacks a required operation",
                              path="required_operation_ids")
        by_fragment = {fragment.id: fragment for fragment in fragments}
        if len(by_fragment) != len(fragments):
            raise SchemaError("physical DAG carrier has duplicate fragments", path="fragments")
        expected = {
            (stream.logical_core, fragment.id, index,
             record.source_global_action_id, record.opcode)
            for fragment in fragments for stream in fragment.core_streams
            for index, record in enumerate(stream.records)
        }
        actual = set()
        for action in self.actions:
            actual.update((action.logical_core, fragment_id, index, action.id, opcode)
                          for fragment_id, index, opcode in action.executable_records)
        if expected != actual:
            raise SchemaError("physical DAG must witness every carrier record exactly",
                              path="full_training_physical_dag.actions")
        source_refs = {
            (stream.logical_core, ref.fragment_id,
             ref.fragment_record_index, ref.source_global_action_id)
            for stream in streams for ref in stream.records
        }
        if {(core, fragment_id, index, action_id)
            for core, fragment_id, index, action_id, _ in actual} != source_refs:
            raise SchemaError("DAG physical actions differ from one linked timeline",
                              path="full_training_physical_dag.actions")


def build_full_training_physical_dag(
    *,
    fragments: tuple[CommandFragment, ...],
    streams: tuple[LinkedCoreStream, ...],
    source_artifact_ids: tuple[str, ...],
    operation_by_action: Mapping[str, tuple[str, str, int, int | None, str]],
    transport_edges: tuple[tuple[str, str], ...],
    state_version_edges: tuple[tuple[str, str], ...] = (),
    required_operation_ids: tuple[str, ...],
) -> FullTrainingPhysicalDAG:
    """Compute on-core and cross-core edges from real executable records."""
    carriers = {fragment.id: fragment for fragment in fragments}
    records = defaultdict(list)
    predecessors = defaultdict(set)
    for stream in streams:
        previous = None
        for ref in stream.records:
            fragment = carriers[ref.fragment_id]
            source = next((local for local in fragment.core_streams
                           if local.logical_core == stream.logical_core), None)
            if source is None:
                raise SchemaError("TRAIN stream references absent carrier core", path=ref.fragment_id)
            record = source.records[ref.fragment_record_index]
            if record.source_global_action_id != ref.source_global_action_id:
                raise SchemaError("TRAIN ref changed source action after cloning", path=ref.fragment_id)
            if ref.source_global_action_id not in operation_by_action:
                raise SchemaError("each action requires a real step/layer operation identity",
                                  path=ref.source_global_action_id)
            key = (ref.source_global_action_id, stream.logical_core)
            records[key].append((ref.fragment_id, ref.fragment_record_index, record.opcode))
            if previous is not None and previous != key:
                predecessors[key].add(previous[0])
            previous = key
    for source, target in transport_edges + state_version_edges:
        if source == target:
            raise SchemaError("physical cross edge must differ", path="transport_edges")
        endpoints = [key for key in records if key[0] == target]
        if len(endpoints) != 1 or not any(key[0] == source for key in records):
            raise SchemaError("TRAIN dependency lacks one physical endpoint", path=target)
        predecessors[endpoints[0]].add(source)
    actions = []
    for (action_id, core), physical in records.items():
        source_ref, operation, step, layer, phase = operation_by_action[action_id]
        actions.append(PhysicalTrainingAction(
            action_id, core, source_ref, operation, phase, step, layer,
            tuple(physical), tuple(sorted(predecessors[(action_id, core)])),
        ))
    artifact = FullTrainingPhysicalDAG.create(
        source_artifact_ids=tuple(sorted(set(source_artifact_ids))),
        actions=tuple(sorted(actions, key=lambda action: action.id)),
        transport_edges=tuple(sorted(set(transport_edges))),
        state_version_edges=tuple(sorted(set(state_version_edges))),
    )
    artifact.validate_against(fragments, streams,
                              required_operation_ids=required_operation_ids)
    return artifact


__all__ = ["FULL_TRAINING_PHYSICAL_DAG_SCHEMA_VERSION", "PhysicalTrainingAction",
           "FullTrainingPhysicalDAG", "build_full_training_physical_dag",
           "full_training_physical_dag_source_id"]
