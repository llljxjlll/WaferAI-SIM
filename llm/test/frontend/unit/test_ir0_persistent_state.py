from __future__ import annotations

import importlib
import unittest
from collections import Counter
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import (
    IR0,
    STATE_ACCESS_SCHEMA_VERSION,
    StateAccess,
    StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateLifetime,
    StateKind,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    loads_dataclass,
)

from _fixtures import valid_ir0, valid_spec


def _rebuild(graph: IR0, **updates: object) -> IR0:
    fields: dict[str, object] = {
        "producer_pass": graph.producer_pass,
        "job": graph.job,
        "instances": graph.instances,
        "nodes": graph.nodes,
        "values": graph.values,
        "edges": graph.edges,
        "fusion_candidates": graph.fusion_candidates,
        "profile": graph.profile,
        "train": graph.train,
        "persistent_states": graph.persistent_states,
        "state_accesses": graph.state_accesses,
    }
    fields.update(updates)
    return IR0.create(**fields)  # type: ignore[arg-type]


def _decl(
    graph: IR0,
    kind: StateKind,
    *,
    tensor_ref: str | None,
    mesh_ref: str | None = None,
    shard_index: int = 0,
) -> PersistentStateDecl:
    identity = PersistentStateIdentity.create(
        kind=kind,
        instance_ref=graph.instances[0].id,
        mesh_ref=mesh_ref or graph.instances[0].meshes[0].id,
        request_ref="request_slot_0"
        if kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
        else None,
        layer_index=0 if kind in (StateKind.KV_KEY, StateKind.KV_VALUE) else None,
        tensor_ref=tensor_ref,
        shard_index=shard_index,
        generation=0,
    )
    permission = {
        StateKind.PARAMETER: PersistentStateAccess.READ_ONLY,
        StateKind.KV_KEY: PersistentStateAccess.READ_WRITE,
        StateKind.KV_VALUE: PersistentStateAccess.READ_WRITE,
        StateKind.OPTIMIZER_RESERVED: PersistentStateAccess.RESERVED,
    }[kind]
    return PersistentStateDecl.create(
        identity=identity,
        shape=(16, 128),
        dtype=graph.values[0].dtype,
        layout="rank_local",
        lifetime=PersistentStateLifetime.PERSISTENT,
        access=permission,
    )


class IR0PersistentStateSchemaTest(unittest.TestCase):
    def test_public_round_trip_stable_id_and_optimizer_reservation(self) -> None:
        graph = valid_ir0()
        parameter = _decl(graph, StateKind.PARAMETER, tensor_ref=graph.values[0].id)
        optimizer = _decl(
            graph, StateKind.OPTIMIZER_RESERVED, tensor_ref=graph.values[0].id
        )
        access = StateAccess.create(
            node_ref=graph.nodes[0].id,
            state_ref=parameter.id,
            mode=StateAccessMode.READ,
            rank=0,
        )
        rebuilt = _rebuild(
            graph,
            persistent_states=(optimizer, parameter),
            state_accesses=(access,),
        )
        rebuilt.validate()
        self.assertEqual(
            rebuilt.persistent_states,
            tuple(
                sorted(
                    (parameter, optimizer),
                    key=lambda item: (item.identity.id, item.id),
                )
            ),
        )
        self.assertEqual(
            loads_dataclass(IR0, canonical_json(rebuilt), path="ir0"), rebuilt
        )
        self.assertEqual(
            loads_dataclass(StateAccess, canonical_json(access), path="access"),
            access,
        )
        self.assertEqual(
            STATE_ACCESS_SCHEMA_VERSION, "wafer_frontend.state_access/v1alpha2"
        )
        public = importlib.import_module("llm.frontend.wafer_frontend.schema")
        for name in (
            "STATE_ACCESS_SCHEMA_VERSION",
            "StateAccess",
            "StateAccessMode",
        ):
            self.assertIn(name, public.__all__)

    def test_dangling_identity_rank_and_permission_fail_closed(self) -> None:
        graph = valid_ir0()
        parameter = _decl(graph, StateKind.PARAMETER, tensor_ref=graph.values[0].id)

        cases = (
            (
                (parameter,),
                (
                    StateAccess.create(
                        node_ref="missing_node",
                        state_ref=parameter.id,
                        mode=StateAccessMode.READ,
                        rank=0,
                    ),
                ),
                "dangling node",
            ),
            (
                (parameter,),
                (
                    StateAccess.create(
                        node_ref=graph.nodes[0].id,
                        state_ref="missing_state",
                        mode=StateAccessMode.READ,
                        rank=0,
                    ),
                ),
                "dangling state",
            ),
            (
                (parameter,),
                (
                    StateAccess.create(
                        node_ref=graph.nodes[0].id,
                        state_ref=parameter.id,
                        mode=StateAccessMode.WRITE,
                        rank=0,
                    ),
                ),
                "exceeds declaration permission",
            ),
            (
                (parameter,),
                (
                    StateAccess.create(
                        node_ref=graph.nodes[0].id,
                        state_ref=parameter.id,
                        mode=StateAccessMode.READ,
                        rank=1,
                    ),
                ),
                "rank must equal",
            ),
        )
        for states, accesses, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    _rebuild(
                        graph, persistent_states=states, state_accesses=accesses
                    ).validate()

        dangling_tensor = _decl(
            graph, StateKind.PARAMETER, tensor_ref="missing_tensor"
        )
        with self.assertRaisesRegex(SchemaError, "dangling tensor"):
            _rebuild(graph, persistent_states=(dangling_tensor,)).validate()
        wrong_mesh = _decl(
            graph,
            StateKind.PARAMETER,
            tensor_ref=graph.values[0].id,
            mesh_ref="missing_mesh",
        )
        with self.assertRaisesRegex(SchemaError, "mesh outside"):
            _rebuild(graph, persistent_states=(wrong_mesh,)).validate()

        valid_access = StateAccess.create(
            node_ref=graph.nodes[0].id,
            state_ref=parameter.id,
            mode=StateAccessMode.READ,
            rank=0,
        )
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            _rebuild(
                graph,
                persistent_states=(parameter,),
                state_accesses=(replace(valid_access, id="forged"),),
            ).validate()


class DensePersistentStateTest(unittest.TestCase):
    @staticmethod
    def _graph(*, tp: int = 2, layers: int = 1) -> IR0:
        raw = valid_spec()
        raw["parallel"]["instances"][0].update(  # type: ignore[index]
            tp=tp, sp=tp > 1
        )
        raw["model"]["L"] = layers  # type: ignore[index]
        spec = from_data(ExperimentSpec, raw, path="spec")
        return logical_expand(build_ir0(spec)).entries[0].graph

    def test_rank_local_parameter_and_kv_counts_shapes_and_bytes(self) -> None:
        graph = self._graph(tp=2, layers=2)
        by_kind = {
            kind: tuple(
                state
                for state in graph.persistent_states
                if state.identity.kind is kind
            )
            for kind in StateKind
        }
        self.assertEqual(len(graph.persistent_states), 38)
        self.assertEqual(len(graph.state_accesses), 38)
        self.assertEqual(len(by_kind[StateKind.PARAMETER]), 30)
        self.assertFalse(by_kind[StateKind.TRAINABLE_PARAMETER])
        self.assertEqual(len(by_kind[StateKind.KV_KEY]), 4)
        self.assertEqual(len(by_kind[StateKind.KV_VALUE]), 4)
        self.assertFalse(by_kind[StateKind.OPTIMIZER_RESERVED])
        self.assertEqual(
            Counter(
                (state.identity.tensor_ref.rsplit(".", 1)[-1], state.shape)
                for state in by_kind[StateKind.PARAMETER]
                if state.identity.tensor_ref is not None
            ),
            Counter(
                {
                    ("w_qkv", (256, 256)): 4,
                    ("w_o", (128, 256)): 4,
                    ("w_gate_up", (256, 512)): 4,
                    ("w_down", (256, 256)): 4,
                    ("w_norm1", (256,)): 4,
                    ("w_norm2", (256,)): 4,
                    ("weight", (256, 512)): 2,
                    ("weight", (256,)): 2,
                    ("weight", (512, 256)): 2,
                }
            ),
        )
        for state in (
            *by_kind[StateKind.KV_KEY],
            *by_kind[StateKind.KV_VALUE],
        ):
            self.assertEqual(state.shape, (32, 1, 64))
            self.assertEqual(state.tensor_bytes, 4096)
            self.assertEqual(state.identity.request_ref, graph.profile.stable_id())
        self.assertEqual(
            sum(state.tensor_bytes for state in graph.persistent_states),
            3_445_760,
        )
        declarations = {state.id: state for state in graph.persistent_states}
        for access in graph.state_accesses:
            declaration = declarations[access.state_ref]
            self.assertEqual(access.rank, declaration.identity.shard_index)
            if declaration.identity.kind is StateKind.PARAMETER:
                self.assertIs(access.mode, StateAccessMode.READ)
            else:
                self.assertIn(
                    access.mode,
                    (
                        StateAccessMode.READ,
                        StateAccessMode.WRITE,
                        StateAccessMode.READ_WRITE,
                    ),
                )
        DenseIR0Validator.validate(graph)

    def test_missing_or_wrong_layer_kv_pair_is_rejected(self) -> None:
        graph = self._graph(tp=2)
        victim = next(
            state
            for state in graph.persistent_states
            if state.identity.kind is StateKind.KV_VALUE
        )
        missing_pair = _rebuild(
            graph,
            persistent_states=tuple(
                state for state in graph.persistent_states if state.id != victim.id
            ),
            state_accesses=tuple(
                access for access in graph.state_accesses if access.state_ref != victim.id
            ),
        )
        with self.assertRaisesRegex(SchemaError, "state declarations are not exact"):
            DenseIR0Validator.validate(missing_pair)

        key = next(
            state
            for state in graph.persistent_states
            if state.identity.kind is StateKind.KV_KEY
        )
        wrong_identity = PersistentStateIdentity.create(
            kind=StateKind.KV_KEY,
            instance_ref=key.identity.instance_ref,
            mesh_ref=key.identity.mesh_ref,
            request_ref=key.identity.request_ref,
            layer_index=key.identity.layer_index + 1,  # type: ignore[operator]
            tensor_ref=None,
            shard_index=key.identity.shard_index,
            generation=key.identity.generation,
        )
        wrong_decl = PersistentStateDecl.create(
            identity=wrong_identity,
            shape=key.shape,
            dtype=key.dtype,
            layout=key.layout,
            lifetime=key.lifetime,
            access=key.access,
        )
        old_access = next(
            access for access in graph.state_accesses if access.state_ref == key.id
        )
        wrong_access = StateAccess.create(
            node_ref=old_access.node_ref,
            state_ref=wrong_decl.id,
            mode=old_access.mode,
            rank=old_access.rank,
        )
        wrong_layer = _rebuild(
            graph,
            persistent_states=tuple(
                wrong_decl if state.id == key.id else state
                for state in graph.persistent_states
            ),
            state_accesses=tuple(
                wrong_access if access.id == old_access.id else access
                for access in graph.state_accesses
            ),
        )
        with self.assertRaisesRegex(SchemaError, "state declarations are not exact"):
            DenseIR0Validator.validate(wrong_layer)


if __name__ == "__main__":
    unittest.main()
