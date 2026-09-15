"""Strict persistent and linked HBM state ABI for AdamW master/m/v/step."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import StateABI
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess, PersistentStateDecl, PersistentStateIdentity,
    PersistentStateLifetime, StateKind,
)


def _state(kind: StateKind) -> tuple[PersistentStateDecl, StateABI]:
    shape = (1,) if kind is StateKind.OPTIMIZER_STEP else (8, 4)
    dtype = DType.INT32 if kind is StateKind.OPTIMIZER_STEP else DType.FP32
    identity = PersistentStateIdentity.create(
        kind=kind, instance_ref="model", mesh_ref="mesh", request_ref=None,
        layer_index=None, tensor_ref="weight", shard_index=0, generation=0,
    )
    declaration = PersistentStateDecl.create(
        identity=identity, shape=shape, dtype=dtype, layout="row_major",
        lifetime=PersistentStateLifetime.PERSISTENT,
        access=PersistentStateAccess.READ_WRITE,
    )
    abi = StateABI.create(
        state_ref=declaration.id, hbm_binding_ref=f"hbm.{kind.value}",
        kind=kind, lifetime=declaration.lifetime, access=declaration.access,
        shape=shape, dtype=dtype, layout=declaration.layout, die_id=0,
        address=0x1000, size_bytes=declaration.tensor_bytes,
        alignment_bytes=64,
    )
    return declaration, abi


class AdamwStateAbiTests(unittest.TestCase):
    def test_four_independent_true_hbm_state_kinds(self) -> None:
        states = tuple(_state(kind) for kind in (
            StateKind.OPTIMIZER_MASTER, StateKind.OPTIMIZER_MOMENT1,
            StateKind.OPTIMIZER_MOMENT2, StateKind.OPTIMIZER_STEP,
        ))
        self.assertEqual(
            tuple(abi.size_bytes for _, abi in states), (128, 128, 128, 4)
        )
        self.assertEqual(states[-1][1].dtype, DType.INT32)
        self.assertEqual(
            len({abi.state_ref for _, abi in states}), 4
        )

    def test_step_requires_scalar_int32_and_optimizer_reserved_cannot_dma(self) -> None:
        _, step = _state(StateKind.OPTIMIZER_STEP)
        for changed in (
            replace(step, dtype=DType.FP32),
            replace(step, shape=(2,), size_bytes=8),
            replace(step, access=PersistentStateAccess.RESERVED),
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(SchemaError):
                    changed.validate("step")
        with self.assertRaises(SchemaError):
            replace(step, kind=StateKind.OPTIMIZER_RESERVED).validate("reserved")


if __name__ == "__main__":
    unittest.main()
