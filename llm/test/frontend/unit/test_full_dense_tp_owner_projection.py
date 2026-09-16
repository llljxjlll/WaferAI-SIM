"""A TP4 Dense parameter derivative must execute on its exact StateABI home."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    _dense_train_tp_owner_placements,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    OpKind, StateAccess, StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind


class DenseTpOwnerProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.node = SimpleNamespace(
            id="wgrad::T0.layer0.w_qkv::tp2::step0__dp0",
            kind=OpKind.GEMM_WEIGHT_WGRAD,
        )
        self.group = SimpleNamespace(placements=tuple(
            SimpleNamespace(rank=rank, die_id=rank) for rank in range(4)
        ))
        self.access = StateAccess.create(
            node_ref=self.node.id, state_ref="real_parameter_state_2",
            rank=2, mode=StateAccessMode.READ,
        )
        self.state = SimpleNamespace(
            id=self.access.state_ref,
            identity=SimpleNamespace(
                kind=StateKind.TRAINABLE_PARAMETER, shard_index=2,
            ),
        )
        self.binding = SimpleNamespace(
            state_ref=self.state.id, die_id=2,
        )
        self.ir1 = SimpleNamespace(
            state_accesses=(self.access,),
            persistent_state_manifest=SimpleNamespace(
                declarations=(self.state,), bindings=(self.binding,),
            ),
        )

    def test_exact_tp4_wgrad_and_optimizer_execute_once_on_hbm_owner(self) -> None:
        self.assertEqual(
            _dense_train_tp_owner_placements(self.ir1, self.node, self.group),
            (self.group.placements[2],),
        )
        optimizer = SimpleNamespace(
            id="sgd_update::T0.layer0.w_qkv::tp2::step1__dp0",
            kind=OpKind.OPTIMIZER_UPDATE,
        )
        optimizer_ir1 = SimpleNamespace(
            state_accesses=(StateAccess.create(
                node_ref=optimizer.id, state_ref=self.state.id,
                rank=2, mode=StateAccessMode.READ_WRITE,
            ),),
            persistent_state_manifest=self.ir1.persistent_state_manifest,
        )
        self.assertEqual(
            _dense_train_tp_owner_placements(
                optimizer_ir1, optimizer, self.group,
            ), (self.group.placements[2],),
        )
        forward = SimpleNamespace(id="T0.layer0.qkv", kind=OpKind.GEMM)
        self.assertEqual(
            _dense_train_tp_owner_placements(self.ir1, forward, self.group),
            self.group.placements,
        )

    def test_missing_state_access_or_wrong_hbm_home_fails_closed(self) -> None:
        missing = replace(self.access, rank=1)
        with self.assertRaisesRegex(SchemaError, "owner StateABI"):
            _dense_train_tp_owner_placements(
                SimpleNamespace(
                    state_accesses=(missing,),
                    persistent_state_manifest=self.ir1.persistent_state_manifest,
                ), self.node, self.group,
            )
        with self.assertRaisesRegex(SchemaError, "owner StateABI"):
            _dense_train_tp_owner_placements(
                SimpleNamespace(
                    state_accesses=(self.access,),
                    persistent_state_manifest=SimpleNamespace(
                        declarations=(self.state,),
                        bindings=(SimpleNamespace(
                            state_ref=self.state.id, die_id=1,
                        ),),
                    ),
                ), self.node, self.group,
            )
        with self.assertRaisesRegex(SchemaError, "one physical StateAccess"):
            _dense_train_tp_owner_placements(
                SimpleNamespace(
                    state_accesses=(),
                    persistent_state_manifest=self.ir1.persistent_state_manifest,
                ), self.node, self.group,
            )


if __name__ == "__main__":
    unittest.main()
