from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir2 import (
    DmaContract,
    OrdinaryNodeOrigin,
    OriginKind,
    SemanticTask,
    SemanticTaskKind,
    StateIoOrigin,
    TensorSlice,
)


def _dma_task(kind: SemanticTaskKind) -> SemanticTask:
    local_value_ref = "state-stage"
    return SemanticTask(
        id="dma-in" if kind is SemanticTaskKind.DMA_IN else "dma-out",
        kind=kind,
        origin_ref=StateIoOrigin(
            kind=OriginKind.STATE_IO,
            state_access_ref="access-0",
            node_ref="node-0",
            rank=0,
        ),
        region_id="state-region",
        op_kind=None,
        member_id=None,
        flow_id=None,
        chunk_id=None,
        collective_step=None,
        source_rank=None,
        destination_rank=None,
        tensor_slice=TensorSlice(
            value_id=local_value_ref,
            offset=(0, 0),
            shape=(2, 2),
        ),
        bytes=8,
        dtype=DType.FP16,
        shape=(2, 2),
        read_values=(
            (local_value_ref,)
            if kind is SemanticTaskKind.DMA_OUT
            else ()
        ),
        write_values=(
            (local_value_ref,)
            if kind is SemanticTaskKind.DMA_IN
            else ()
        ),
        compute=None,
        reduction=None,
        sync=None,
        deps=() if kind is SemanticTaskKind.DMA_IN else ("access-task",),
        dma=DmaContract(
            state_ref="state-0",
            local_value_ref=local_value_ref,
            state_offset_bytes=0,
            access_task_refs=("access-task",),
        ),
    )


class IR2StateDmaSchemaTest(unittest.TestCase):
    def test_dma_in_has_one_local_write_endpoint(self) -> None:
        _dma_task(SemanticTaskKind.DMA_IN).validate("task")

    def test_dma_out_has_one_local_read_endpoint(self) -> None:
        _dma_task(SemanticTaskKind.DMA_OUT).validate("task")

    def test_dma_in_rejects_two_buffer_copy_shape(self) -> None:
        task = _dma_task(SemanticTaskKind.DMA_IN)
        with self.assertRaises(SchemaError):
            replace(
                task,
                read_values=(task.dma.local_value_ref,),
                write_values=(task.dma.local_value_ref,),
            ).validate("task")

    def test_dma_requires_state_io_origin(self) -> None:
        task = _dma_task(SemanticTaskKind.DMA_IN)
        with self.assertRaises(SchemaError):
            replace(
                task,
                origin_ref=OrdinaryNodeOrigin(
                    kind=OriginKind.ORDINARY,
                    op_id="node-0",
                    rank=0,
                ),
            ).validate("task")

    def test_non_dma_rejects_state_contract(self) -> None:
        task = _dma_task(SemanticTaskKind.DMA_IN)
        with self.assertRaises(SchemaError):
            replace(task, kind=SemanticTaskKind.LOCAL_COPY).validate("task")

    def test_dma_accepts_nonzero_directional_state_offset(self) -> None:
        task = _dma_task(SemanticTaskKind.DMA_IN)
        assert task.dma is not None
        candidate = replace(
            task,
            dma=replace(task.dma, state_offset_bytes=2),
        )
        candidate.validate("task")
        assert candidate.dma is not None
        self.assertEqual(candidate.dma.state_offset_bytes, 2)

    def test_dma_rejects_non_tight_payload_bytes(self) -> None:
        task = _dma_task(SemanticTaskKind.DMA_OUT)
        with self.assertRaises(SchemaError):
            replace(task, bytes=7).validate("task")


if __name__ == "__main__":
    unittest.main()
