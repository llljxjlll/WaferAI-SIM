"""Derive CE-backward wire shape from a physical Dense CE forward.

The legacy factory reuses the *value* of forward loss as upstream only for a
timing surrogate.  A complete training program must call the independent
dLoss factory and explicitly seed that separate FP32 per-row buffer.
"""

from __future__ import annotations

import struct

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    ProgramSymbol,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    SemanticOperandId,
)
from ..schema.common import stable_artifact_id


_SCHEMA = "wafer_frontend.dense_training_ce_backward/v1alpha1"


def build_dense_training_ce_backward_record(
    forward_record: RelocatableRecord,
    *,
    logits: ProgramSymbol,
    labels: ProgramSymbol,
    upstream: ProgramSymbol,
    logits_gradient: ProgramSymbol,
    independent_loss_gradient: bool = False,
) -> RelocatableRecord:
    """Bind CE backward to real logits/labels; default is a timing surrogate.

    The native ISA writes FP16 logits gradient.  The later WGRAD stage must
    explicitly cast to FP32 when it needs an FP32 upstream value.  Unless
    independent_loss_gradient=True, the forward loss output is incorrectly
    interpreted as dLoss and must never be released as full-model training.
    """

    forward_record.validate("forward_ce")
    if forward_record.opcode is not RecordOpcode.CROSS_ENTROPY_FORWARD:
        raise SchemaError("requires a physical CE forward", path="forward_record")
    if (forward_record.operands[4].symbol_ref != logits.id
            or forward_record.operands[5].symbol_ref != labels.id):
        raise SchemaError(
            "CE backward logits/labels differ from physical forward",
            path="forward_record.operands",
        )
    forward_loss_ref = forward_record.operands[6].symbol_ref
    if independent_loss_gradient and upstream.id == forward_loss_ref:
        raise SchemaError(
            "CE backward upstream gradient must be independent of forward loss",
            path="forward_record.operands",
        )
    if not independent_loss_gradient and upstream.id != forward_loss_ref:
        raise SchemaError(
            "legacy timing surrogate must reuse the forward loss symbol",
            path="forward_record.operands",
        )
    symbols = (logits, labels, upstream, logits_gradient)
    if len({symbol.id for symbol in symbols}) != 4:
        raise SchemaError("CE backward must use four distinct SRAM symbols", path="symbols")
    literals = {
        operand.name: operand.literal_value
        for operand in forward_record.operands
        if operand.literal_value is not None
    }
    if (
        literals["logits_datatype"] != 1
        or literals["label_datatype"] != 2
        or literals["loss_datatype"] != 3
        or literals["reduction"] != 0
    ):
        raise SchemaError("forward CE dtype/reduction is incompatible", path="forward_record")
    rows = literals["rank_rows"]
    action = stable_artifact_id(
        "dense_training_ce_backward_action",
        {
            "forward_action": forward_record.source_global_action_id,
            "symbols": tuple(symbol.id for symbol in symbols),
            "rows": rows,
            "tp": literals["tp_degree"],
            "vocab": literals["vocab_size"],
        },
        schema_version=_SCHEMA,
    )
    record = RelocatableRecord(
        action,
        RecordOpcode.CROSS_ENTROPY_BACKWARD,
        (
            RecordOperand.literal("logits_datatype", 1),
            RecordOperand.literal("label_datatype", 2),
            RecordOperand.literal("upstream_datatype", 3),
            RecordOperand.literal("output_datatype", 1),
            RecordOperand.literal("reduction", 0),
            RecordOperand.literal("upstream_mode", 1),
            RecordOperand.address("logits_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, logits.id),
            RecordOperand.address("labels_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, labels.id),
            RecordOperand.address("upstream_address", SemanticOperandId.COMPUTE_AUX_ADDRESS, upstream.id),
            RecordOperand.address("logits_grad_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, logits_gradient.id),
            RecordOperand.literal("logical_rows", literals["logical_rows"]),
            RecordOperand.literal("rank_rows", rows),
            RecordOperand.literal("tp_degree", literals["tp_degree"]),
            RecordOperand.literal("vocab_size", literals["vocab_size"]),
            RecordOperand.literal("upstream_elements", rows),
        ),
    )
    record.validate("physical_ce_backward")
    return record


def build_dense_training_ce_backward_from_loss_gradient(
    forward_record: RelocatableRecord,
    *,
    logits: ProgramSymbol,
    labels: ProgramSymbol,
    loss_gradient: ProgramSymbol,
    logits_gradient: ProgramSymbol,
) -> RelocatableRecord:
    """Use an independent FP32 per-row dLoss, preserving real logits/labels."""
    return build_dense_training_ce_backward_record(
        forward_record,
        logits=logits,
        labels=labels,
        upstream=loss_gradient,
        logits_gradient=logits_gradient,
        independent_loss_gradient=True,
    )


def dense_training_per_row_loss_gradient_seed(rank_rows: int) -> bytes:
    """Explicit nonzero dLoss/dLoss=1.0 per row for timing ProgramIO."""
    if type(rank_rows) is not int or not 0 < rank_rows <= 1 << 20:
        raise SchemaError("invalid rank-local loss-gradient row count", path="rank_rows")
    return struct.pack("<f", 1.0) * rank_rows


__all__ = [
    "build_dense_training_ce_backward_record",
    "build_dense_training_ce_backward_from_loss_gradient",
    "dense_training_per_row_loss_gradient_seed",
]
