"""Pure analytic producer for S2-Lite LM-head-only training."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.lite_train import (
    S2LiteCeBackwardMetrics,
    S2LiteLmHeadTrainContract,
    S2LiteLmHeadTrainOracle,
    S2LiteLmHeadWgradMetrics,
    S2LiteSgdMetrics,
)


def build_s2_lite_lm_head_train_oracle(
    contract: S2LiteLmHeadTrainContract,
) -> S2LiteLmHeadTrainOracle:
    """Derive the complete Lite work/byte oracle without consulting a graph."""

    if type(contract) is not S2LiteLmHeadTrainContract:
        raise SchemaError(
            "must be an S2LiteLmHeadTrainContract", path="contract"
        )
    contract.validate("contract")

    rows = contract.micro_batch_size * contract.sequence_length
    hidden_elements = rows * contract.hidden_size
    logits_elements = rows * contract.vocabulary_size
    weight_elements = contract.hidden_size * contract.vocabulary_size

    hidden_bytes = 2 * hidden_elements
    logits_bytes = 2 * logits_elements
    labels_bytes = 4 * rows
    loss_gradient_bytes = 4 * rows
    weight_bytes = 2 * weight_elements
    weight_gradient_bytes = 4 * weight_elements

    ce_backward = S2LiteCeBackwardMetrics(
        element_count=logits_elements,
        logits_read_bytes=logits_bytes,
        labels_read_bytes=labels_bytes,
        loss_gradient_read_bytes=loss_gradient_bytes,
        logits_gradient_write_bytes=logits_bytes,
    )
    lm_head_wgrad = S2LiteLmHeadWgradMetrics(
        weight_element_count=weight_elements,
        hidden_read_bytes=hidden_bytes,
        logits_gradient_read_bytes=logits_bytes,
        weight_gradient_write_bytes=weight_gradient_bytes,
        floating_point_ops=2 * rows * contract.hidden_size * contract.vocabulary_size,
    )
    sgd_update = S2LiteSgdMetrics(
        element_count=weight_elements,
        weight_read_bytes=weight_bytes,
        gradient_read_bytes=weight_gradient_bytes,
        updated_weight_write_bytes=weight_bytes,
        floating_point_ops=2 * weight_elements,
    )
    return S2LiteLmHeadTrainOracle.create(
        source_contract_id=contract.id,
        source_spec_digest=contract.source_spec_digest,
        case_id=contract.case_id,
        stages=contract.stages,
        dependency_count=len(contract.stages) - 1,
        logical_rows=rows,
        hidden_elements=hidden_elements,
        logits_elements=logits_elements,
        lm_head_weight_elements=weight_elements,
        lm_head_weight_bytes=weight_bytes,
        lm_head_weight_gradient_bytes=weight_gradient_bytes,
        ce_backward=ce_backward,
        lm_head_wgrad=lm_head_wgrad,
        sgd_update=sgd_update,
        total_read_bytes=(
            ce_backward.logits_read_bytes
            + ce_backward.labels_read_bytes
            + ce_backward.loss_gradient_read_bytes
            + lm_head_wgrad.hidden_read_bytes
            + lm_head_wgrad.logits_gradient_read_bytes
            + sgd_update.weight_read_bytes
            + sgd_update.gradient_read_bytes
        ),
        total_write_bytes=(
            ce_backward.logits_gradient_write_bytes
            + lm_head_wgrad.weight_gradient_write_bytes
            + sgd_update.updated_weight_write_bytes
        ),
        total_floating_point_ops=(
            lm_head_wgrad.floating_point_ops + sgd_update.floating_point_ops
        ),
    )


__all__ = ["build_s2_lite_lm_head_train_oracle"]
