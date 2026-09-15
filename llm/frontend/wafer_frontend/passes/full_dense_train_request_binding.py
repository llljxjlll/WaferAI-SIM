"""Bind user-visible Dense TRAIN intent to the exact forward/source oracle.

The old ExperimentSpec is forward-only.  A valid unified TRAIN request does
not turn that old source, a gradient requirement, or an optimizer motif into a
complete run; physical source lineage and runtime coverage remain mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.full_dense_gradient_requirements import (
    DenseFullTrainRequirements,
)
from ..schema.serde import canonical_digest
from ..schema.workload_run import (
    WorkloadFamily,
    WorkloadModelArchitecture,
    WorkloadOptimizerKind,
    WorkloadRunRequest,
)


@dataclass(frozen=True, slots=True)
class DenseFullTrainRequestBinding:
    """Stable input lineage; this carrier deliberately has no SUCCESS flag."""

    case_id: str
    request_digest: str
    source_plan_id: str
    source_plan_digest: str
    requirements_digest: str


def bind_dense_full_train_request(
    request: WorkloadRunRequest,
    source_plan: FlexibleDenseTrainPlan,
    requirements: DenseFullTrainRequirements,
) -> DenseFullTrainRequestBinding:
    if type(request) is not WorkloadRunRequest:
        raise SchemaError("requires a versioned full workload request", path="request")
    request.validate("request")
    if type(requirements) is not DenseFullTrainRequirements:
        raise SchemaError("requires independently derived requirements", path="requirements")
    requirements.validate_against(source_plan)
    if request.family is not WorkloadFamily.DENSE_TRAINING:
        raise SchemaError("requires Dense full training", path="request.family")
    if request.execution.functional:
        raise UnsupportedFeatureError(
            "complete training functional oracle has not been implemented",
            path="request.execution.functional",
        )
    if not request.execution.timing:
        raise SchemaError("training requires timing execution", path="request.execution")
    experiment = source_plan.source_experiment
    source_model = experiment.model
    source_train = experiment.workload.train
    assert source_train is not None
    model = request.model
    expected_model = (
        WorkloadModelArchitecture.LLAMA_DENSE,
        source_model.V,
        source_model.H,
        source_model.I,
        source_model.L,
        source_model.NH,
        source_model.KVH,
        source_model.DH,
        source_model.max_position_embeddings,
        source_model.dtype,
    )
    actual_model = (
        model.architecture,
        model.vocabulary_size,
        model.hidden_size,
        model.intermediate_size,
        model.num_layers,
        model.num_attention_heads,
        model.num_kv_heads,
        model.head_dim,
        model.max_sequence_length,
        model.dtype,
    )
    if actual_model != expected_model or model.num_experts or source_model.moe is not None:
        raise SchemaError("unified case model differs from exact source forward IR0", path="request.model")
    training = request.steps.training
    assert training is not None and request.optimizer is not None
    if (
        training.step_count != requirements.steps
        or training.global_batch_size != source_train.global_batch
        or training.micro_batch_size != source_train.micro_batch
        or training.micro_batch_count != source_train.structure.micro_batch_count
        or training.sequence_length != source_train.seq_len
    ):
        raise SchemaError("TRAIN steps/batch/sequence differ from source graph", path="request.steps")
    if (
        request.mesh.rows != source_plan.spec.mesh.rows
        or request.mesh.columns != source_plan.spec.mesh.columns
        or request.parallel.tp != source_plan.spec.tp_degree
        or request.parallel.dp != source_plan.spec.dp_degree
        or request.parallel.ep != 1
        or request.parallel.pp != 1
        or request.parallel.active_die_ids not in (
            (), tuple(range(source_plan.spec.mesh.rank_count))
        )
    ):
        raise SchemaError("physical rank/TP/DP mapping differs from source", path="request.parallel")
    optimizer = request.optimizer
    if (
        optimizer.kind is not WorkloadOptimizerKind.SGD
        or optimizer.learning_rate != source_plan.spec.learning_rate
        or optimizer.weight_decay != 0.0
    ):
        raise SchemaError("SGD contract differs from source parameter plan", path="request.optimizer")
    return DenseFullTrainRequestBinding(
        case_id=request.case_id,
        request_digest=request.digest,
        source_plan_id=source_plan.id,
        source_plan_digest=canonical_digest(source_plan),
        requirements_digest=canonical_digest(requirements),
    )
