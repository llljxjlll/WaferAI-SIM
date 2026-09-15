"""Independent source-backed requirements for a complete Dense train program.

This is a *requirement oracle*, not a backward lowering or success receipt.
The legacy flexible train plan explicitly carries non-materialized backward
labels; no caller may infer physical gradient coverage from these names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from ..errors import SchemaError
from .common import DType
from .flexible_dense_train import FlexibleDenseTrainPlan
from .ir0 import CrossEntropyForwardWorkload, OpKind, OpPhase
from .persistent_state import StateKind
from .serde import canonical_digest


class DenseGradientLossObjective(str, Enum):
    """Sum every independently seeded per-row CE loss, without a mean."""

    PER_ROW_CE_SUM = "per_row_ce_sum"


class DenseGradientDPReduction(str, Enum):
    """Use the backend's rank-major FP32 SUM, not an implied MEAN."""

    FP32_RANK_MAJOR_SUM = "fp32_rank_major_sum"


@dataclass(frozen=True, slots=True)
class DenseRequiredGradientPath:
    """What *one real physical rank* must demonstrate for one parameter/step."""

    step: int
    parameter_state_ref: str
    rank: int
    read_version: int
    write_version: int
    tp_shard_index: int
    dp_group_ranks: tuple[int, ...]
    forward_op_refs: tuple[str, ...]
    backward_producer_refs: tuple[str, ...]
    named_wgrad_op_ref: str
    named_sync_op_ref: str
    named_optimizer_op_ref: str
    named_store_op_ref: str
    loss_gradient_seed_ref: str
    gradient_bytes: int
    weight_bytes: int


@dataclass(frozen=True, slots=True)
class DenseFullTrainRequirements:
    """A complete per-rank/per-step checklist bound to validated forward IR0."""

    source_plan_id: str
    source_plan_digest: str
    source_forward_graph_digest: str
    steps: int
    forward_ce_ref: str
    backward_ce_ref: str
    loss_gradient_seed_ref: str
    loss_gradient_seed_dtype: DType
    loss_gradient_seed_per_row: float
    loss_objective: DenseGradientLossObjective
    dp_reduction: DenseGradientDPReduction
    optimizer_gradient_normalization: bool
    forward_loss_value_ref: str
    required_forward_refs: tuple[str, ...]
    required_backbone_backward_refs: tuple[str, ...]
    paths: tuple[DenseRequiredGradientPath, ...]

    @property
    def required_gradient_producers(
        self,
    ) -> Mapping[tuple[int, str, int], DenseRequiredGradientPath]:
        """Step, *source state*, physical die/rank -> externally provable chain."""

        return MappingProxyType(
            {
                (path.step, path.parameter_state_ref, path.rank): path
                for path in self.paths
            }
        )

    def validate_against(
        self, plan: FlexibleDenseTrainPlan,
        path: str = "dense_full_train_requirements",
    ) -> None:
        """Re-derive independent requirements; do not trust producer labels."""

        expected = build_dense_full_train_requirements(plan, steps=self.steps)
        if self != expected:
            raise SchemaError(
                "gradient producer/owner/version/source contract drifted",
                path=path,
            )
        mapping = self.required_gradient_producers
        if len(mapping) != len(self.paths):
            raise SchemaError("duplicate parameter/step/rank path", path=f"{path}.paths")


def build_dense_full_train_requirements(
    plan: FlexibleDenseTrainPlan,
    *,
    steps: int = 2,
) -> DenseFullTrainRequirements:
    """Derive *unmet until proven* Dense training requirements from real IR0.

    Producer names describe the required physical operation provenance.  They
    do not mean that a timing-only MATMUL motif implements embedding, RMSNorm,
    RoPE or attention backward, nor that an SGD opcode consumes these gradients.
    """

    if type(plan) is not FlexibleDenseTrainPlan:
        raise SchemaError("requires production Dense training plan", path="plan")
    plan.validate()
    if type(steps) is not int or steps < 2:
        raise SchemaError("full training requires at least two steps", path="steps")
    graph = plan.forward_graph
    if (
        plan.source_experiment.model.L < 2
        or graph.instances[0].parallel.tp != plan.spec.tp_degree
        or graph.instances[0].parallel.dp != plan.spec.dp_degree
        or plan.full_model_backward_materialized
    ):
        raise SchemaError("requires source-backed multi-layer non-materialized plan", path="plan")
    ce_nodes = tuple(node for node in graph.nodes if node.kind is OpKind.CE_FORWARD)
    if (
        len(ce_nodes) != 1
        or type(ce_nodes[0].workload) is not CrossEntropyForwardWorkload
        or any(node.phase is not OpPhase.FWD for node in graph.nodes)
    ):
        raise SchemaError("requires complete forward/loss source graph", path="plan.forward_graph")
    ce = ce_nodes[0]
    values = {value.id: value for value in graph.values}
    loss = values[ce.outputs[0]]
    if loss.producer != ce.id or loss.dtype is not DType.FP32:
        raise SchemaError("source CE loss producer drifted", path="plan.forward_graph")
    state_ids = {state.id for state in graph.persistent_states}
    template_ids = {item.state_ref for item in plan.parameter_templates}
    if (
        not state_ids
        or state_ids != template_ids
        or any(state.identity.kind is not StateKind.PARAMETER for state in graph.persistent_states)
    ):
        raise SchemaError("all persistent parameter shards must be trainable", path="plan")
    seed_ref = f"{ce.instance_id}.loss_gradient"
    if seed_ref in values or seed_ref == loss.id:
        raise SchemaError("dLoss must be an independent input", path="plan.forward_graph")
    forward_node_ids = {node.id for node in graph.nodes}
    rank_count = plan.spec.mesh.rank_count
    paths: list[DenseRequiredGradientPath] = []
    for step in range(steps):
        for template in plan.parameter_templates:
            if (
                not set(template.forward_consumer_refs) <= forward_node_ids
                or len(template.backward_node_refs) != len(template.forward_consumer_refs)
                or template.gradient_bytes != 2 * template.weight_bytes
            ):
                raise SchemaError("parameter derivative source/FP32 extent invalid", path="plan")
            column = template.tp_shard_index
            dp_group = tuple(
                row * plan.spec.tp_degree + column
                for row in range(plan.spec.dp_degree)
            )
            if template.owner_ranks != dp_group:
                raise SchemaError("parameter ownership misses DP replicas", path="plan")
            for rank in dp_group:
                if rank >= rank_count:
                    raise SchemaError("parameter owner escapes physical mesh", path="plan")
                paths.append(
                    DenseRequiredGradientPath(
                        step=step,
                        parameter_state_ref=template.state_ref,
                        rank=rank,
                        read_version=step,
                        write_version=step + 1,
                        tp_shard_index=column,
                        dp_group_ranks=dp_group,
                        forward_op_refs=template.forward_consumer_refs,
                        backward_producer_refs=template.backward_node_refs,
                        named_wgrad_op_ref=template.wgrad_ref,
                        named_sync_op_ref=(
                            f"dp_sync::{template.state_ref}::tp{column}::step{step}"
                            if len(dp_group) > 1
                            else f"local_sync::{template.state_ref}::r{rank}::step{step}"
                        ),
                        named_optimizer_op_ref=(
                            f"sgd::{template.state_ref}::r{rank}::step{step}"
                        ),
                        named_store_op_ref=(
                            f"store::{template.state_ref}::r{rank}::step{step}"
                        ),
                        loss_gradient_seed_ref=seed_ref,
                        gradient_bytes=template.gradient_bytes,
                        weight_bytes=template.weight_bytes,
                    )
                )
    requirements = DenseFullTrainRequirements(
        source_plan_id=plan.id,
        source_plan_digest=canonical_digest(plan),
        source_forward_graph_digest=canonical_digest(graph),
        steps=steps,
        forward_ce_ref=ce.id,
        backward_ce_ref=f"{ce.id}_backward",
        loss_gradient_seed_ref=seed_ref,
        loss_gradient_seed_dtype=DType.FP32,
        loss_gradient_seed_per_row=1.0,
        loss_objective=DenseGradientLossObjective.PER_ROW_CE_SUM,
        dp_reduction=DenseGradientDPReduction.FP32_RANK_MAJOR_SUM,
        optimizer_gradient_normalization=False,
        forward_loss_value_ref=loss.id,
        required_forward_refs=tuple(node.id for node in graph.nodes),
        required_backbone_backward_refs=tuple(
            f"backward::{node.id}" for node in reversed(graph.nodes)
            if node.kind is not OpKind.CE_FORWARD
        ),
        paths=tuple(paths),
    )
    if len(requirements.required_gradient_producers) != len(paths):
        raise SchemaError("duplicate physical parameter paths", path="paths")
    return requirements
