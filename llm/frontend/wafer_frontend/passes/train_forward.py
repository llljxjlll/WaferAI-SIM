"""Build the N6.1 forward-only dense training graph."""

from __future__ import annotations

from ..schema.common import MeshAxisName, ProfileKey
from ..schema.experiment import ExperimentSpec, InstanceRole, WorkloadMode
from ..schema.ir0 import (
    DeviceMesh,
    IR0,
    JobKind,
    LogicalInstance,
    LogicalRole,
    MeshAxis,
    ParallelAxes,
)
from ..schema.logical import DenseModelShape
from .logical_expand import _expand_dense_graph


def build_train_forward_ir0(spec: ExperimentSpec) -> IR0:
    """Expand one validated forward-only TRAIN spec without entering placement."""

    spec.validate("spec")
    if spec.workload.mode is not WorkloadMode.TRAIN:
        raise ValueError("build_train_forward_ir0 requires a TRAIN workload")
    train = spec.workload.train
    assert train is not None
    instance_spec = spec.parallel.instances[0]
    if instance_spec.role is not InstanceRole.TRAIN:
        raise ValueError("build_train_forward_ir0 requires role=TRAIN")

    model_spec = spec.model
    model = DenseModelShape(
        vocab_size=model_spec.V,
        hidden_size=model_spec.H,
        intermediate_size=model_spec.I,
        num_layers=model_spec.L,
        num_heads=model_spec.NH,
        num_kv_heads=model_spec.KVH,
        head_dim=model_spec.DH,
        rotary_dim=model_spec.rotary_dim,
        dtype=model_spec.dtype,
        tie_word_embeddings=model_spec.tie_word_embeddings,
        rms_norm_epsilon=model_spec.rms_norm_epsilon,
        rope_theta=model_spec.rope_theta,
        max_position_embeddings=model_spec.max_position_embeddings,
    )
    mesh = DeviceMesh(
        id=f"{instance_spec.id}.mesh.tp",
        axes=(MeshAxis(MeshAxisName.TP, instance_spec.tp),),
    )
    instance = LogicalInstance(
        id=instance_spec.id,
        role=LogicalRole.TRAIN,
        replicas=instance_spec.replicas,
        parallel=ParallelAxes(
            tp=instance_spec.tp,
            sp=instance_spec.sp,
            dp=instance_spec.dp,
            pp=instance_spec.pp,
            ep=instance_spec.ep,
        ),
        meshes=(mesh,),
    )
    tokens = train.micro_batch * train.seq_len
    profile = ProfileKey(
        prefill_tokens=tokens,
        decode_tokens=0,
        num_seqs=train.micro_batch,
        context_sum=tokens,
        context_max=train.seq_len,
        kv_pages=0,
        expert_load=None,
    )
    graph = _expand_dense_graph(
        model,
        instance,
        profile,
        None,
        job=JobKind.TRAIN,
        train=train.structure,
        infer_output=None,
        producer_pass="train_forward_expand",
    )
    graph.validate("train_forward_ir0")
    from .validate_ir0 import DenseIR0Validator
    from .train_forward_oracle import build_train_forward_oracle

    DenseIR0Validator.validate(graph, "train_forward_ir0")
    build_train_forward_oracle(spec).validate_against_ir0(
        spec,
        graph,
        path="train_forward_oracle",
    )
    return graph
