"""Build the profile-independent Dense IR-0 template from a validated spec."""

from __future__ import annotations

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import MeshAxisName, ProfileKey
from ..schema.experiment import (
    ExperimentSpec,
    InstanceRole,
    ParallelInstanceSpec,
    WorkloadMode,
)
from ..schema.ir0 import (
    DeviceMesh,
    JobKind,
    LogicalInstance,
    LogicalRole,
    MeshAxis,
    ParallelAxes,
)
from ..schema.logical import (
    DenseLayerKind,
    DenseLayerTemplate,
    DenseModelShape,
    ElementwiseKind,
    IR0Template,
    NormKind,
    ProfileEntry,
    SequenceParallelSpec,
)
from ..schema.stage3_profile import Stage3StaticProfile


_LOGICAL_ROLE = {
    InstanceRole.PREFILL: LogicalRole.PREFILL,
    InstanceRole.DECODE: LogicalRole.DECODE,
    InstanceRole.BOTH: LogicalRole.BOTH,
}


def _unsupported(message: str, *, path: str) -> None:
    raise UnsupportedFeatureError(message, path=path)


def _source_profiles(
    spec: ExperimentSpec,
) -> tuple[tuple[ProfileKey, float, str], ...]:
    infer = spec.workload.infer
    if infer.profile is not None:
        return ((infer.profile, 1.0, "spec.workload.infer.profile"),)
    assert infer.shape_dist is not None
    return tuple(
        (
            entry.key,
            entry.weight,
            f"spec.workload.infer.shape_dist.profiles[{index}].key",
        )
        for index, entry in enumerate(infer.shape_dist.profiles)
    )


def _validate_tp_sp(*, tp: int, sp: bool, path: str) -> None:
    if tp == 1 and not sp:
        return
    if tp > 1 and sp:
        return
    if tp == 1:
        _unsupported(
            "sequence parallelism over TP=1 is a meaningless no-op",
            path=f"{path}.sp",
        )
    _unsupported(
        "TP>1 without sequence parallelism requires AllReduce, which is outside the MVP",
        path=f"{path}.sp",
    )


def _validate_attention_profile(
    key: ProfileKey,
    exact_profile: Stage3StaticProfile | None,
    *,
    path: str,
) -> None:
    if exact_profile is not None:
        exact_profile.validate(f"{path}.exact_profile")
        if exact_profile.key != key:
            raise SchemaError(
                "must equal the source ProfileKey",
                path=f"{path}.exact_profile.key",
            )
        return
    prefill = key.prefill_tokens
    decode = key.decode_tokens
    if prefill and decode:
        _unsupported(
            "mixed prefill/decode profiles do not uniquely determine attention work",
            path=path,
        )
    if prefill:
        if key.num_seqs != 1:
            _unsupported(
                "prefill MVP requires exactly one sequence",
                path=f"{path}.num_seqs",
            )
        if key.context_sum != prefill or key.context_max != prefill:
            _unsupported(
                "prefill MVP requires context_sum == context_max == prefill_tokens",
                path=f"{path}.context_sum",
            )
        return
    if decode != key.num_seqs:
        _unsupported(
            "decode MVP requires exactly one query token per sequence",
            path=f"{path}.decode_tokens",
        )


def _build_template_for_instance_spec(
    spec: ExperimentSpec,
    instance_spec: ParallelInstanceSpec,
    profiles: tuple[ProfileEntry, ...],
) -> IR0Template:
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
    logical_instance = LogicalInstance(
        id=instance_spec.id,
        role=_LOGICAL_ROLE[instance_spec.role],
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
    sequence_parallel = SequenceParallelSpec(
        enabled=instance_spec.sp,
        reuse_axis=MeshAxisName.TP if instance_spec.sp else None,
    )
    layer = DenseLayerTemplate(
        id=f"{instance_spec.id}.dense_block",
        kind=DenseLayerKind.LLAMA_DENSE_BLOCK_V1,
        norm=NormKind.RMS_NORM,
        activation=ElementwiseKind.SWIGLU,
        has_bias=False,
    )
    result = IR0Template.create(
        job=JobKind.INFER,
        model=model,
        instance=logical_instance,
        sequence_parallel=sequence_parallel,
        layer=layer,
        infer_output=spec.workload.infer.output,
        profiles=profiles,
    )
    result.validate("ir0_template")
    return result


def build_ir0_template_for_profile(
    spec: ExperimentSpec,
    *,
    instance_ref: str,
    exact_profile: Stage3StaticProfile,
) -> IR0Template:
    """Build one exact per-instance template for a selected Stage 4 phase."""

    spec.validate("spec")
    if spec.workload.mode is WorkloadMode.TRAIN:
        _unsupported(
            "N6.1 train graph expansion is not enabled",
            path="spec.workload.mode",
        )
    if type(exact_profile) is not Stage3StaticProfile:
        raise SchemaError(
            "must be a Stage3StaticProfile", path="exact_profile"
        )
    exact_profile.validate("exact_profile")
    instances = {
        instance.id: instance for instance in spec.parallel.instances
    }
    instance_spec = instances.get(instance_ref)
    if instance_spec is None:
        raise SchemaError(
            "references an unknown instance", path="instance_ref"
        )
    if exact_profile.key.stable_id() not in {
        profile.stable_id() for profile in spec.workload.infer.profiles()
    }:
        raise SchemaError(
            "profile key is absent from the ExperimentSpec",
            path="exact_profile.key",
        )
    _validate_tp_sp(
        tp=instance_spec.tp,
        sp=instance_spec.sp,
        path=f"spec.parallel.instances[{instance_ref!r}]",
    )
    _validate_attention_profile(
        exact_profile.key,
        exact_profile,
        path="exact_profile.key",
    )
    profiles = (
        ProfileEntry.create(
            key=exact_profile.key,
            weight=1.0,
            exact_profile=exact_profile,
        ),
    )
    return _build_template_for_instance_spec(spec, instance_spec, profiles)


def build_ir0(
    spec: ExperimentSpec,
    *,
    exact_profiles: tuple[Stage3StaticProfile, ...] = (),
) -> IR0Template:
    """Return a canonical Dense template without materializing an IR-0 graph."""

    spec.validate("spec")
    if spec.workload.mode is WorkloadMode.TRAIN:
        _unsupported(
            "N6.1 train graph expansion is not enabled",
            path="spec.workload.mode",
        )
    instance_spec = spec.parallel.instances[0]
    instance_path = "spec.parallel.instances[0]"
    _validate_tp_sp(
        tp=instance_spec.tp,
        sp=instance_spec.sp,
        path=instance_path,
    )

    source_profiles = _source_profiles(spec)
    exact_by_key: dict[str, Stage3StaticProfile] = {}
    source_keys = {key.stable_id() for key, _weight, _path in source_profiles}
    for index, exact_profile in enumerate(exact_profiles):
        if type(exact_profile) is not Stage3StaticProfile:
            raise SchemaError(
                "must be a Stage3StaticProfile",
                path=f"exact_profiles[{index}]",
            )
        exact_profile.validate(f"exact_profiles[{index}]")
        key_id = exact_profile.key.stable_id()
        if key_id not in source_keys:
            raise SchemaError(
                "profile key is absent from the ExperimentSpec",
                path=f"exact_profiles[{index}].key",
            )
        if key_id in exact_by_key:
            raise SchemaError(
                "contains a duplicate ProfileKey",
                path=f"exact_profiles[{index}].key",
            )
        exact_by_key[key_id] = exact_profile
    for key, _weight, profile_path in source_profiles:
        _validate_attention_profile(
            key,
            exact_by_key.get(key.stable_id()),
            path=profile_path,
        )
    profiles = tuple(
        sorted(
            (
                ProfileEntry.create(
                    key=key,
                    weight=weight,
                    exact_profile=exact_by_key.get(key.stable_id()),
                )
                for key, weight, _path in source_profiles
            ),
            key=lambda entry: entry.profile_id,
        )
    )

    return _build_template_for_instance_spec(spec, instance_spec, profiles)
