from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from llm.frontend.wafer_frontend.errors import UnsupportedFeatureError
from llm.frontend.wafer_frontend.legacy_dense_backend import (
    LegacyDenseBackendScope,
    assess_legacy_dense_backend,
    require_legacy_dense_backend,
    run_legacy_dense_compiler_canary,
)
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTier, MemoryTierCapacity
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, from_data
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadCapabilityLevel,
    WorkloadExecutionSpec,
    WorkloadFamily,
    WorkloadFamilyCapability,
    WorkloadInferenceSteps,
    WorkloadMemoryMode,
    WorkloadMemoryPolicy,
    WorkloadMeshSpec,
    WorkloadModelArchitecture,
    WorkloadModelSpec,
    WorkloadOptimizerKind,
    WorkloadOptimizerSpec,
    WorkloadParallelSpec,
    WorkloadRunCapability,
    WorkloadRunRequest,
    WorkloadStepSpec,
    WorkloadTrainingSteps,
)
from llm.frontend.wafer_frontend.workload_runner import (
    WorkloadRunnerStage,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces, valid_spec


def _legacy_spec(*, layers: int = 2, prefill: int = 4, decode: int = 2) -> ExperimentSpec:
    data = deepcopy(valid_spec())
    data["model"].update(
        {
            "V": 128,
            "H": 32,
            "I": 64,
            "NH": 4,
            "KVH": 2,
            "DH": 8,
            "rotary_dim": 8,
            "L": layers,
            "max_position_embeddings": 128,
        }
    )
    profile = data["workload"]["infer"]["profile"]
    profile.update(
        {
            "prefill_tokens": prefill,
            "decode_tokens": decode,
            "num_seqs": 1,
            "context_sum": prefill + decode,
            "context_max": prefill + decode,
            "kv_pages": 1,
        }
    )
    data["parallel"]["instances"][0]["role"] = (
        "both" if decode else "prefill"
    )
    return from_data(ExperimentSpec, data, path="spec")


def _model(*, layers: int = 2, moe: bool = False) -> WorkloadModelSpec:
    return WorkloadModelSpec(
        architecture=(
            WorkloadModelArchitecture.LLAMA_MOE
            if moe
            else WorkloadModelArchitecture.LLAMA_DENSE
        ),
        vocabulary_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=layers,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_sequence_length=128,
        dtype=DType.FP16,
        num_experts=4 if moe else 0,
        experts_per_token=1 if moe else 0,
    )


def _request(
    *,
    family: WorkloadFamily = WorkloadFamily.DENSE_INFERENCE,
    layers: int = 2,
    prefill: int = 4,
    decode: int = 2,
    memory_mode: WorkloadMemoryMode = WorkloadMemoryMode.RESIDENT_HBM,
) -> WorkloadRunRequest:
    if family.is_training:
        steps = WorkloadStepSpec(
            training=WorkloadTrainingSteps(2, 1, 1, 1, prefill)
        )
        optimizer = WorkloadOptimizerSpec(WorkloadOptimizerKind.SGD, 0.01)
    else:
        steps = WorkloadStepSpec(
            inference=WorkloadInferenceSteps(prefill, decode, 1)
        )
        optimizer = None
    return WorkloadRunRequest.create(
        family=family,
        model=_model(layers=layers, moe=family.is_moe),
        steps=steps,
        mesh=WorkloadMeshSpec(1, 2),
        parallel=WorkloadParallelSpec(tp=2),
        memory=WorkloadMemoryPolicy(mode=memory_mode),
        optimizer=optimizer,
        execution=WorkloadExecutionSpec(timing=True, functional=False),
    )


def _capability() -> WorkloadRunCapability:
    supported = WorkloadCapabilityLevel.SUPPORTED
    return WorkloadRunCapability.create(
        max_mesh_rows=2,
        max_mesh_columns=2,
        max_mesh_ranks=4,
        families=tuple(
            WorkloadFamilyCapability(
                family=family,
                full_model=supported,
                motif=supported,
                baseline=supported,
                optimized=supported,
                lowering=supported,
                runtime=supported,
                timing=supported,
                functional=supported,
                capacity=supported,
                multi_step=supported,
                remote_hbm=supported,
                external_offload=supported,
                sgd_optimizer=supported,
                adamw_optimizer=supported,
                repeatability=supported,
            )
            for family in WorkloadFamily
        ),
    )


def _capacities() -> tuple[MemoryTierCapacity, ...]:
    return tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{rank}",
            base_address=0,
            capacity_bytes=1 << 30,
            alignment_bytes=64,
        )
        for rank in range(2)
    )


def _materialize(request: WorkloadRunRequest):
    return materialize_workload_preflight(
        request,
        _capability(),
        capacities=_capacities(),
    )


class LegacyDenseBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fabric = physical_fabric_from_data(minimal_hardware(2, 1, sram_bytes=65536))

    def test_real_compile_rect_mesh_canary_emits_verified_stage_files(self) -> None:
        spec = _legacy_spec(decode=0)
        with tempfile.TemporaryDirectory() as raw:
            work_dir = Path(raw)
            stage = run_legacy_dense_compiler_canary(
                spec,
                self.fabric,
                valid_hbm_address_spaces(self.fabric),
                work_dir=work_dir,
                input_digest=canonical_digest(spec),
            )
            self.assertIs(stage.stage, WorkloadRunnerStage.ADAPTER)
            self.assertEqual(len(stage.files), 4)
            stage.verify_files(work_dir)
            capability = (work_dir / "adapter/capability_report.json").read_text()
            self.assertIn('"dense_workload_complete":false', capability)
            self.assertIn('"selected_chain":"naive_fixed/v1"', capability)

            (work_dir / stage.files[0].relative_path).write_text("truncated")
            with self.assertRaisesRegex(Exception, "digest changed"):
                stage.verify_files(work_dir)

    def test_p3_two_layer_prefill_decode_is_explicitly_unsupported(self) -> None:
        request = _request(layers=2, prefill=8, decode=2)
        decision = assess_legacy_dense_backend(
            request,
            _materialize(request),
            _legacy_spec(layers=2, prefill=8, decode=2),
            self.fabric,
        )
        self.assertIs(decision.scope, LegacyDenseBackendScope.UNSUPPORTED)
        self.assertIn("legacy.mixed_prefill_decode_unsupported", decision.reasons)
        self.assertFalse(decision.full_model_ready)
        self.assertIsNone(decision.selected_chain)

    def test_non_dense_training_remote_and_partial_mesh_fail_closed(self) -> None:
        remote = _request(memory_mode=WorkloadMemoryMode.REMOTE_HBM)
        with self.assertRaisesRegex(
            UnsupportedFeatureError,
            "remote HBM placement/staging",
        ):
            _materialize(remote)

        cases = (
            _request(family=WorkloadFamily.DENSE_TRAINING),
            _request(family=WorkloadFamily.MOE_INFERENCE),
            WorkloadRunRequest.create(
                family=WorkloadFamily.DENSE_INFERENCE,
                model=_model(),
                steps=WorkloadStepSpec(
                    inference=WorkloadInferenceSteps(4, 2, 1)
                ),
                mesh=WorkloadMeshSpec(1, 2),
                parallel=WorkloadParallelSpec(tp=1, active_die_ids=(1,)),
                execution=WorkloadExecutionSpec(timing=True, functional=False),
            ),
        )
        for request in cases:
            with self.subTest(family=request.family, memory=request.memory.mode):
                decision = assess_legacy_dense_backend(
                    request,
                    _materialize(request),
                    _legacy_spec(),
                    self.fabric,
                )
                self.assertIs(decision.scope, LegacyDenseBackendScope.UNSUPPORTED)
                self.assertTrue(decision.reasons)
                with self.assertRaisesRegex(
                    UnsupportedFeatureError,
                    "outside legacy Dense static-profile subset",
                ):
                    require_legacy_dense_backend(
                        request,
                        _materialize(request),
                        _legacy_spec(),
                        self.fabric,
                    )


if __name__ == "__main__":
    unittest.main()
