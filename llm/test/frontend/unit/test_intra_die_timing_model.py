from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from functools import lru_cache
import unittest

from llm.frontend.wafer_frontend.cli import load_experiment_spec
from llm.frontend.wafer_frontend.compiler import compile_naive
from llm.frontend.wafer_frontend.passes import (
    load_physical_fabric_and_hbm_address_spaces,
)
from llm.frontend.wafer_frontend.policies.intra_die_timing_model import (
    DEFAULT_INTRA_DIE_HARDWARE_DIGEST,
    DEFAULT_INTRA_DIE_SIMULATION_DIGEST,
    create_intra_die_timing_model,
    estimate_task_cycles,
    resolve_intra_die_timing_model,
)
from llm.frontend.wafer_frontend.policies.intra_die_v2_search import (
    evaluate_intra_die_v2_candidates,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
from llm.frontend.wafer_frontend.policies.optimized_intra_die import (
    OptimizedIntraDieConfig,
    OptimizedIntraDiePolicy,
    _critical_path_orders,
    _estimated_order_makespan,
)
from llm.frontend.wafer_frontend.schema.intra_die_refine import (
    IntraDieOptimizationMode,
    IntraDieOptimizationOptions,
)
from llm.frontend.wafer_frontend.schema.intra_die_timing_model import (
    IntraDieTimingModel,
)
from llm.frontend.wafer_frontend.schema.intra_die_v2_search import (
    IntraDieV2CandidateKind,
)
from llm.frontend.wafer_frontend.schema.n5 import ProjectedIR2Bundle
from llm.frontend.wafer_frontend.schema.serde import from_data, to_primitive

from test_split_k_intra_die_refine import _source_projection


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x1.json"
_SIMULATION = (
    _ROOT
    / "notes/frontend/baselines/stage2-dense-forward-v1/stage1a/e1/inputs/simulation.json"
)
_MAPPING = _ROOT / "llm/test/mapping_config/default_mapping.spec"
_WORKLOAD = _ROOT / "notes/frontend/examples/intra_die_perf_compute_bound.yaml"


@lru_cache(maxsize=1)
def _frozen_projected():
    spec = load_experiment_spec(_WORKLOAD)
    fabric, hbm = load_physical_fabric_and_hbm_address_spaces(_HARDWARE, _MAPPING)
    compilation = compile_naive(
        spec, fabric, hbm_address_spaces=hbm,
        intra_die_refine_options=IntraDieOptimizationOptions(
            mode=IntraDieOptimizationMode.OFF
        ),
    )
    return next(
        artifact for artifact in compilation.artifacts
        if type(artifact) is ProjectedIR2Bundle
    ).entries[0]


class IntraDieTimingModelTest(unittest.TestCase):
    def test_versioned_table_is_stable_and_serde_exact(self) -> None:
        first = create_intra_die_timing_model(
            hardware_digest=DEFAULT_INTRA_DIE_HARDWARE_DIGEST,
            simulation_digest=DEFAULT_INTRA_DIE_SIMULATION_DIGEST,
        )
        second = create_intra_die_timing_model(
            hardware_digest=DEFAULT_INTRA_DIE_HARDWARE_DIGEST,
            simulation_digest=DEFAULT_INTRA_DIE_SIMULATION_DIGEST,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            from_data(IntraDieTimingModel, to_primitive(first), path="model"),
            first,
        )
        self.assertEqual(
            hashlib.sha256(_HARDWARE.read_bytes()).hexdigest(),
            first.hardware_digest,
        )
        self.assertEqual(
            hashlib.sha256(_SIMULATION.read_bytes()).hexdigest(),
            first.simulation_digest,
        )

    def test_digest_mismatch_fails_closed(self) -> None:
        graph, _projection = _source_projection()
        model = create_intra_die_timing_model(
            hardware_digest=DEFAULT_INTRA_DIE_HARDWARE_DIGEST,
            simulation_digest=DEFAULT_INTRA_DIE_SIMULATION_DIGEST,
        )
        with self.assertRaisesRegex(Exception, "hardware digest mismatch"):
            resolve_intra_die_timing_model(
                graph, model, hardware_digest="1" * 64,
                simulation_digest=DEFAULT_INTRA_DIE_SIMULATION_DIGEST,
            )
        with self.assertRaisesRegex(Exception, "simulation digest mismatch"):
            resolve_intra_die_timing_model(
                graph, model,
                hardware_digest=DEFAULT_INTRA_DIE_HARDWARE_DIGEST,
                simulation_digest="2" * 64,
            )
        with self.assertRaisesRegex(Exception, "unstable artifact id"):
            replace(model, compute_setup_cycles=model.compute_setup_cycles + 1).validate()

    def test_task_cycle_estimate_is_deterministic(self) -> None:
        graph, projection = _source_projection()
        task = next(
            task for dag in projection.dags for task in dag.tasks
            if task.compute is not None
        )
        first = estimate_task_cycles(task, graph)
        second = estimate_task_cycles(task, graph)
        self.assertEqual(first, second)
        self.assertGreater(first, 1)

    def test_frozen_default_profile_prediction_rejects_slow_split2(self) -> None:
        projected = _frozen_projected()
        decision = evaluate_intra_die_v2_candidates(
            projected.projection,
            projected.graph,
            IntraDieOptimizationOptions(
                mode=IntraDieOptimizationMode.AUTO,
                allowed_candidates=("identity", "split_k"),
                split_k_parts=(2, 4),
            ),
            hardware_digest=hashlib.sha256(_HARDWARE.read_bytes()).hexdigest(),
            simulation_digest=hashlib.sha256(_SIMULATION.read_bytes()).hexdigest(),
        )
        identity = next(
            candidate for candidate in decision.candidates
            if candidate.kind is IntraDieV2CandidateKind.IDENTITY
        )
        split2 = next(
            candidate for candidate in decision.rejected_candidates
            if candidate.candidate_name == "split_k"
            and candidate.split_k_parts == 2
        )
        self.assertEqual(identity.analytic_cost.predicted_makespan_cycles, 14_760)
        self.assertEqual(decision.selected_candidate_ref, identity.id)
        self.assertEqual(decision.selection_reason, "auto_identity_no_profitable_candidate")
        self.assertEqual(split2.reason, "break_even_not_met")
        self.assertIsNotNone(split2.candidate_predicted_makespan_cycles)
        self.assertLessEqual(
            abs(split2.candidate_predicted_makespan_cycles - 18_070) / 18_070,
            0.20,
        )
        self.assertEqual(decision.simulator_calls_during_search, 0)
        self.assertEqual(decision.hardware_digest, DEFAULT_INTRA_DIE_HARDWARE_DIGEST)
        self.assertEqual(decision.simulation_digest, DEFAULT_INTRA_DIE_SIMULATION_DIGEST)


    def test_cycle_weighted_order_is_stable_legal_and_no_regression(self) -> None:
        projected = _frozen_projected()
        baseline = NaiveIntraDiePolicy().schedule(
            projected.projection, projected.graph
        )
        for dag, schedule in zip(
            projected.projection.dags, baseline.schedules, strict=True
        ):
            first = _critical_path_orders(dag, schedule, projected.graph)
            second = _critical_path_orders(dag, schedule, projected.graph)
            self.assertEqual(first, second)
            task_by_id = {task.id: task for task in dag.tasks}
            for order in first:
                position = {task_id: index for index, task_id in enumerate(order.task_ids)}
                for task_id in order.task_ids:
                    for dependency in task_by_id[task_id].deps:
                        if dependency in position:
                            self.assertLess(position[dependency], position[task_id])
        optimized = OptimizedIntraDiePolicy(OptimizedIntraDieConfig(
            critical_path_order=True, lifetime_reuse=False, bank_stagger=False,
        )).schedule(projected.projection, projected.graph)
        for dag, before, after in zip(
            projected.projection.dags, baseline.schedules, optimized.schedules,
            strict=True,
        ):
            self.assertLessEqual(
                _estimated_order_makespan(dag, after.core_orders, projected.graph),
                _estimated_order_makespan(dag, before.core_orders, projected.graph),
            )


if __name__ == "__main__":
    unittest.main()
