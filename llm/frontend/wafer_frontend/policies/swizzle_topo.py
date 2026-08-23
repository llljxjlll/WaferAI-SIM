"""Production entry point for typed inter-die Swizzle planning."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.common import ProfileKey, stable_artifact_id
from ..schema.ir0 import FusionImpl
from ..schema.ir1 import FusedOpSkeleton, IR1
from ..schema.n4 import FUSED_OP_SKELETON_SCHEMA_VERSION
from ..schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleConstraints,
    SwizzleDecision,
    SwizzleEfficiencyPoint,
    SwizzleHardwareProfile,
)
from ..schema.swizzle_plan import SwizzleFusionPlan
from .swizzle.cost import build_unfused_baseline
from .swizzle.decide import decide_swizzle
from .swizzle.enumerate import (
    CandidateGenerator,
    enumerate_drafts,
    materialize_drafts,
)
from .swizzle.meshslice_2d import generate_meshslice_2d_drafts
from .swizzle.problem import build_swizzle_problem
from .swizzle.semantics import analyze_semantics
from .swizzle.wang_1d import generate_wang_1d_drafts
from .swizzle.materialize_ir1 import materialize_swizzle_plan
from .swizzle.materialize import force_swizzle_deployment


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


def _skeleton(candidate: object, ir1: IR1, *, path: str) -> FusedOpSkeleton:
    members = getattr(candidate, "members")
    nodes = {node.id: node for node in ir1.nodes}
    if len(members) != 2 or any(member not in nodes for member in members):
        _fail("candidate must contain two existing IR1 members", f"{path}.members")
    first, second = (nodes[member] for member in members)
    if (
        first.instance_id != second.instance_id
        or first.stage != second.stage
        or first.phase is not second.phase
        or first.execution_group_ref != second.execution_group_ref
    ):
        _fail(
            "candidate members must share instance, stage, phase and group",
            f"{path}.members",
        )
    if getattr(candidate, "impl") is not FusionImpl.NONE:
        _fail("candidate implementation must remain NONE", f"{path}.impl")
    semantic_key = {
        "fusion_ref": getattr(candidate, "id"),
        "member_node_ids": members,
        "boundary_inputs": getattr(candidate, "boundary_inputs"),
        "boundary_outputs": getattr(candidate, "boundary_outputs"),
        "semantic_contract": getattr(candidate, "semantic_contract"),
        "impl": FusionImpl.NONE,
    }
    result = FusedOpSkeleton(
        id=stable_artifact_id(
            "fused_op_skeleton",
            semantic_key,
            schema_version=FUSED_OP_SKELETON_SCHEMA_VERSION,
        ),
        fusion_ref=getattr(candidate, "id"),
        instance_id=first.instance_id,
        member_node_ids=members,
        boundary_inputs=getattr(candidate, "boundary_inputs"),
        boundary_outputs=getattr(candidate, "boundary_outputs"),
        semantic_contract=getattr(candidate, "semantic_contract"),
        impl=FusionImpl.NONE,
    )
    result.validate(path)
    return result


class SwizzleFusionPartition:
    """Select all typed GEMM/collective patterns without choosing an algorithm."""

    def run(self, ir1: IR1) -> tuple[FusedOpSkeleton, ...]:
        if type(ir1) is not IR1:
            _fail("must be an IR1", "ir1")
        ir1.validate("ir1")
        if ir1.fused_op_skeletons:
            _fail("must be empty before fusion partition", "ir1.fused_op_skeletons")
        claimed: set[str] = set()
        result: list[FusedOpSkeleton] = []
        for index, candidate in enumerate(ir1.fusion_candidates):
            overlap = claimed.intersection(candidate.members)
            if overlap:
                _fail(
                    "fusion candidates must not share members",
                    f"ir1.fusion_candidates[{index}].members",
                )
            claimed.update(candidate.members)
            result.append(
                _skeleton(
                    candidate,
                    ir1,
                    path=f"ir1.fusion_candidates[{index}]",
                )
            )
        return tuple(result)


@dataclass(frozen=True, slots=True)
class SwizzlePlanner:
    """Generate, cost and select a Swizzle candidate for one bound skeleton."""

    hardware_profile: SwizzleHardwareProfile
    constraints: SwizzleConstraints
    force_deployment: bool = False
    generators: tuple[CandidateGenerator, ...] = (
        generate_wang_1d_drafts,
        generate_meshslice_2d_drafts,
    )

    def validate(self, path: str = "swizzle_planner") -> None:
        self.hardware_profile.validate(f"{path}.hardware_profile")
        self.constraints.validate(f"{path}.constraints")
        if type(self.force_deployment) is not bool:
            _fail("must be bool", f"{path}.force_deployment")
        if type(self.generators) is not tuple or not self.generators:
            _fail("must contain candidate generators", f"{path}.generators")
        if len({id(generator) for generator in self.generators}) != len(
            self.generators
        ):
            _fail("candidate generators must be unique", f"{path}.generators")

    def decide(
        self,
        ir1: IR1,
        fused_op: FusedOpSkeleton,
        profile: ProfileKey | None = None,
    ) -> SwizzleDecision:
        self.validate()
        if profile is not None:
            profile.validate("profile")
            if profile != ir1.profile and not any(
                binding.instance_ref == fused_op.instance_id
                and binding.profile == profile
                for binding in ir1.instance_profiles
            ):
                _fail("profile does not own the fused op", "profile")
        problem = build_swizzle_problem(
            ir1,
            fused_op,
            self.hardware_profile,
            self.constraints,
        )
        witness = analyze_semantics(
            problem.pattern,
            problem.gemm,
            problem.collective,
            boundary_input_refs=fused_op.boundary_inputs,
            boundary_output_refs=fused_op.boundary_outputs,
        )
        drafts = enumerate_drafts(problem, witness, self.generators)
        candidates = materialize_drafts(problem, drafts)
        baseline = build_unfused_baseline(problem, witness)
        decision = decide_swizzle(problem, baseline, candidates)
        decision.validate("swizzle_decision")
        return decision

    def plan(
        self,
        ir1: IR1,
        fused_op: FusedOpSkeleton,
        profile: ProfileKey,
    ) -> SwizzleFusionPlan:
        if self.force_deployment:
            return self.plan_forced(ir1, fused_op, profile)
        return materialize_swizzle_plan(
            ir1,
            self.decide(ir1, fused_op, profile),
            profile,
        )

    def plan_forced(
        self,
        ir1: IR1,
        fused_op: FusedOpSkeleton,
        profile: ProfileKey,
        *,
        candidate_ref: str | None = None,
    ) -> SwizzleFusionPlan:
        """Deploy a ranked fused candidate while retaining economic evidence."""

        decision = self.decide(ir1, fused_op, profile)
        adapter = force_swizzle_deployment(
            decision,
            candidate_ref=candidate_ref,
        )
        return materialize_swizzle_plan(
            ir1,
            decision,
            profile,
            deployment_selection=adapter.deployment_selection,
        )

    def decide_all(self, ir1: IR1) -> tuple[SwizzleDecision, ...]:
        if type(ir1) is not IR1:
            _fail("must be an IR1", "ir1")
        ir1.validate("ir1")
        return tuple(self.decide(ir1, skeleton) for skeleton in ir1.fused_op_skeletons)


__all__ = ["SwizzleFusionPartition", "SwizzlePlanner"]
