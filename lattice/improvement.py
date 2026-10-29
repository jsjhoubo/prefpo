"""Closed-loop continuous policy improvement for unlabeled deployment traffic."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable

from prefpo.lattice.adversarial import (
    AdversarialBeamSearch,
    AdversarialSearchResult,
    deduplicate_search_results,
)
from prefpo.lattice.config import ImprovementConfig
from prefpo.lattice.models import (
    Conversation,
    LabeledConversation,
    RuleSet,
    SafetyLabel,
)
from prefpo.lattice.optimization import (
    PolicyOptimizer,
    evaluate_policy_async,
)
from prefpo.lattice.risk import DualCheckRiskAssessor, RiskAssessment


@dataclass(frozen=True)
class ImprovementIteration:
    """Risk and search evidence, candidate result, and promotion decision."""

    iteration: int
    before_fingerprint: str
    after_fingerprint: str
    baseline_validation_f1: float
    candidate_validation_f1: float
    promoted: bool
    missed_threats: int
    false_alarms: int
    search_nodes: int
    evidence_count: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuousImprovementResult:
    """Best deployed policy plus complete closed-loop evidence."""

    best_policy: RuleSet
    validation_f1: float
    history: tuple[ImprovementIteration, ...]
    risk_assessments: tuple[RiskAssessment, ...]
    searches: tuple[AdversarialSearchResult, ...]
    stop_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "best_policy": self.best_policy.to_dict(),
            "validation_f1": self.validation_f1,
            "history": [item.to_dict() for item in self.history],
            "risk_assessments": [item.to_dict() for item in self.risk_assessments],
            "searches": [item.to_dict() for item in self.searches],
            "stop_reason": self.stop_reason,
        }


class ContinuousImprovement:
    """Assess, search, propose, validate, and promote policy revisions."""

    def __init__(
        self,
        risk_assessor: DualCheckRiskAssessor,
        searcher: AdversarialBeamSearch,
        optimizer: PolicyOptimizer,
        config: ImprovementConfig | None = None,
    ) -> None:
        self.risk_assessor = risk_assessor
        self.searcher = searcher
        self.optimizer = optimizer
        self.config = config or ImprovementConfig()

    async def run(
        self,
        deployed: Iterable[Conversation | LabeledConversation],
        validation: Iterable[LabeledConversation],
        current_policy: RuleSet,
    ) -> ContinuousImprovementResult:
        deployment = list(deployed)
        validation_set = list(validation)
        if not deployment:
            raise ValueError("Continuous improvement requires deployment conversations")
        if not validation_set:
            raise ValueError("Continuous improvement requires held-out validation labels")
        deployment_ids = {
            item.conversation_id for item in deployment
        }
        validation_ids = {item.conversation_id for item in validation_set}
        overlap = deployment_ids.intersection(validation_ids)
        if overlap:
            raise ValueError(f"Deployment and validation ids overlap: {sorted(overlap)}")

        baseline_report, _ = await evaluate_policy_async(
            validation_set, current_policy, self.optimizer.evaluator
        )
        best_f1 = baseline_report.overall.f1
        best_policy = current_policy
        history: list[ImprovementIteration] = []
        risk_reports: list[RiskAssessment] = []
        searches: list[AdversarialSearchResult] = []
        stop_reason = "maximum_iterations"

        for iteration in range(self.config.iterations):
            assessment = await self.risk_assessor.assess_async(deployment, best_policy)
            risk_reports.append(assessment)
            if not assessment.discrepancies:
                stop_reason = "no_dual_check_discrepancies"
                break
            assessment_for_iteration = _filter_risk_assessment(
                assessment, self.config.risk_threshold
            )
            if not assessment_for_iteration.discrepancies:
                stop_reason = "no_discrepancies_above_risk_threshold"
                break

            seeds = _select_discrepancy_seeds(deployment, assessment_for_iteration)
            iteration_searches = []
            for discrepancy, conversation in seeds:
                result = await self.searcher.search(
                    conversation,
                    best_policy,
                    category=discrepancy.category,
                )
                searches.append(result)
                iteration_searches.append(result)

            nodes = deduplicate_search_results(iteration_searches)
            evidence = await _build_pseudo_labeled_evidence(
                deployment,
                assessment_for_iteration,
                nodes,
                self.risk_assessor,
            )
            if len(evidence) < self.config.minimum_evidence:
                history.append(
                    ImprovementIteration(
                        iteration=iteration,
                        before_fingerprint=best_policy.fingerprint,
                        after_fingerprint=best_policy.fingerprint,
                        baseline_validation_f1=best_f1,
                        candidate_validation_f1=best_f1,
                        promoted=False,
                        missed_threats=len(assessment.missed_threats),
                        false_alarms=len(assessment.false_alarms),
                        search_nodes=sum(item.nodes_expanded for item in iteration_searches),
                        evidence_count=len(evidence),
                        reason="insufficient_pseudo_labeled_evidence",
                    )
                )
                stop_reason = "insufficient_evidence"
                break

            before_fingerprint = best_policy.fingerprint
            baseline_f1 = best_f1
            construction = await self.optimizer.construct(
                evidence,
                validation_set,
                initial_policy=best_policy,
            )
            candidate = construction.best_policy
            candidate_f1 = construction.best_report.overall.f1
            promoted = (
                candidate_f1 >= best_f1
                if self.config.require_non_decreasing_f1
                else candidate_f1 > best_f1
            )
            reason = "candidate_promoted" if promoted else "candidate_rejected_validation_regression"
            if promoted:
                best_policy = candidate
                best_f1 = candidate_f1
            history.append(
                ImprovementIteration(
                    iteration=iteration,
                    before_fingerprint=before_fingerprint,
                    after_fingerprint=candidate.fingerprint if promoted else best_policy.fingerprint,
                    baseline_validation_f1=baseline_f1,
                    candidate_validation_f1=candidate_f1,
                    promoted=promoted,
                    missed_threats=len(assessment.missed_threats),
                    false_alarms=len(assessment.false_alarms),
                    search_nodes=sum(item.nodes_expanded for item in iteration_searches),
                    evidence_count=len(evidence),
                    reason=reason,
                )
            )
            if not promoted:
                stop_reason = "candidate_rejected_validation_regression"
                break
        return ContinuousImprovementResult(
            best_policy=best_policy,
            validation_f1=best_f1,
            history=tuple(history),
            risk_assessments=tuple(risk_reports),
            searches=tuple(searches),
            stop_reason=stop_reason,
        )


def _select_discrepancy_seeds(
    deployment: list[Conversation | LabeledConversation],
    assessment: RiskAssessment,
) -> list[tuple[object, Conversation]]:
    by_id = {
        item.conversation_id: item.conversation if isinstance(item, LabeledConversation) else item
        for item in deployment
    }
    return [
        (discrepancy, by_id[discrepancy.conversation_id])
        for discrepancy in assessment.discrepancies
        if discrepancy.conversation_id in by_id
    ]


def _filter_risk_assessment(
    assessment: RiskAssessment,
    minimum_risk: float,
) -> RiskAssessment:
    discrepancies = tuple(
        discrepancy
        for discrepancy in assessment.discrepancies
        if discrepancy.general_risk >= minimum_risk
    )
    return RiskAssessment(
        discrepancies=discrepancies,
        general_predictions=assessment.general_predictions,
        specialized_predictions=assessment.specialized_predictions,
        general_metrics=assessment.general_metrics,
        specialized_metrics=assessment.specialized_metrics,
    )


async def _build_pseudo_labeled_evidence(
    deployment: list[Conversation | LabeledConversation],
    assessment: RiskAssessment,
    nodes: tuple,
    assessor: DualCheckRiskAssessor,
) -> list[LabeledConversation]:
    records: list[LabeledConversation] = []
    for discrepancy in assessment.discrepancies:
        source = next(
            (
                item.conversation if isinstance(item, LabeledConversation) else item
                for item in deployment
                if item.conversation_id == discrepancy.conversation_id
            ),
            None,
        )
        if source is None:
            continue
        records.append(
            LabeledConversation(
                conversation=source,
                label=SafetyLabel.HARMFUL
                if discrepancy.kind == "missed_threat"
                else SafetyLabel.BENIGN,
                category=discrepancy.category,
                rationale=f"Dual-check pseudo-label: {discrepancy.kind}",
            )
        )
    for node in nodes:
        pseudo_label = (await assessor.general_evaluator.evaluate_async(
            node.conversation, assessor.general_policy
        )).label
        digest = hashlib.sha1(node.conversation.transcript.encode("utf-8")).hexdigest()[:10]
        conversation = Conversation(
            conversation_id=f"adversarial:{node.node_id}:{digest}",
            turns=node.conversation.turns,
            domain=node.conversation.domain,
            metadata={
                **node.conversation.metadata,
                "synthetic": True,
                "adversarial_strategy": node.strategy,
                "pseudo_label_source": "general_policy",
            },
        )
        records.append(
            LabeledConversation(
                conversation=conversation,
                label=pseudo_label,
                category=node.category,
                rationale=f"Adversarial search outcome: {node.outcome}",
            )
        )
    # A deployment record can occur in more than one discrepancy source only
    # once; stable IDs preserve the provenance of the original observation.
    unique: dict[str, LabeledConversation] = {}
    for item in records:
        unique.setdefault(item.conversation_id, item)
    return list(unique.values())
