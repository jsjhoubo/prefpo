"""Dual-check risk assessment between a deployed policy and a broad baseline."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Iterable

from prefpo.lattice.evaluation import PolicyDecision, PolicyEvaluator
from prefpo.lattice.models import Conversation, LabeledConversation, PolicyRule, RuleSet, SafetyLabel
from prefpo.lattice.optimization import evaluate_policy
from prefpo.lattice.rules import _TEMPLATES


def build_general_safety_policy() -> RuleSet:
    """Create the broad baseline used to identify specialized-policy coverage gaps."""
    categories = (
        "self_harm",
        "violence",
        "harassment",
        "privacy",
        "medical",
        "misinformation",
        "general",
    )
    return RuleSet(
        tuple(
            PolicyRule(
                rule_id=f"general-{category}",
                text=_TEMPLATES[category],
                categories=(category,),
                source="general_baseline",
                confidence=0.75,
            )
            for category in categories
        )
    )


@dataclass(frozen=True)
class RiskDiscrepancy:
    """A general/specialized disagreement needing review or adversarial probing."""

    conversation_id: str
    category: str
    kind: str
    transcript: str
    general_rule_ids: tuple[str, ...]
    specialized_rule_ids: tuple[str, ...]
    general_risk: float
    specialized_risk: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RiskAssessment:
    """Dual-check predictions and discrepancy set for one deployment batch."""

    discrepancies: tuple[RiskDiscrepancy, ...]
    general_predictions: tuple[SafetyLabel, ...]
    specialized_predictions: tuple[SafetyLabel, ...]
    general_metrics: dict[str, object] | None = None
    specialized_metrics: dict[str, object] | None = None

    @property
    def missed_threats(self) -> tuple[RiskDiscrepancy, ...]:
        return tuple(item for item in self.discrepancies if item.kind == "missed_threat")

    @property
    def false_alarms(self) -> tuple[RiskDiscrepancy, ...]:
        return tuple(item for item in self.discrepancies if item.kind == "false_alarm")

    def to_dict(self) -> dict[str, object]:
        return {
            "discrepancies": [item.to_dict() for item in self.discrepancies],
            "general_predictions": [int(item) for item in self.general_predictions],
            "specialized_predictions": [int(item) for item in self.specialized_predictions],
            "general_metrics": self.general_metrics,
            "specialized_metrics": self.specialized_metrics,
        }


class DualCheckRiskAssessor:
    """Compare the active policy with a broader, independently configured policy."""

    def __init__(
        self,
        specialized_evaluator: PolicyEvaluator,
        *,
        general_policy: RuleSet | None = None,
        general_evaluator: PolicyEvaluator | None = None,
    ) -> None:
        self.specialized_evaluator = specialized_evaluator
        self.general_policy = general_policy or build_general_safety_policy()
        self.general_evaluator = general_evaluator or PolicyEvaluator()

    def assess(
        self,
        conversations: Iterable[Conversation | LabeledConversation],
        specialized_policy: RuleSet,
    ) -> RiskAssessment:
        items = list(conversations)
        discrepancies: list[RiskDiscrepancy] = []
        general_decisions: list[PolicyDecision] = []
        specialized_decisions: list[PolicyDecision] = []
        for item in items:
            conversation = item.conversation if isinstance(item, LabeledConversation) else item
            general = self.general_evaluator.evaluate(conversation, self.general_policy)
            specialized = self.specialized_evaluator.evaluate(conversation, specialized_policy)
            general_decisions.append(general)
            specialized_decisions.append(specialized)
            if general.label == specialized.label:
                continue
            if general.label == SafetyLabel.HARMFUL:
                kind = "missed_threat"
            else:
                kind = "false_alarm"
            category = (
                item.category if isinstance(item, LabeledConversation)
                else str(conversation.metadata.get("category", "general"))
            )
            discrepancies.append(
                RiskDiscrepancy(
                    conversation_id=conversation.conversation_id,
                    category=category,
                    kind=kind,
                    transcript=conversation.transcript,
                    general_rule_ids=general.triggered_rule_ids,
                    specialized_rule_ids=specialized.triggered_rule_ids,
                    general_risk=general.risk_score,
                    specialized_risk=specialized.risk_score,
                )
            )

        general_metrics = None
        specialized_metrics = None
        labeled = [item for item in items if isinstance(item, LabeledConversation)]
        if labeled and len(labeled) == len(items):
            general_report, _ = evaluate_policy(
                labeled, self.general_policy, self.general_evaluator
            )
            specialized_report, _ = evaluate_policy(
                labeled, specialized_policy, self.specialized_evaluator
            )
            general_metrics = general_report.overall.to_dict()
            specialized_metrics = specialized_report.overall.to_dict()
        return RiskAssessment(
            discrepancies=tuple(discrepancies),
            general_predictions=tuple(decision.label for decision in general_decisions),
            specialized_predictions=tuple(
                decision.label for decision in specialized_decisions
            ),
            general_metrics=general_metrics,
            specialized_metrics=specialized_metrics,
        )

    async def assess_async(
        self,
        conversations: Iterable[Conversation | LabeledConversation],
        specialized_policy: RuleSet,
    ) -> RiskAssessment:
        items = list(conversations)
        pairs = await asyncio.gather(
            *(
                self._evaluate_pair(
                    item.conversation if isinstance(item, LabeledConversation) else item,
                    specialized_policy,
                )
                for item in items
            )
        )
        general_decisions = [pair[0] for pair in pairs]
        specialized_decisions = [pair[1] for pair in pairs]
        discrepancies: list[RiskDiscrepancy] = []
        for item, general, specialized in zip(items, general_decisions, specialized_decisions):
            if general.label == specialized.label:
                continue
            conversation = item.conversation if isinstance(item, LabeledConversation) else item
            kind = "missed_threat" if general.label == SafetyLabel.HARMFUL else "false_alarm"
            category = (
                item.category if isinstance(item, LabeledConversation)
                else str(conversation.metadata.get("category", "general"))
            )
            discrepancies.append(
                RiskDiscrepancy(
                    conversation_id=conversation.conversation_id,
                    category=category,
                    kind=kind,
                    transcript=conversation.transcript,
                    general_rule_ids=general.triggered_rule_ids,
                    specialized_rule_ids=specialized.triggered_rule_ids,
                    general_risk=general.risk_score,
                    specialized_risk=specialized.risk_score,
                )
            )
        general_metrics = None
        specialized_metrics = None
        labeled = [item for item in items if isinstance(item, LabeledConversation)]
        if labeled and len(labeled) == len(items):
            actual = [item.label for item in labeled]
            from prefpo.lattice.metrics import binary_metrics

            general_metrics = binary_metrics(
                actual, [decision.label for decision in general_decisions]
            ).to_dict()
            specialized_metrics = binary_metrics(
                actual, [decision.label for decision in specialized_decisions]
            ).to_dict()
        return RiskAssessment(
            discrepancies=tuple(discrepancies),
            general_predictions=tuple(decision.label for decision in general_decisions),
            specialized_predictions=tuple(
                decision.label for decision in specialized_decisions
            ),
            general_metrics=general_metrics,
            specialized_metrics=specialized_metrics,
        )

    async def _evaluate_pair(
        self,
        conversation: Conversation,
        policy: RuleSet,
    ):
        return await asyncio.gather(
            self.general_evaluator.evaluate_async(conversation, self.general_policy),
            self.specialized_evaluator.evaluate_async(conversation, policy),
        )
