"""Iterative rule-set optimization with F1-monotone promotion and rollback."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Iterable

from prefpo.lattice.config import ConstructionConfig
from prefpo.lattice.evaluation import PolicyDecision, PolicyEvaluator
from prefpo.lattice.metrics import EvaluationReport, evaluate_predictions
from prefpo.lattice.models import LabeledConversation, PolicyRule, RuleSet, SafetyLabel
from prefpo.lattice.rules import RuleConsolidator, RuleGenerator, infer_category
from prefpo.lattice.simulation import ConversationSimulator


@dataclass(frozen=True)
class PolicyError:
    """An evaluation error linked to the rule decision that produced it."""

    conversation_id: str
    category: str
    truth: SafetyLabel
    prediction: SafetyLabel
    triggered_rule_ids: tuple[str, ...]
    transcript: str

    @property
    def kind(self) -> str:
        if self.truth == SafetyLabel.HARMFUL and self.prediction == SafetyLabel.BENIGN:
            return "false_negative"
        if self.truth == SafetyLabel.BENIGN and self.prediction == SafetyLabel.HARMFUL:
            return "false_positive"
        return "correct"


@dataclass(frozen=True)
class ConstructionIteration:
    """Auditable before/after state for a construction update."""

    iteration: int
    baseline_f1: float
    candidate_f1: float
    promoted: bool
    stop_reason: str | None
    baseline_rule_count: int
    candidate_rule_count: int
    simulated_examples: int
    errors: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConstructionResult:
    """Best accepted policy and full construction trace."""

    best_policy: RuleSet
    best_report: EvaluationReport
    history: tuple[ConstructionIteration, ...]
    initial_generation_categories: tuple[str, ...]
    stop_reason: str
    validation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "best_policy": self.best_policy.to_dict(),
            "best_report": self.best_report.to_dict(),
            "history": [item.to_dict() for item in self.history],
            "initial_generation_categories": list(self.initial_generation_categories),
            "stop_reason": self.stop_reason,
            "validation_ids": list(self.validation_ids),
        }


def evaluate_policy(
    examples: Iterable[LabeledConversation],
    policy: RuleSet,
    evaluator: PolicyEvaluator,
    *,
    bootstrap_repetitions: int = 0,
    seed: int = 42,
) -> tuple[EvaluationReport, tuple[PolicyDecision, ...]]:
    """Evaluate one policy and return metrics plus detailed decisions."""
    items = list(examples)
    decisions = tuple(evaluator.evaluate(item.conversation, policy) for item in items)
    report = evaluate_predictions(
        [item.label for item in items],
        [decision.label for decision in decisions],
        domains=[item.domain for item in items],
        categories=[item.category for item in items],
        conversation_ids=[item.conversation_id for item in items],
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    return report, decisions


async def evaluate_policy_async(
    examples: Iterable[LabeledConversation],
    policy: RuleSet,
    evaluator: PolicyEvaluator,
    *,
    bootstrap_repetitions: int = 0,
    seed: int = 42,
) -> tuple[EvaluationReport, tuple[PolicyDecision, ...]]:
    """Asynchronous equivalent for model-backed semantic evaluators."""
    items = list(examples)
    decisions = tuple(
        await asyncio.gather(
            *(
                evaluator.evaluate_async(item.conversation, policy)
                for item in items
            )
        )
    )
    report = evaluate_predictions(
        [item.label for item in items],
        [decision.label for decision in decisions],
        domains=[item.domain for item in items],
        categories=[item.category for item in items],
        conversation_ids=[item.conversation_id for item in items],
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    return report, decisions


class PolicyOptimizer:
    """Construct and refine policies using observed error modes.

    The non-decreasing F1 acceptance criterion is paper-specified. The lexical
    category inference and targeted wording templates are reconstructed,
    inspectable implementation choices.
    """

    def __init__(
        self,
        evaluator: PolicyEvaluator,
        *,
        generator: RuleGenerator | None = None,
        consolidator: RuleConsolidator | None = None,
        config: ConstructionConfig | None = None,
        simulator: ConversationSimulator | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.generator = generator or RuleGenerator()
        self.consolidator = consolidator or RuleConsolidator()
        self.config = config or ConstructionConfig()
        self.simulator = simulator

    async def construct(
        self,
        train: Iterable[LabeledConversation],
        validation: Iterable[LabeledConversation],
        *,
        initial_policy: RuleSet | None = None,
    ) -> ConstructionResult:
        training = list(train)
        validation_set = list(validation)
        if not training:
            raise ValueError("Construction requires at least one labeled training example")
        if not validation_set:
            raise ValueError("Construction requires a non-empty held-out validation set")
        _ensure_unique_ids(training, validation_set)

        generation = await self.generator.generate(training)
        current = initial_policy if initial_policy is not None else generation.policy
        if self.config.consolidate_every_iteration:
            current = (await self.consolidator.consolidate(current)).policy
        best_policy = current
        best_report, _ = await evaluate_policy_async(validation_set, best_policy, self.evaluator)
        history: list[ConstructionIteration] = []
        stop_reason = "maximum_iterations"
        if best_report.overall.f1 >= self.config.target_f1:
            stop_reason = "target_reached_before_first_iteration"
            return ConstructionResult(
                best_policy,
                best_report,
                (),
                generation.generated_categories,
                stop_reason,
                tuple(item.conversation_id for item in validation_set),
            )

        for iteration in range(self.config.iterations):
            simulated_count = 0
            expanded_training = training
            if self.simulator is not None:
                simulations = await self.simulator.simulate(training, current)
                expanded_training = training + [record.example for record in simulations]
                simulated_count = len(simulations)
            _, train_decisions = await evaluate_policy_async(
                expanded_training, current, self.evaluator
            )
            errors = _collect_errors(expanded_training, train_decisions)
            baseline_f1 = best_report.overall.f1
            candidate = await self._targeted_update(
                current,
                expanded_training,
                errors,
            )
            if self.config.consolidate_every_iteration:
                candidate = (await self.consolidator.consolidate(candidate)).policy
            candidate_report, _ = await evaluate_policy_async(
                validation_set, candidate, self.evaluator
            )
            baseline_rule_count = len(best_policy)
            promoted = candidate_report.overall.f1 >= best_report.overall.f1
            stop_iteration: str | None = None
            if promoted:
                current = candidate
                best_policy = candidate
                best_report = candidate_report
                if best_report.overall.f1 >= self.config.target_f1:
                    stop_iteration = "target_f1_reached"
                    stop_reason = stop_iteration
            else:
                current = best_policy
                stop_iteration = "candidate_rejected_non_improving_validation_f1"
            history.append(
                ConstructionIteration(
                    iteration=iteration,
                    baseline_f1=baseline_f1,
                    candidate_f1=candidate_report.overall.f1,
                    promoted=promoted,
                    stop_reason=stop_iteration,
                    baseline_rule_count=baseline_rule_count,
                    candidate_rule_count=len(candidate),
                    simulated_examples=simulated_count,
                    errors=tuple(_error_to_dict(error) for error in errors),
                )
            )
            if stop_iteration == "target_f1_reached":
                break

        return ConstructionResult(
            best_policy=best_policy,
            best_report=best_report,
            history=tuple(history),
            initial_generation_categories=generation.generated_categories,
            stop_reason=stop_reason,
            validation_ids=tuple(item.conversation_id for item in validation_set),
        )

    async def _targeted_update(
        self,
        policy: RuleSet,
        evidence: list[LabeledConversation],
        errors: list[PolicyError],
    ) -> RuleSet:
        false_negatives = [error for error in errors if error.kind == "false_negative"]
        false_positives = [error for error in errors if error.kind == "false_positive"]
        corrected = policy

        if false_negatives:
            existing_categories = {category for rule in corrected.rules for category in rule.categories}
            new_categories = {
                error.category for error in false_negatives if error.category not in existing_categories
            }
            if new_categories:
                new_evidence = [
                    item
                    for item in evidence
                    if item.label == SafetyLabel.HARMFUL
                    and infer_category(item) in new_categories
                ]
                generated = await self.generator.generate(new_evidence)
                additions = [
                    rule for rule in generated.policy.rules
                    if rule.categories[0] in new_categories
                ]
                if additions:
                    corrected = corrected.evolve(add=additions)
            else:
                # Broaden the rule associated with each missed category without
                # weakening its safety action or inventing operational details.
                for category in sorted({error.category for error in false_negatives}):
                    matching = [
                        rule for rule in corrected.rules if category in rule.categories
                    ]
                    if not matching:
                        continue
                    selected = max(matching, key=lambda rule: rule.confidence)
                    broader = _broaden_rule(selected, category)
                    corrected = corrected.evolve(replace=[broader])

        if false_positives:
            current_by_id = {rule.rule_id: rule for rule in corrected.rules}
            for error in false_positives:
                for rule_id in error.triggered_rule_ids:
                    rule = current_by_id.get(rule_id)
                    if rule is not None:
                        refined = _refine_rule(rule)
                        corrected = corrected.evolve(replace=[refined])
                        current_by_id[rule_id] = refined

        if self.config.remove_unused_rules:
            _, updated_decisions = await evaluate_policy_async(
                evidence, corrected, self.evaluator
            )
            used = {
                rule_id
                for decision in updated_decisions
                for rule_id in decision.triggered_rule_ids
            }
            removable = [
                rule.rule_id
                for rule in corrected.rules
                if rule.rule_id not in used
            ]
            if removable:
                corrected = corrected.evolve(remove=removable)
        return corrected


def _broaden_rule(rule: PolicyRule, category: str) -> PolicyRule:
    broadening = {
        "self_harm": (
            "Include indirect expressions of distress or requests for help, and respond "
            "supportively without requiring explicit wording."
        ),
        "violence": (
            "Include implied threats and requests framed as hypotheticals when intent "
            "still concerns harm to a person."
        ),
        "harassment": (
            "Include coded, indirect, or context-shifted requests for targeted abuse."
        ),
        "privacy": (
            "Include indirect attempts to locate, infer, or expose private identifying data."
        ),
        "medical": (
            "Include indirect requests for individualized diagnosis or treatment decisions."
        ),
        "misinformation": (
            "Include requests to give unsupported claims unwarranted certainty or credibility."
        ),
    }
    addition = broadening.get(category, "Include indirect and context-shifted forms of the same risk.")
    if addition.casefold() in rule.text.casefold():
        return rule.revise(rule.text, source="lattice_broadened")
    return rule.revise(f"{rule.text.rstrip()} {addition}", source="lattice_broadened")


def _refine_rule(rule: PolicyRule) -> PolicyRule:
    qualifier = (
        "Apply this only when the conversation indicates an actual request or risk; "
        "do not trigger solely on quotation, benign fiction, education, prevention, "
        "or a request for support."
    )
    text = rule.text
    if qualifier.casefold() not in text.casefold():
        text = f"{text.rstrip()} {qualifier}"
    return rule.revise(text, source="lattice_refined")


def _collect_errors(
    examples: list[LabeledConversation],
    decisions: tuple[PolicyDecision, ...],
) -> list[PolicyError]:
    return [
        PolicyError(
            conversation_id=item.conversation_id,
            category=infer_category(item),
            truth=item.label,
            prediction=decision.label,
            triggered_rule_ids=decision.triggered_rule_ids,
            transcript=item.conversation.transcript,
        )
        for item, decision in zip(examples, decisions)
        if item.label != decision.label
    ]


def _error_to_dict(error: PolicyError) -> dict[str, object]:
    return {
        "conversation_id": error.conversation_id,
        "category": error.category,
        "kind": error.kind,
        "triggered_rule_ids": list(error.triggered_rule_ids),
    }


def _ensure_unique_ids(
    train: list[LabeledConversation],
    validation: list[LabeledConversation],
) -> None:
    train_ids = [item.conversation_id for item in train]
    validation_ids = [item.conversation_id for item in validation]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Training conversation ids must be unique")
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Validation conversation ids must be unique")
    overlap = set(train_ids).intersection(validation_ids)
    if overlap:
        raise ValueError(f"Training and validation ids overlap: {sorted(overlap)}")
