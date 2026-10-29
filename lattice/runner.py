"""End-to-end orchestration for reproducible construction and improvement runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from prefpo.lattice.adapters import ModelAdapter, OfflineModelAdapter, PrefPOModelAdapter
from prefpo.lattice.config import ExperimentConfig
from prefpo.lattice.datasets import (
    load_conversations,
    load_unlabeled_conversations,
    split_conversations,
    validate_domain_disjointness,
)
from prefpo.lattice.evaluation import PolicyEvaluator
from prefpo.lattice.improvement import ContinuousImprovement, ContinuousImprovementResult
from prefpo.lattice.models import Conversation, LabeledConversation, RuleSet
from prefpo.lattice.optimization import (
    ConstructionResult,
    PolicyOptimizer,
    evaluate_policy_async,
)
from prefpo.lattice.risk import DualCheckRiskAssessor
from prefpo.lattice.rules import RuleConsolidator, RuleGenerator
from prefpo.lattice.simulation import ConversationSimulator
from prefpo.lattice.tracking import ExperimentTracker


@dataclass(frozen=True)
class LatticeResult:
    """Returned artifacts from a complete research experiment."""

    run_id: str
    construction: ConstructionResult
    improvement: ContinuousImprovementResult | None
    test_metrics: dict[str, object] | None
    run_directory: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "construction": self.construction.to_dict(),
            "improvement": self.improvement.to_dict() if self.improvement else None,
            "test_metrics": self.test_metrics,
            "run_directory": str(self.run_directory),
        }


class LatticeExperiment:
    """Dependency-injected experiment runner; no remote data or model calls by default."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        adapter: ModelAdapter | None = None,
        evaluator: PolicyEvaluator | None = None,
    ) -> None:
        self.config = config
        if adapter is None:
            if config.use_offline_adapter:
                adapter = OfflineModelAdapter(random_seed=config.seed)
            else:
                adapter = PrefPOModelAdapter(
                    config.rule_generation_model or config.model
                )
        self.adapter = adapter
        if evaluator is None:
            evaluator = PolicyEvaluator(
                adapter=adapter if not config.use_deterministic_evaluator else None,
                semantic_mode=(
                    "lexical" if config.use_deterministic_evaluator else "model"
                ),
            )
        self.evaluator = evaluator

    async def run(
        self,
        train: Iterable[LabeledConversation] | None = None,
        validation: Iterable[LabeledConversation] | None = None,
        deployment: Iterable[Conversation | LabeledConversation] | None = None,
        test: Iterable[LabeledConversation] | None = None,
    ) -> LatticeResult:
        training, validation_set, deployment_set, test_set = self._resolve_data(
            train, validation, deployment, test
        )
        tracker = ExperimentTracker(self.config)
        try:
            tracker.log_event(
                "dataset_loaded",
                {
                    "train_size": len(training),
                    "validation_size": len(validation_set),
                    "deployment_size": len(deployment_set),
                    "test_size": len(test_set),
                },
            )
            simulator = ConversationSimulator(
                self.adapter,
                self.evaluator,
                self.config.construction.simulation,
            )
            generator_adapter = (
                self.adapter
                if self.config.construction.allow_llm_rule_generation
                or isinstance(self.adapter, OfflineModelAdapter)
                else None
            )
            optimizer = PolicyOptimizer(
                self.evaluator,
                generator=RuleGenerator(
                    adapter=generator_adapter,
                    minimum_confidence=self.config.construction.minimum_rule_confidence,
                ),
                consolidator=RuleConsolidator(),
                config=self.config.construction,
                simulator=simulator,
            )
            construction = await optimizer.construct(training, validation_set)
            tracker.log_policy(
                "construction_best",
                construction.best_policy,
                metrics=construction.best_report,
            )
            tracker.log_metrics("construction_validation", construction.best_report)
            tracker.log_event("construction_complete", construction.to_dict())

            improvement_result = None
            if deployment_set and self.config.improvement.iterations > 0:
                risk_assessor = DualCheckRiskAssessor(self.evaluator)
                from prefpo.lattice.adversarial import AdversarialBeamSearch

                searcher = AdversarialBeamSearch(
                    self.adapter,
                    self.evaluator,
                    risk_assessor,
                    self.config.improvement.adversarial,
                )
                improver = ContinuousImprovement(
                    risk_assessor,
                    searcher,
                    optimizer,
                    self.config.improvement,
                )
                improvement_result = await improver.run(
                    deployment_set,
                    validation_set,
                    construction.best_policy,
                )
                tracker.log_policy("continuous_improvement_best", improvement_result.best_policy)
                tracker.log_event("continuous_improvement_complete", improvement_result.to_dict())

            selected_policy = (
                improvement_result.best_policy
                if improvement_result is not None
                else construction.best_policy
            )
            test_metrics = None
            if test_set:
                test_report, _ = await evaluate_policy_async(
                    test_set, selected_policy, self.evaluator
                )
                test_metrics = test_report.to_dict()
                tracker.log_metrics("test", test_report)
            summary = {
                "run_id": tracker.run_id,
                "policy_fingerprint": selected_policy.fingerprint,
                "rule_count": len(selected_policy),
                "construction_validation_f1": construction.best_report.overall.f1,
                "test_f1": (
                    test_metrics["overall"]["f1"] if test_metrics is not None else None
                ),
                "improvement_iterations": (
                    len(improvement_result.history) if improvement_result else 0
                ),
            }
            tracker.complete(summary)
            return LatticeResult(
                run_id=tracker.run_id,
                construction=construction,
                improvement=improvement_result,
                test_metrics=test_metrics,
                run_directory=tracker.run_dir,
            )
        except Exception as exc:
            tracker.log_event(
                "run_failed",
                {"exception_type": type(exc).__name__, "message": str(exc)},
            )
            raise

    def _resolve_data(
        self,
        train: Iterable[LabeledConversation] | None,
        validation: Iterable[LabeledConversation] | None,
        deployment: Iterable[Conversation | LabeledConversation] | None,
        test: Iterable[LabeledConversation] | None,
    ) -> tuple[
        list[LabeledConversation],
        list[LabeledConversation],
        list[Conversation | LabeledConversation],
        list[LabeledConversation],
    ]:
        if train is not None:
            training = list(train)
        elif self.config.dataset.train_path:
            training = load_conversations(
                self.config.dataset.train_path,
                format=self.config.dataset.format,
            )
        else:
            raise ValueError("Provide train data or dataset.train_path")
        if validation is not None:
            validation_set = list(validation)
        elif self.config.dataset.validation_path:
            validation_set = load_conversations(
                self.config.dataset.validation_path,
                format=self.config.dataset.format,
            )
        else:
            training, validation_set = split_conversations(
                training,
                validation_fraction=self.config.dataset.validation_fraction,
                seed=self.config.dataset.seed,
            )
        if self.config.dataset.require_domain_disjointness:
            validate_domain_disjointness(training, validation_set)
        if deployment is not None:
            deployment_set = list(deployment)
        elif self.config.dataset.deployment_path:
            deployment_set = load_unlabeled_conversations(
                self.config.dataset.deployment_path,
                format=self.config.dataset.format,
            )
        else:
            deployment_set = []
        if test is not None:
            test_set = list(test)
        else:
            test_set = []
        return training, validation_set, deployment_set, test_set


def run_experiment(
    config_path: str | Path,
    *,
    train: Iterable[LabeledConversation] | None = None,
    validation: Iterable[LabeledConversation] | None = None,
    deployment: Iterable[Conversation | LabeledConversation] | None = None,
) -> LatticeResult:
    """Synchronous convenience entry point for scripts."""
    import asyncio

    config = ExperimentConfig.from_yaml(config_path)
    return asyncio.run(
        LatticeExperiment(config).run(
            train=train,
            validation=validation,
            deployment=deployment,
        )
    )
