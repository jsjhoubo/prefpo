"""PrefPO Grader adapter for evaluating Lattice policies on Sample records."""

from __future__ import annotations

import asyncio

from prefpo.config import ModelConfig
from prefpo.grading.base import GradeResult, Grader
from prefpo.lattice.datasets import from_prefpo_samples
from prefpo.lattice.evaluation import PolicyEvaluator
from prefpo.lattice.metrics import evaluate_predictions
from prefpo.lattice.models import RuleSet
from prefpo.types import Prompt, Sample


class PolicySetGrader(Grader):
    """Expose deterministic policy metrics through PrefPO's existing Grader API.

    This lets existing PrefPO reports compare task prompts against a fixed
    safety-policy baseline without duplicating optimizer orchestration.
    """

    def __init__(self, policy: RuleSet, evaluator: PolicyEvaluator | None = None) -> None:
        self.policy = policy
        self.evaluator = evaluator or PolicyEvaluator()

    async def grade(
        self,
        prompt: Prompt,
        samples: list[Sample] | None,
        model_config: ModelConfig,
        semaphore: asyncio.Semaphore,
    ) -> GradeResult:
        del prompt, model_config, semaphore
        if not samples:
            raise ValueError("PolicySetGrader requires non-empty labeled samples")
        records = from_prefpo_samples(samples)
        decisions = await asyncio.gather(
            *(
                self.evaluator.evaluate_async(item.conversation, self.policy)
                for item in records
            )
        )
        report = evaluate_predictions(
            [item.label for item in records],
            [decision.label for decision in decisions],
            domains=[item.domain for item in records],
            categories=[item.category for item in records],
            conversation_ids=[item.conversation_id for item in records],
        )
        return GradeResult(
            score=report.overall.f1,
            n=report.overall.support,
            per_sample=list(report.prediction_rows),
            outputs=[
                {
                    "conversation_id": item.conversation_id,
                    "triggered_rule_ids": list(decision.triggered_rule_ids),
                    "risk_score": decision.risk_score,
                }
                for item, decision in zip(records, decisions)
            ],
        )

    def check_output(self, output: str, prompt_text: str | None = None) -> dict:
        from prefpo.lattice.models import Conversation, ConversationTurn

        conversation = Conversation(
            conversation_id="standalone-output",
            turns=(ConversationTurn("user", prompt_text or "User request", 0),
                   ConversationTurn("assistant", output, 1)),
        )
        decision = self.evaluator.evaluate(conversation, self.policy)
        return {
            "passed": decision.label.value == 0,
            "triggered_rule_ids": list(decision.triggered_rule_ids),
            "risk_score": decision.risk_score,
        }
