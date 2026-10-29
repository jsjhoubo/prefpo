"""Multi-turn conversation simulation conditioned on a current policy."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from prefpo.lattice.adapters import ModelAdapter, conversation_payload
from prefpo.lattice.config import SimulationConfig
from prefpo.lattice.evaluation import PolicyEvaluator
from prefpo.lattice.models import Conversation, LabeledConversation, RuleSet


@dataclass(frozen=True)
class SimulationRecord:
    """A synthetic transcript with source, persona, and inherited label provenance."""

    example: LabeledConversation
    source_conversation_id: str
    persona: str
    simulated_turns: int


class ConversationSimulator:
    """Extend representative labeled dialogue into reproducible multi-turn trials."""

    def __init__(
        self,
        adapter: ModelAdapter,
        evaluator: PolicyEvaluator,
        config: SimulationConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.evaluator = evaluator
        self.config = config or SimulationConfig()

    async def simulate(
        self,
        examples: Iterable[LabeledConversation],
        policy: RuleSet,
    ) -> list[SimulationRecord]:
        """Create `samples_per_example` continuations for each labeled record."""
        rng = random.Random(self.config.random_seed)
        records: list[SimulationRecord] = []
        for source in examples:
            for sample_number in range(self.config.samples_per_example):
                persona = rng.choice(self.config.personas)
                conversation = source.conversation
                for simulated_turn in range(self.config.turns):
                    response = await self.adapter.run(
                        "simulate_user_turn",
                        {
                            **conversation_payload(conversation),
                            "persona": persona,
                            "turn_index": simulated_turn,
                            "sample_number": sample_number,
                            "category": source.category,
                            "inherited_label": int(source.label),
                            "policy": [rule.to_dict() for rule in policy.rules],
                        },
                    )
                    user_text = response.get("text")
                    if not isinstance(user_text, str) or not user_text.strip():
                        raise ValueError(
                            f"Simulator returned empty user turn for {source.conversation_id}"
                        )
                    conversation = conversation.append("user", user_text.strip())
                    if self.config.include_assistant_turns:
                        decision = await self.evaluator.evaluate_async(conversation, policy)
                        assistant = await self.adapter.run(
                            "assistant_response",
                            {
                                **conversation_payload(conversation),
                                "policy_triggered": bool(decision.triggered_rule_ids),
                                "triggered_rule_ids": list(decision.triggered_rule_ids),
                                "policy": [rule.to_dict() for rule in policy.rules],
                            },
                        )
                        assistant_text = assistant.get("text")
                        if not isinstance(assistant_text, str) or not assistant_text.strip():
                            raise ValueError(
                                f"Simulator returned empty assistant turn for "
                                f"{source.conversation_id}"
                            )
                        conversation = conversation.append("assistant", assistant_text.strip())
                synthetic = LabeledConversation(
                    conversation=Conversation(
                        conversation_id=(
                            f"{source.conversation_id}:sim:{sample_number}:{persona}"
                        ),
                        turns=conversation.turns,
                        domain=source.domain,
                        metadata={
                            **source.conversation.metadata,
                            "synthetic": True,
                            "source_conversation_id": source.conversation_id,
                            "persona": persona,
                            "simulated_turns": self.config.turns,
                        },
                    ),
                    label=source.label,
                    category=source.category,
                    rationale=source.rationale,
                )
                records.append(
                    SimulationRecord(
                        example=synthetic,
                        source_conversation_id=source.conversation_id,
                        persona=persona,
                        simulated_turns=self.config.turns,
                    )
                )
        return records
