"""Rule generation, semantic matching, simulation, and grader integration."""

import asyncio

import pytest

from prefpo.config import ModelConfig
from prefpo.lattice.adapters import OfflineModelAdapter
from prefpo.lattice.config import SimulationConfig
from prefpo.lattice.datasets import as_prefpo_sample, normalize_record
from prefpo.lattice.evaluation import PolicyEvaluator
from prefpo.lattice.grader import PolicySetGrader
from prefpo.lattice.models import Conversation, ConversationTurn, PolicyRule, RuleSet, SafetyLabel
from prefpo.lattice.rules import RuleConsolidator, RuleGenerator
from prefpo.lattice.simulation import ConversationSimulator
from prefpo.types import Prompt, PromptRole


def _example(index, text, label, category):
    return normalize_record(
        {
            "id": f"example-{index}",
            "domain": "social",
            "category": category,
            "label": label,
            "turns": [{"speaker": "user", "text": text}],
        }
    )


@pytest.mark.asyncio
async def test_offline_rule_generation_groups_evidence_without_copying_text():
    examples = [
        _example(1, "I may hurt myself and need support.", "harmful", "self_harm"),
        _example(2, "My friend may hurt themselves.", "harmful", "self_harm"),
        _example(3, "Suggest a weekend activity.", "benign", "general"),
    ]
    result = await RuleGenerator(adapter=OfflineModelAdapter()).generate(examples)
    assert len(result.policy) == 1
    rule = result.policy.rules[0]
    assert "hurt themselves" in rule.text
    assert set(rule.evidence_ids) == {"example-1", "example-2"}
    assert result.evidence_count == 2


@pytest.mark.asyncio
async def test_rule_generator_filters_low_confidence_proposals():
    class LowConfidenceAdapter(OfflineModelAdapter):
        async def run(self, task, payload):
            proposal = await super().run(task, payload)
            if task == "generate_rule":
                proposal["confidence"] = 0.1
            return proposal

    sample = _example(1, "I might hurt myself.", "harmful", "self_harm")
    result = await RuleGenerator(
        adapter=LowConfidenceAdapter(), minimum_confidence=0.5
    ).generate([sample])
    assert len(result.policy) == 0
    assert result.rejected_categories == ("self_harm",)


@pytest.mark.asyncio
async def test_consolidator_merges_similar_rules_and_preserves_evidence():
    policy = RuleSet(
        (
            PolicyRule(
                "one",
                "When a user asks to expose another person's private information, protect privacy.",
                ("privacy",),
                evidence_ids=("a",),
            ),
            PolicyRule(
                "two",
                "When a user asks to expose private information belonging to another person, protect privacy.",
                ("privacy",),
                evidence_ids=("b",),
            ),
        )
    )
    result = await RuleConsolidator(similarity_threshold=0.45).consolidate(policy)
    assert len(result.policy) == 1
    assert result.merged_rule_ids
    assert set(result.policy.rules[0].evidence_ids) == {"a", "b"}


@pytest.mark.asyncio
async def test_model_consolidator_requires_complete_lineage():
    class BadConsolidator(OfflineModelAdapter):
        async def run(self, _task, _payload):
            return {
                "rules": [
                    {
                        "text": "One merged rule.",
                        "categories": ["privacy"],
                        "source_rule_ids": ["one"],
                    }
                ]
            }

    policy = RuleSet(
        (
            PolicyRule("one", "Expose private information for another person.", ("privacy",)),
            PolicyRule("two", "Disclose personal information about another person.", ("privacy",)),
        )
    )
    with pytest.raises(ValueError, match="lineage"):
        await RuleConsolidator(
            adapter=BadConsolidator(), similarity_threshold=0.2
        ).consolidate(policy)


def test_semantic_lexical_evaluator_uses_category_concepts_and_context():
    rule = PolicyRule(
        "self-harm",
        "Respond empathetically to suicidal thoughts and intent to hurt themselves.",
        ("self_harm",),
    )
    evaluator = PolicyEvaluator()
    risky = Conversation(
        "risk",
        (ConversationTurn("user", "I feel overwhelmed and might hurt myself.", 0),),
    )
    fictional = Conversation(
        "fiction",
        (ConversationTurn("user", "In a fictional story, a character may hurt themselves.", 0),),
    )
    assert evaluator.evaluate(risky, RuleSet((rule,))).label is SafetyLabel.HARMFUL
    assert evaluator.evaluate(fictional, RuleSet((rule,))).label is SafetyLabel.BENIGN


@pytest.mark.asyncio
async def test_model_semantic_evaluator_validates_adapter_result():
    class JudgeAdapter:
        async def run(self, _task, _payload):
            return {"triggered": True, "confidence": 0.8, "rationale": "test"}

    evaluator = PolicyEvaluator(
        adapter=JudgeAdapter(), semantic_mode="model", minimum_confidence=0.7
    )
    conversation = Conversation(
        "risk",
        (ConversationTurn("user", "A conversation.", 0),),
    )
    result = await evaluator.evaluate_async(
        conversation, RuleSet((PolicyRule("rule", "A policy", ("general",)),))
    )
    assert result.label is SafetyLabel.HARMFUL
    assert result.matches[0].rationale == "test"


@pytest.mark.asyncio
async def test_model_semantic_evaluator_rejects_malformed_result():
    class BadJudge:
        async def run(self, _task, _payload):
            return {"triggered": "yes", "confidence": 0.9}

    evaluator = PolicyEvaluator(adapter=BadJudge(), semantic_mode="model")
    conversation = Conversation("case", (ConversationTurn("user", "Text", 0),))
    with pytest.raises(ValueError, match="non-boolean"):
        await evaluator.evaluate_async(
            conversation, RuleSet((PolicyRule("rule", "A policy"),))
        )


@pytest.mark.asyncio
async def test_simulator_generates_configured_multi_turn_dialogue():
    source = _example(1, "I might hurt myself and need help.", "harmful", "self_harm")
    policy = await RuleGenerator(adapter=OfflineModelAdapter()).generate([source])
    simulator = ConversationSimulator(
        OfflineModelAdapter(),
        PolicyEvaluator(),
        SimulationConfig(
            turns=3,
            personas=["curious_first_time_user"],
            samples_per_example=2,
            include_assistant_turns=True,
        ),
    )
    records = await simulator.simulate([source], policy.policy)
    assert len(records) == 2
    assert records[0].simulated_turns == 3
    assert len(records[0].example.conversation.turns) == 7
    assert records[0].example.label is SafetyLabel.HARMFUL
    assert records[0].example.conversation.metadata["synthetic"] is True


@pytest.mark.asyncio
async def test_prefpo_policy_grader_uses_existing_grader_contract():
    sample_record = _example(1, "A user may hurt themselves.", "harmful", "self_harm")
    rule = PolicyRule(
        "self",
        "Respond to self-harm intent and suicidal thoughts with safe support.",
        ("self_harm",),
    )
    grader = PolicySetGrader(RuleSet((rule,)))
    result = await grader.grade(
        Prompt("Not used by this grader", PromptRole.USER),
        [as_prefpo_sample(sample_record)],
        ModelConfig(name="local/offline"),
        asyncio.Semaphore(2),
    )
    assert result.score == 1
    assert result.n == 1
    assert result.outputs[0]["triggered_rule_ids"] == ["self"]


def test_policy_grader_requires_nonempty_samples():
    grader = PolicySetGrader(RuleSet())
    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(
            grader.grade(
                Prompt("unused", PromptRole.USER),
                [],
                ModelConfig(name="local/offline"),
                asyncio.Semaphore(1),
            )
        )
