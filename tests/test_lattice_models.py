"""Model and configuration invariants for the Lattice research package."""

import pytest
from pydantic import ValidationError

from prefpo.lattice.config import (
    AdversarialConfig,
    ConstructionConfig,
    ExperimentConfig,
    ImprovementConfig,
    SimulationConfig,
)
from prefpo.lattice.models import (
    Conversation,
    ConversationTurn,
    LabeledConversation,
    PolicyRule,
    RuleSet,
    SafetyLabel,
    stable_rule_id,
)


def _conversation(conversation_id="case-1", text="A benign request.") -> Conversation:
    return Conversation(
        conversation_id,
        (ConversationTurn("user", text, 0),),
        domain="social",
    )


def test_safety_label_parsing_accepts_common_dataset_values():
    assert SafetyLabel.parse("harmful") is SafetyLabel.HARMFUL
    assert SafetyLabel.parse("non-harmful") is SafetyLabel.BENIGN
    assert SafetyLabel.parse(True) is SafetyLabel.HARMFUL
    assert SafetyLabel.parse(0) is SafetyLabel.BENIGN


@pytest.mark.parametrize("value", ["maybe", 4, -1])
def test_safety_label_rejects_ambiguous_values(value):
    with pytest.raises(ValueError):
        SafetyLabel.parse(value)


def test_conversation_requires_monotonic_turn_indices():
    with pytest.raises(ValueError, match="strictly increasing"):
        Conversation(
            "duplicate",
            (
                ConversationTurn("user", "First turn", 2),
                ConversationTurn("assistant", "Second turn", 1),
            ),
        )


def test_conversation_append_is_immutable_and_preserves_metadata():
    base = Conversation(
        "case",
        (ConversationTurn("user", "A request", 0),),
        metadata={"source": "fixture"},
    )
    extended = base.append("assistant", "A response")
    assert len(base.turns) == 1
    assert len(extended.turns) == 2
    assert extended.turns[-1].turn_index == 1
    assert extended.metadata == base.metadata


def test_conversation_from_dict_accepts_chat_message_shape():
    conversation = Conversation.from_dict(
        {
            "id": "chat-1",
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
        }
    )
    assert conversation.conversation_id == "chat-1"
    assert conversation.turns[1].speaker == "assistant"
    assert conversation.transcript == "user: Question\nassistant: Answer"


def test_labeled_conversation_normalizes_label_and_metadata():
    record = LabeledConversation(
        _conversation(),
        "safe",
        category="General",
        rationale="clear",
    )
    assert record.label is SafetyLabel.BENIGN
    assert record.category == "general"
    assert record.to_dict()["label"] == 0


def test_policy_rule_revision_preserves_lineage_and_increments_version():
    rule = PolicyRule("risk", "Avoid enabling harm", ("violence",), evidence_ids=("ex-1",))
    revised = rule.revise("Decline harmful requests and redirect safely")
    assert revised.rule_id == rule.rule_id
    assert revised.version == rule.version + 1
    assert revised.evidence_ids == rule.evidence_ids


def test_policy_rule_rejects_bad_confidence_and_empty_text():
    with pytest.raises(ValueError, match="confidence"):
        PolicyRule("risk", "Rule text", confidence=1.1)
    with pytest.raises(ValueError, match="must not be empty"):
        PolicyRule("risk", "   ")


def test_rule_set_fingerprint_is_stable_and_content_sensitive():
    first = RuleSet((PolicyRule("a", "Policy A", ("privacy",)),))
    same = RuleSet((PolicyRule("a", "Policy A", ("privacy",)),))
    changed = RuleSet((PolicyRule("a", "Policy B", ("privacy",)),))
    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != changed.fingerprint


def test_rule_set_evolve_supports_add_replace_and_remove():
    original = RuleSet(
        (
            PolicyRule("one", "First policy", ("privacy",)),
            PolicyRule("two", "Second policy", ("violence",)),
        )
    )
    evolved = original.evolve(
        replace=[original.rules[0].revise("Revised first policy")],
        add=[PolicyRule("three", "Third policy", ("medical",))],
        remove=["two"],
    )
    assert [rule.rule_id for rule in evolved.rules] == ["one", "three"]
    assert evolved.get("one").version == 2
    assert evolved.parent_fingerprint == original.fingerprint
    assert evolved.revision == original.revision + 1


def test_rule_set_rejects_duplicate_ids_and_invalid_changes():
    with pytest.raises(ValueError, match="unique"):
        RuleSet((PolicyRule("same", "First"), PolicyRule("same", "Second")))
    original = RuleSet((PolicyRule("one", "Existing"),))
    with pytest.raises(ValueError, match="unknown"):
        original.evolve(replace=[PolicyRule("missing", "No rule")])
    with pytest.raises(ValueError, match="already exists"):
        original.evolve(add=[PolicyRule("one", "Duplicate id")])


def test_stable_rule_ids_are_repeatable_and_readable():
    assert stable_rule_id("Avoid exposing private information") == stable_rule_id(
        "Avoid exposing private information"
    )
    assert stable_rule_id("Avoid exposing private information").startswith("rule-avoid-")


def test_research_config_defaults_match_documented_paper_settings():
    construction = ConstructionConfig()
    improvement = ImprovementConfig()
    assert construction.iterations == 10
    assert construction.target_f1 == 0.9
    assert construction.simulation.turns == 3
    assert improvement.adversarial.beam_width == 3
    assert improvement.adversarial.depth == 10
    assert AdversarialConfig().maximum_nodes >= 30


def test_config_rejects_invalid_limits_and_extra_fields():
    with pytest.raises(ValidationError):
        SimulationConfig(turns=0)
    with pytest.raises(ValidationError):
        AdversarialConfig(beam_width=0)
    with pytest.raises(ValidationError):
        ExperimentConfig(name="bad", output_dir="results", unrecognized=True)


def test_experiment_config_yaml_round_trip(tmp_path):
    config = ExperimentConfig(name="yaml-round-trip", output_dir=str(tmp_path))
    path = tmp_path / "config.yaml"
    config.write_yaml(path)
    loaded = ExperimentConfig.from_yaml(path)
    assert loaded.name == config.name
    assert loaded.model.name == config.model.name
    assert loaded.construction.simulation.personas == config.construction.simulation.personas


@pytest.mark.parametrize(
    "path",
    [
        "research_configs/construction.yaml",
        "research_configs/continuous_improvement.yaml",
    ],
)
def test_shipped_research_configs_validate(path):
    config = ExperimentConfig.from_yaml(path)
    assert config.use_offline_adapter
    assert config.dataset.train_path


def test_experiment_config_requires_validation_path_for_domain_holdout():
    with pytest.raises(ValidationError, match="validation_path"):
        ExperimentConfig(
            dataset={"require_domain_disjointness": True},
        )
