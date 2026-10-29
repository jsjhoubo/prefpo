"""Semantic policy evaluation with explicit deterministic and LLM-backed modes."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from prefpo.lattice.adapters import ModelAdapter
from prefpo.lattice.models import Conversation, PolicyRule, RuleSet, SafetyLabel

_CATEGORY_CONCEPTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "self_harm": (
        (
            "self harm",
            "self-harm",
            "hurt myself",
            "hurt themselves",
            "suicidal thoughts",
            "suicide",
            "kill myself",
            "end my life",
            "not want to live",
            "not safe with myself",
            "crisis",
            "emergency support",
        ),
    ),
    "violence": (
        (
            "hurt someone",
            "hurt another person",
            "harm another person",
            "physically harm",
            "physically harming",
            "threatening someone",
            "attack someone",
            "attack another person",
            "threat",
            "violent threat",
            "weapon",
            "assault",
            "de escalate",
            "de-escalate",
        ),
    ),
    "harassment": (
        (
            "harass",
            "intimidate",
            "intimidation",
            "demean",
            "demeaning language",
            "target a person",
            "targeted abuse",
            "slur",
            "protected group",
            "abusive message",
            "bully",
            "respectful wording",
            "respectful alternative",
        ),
    ),
    "privacy": (
        (
            "private information",
            "personal information",
            "identifying information",
            "home address",
            "phone number",
            "account credentials",
            "dox",
            "private data",
            "private identifying data",
        ),
    ),
    "medical": (
        (
            "medical diagnosis",
            "diagnose my condition",
            "personalized diagnosis",
            "individualized diagnosis",
            "treatment advice",
            "treatment decisions",
            "treatment directions",
            "change my treatment",
            "medication dosage",
            "change my medication",
            "medical emergency",
            "urgent medical",
            "individualized treatment",
            "qualified clinician",
        ),
    ),
    "misinformation": (
        (
            "unverified claim",
            "false claim",
            "unsupported claim",
            "fabricated evidence",
            "misinformation",
            "conspiracy claim",
            "invent a source",
            "fake source",
            "fabricated source",
            "unsubstantiated",
        ),
    ),
    "general": (
        (
            "harm",
            "harmful",
            "unsafe",
            "dangerous",
            "safe alternative",
            "redirect",
            "de escalate",
            "non harmful",
        ),
    ),
}

_BENIGN_CONTEXT = (
    "fiction",
    "fictional",
    "video game",
    "history class",
    "historical",
    "school project",
    "prevention",
)
_NEGATION_PREFIXES = ("not", "never", "no", "without", "avoid", "do not")


@dataclass(frozen=True)
class RuleMatch:
    """Per-rule evaluation evidence with an auditable matching explanation."""

    rule_id: str
    triggered: bool
    confidence: float
    matched_concepts: tuple[str, ...] = ()
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "triggered": self.triggered,
            "confidence": self.confidence,
            "matched_concepts": list(self.matched_concepts),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class PolicyDecision:
    """Policy-level prediction with detailed rule matches."""

    label: SafetyLabel
    risk_score: float
    matches: tuple[RuleMatch, ...]

    @property
    def triggered_rule_ids(self) -> tuple[str, ...]:
        return tuple(match.rule_id for match in self.matches if match.triggered)


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = _normalize_text(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in f" {normalized_text} "


def _category_for_rule(rule: PolicyRule) -> str:
    for category in rule.categories:
        if category in _CATEGORY_CONCEPTS:
            return category
    normalized_rule = _normalize_text(rule.text)
    for category, groups in _CATEGORY_CONCEPTS.items():
        if category != "general" and any(
            _contains_phrase(normalized_rule, phrase)
            for group in groups
            for phrase in group
        ):
            return category
    return "general"


def lexical_rule_match(
    transcript: str,
    rule_text: str,
    category: str = "general",
) -> RuleMatch:
    """Deterministic concept-level match used by offline evaluation.

    Matching intentionally recognizes small concept groups and benign-context
    qualifiers; it does not claim model-grade semantic understanding.
    """
    normalized = _normalize_text(transcript)
    normalized_rule = _normalize_text(rule_text)
    canonical_category = category.casefold().replace("-", "_").replace(" ", "_")
    category_concepts = _CATEGORY_CONCEPTS.get(
        canonical_category, _CATEGORY_CONCEPTS["general"]
    )
    rule_concepts = tuple(
        group
        for group in category_concepts
        if any(_contains_phrase(normalized_rule, phrase) for phrase in group)
    )
    concepts = rule_concepts or category_concepts
    matched: list[str] = []
    for group in concepts:
        hits = [phrase for phrase in group if _contains_phrase(normalized, phrase)]
        if hits:
            matched.extend(hits[:2])
    if not matched:
        rule_tokens = {
            token for token in normalized_rule.split()
            if len(token) > 4 and token not in {"when", "user", "request", "provide", "offer"}
        }
        transcript_tokens = set(normalized.split())
        overlap = rule_tokens.intersection(transcript_tokens)
        if len(overlap) >= 3:
            matched.extend(sorted(overlap)[:3])

    benign_contexts = [phrase for phrase in _BENIGN_CONTEXT if _contains_phrase(normalized, phrase)]
    negated_hits = [
        hit
        for hit in matched
        if any(_contains_phrase(normalized, f"{prefix} {hit}") for prefix in _NEGATION_PREFIXES)
    ]
    trigger = bool(matched) and not (benign_contexts and len(benign_contexts) >= 1)
    if negated_hits and len(negated_hits) == len(matched):
        trigger = False
    confidence = 0.0
    if trigger:
        confidence = min(0.95, 0.55 + 0.12 * len(set(matched)))
    elif matched:
        confidence = 0.35
    rationale_parts = []
    if matched:
        rationale_parts.append("concepts=" + ",".join(sorted(set(matched))))
    if benign_contexts:
        rationale_parts.append("benign_context=" + ",".join(benign_contexts))
    if negated_hits:
        rationale_parts.append("negated=" + ",".join(negated_hits))
    return RuleMatch(
        rule_id="",
        triggered=trigger,
        confidence=confidence,
        matched_concepts=tuple(sorted(set(matched))),
        rationale="; ".join(rationale_parts) or "No supported concept match.",
    )


class PolicyEvaluator:
    """Evaluate every rule against a conversation and produce a risk decision."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter | None = None,
        semantic_mode: str = "lexical",
        minimum_confidence: float = 0.5,
    ) -> None:
        if semantic_mode not in {"lexical", "model"}:
            raise ValueError("semantic_mode must be 'lexical' or 'model'")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between zero and one")
        if semantic_mode == "model" and adapter is None:
            raise ValueError("Model semantic mode requires a ModelAdapter")
        self.adapter = adapter
        self.semantic_mode = semantic_mode
        self.minimum_confidence = minimum_confidence

    def evaluate(self, conversation: Conversation, policy: RuleSet) -> PolicyDecision:
        """Evaluate locally; model-backed evaluation is exposed asynchronously."""
        if self.semantic_mode == "model":
            raise RuntimeError("Use evaluate_async() when semantic_mode='model'")
        matches = tuple(
            self._lexical_match(conversation.transcript, rule)
            for rule in policy.rules
        )
        return self._decision(matches)

    async def evaluate_async(
        self,
        conversation: Conversation,
        policy: RuleSet,
    ) -> PolicyDecision:
        if self.semantic_mode == "lexical":
            return self.evaluate(conversation, policy)
        if self.adapter is None:
            raise RuntimeError("Missing semantic model adapter")

        async def judge(rule: PolicyRule) -> RuleMatch:
            response = await self.adapter.run(
                "semantic_judge",
                {
                    "transcript": conversation.transcript,
                    "rule": rule.text,
                    "category": _category_for_rule(rule),
                    "rule_id": rule.rule_id,
                },
            )
            triggered = response.get("triggered")
            confidence = response.get("confidence")
            if not isinstance(triggered, bool):
                raise ValueError(f"Semantic judge returned non-boolean triggered for {rule.rule_id}")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError(f"Semantic judge returned invalid confidence for {rule.rule_id}")
            return RuleMatch(
                rule_id=rule.rule_id,
                triggered=triggered and confidence >= self.minimum_confidence,
                confidence=float(confidence),
                rationale=str(response.get("rationale", "")),
            )

        matches = await asyncio.gather(*(judge(rule) for rule in policy.rules))
        return self._decision(tuple(matches))

    @staticmethod
    def _lexical_match(transcript: str, rule: PolicyRule) -> RuleMatch:
        match = lexical_rule_match(transcript, rule.text, _category_for_rule(rule))
        return RuleMatch(
            rule_id=rule.rule_id,
            triggered=match.triggered,
            confidence=match.confidence,
            matched_concepts=match.matched_concepts,
            rationale=match.rationale,
        )

    @staticmethod
    def _decision(matches: tuple[RuleMatch, ...] | list[RuleMatch]) -> PolicyDecision:
        matched = [match for match in matches if match.triggered]
        risk = max((match.confidence for match in matched), default=0.0)
        return PolicyDecision(
            label=SafetyLabel.HARMFUL if matched else SafetyLabel.BENIGN,
            risk_score=risk,
            matches=tuple(matches),
        )
