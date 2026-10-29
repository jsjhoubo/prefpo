"""Typed, serializable data models used by the Lattice research pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable


class SafetyLabel(IntEnum):
    """Binary conversation label: benign is zero and harmful is one."""

    BENIGN = 0
    HARMFUL = 1

    @classmethod
    def parse(cls, value: SafetyLabel | str | int | bool) -> SafetyLabel:
        """Parse common dataset encodings without guessing ambiguous values."""
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.HARMFUL if value else cls.BENIGN
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError as exc:
                raise ValueError(f"Safety label integer must be 0 or 1, got {value}") from exc
        normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
        aliases = {
            "0": cls.BENIGN,
            "benign": cls.BENIGN,
            "safe": cls.BENIGN,
            "non_harmful": cls.BENIGN,
            "not_harmful": cls.BENIGN,
            "1": cls.HARMFUL,
            "harmful": cls.HARMFUL,
            "unsafe": cls.HARMFUL,
            "prosocial_violation": cls.HARMFUL,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported safety label: {value!r}") from exc


def _clean_text(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    return cleaned


def _json_safe_mapping(value: dict[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a mapping with string keys")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain finite JSON-serializable values") from exc
    return dict(value)


@dataclass(frozen=True)
class ConversationTurn:
    """One user or assistant message in a conversation transcript."""

    speaker: str
    text: str
    turn_index: int = 0

    def __post_init__(self) -> None:
        speaker = _clean_text(self.speaker, name="speaker").casefold()
        if speaker not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"Unsupported conversation speaker: {speaker!r}")
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")
        object.__setattr__(self, "speaker", speaker)
        object.__setattr__(self, "text", _clean_text(self.text, name="turn text"))

    def to_dict(self) -> dict[str, Any]:
        return {"speaker": self.speaker, "text": self.text, "turn_index": self.turn_index}

    @classmethod
    def from_dict(cls, value: dict[str, Any], fallback_index: int = 0) -> ConversationTurn:
        speaker = value.get("speaker", value.get("role"))
        text = value.get("text", value.get("content"))
        if speaker is None or text is None:
            raise ValueError("Each turn requires speaker/role and text/content")
        return cls(str(speaker), str(text), int(value.get("turn_index", fallback_index)))


@dataclass(frozen=True)
class Conversation:
    """An ordered, non-empty multi-turn transcript."""

    conversation_id: str
    turns: tuple[ConversationTurn, ...]
    domain: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "conversation_id", _clean_text(self.conversation_id, name="conversation_id")
        )
        if not self.turns:
            raise ValueError("A conversation must contain at least one turn")
        turns = tuple(self.turns)
        if any(not isinstance(turn, ConversationTurn) for turn in turns):
            raise TypeError("turns must contain ConversationTurn instances")
        if any(a.turn_index >= b.turn_index for a, b in zip(turns, turns[1:])):
            raise ValueError("turn_index values must be strictly increasing")
        object.__setattr__(self, "turns", turns)
        object.__setattr__(self, "domain", _clean_text(self.domain, name="domain").casefold())
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata, name="metadata"))

    @property
    def transcript(self) -> str:
        return "\n".join(f"{turn.speaker}: {turn.text}" for turn in self.turns)

    @property
    def user_turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(turn for turn in self.turns if turn.speaker == "user")

    def append(self, speaker: str, text: str) -> Conversation:
        """Return a new conversation with the next turn appended."""
        next_index = self.turns[-1].turn_index + 1
        return Conversation(
            conversation_id=self.conversation_id,
            turns=self.turns + (ConversationTurn(speaker, text, next_index),),
            domain=self.domain,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "domain": self.domain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Conversation:
        raw_turns = value.get("turns")
        if raw_turns is None:
            raw_turns = value.get("messages")
        if not isinstance(raw_turns, list):
            raise ValueError("Conversation requires a turns/messages list")
        turns = tuple(ConversationTurn.from_dict(turn, i) for i, turn in enumerate(raw_turns))
        return cls(
            conversation_id=str(value.get("conversation_id", value.get("id", ""))),
            turns=turns,
            domain=str(value.get("domain", "general")),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class LabeledConversation:
    """Conversation paired with a binary label and optional rationale/category."""

    conversation: Conversation
    label: SafetyLabel
    category: str = "general"
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        object.__setattr__(self, "label", SafetyLabel.parse(self.label))
        object.__setattr__(self, "category", _clean_text(self.category, name="category").casefold())
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")

    @property
    def conversation_id(self) -> str:
        return self.conversation.conversation_id

    @property
    def domain(self) -> str:
        return self.conversation.domain

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.conversation.to_dict(),
            "label": int(self.label),
            "category": self.category,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LabeledConversation:
        conversation = Conversation.from_dict(value)
        label = value.get("label", value.get("is_harmful", value.get("safety_label")))
        if label is None:
            raise ValueError("Labeled conversation requires label/is_harmful/safety_label")
        return cls(
            conversation=conversation,
            label=SafetyLabel.parse(label),
            category=str(value.get("category", value.get("topic", "general"))),
            rationale=str(value.get("rationale", "")),
        )


@dataclass(frozen=True)
class PolicyRule:
    """A versioned natural-language rule with provenance."""

    rule_id: str
    text: str
    categories: tuple[str, ...] = ("general",)
    source: str = "seed"
    confidence: float = 1.0
    evidence_ids: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _clean_text(self.rule_id, name="rule_id"))
        object.__setattr__(self, "text", _clean_text(self.text, name="rule text"))
        normalized = tuple(dict.fromkeys(c.strip().casefold() for c in self.categories if c.strip()))
        if not normalized:
            normalized = ("general",)
        object.__setattr__(self, "categories", normalized)
        object.__setattr__(self, "evidence_ids", tuple(dict.fromkeys(self.evidence_ids)))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.version < 1:
            raise ValueError("version must be at least 1")

    def revise(self, text: str, *, source: str | None = None) -> PolicyRule:
        """Create a new revision while preserving the original rule identity."""
        return PolicyRule(
            rule_id=self.rule_id,
            text=text,
            categories=self.categories,
            source=source or self.source,
            confidence=self.confidence,
            evidence_ids=self.evidence_ids,
            version=self.version + 1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "text": self.text,
            "categories": list(self.categories),
            "source": self.source,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyRule:
        categories = value.get("categories", ("general",))
        evidence = value.get("evidence_ids", ())
        return cls(
            rule_id=str(value["rule_id"]),
            text=str(value["text"]),
            categories=tuple(str(item) for item in categories),
            source=str(value.get("source", "imported")),
            confidence=float(value.get("confidence", 1.0)),
            evidence_ids=tuple(str(item) for item in evidence),
            version=int(value.get("version", 1)),
        )


@dataclass(frozen=True)
class RuleSet:
    """Immutable set-like policy representation with a stable fingerprint."""

    rules: tuple[PolicyRule, ...] = ()
    revision: int = 0
    parent_fingerprint: str | None = None

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if any(not isinstance(rule, PolicyRule) for rule in rules):
            raise TypeError("rules must contain PolicyRule instances")
        ids = [rule.rule_id for rule in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Rule ids must be unique within a RuleSet")
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        object.__setattr__(self, "rules", rules)

    def __iter__(self):
        return iter(self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            [rule.to_dict() for rule in self.rules], sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, rule_id: str) -> PolicyRule | None:
        return next((rule for rule in self.rules if rule.rule_id == rule_id), None)

    def evolve(
        self,
        *,
        add: Iterable[PolicyRule] = (),
        replace: Iterable[PolicyRule] = (),
        remove: Iterable[str] = (),
    ) -> RuleSet:
        """Return a child revision with explicit additions, replacements, and removals."""
        replacement_rules = list(replace)
        addition_rules = list(add)
        replacements = {rule.rule_id: rule for rule in replacement_rules}
        additions = {rule.rule_id: rule for rule in addition_rules}
        if len(replacements) != len(replacement_rules):
            raise ValueError("Replacement rule ids must be unique")
        if len(additions) != len(addition_rules):
            raise ValueError("Added rule ids must be unique")
        removed = set(remove)
        overlap = removed.intersection(replacements)
        if overlap:
            raise ValueError(f"Cannot replace and remove rule ids: {sorted(overlap)}")
        ordered: list[PolicyRule] = []
        seen: set[str] = set()
        for rule in self.rules:
            if rule.rule_id in removed:
                continue
            updated = replacements.get(rule.rule_id, rule)
            ordered.append(updated)
            seen.add(updated.rule_id)
        for rule_id, rule in replacements.items():
            if rule_id not in seen:
                raise ValueError(f"Cannot replace unknown rule id: {rule_id}")
        for rule_id, rule in additions.items():
            if rule_id in seen:
                raise ValueError(f"Rule id already exists: {rule_id}")
            ordered.append(rule)
            seen.add(rule_id)
        return RuleSet(tuple(ordered), self.revision + 1, self.fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "parent_fingerprint": self.parent_fingerprint,
            "fingerprint": self.fingerprint,
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RuleSet:
        return cls(
            rules=tuple(PolicyRule.from_dict(item) for item in value.get("rules", [])),
            revision=int(value.get("revision", 0)),
            parent_fingerprint=value.get("parent_fingerprint"),
        )


def stable_rule_id(text: str, *, prefix: str = "rule") -> str:
    """Create readable, reproducible rule identifiers for generated candidates."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")[:32] or "policy"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{slug}-{digest}"
