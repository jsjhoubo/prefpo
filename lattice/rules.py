"""Grounded guardrail generation, quality checks, and redundancy consolidation."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from prefpo.lattice.adapters import ModelAdapter
from prefpo.lattice.models import (
    LabeledConversation,
    PolicyRule,
    RuleSet,
    SafetyLabel,
    stable_rule_id,
)

_CATEGORY_ALIASES = {
    "suicide": "self_harm",
    "self-harm": "self_harm",
    "self harm": "self_harm",
    "violence": "violence",
    "threat": "violence",
    "harassment": "harassment",
    "hate": "harassment",
    "privacy": "privacy",
    "doxxing": "privacy",
    "medical": "medical",
    "health": "medical",
    "misinformation": "misinformation",
}

_TEMPLATES = {
    "self_harm": (
        "When a user signals suicidal thoughts or intent to hurt themselves or requests help doing so, "
        "respond with empathy, encourage immediate support from a trusted person or "
        "qualified service, and do not provide harmful instructions."
    ),
    "violence": (
        "When a user seeks help threatening someone or physically harming someone, do not "
        "assist with the harm; de-escalate and offer a safe, lawful alternative."
    ),
    "harassment": (
        "When a user requests targeted abuse, intimidation, or demeaning language, "
        "decline the abusive content and offer respectful wording."
    ),
    "privacy": (
        "When a user asks to identify, locate, or expose another person's private data, "
        "do not help obtain or disclose it; suggest consent-based alternatives."
    ),
    "medical": (
        "For consequential medical decisions, provide general context only, avoid "
        "personalized diagnosis, treatment decisions, or treatment directions, and "
        "recommend a qualified clinician."
    ),
    "misinformation": (
        "When a user asks to state an unverified claim as fact, distinguish evidence "
        "from speculation, express uncertainty, and avoid fabricated sources."
    ),
    "general": (
        "When a request would foreseeably cause harm, do not provide enabling details "
        "and redirect to a safe, relevant alternative."
    ),
}


def normalize_category(value: str) -> str:
    key = re.sub(r"\s+", " ", value.strip().casefold())
    return _CATEGORY_ALIASES.get(key, re.sub(r"[^a-z0-9]+", "_", key).strip("_") or "general")


def infer_category(example: LabeledConversation) -> str:
    """Choose an explicit category or infer a narrow category from safe metadata."""
    category = normalize_category(example.category)
    if category != "general":
        return category
    metadata_category = example.conversation.metadata.get("category")
    if isinstance(metadata_category, str):
        category = normalize_category(metadata_category)
        if category != "general":
            return category
    rationale = example.rationale.casefold()
    for candidate in _CATEGORY_ALIASES:
        if candidate in rationale:
            return normalize_category(candidate)
    return "general"


def _rule_template(category: str, evidence: list[LabeledConversation]) -> str:
    base = _TEMPLATES.get(category, _TEMPLATES["general"])
    rationales = [item.rationale.strip() for item in evidence if item.rationale.strip()]
    if not rationales:
        return base
    # Evidence is represented as categories and IDs in provenance; user text itself
    # is intentionally not copied into a reusable policy.
    return base


@dataclass(frozen=True)
class GenerationReport:
    """Rules created from evidence and examples omitted by confidence filtering."""

    policy: RuleSet
    evidence_count: int
    generated_categories: tuple[str, ...]
    rejected_categories: tuple[str, ...]


class RuleGenerator:
    """Generate a compact rule per harmful category from labeled examples."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter | None = None,
        minimum_confidence: float = 0.55,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between zero and one")
        self.adapter = adapter
        self.minimum_confidence = minimum_confidence

    async def generate(self, examples: Iterable[LabeledConversation]) -> GenerationReport:
        harmful = [item for item in examples if item.label == SafetyLabel.HARMFUL]
        if not harmful:
            return GenerationReport(RuleSet(), 0, (), ())
        by_category: dict[str, list[LabeledConversation]] = defaultdict(list)
        for item in harmful:
            by_category[infer_category(item)].append(item)

        generated: list[PolicyRule] = []
        rejected: list[str] = []
        for category in sorted(by_category):
            evidence = by_category[category]
            fallback_text = _rule_template(category, evidence)
            if self.adapter is None:
                proposal: dict[str, Any] = {
                    "text": fallback_text,
                    "categories": [category],
                    "confidence": min(0.9, 0.65 + 0.05 * (len(evidence) - 1)),
                }
            else:
                proposal = await self.adapter.run(
                    "generate_rule",
                    {
                        "category": category,
                        "evidence_count": len(evidence),
                        "evidence_ids": [item.conversation_id for item in evidence],
                        "rationales": [
                            item.rationale[:500] for item in evidence if item.rationale.strip()
                        ][:8],
                        "current_rule_template": fallback_text,
                    },
                )
            text = proposal.get("text")
            confidence = proposal.get("confidence", 0.5)
            categories = proposal.get("categories", [category])
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Rule generator returned empty text for category {category}")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError(f"Rule generator returned invalid confidence for {category}")
            if float(confidence) < self.minimum_confidence:
                rejected.append(category)
                continue
            if not isinstance(categories, list) or not categories:
                raise ValueError(f"Rule generator returned invalid categories for {category}")
            normalized_categories = tuple(normalize_category(str(item)) for item in categories)
            if category not in normalized_categories:
                raise ValueError(
                    f"Rule generator omitted the evidence category {category!r}"
                )
            rule = PolicyRule(
                rule_id=stable_rule_id(f"{category}:{text}"),
                text=text.strip(),
                categories=normalized_categories,
                source="lattice_construction",
                confidence=float(confidence),
                evidence_ids=tuple(item.conversation_id for item in evidence),
            )
            generated.append(rule)

        return GenerationReport(
            policy=RuleSet(tuple(generated)),
            evidence_count=len(harmful),
            generated_categories=tuple(rule.categories[0] for rule in generated),
            rejected_categories=tuple(rejected),
        )


@dataclass(frozen=True)
class ConsolidationReport:
    """Consolidated policy and exact source-to-result lineage."""

    policy: RuleSet
    merged_rule_ids: tuple[str, ...]
    retained_rule_ids: tuple[str, ...]
    strategy: str


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 2 and token not in {"the", "and", "for", "when", "with", "that", "from"}
    }


def _similarity(left: PolicyRule, right: PolicyRule) -> float:
    left_categories = set(left.categories)
    right_categories = set(right.categories)
    if not left_categories.intersection(right_categories):
        return 0.0
    left_tokens = _tokens(left.text)
    right_tokens = _tokens(right.text)
    union = left_tokens.union(right_tokens)
    return len(left_tokens.intersection(right_tokens)) / len(union) if union else 0.0


class RuleConsolidator:
    """Consolidate duplicate rules while retaining explicit provenance."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter | None = None,
        similarity_threshold: float = 0.62,
        maximum_rules: int = 100,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between zero and one")
        if maximum_rules < 1:
            raise ValueError("maximum_rules must be at least one")
        self.adapter = adapter
        self.similarity_threshold = similarity_threshold
        self.maximum_rules = maximum_rules

    async def consolidate(self, policy: RuleSet) -> ConsolidationReport:
        if len(policy) <= 1:
            return ConsolidationReport(
                policy, (), tuple(rule.rule_id for rule in policy.rules), "no-op"
            )
        clusters = self._clusters(policy.rules)
        merged: list[PolicyRule] = []
        removed: list[str] = []
        retained: list[str] = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(cluster[0])
                retained.append(cluster[0].rule_id)
                continue
            representative = max(cluster, key=lambda rule: (rule.confidence, -rule.version))
            categories = tuple(sorted({category for rule in cluster for category in rule.categories}))
            evidence = tuple(dict.fromkeys(
                evidence_id for rule in cluster for evidence_id in rule.evidence_ids
            ))
            text = representative.text
            if self.adapter is not None:
                response = await self.adapter.run(
                    "consolidate_rules",
                    {
                        "rules": [rule.to_dict() for rule in cluster],
                        "task": "Merge overlapping rules without dropping distinct coverage.",
                    },
                )
                candidates = response.get("rules")
                if not isinstance(candidates, list) or len(candidates) != 1:
                    raise ValueError("Consolidator must return exactly one rule for a duplicate cluster")
                candidate = candidates[0]
                if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
                    raise ValueError("Consolidator returned a malformed merged rule")
                text = candidate["text"].strip()
                if not text:
                    raise ValueError("Consolidator returned empty merged rule text")
                returned_categories = candidate.get("categories", categories)
                if not isinstance(returned_categories, list) or not returned_categories:
                    raise ValueError("Consolidator returned no categories")
                categories = tuple(
                    sorted({normalize_category(str(item)) for item in returned_categories})
                )
                returned_sources = set(candidate.get("source_rule_ids", []))
                expected_sources = {rule.rule_id for rule in cluster}
                if returned_sources != expected_sources:
                    raise ValueError("Consolidator source_rule_ids did not preserve cluster lineage")
            retained_rule = PolicyRule(
                rule_id=representative.rule_id,
                text=text,
                categories=categories,
                source="consolidated",
                confidence=max(rule.confidence for rule in cluster),
                evidence_ids=evidence,
                version=max(rule.version for rule in cluster) + 1,
            )
            merged.append(retained_rule)
            retained.append(retained_rule.rule_id)
            removed.extend(rule.rule_id for rule in cluster if rule.rule_id != representative.rule_id)

        if len(merged) > self.maximum_rules:
            ranked = sorted(merged, key=lambda rule: (rule.confidence, len(rule.evidence_ids)))
            to_remove = {rule.rule_id for rule in ranked[: len(merged) - self.maximum_rules]}
            # Never discard the only policy covering a category.
            for category in {cat for rule in merged for cat in rule.categories}:
                if not any(category in rule.categories for rule in merged if rule.rule_id not in to_remove):
                    candidates = [rule for rule in merged if category in rule.categories]
                    to_remove.discard(max(candidates, key=lambda rule: rule.confidence).rule_id)
            removed.extend(sorted(to_remove))
            merged = [rule for rule in merged if rule.rule_id not in to_remove]
            retained = [rule.rule_id for rule in merged]

        result = RuleSet(tuple(merged), policy.revision + 1, policy.fingerprint)
        strategy = "model-assisted" if self.adapter is not None else "category-and-jaccard"
        return ConsolidationReport(result, tuple(removed), tuple(retained), strategy)

    def _clusters(self, rules: tuple[PolicyRule, ...]) -> list[list[PolicyRule]]:
        remaining = list(rules)
        clusters: list[list[PolicyRule]] = []
        while remaining:
            representative = remaining.pop(0)
            cluster = [representative]
            still_remaining: list[PolicyRule] = []
            for candidate in remaining:
                if _similarity(representative, candidate) >= self.similarity_threshold:
                    cluster.append(candidate)
                else:
                    still_remaining.append(candidate)
            clusters.append(cluster)
            remaining = still_remaining
        return clusters
