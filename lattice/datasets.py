"""Local conversation dataset readers, validation, split, and conversion helpers."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

from prefpo.lattice.models import (
    Conversation,
    ConversationTurn,
    LabeledConversation,
    SafetyLabel,
)
from prefpo.types import Sample

DatasetFormat = Literal["jsonl", "json", "csv"]


def _parse_messages(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("messages/turns column must contain JSON") from exc
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("messages"), list):
            return raw["messages"]
        if isinstance(raw.get("turns"), list):
            return raw["turns"]
    raise ValueError("Conversation record must contain a list of turns/messages")


def normalize_record(record: dict[str, Any], *, index: int = 0) -> LabeledConversation:
    """Normalize native Lattice and common dialogue benchmark record shapes."""
    if not isinstance(record, dict):
        raise TypeError(f"Dataset row {index} must be a mapping")
    normalized = dict(record)
    turns = normalized.get("turns", normalized.get("messages"))
    if turns is None and "dialogue" in normalized:
        turns = normalized["dialogue"]
    if turns is None and "conversation" in normalized:
        turns = normalized["conversation"]
    if turns is not None and not isinstance(turns, list):
        turns = _parse_messages(turns)
    if turns is None and "text" in normalized:
        turns = [{"speaker": "user", "text": str(normalized["text"]), "turn_index": 0}]
    if turns is None:
        raise ValueError(f"Dataset row {index} has no turns, messages, dialogue, or text field")

    label = normalized.get(
        "label",
        normalized.get("is_harmful", normalized.get("safety_label", normalized.get("unsafe"))),
    )
    if label is None and "response_type" in normalized:
        label = normalized["response_type"]
    if label is None:
        raise ValueError(f"Dataset row {index} has no binary safety label")
    safe_turns = []
    for turn_index, turn in enumerate(turns):
        if isinstance(turn, str):
            safe_turns.append({"speaker": "user", "text": turn, "turn_index": turn_index})
        elif isinstance(turn, dict):
            speaker = turn.get("speaker", turn.get("role", "user"))
            text = turn.get("text", turn.get("content"))
            if text is None:
                raise ValueError(f"Dataset row {index}, turn {turn_index} has no text/content")
            safe_turns.append(
                {
                    "speaker": str(speaker),
                    "text": str(text),
                    "turn_index": int(turn.get("turn_index", turn_index)),
                }
            )
        else:
            raise TypeError(f"Dataset row {index}, turn {turn_index} must be text or mapping")
    metadata = normalized.get("metadata") or {}
    known_keys = {
        "id", "conversation_id", "turns", "messages", "dialogue", "conversation",
        "text", "label", "is_harmful", "safety_label", "unsafe", "response_type",
        "category", "topic", "domain", "rationale", "metadata",
    }
    if not isinstance(metadata, dict):
        raise TypeError(f"Dataset row {index} metadata must be a mapping")
    extra = {key: value for key, value in normalized.items() if key not in known_keys}
    metadata = {**extra, **metadata}
    conversation = Conversation(
        conversation_id=str(normalized.get("conversation_id", normalized.get("id", index))),
        turns=tuple(ConversationTurn.from_dict(turn, fallback_index=i) for i, turn in enumerate(safe_turns)),
        domain=str(normalized.get("domain", "general")),
        metadata=metadata,
    )
    return LabeledConversation(
        conversation=conversation,
        label=SafetyLabel.parse(label),
        category=str(normalized.get("category", normalized.get("topic", "general"))),
        rationale=str(normalized.get("rationale", "")),
    )


def load_conversations(
    path: str | Path,
    *,
    format: DatasetFormat | None = None,
    limit: int | None = None,
) -> list[LabeledConversation]:
    """Read JSONL, JSON arrays, or CSV records from a local path.

    No network access or implicit Hugging Face download occurs. CSV `turns` or
    `messages` cells may contain JSON arrays.
    """
    records = _read_dataset_records(path, format=format, limit=limit)
    samples = [normalize_record(record, index=index) for index, record in enumerate(records)]
    seen: set[str] = set()
    for sample in samples:
        if sample.conversation_id in seen:
            raise ValueError(f"Duplicate conversation id: {sample.conversation_id}")
        seen.add(sample.conversation_id)
    return samples


def load_unlabeled_conversations(
    path: str | Path,
    *,
    format: DatasetFormat | None = None,
    limit: int | None = None,
) -> list[Conversation]:
    """Read deployment transcripts without requiring or inventing a gold label."""
    records = _read_dataset_records(path, format=format, limit=limit)
    conversations: list[Conversation] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Dataset row {index} must be a mapping")
        raw_turns = record.get(
            "turns",
            record.get("messages", record.get("dialogue", record.get("conversation"))),
        )
        if raw_turns is None and "text" in record:
            raw_turns = [{"speaker": "user", "text": str(record["text"])}]
        turns = _parse_messages(raw_turns)
        normalized_turns = []
        for turn_index, turn in enumerate(turns):
            if isinstance(turn, str):
                normalized_turns.append(
                    ConversationTurn("user", turn, turn_index)
                )
            elif isinstance(turn, dict):
                normalized_turns.append(
                    ConversationTurn.from_dict(turn, fallback_index=turn_index)
                )
            else:
                raise TypeError(f"Dataset row {index}, turn {turn_index} is invalid")
        metadata = record.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise TypeError(f"Dataset row {index} metadata must be a mapping")
        conversation = Conversation(
            conversation_id=str(record.get("conversation_id", record.get("id", index))),
            turns=tuple(normalized_turns),
            domain=str(record.get("domain", "general")),
            metadata=metadata,
        )
        if conversation.conversation_id in seen:
            raise ValueError(f"Duplicate conversation id: {conversation.conversation_id}")
        seen.add(conversation.conversation_id)
        conversations.append(conversation)
    return conversations


def _read_dataset_records(
    path: str | Path,
    *,
    format: DatasetFormat | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    source = Path(path)
    selected_format = format or source.suffix.lstrip(".").casefold()
    if selected_format not in {"jsonl", "json", "csv"}:
        raise ValueError(f"Unsupported conversation dataset format: {selected_format}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if not source.is_file():
        raise FileNotFoundError(source)
    if selected_format == "jsonl":
        records = []
        with source.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {source}"
                    ) from exc
    elif selected_format == "json":
        with source.open(encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, list):
            raise ValueError("JSON conversation dataset must be a list of objects")
        records = raw
    else:
        with source.open(encoding="utf-8-sig", newline="") as file:
            records = list(csv.DictReader(file))
    if limit is not None:
        records = records[:limit]
    return records


def split_conversations(
    conversations: Iterable[LabeledConversation],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
    stratify_by: Literal["label", "domain", "label_domain"] = "label",
) -> tuple[list[LabeledConversation], list[LabeledConversation]]:
    """Create reproducible approximate-stratified train/validation partitions."""
    items = list(conversations)
    if len(items) < 2:
        raise ValueError("At least two conversations are required for a split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if stratify_by not in {"label", "domain", "label_domain"}:
        raise ValueError(f"Unsupported stratification key: {stratify_by}")
    groups: dict[str, list[LabeledConversation]] = defaultdict(list)
    for item in items:
        if stratify_by == "label":
            key = str(int(item.label))
        elif stratify_by == "domain":
            key = item.domain
        else:
            key = f"{int(item.label)}:{item.domain}"
        groups[key].append(item)
    rng = random.Random(seed)
    train: list[LabeledConversation] = []
    validation: list[LabeledConversation] = []
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        n_validation = round(len(group) * validation_fraction)
        if len(group) > 1:
            n_validation = min(max(1, n_validation), len(group) - 1)
        else:
            n_validation = int(validation_fraction >= 0.5)
        validation.extend(group[:n_validation])
        train.extend(group[n_validation:])
    rng.shuffle(train)
    rng.shuffle(validation)
    if not train or not validation:
        raise ValueError("Split produced an empty partition; provide more examples")
    _assert_disjoint(train, validation)
    return train, validation


def _assert_disjoint(*partitions: Iterable[LabeledConversation]) -> None:
    seen: set[str] = set()
    for partition_index, partition in enumerate(partitions):
        for item in partition:
            if item.conversation_id in seen:
                raise ValueError(
                    f"Conversation id {item.conversation_id!r} appears in multiple splits "
                    f"(including partition {partition_index})"
                )
            seen.add(item.conversation_id)


def validate_domain_disjointness(
    train: Iterable[LabeledConversation],
    validation: Iterable[LabeledConversation],
) -> None:
    """Require that evaluation is performed on domains absent from training."""
    train_domains = {item.domain for item in train}
    validation_domains = {item.domain for item in validation}
    overlap = train_domains.intersection(validation_domains)
    if overlap:
        raise ValueError(f"Train and validation domains overlap: {sorted(overlap)}")


def as_prefpo_sample(item: LabeledConversation, *, index: int = 0) -> Sample:
    """Bridge the conversation record to PrefPO's existing Grader interface."""
    return Sample(
        index=index,
        question=item.conversation.transcript,
        target=str(int(item.label)),
        metadata={
            "conversation_id": item.conversation_id,
            "domain": item.domain,
            "category": item.category,
        },
    )


def from_prefpo_samples(samples: Iterable[Sample]) -> list[LabeledConversation]:
    """Reconstruct labeled conversations from samples created by as_prefpo_sample."""
    converted: list[LabeledConversation] = []
    for sample in samples:
        turns = []
        for index, line in enumerate(sample.question.splitlines()):
            speaker, delimiter, text = line.partition(": ")
            if delimiter:
                turns.append(ConversationTurn(speaker, text, index))
            elif line.strip():
                turns.append(ConversationTurn("user", line, index))
        if not turns:
            raise ValueError(f"PrefPO sample {sample.index} has no parseable transcript")
        converted.append(
            LabeledConversation(
                conversation=Conversation(
                    str(sample.metadata.get("conversation_id", sample.index)),
                    tuple(turns),
                    str(sample.metadata.get("domain", "general")),
                ),
                label=SafetyLabel.parse(sample.target),
                category=str(sample.metadata.get("category", "general")),
            )
        )
    return converted
