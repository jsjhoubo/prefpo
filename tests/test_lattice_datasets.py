"""Local data import, split, and PrefPO bridge behavior."""

import csv
import json

import pytest

from prefpo.lattice.datasets import (
    as_prefpo_sample,
    from_prefpo_samples,
    load_conversations,
    load_unlabeled_conversations,
    normalize_record,
    split_conversations,
    validate_domain_disjointness,
)
from prefpo.lattice.models import SafetyLabel


def _row(index, label, *, domain="social", category="general"):
    return {
        "id": f"case-{index}",
        "domain": domain,
        "category": category,
        "label": label,
        "turns": [{"role": "user", "content": f"Question {index}"}],
    }


def test_load_jsonl_preserves_labels_domains_and_turns(tmp_path):
    path = tmp_path / "conversations.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(_row(i, "harmful" if i % 2 else "benign"))
            for i in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_conversations(path)
    assert len(samples) == 4
    assert samples[1].label is SafetyLabel.HARMFUL
    assert samples[0].conversation.turns[0].text == "Question 0"


def test_load_jsonl_reports_invalid_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"label":"benign","turns":[]}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_conversations(path)


def test_load_json_array_and_limit(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps([_row(i, i % 2) for i in range(5)]), encoding="utf-8")
    items = load_conversations(path, limit=2)
    assert [item.conversation_id for item in items] == ["case-0", "case-1"]


def test_load_csv_with_json_turns(tmp_path):
    path = tmp_path / "dataset.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "label", "messages", "domain"])
        writer.writeheader()
        writer.writerow(
            {
                "id": "csv-1",
                "label": "safe",
                "domain": "social",
                "messages": json.dumps([{"role": "user", "content": "Hello"}]),
            }
        )
    assert load_conversations(path)[0].label is SafetyLabel.BENIGN


def test_load_unlabeled_deployment_without_gold_labels(tmp_path):
    path = tmp_path / "deployment.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "deploy-1",
                "domain": "ethics",
                "messages": [{"role": "user", "content": "A deployment transcript"}],
            }
        ),
        encoding="utf-8",
    )
    records = load_unlabeled_conversations(path)
    assert len(records) == 1
    assert records[0].domain == "ethics"
    assert records[0].transcript == "user: A deployment transcript"


def test_unlabeled_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicate.json"
    row = {
        "id": "duplicate",
        "turns": [{"speaker": "user", "text": "Question"}],
    }
    path.write_text(json.dumps([row, row]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_unlabeled_conversations(path)


def test_split_is_deterministic_stratified_and_disjoint():
    rows = [
        normalize_record(_row(i, "harmful" if i % 2 else "benign"))
        for i in range(30)
    ]
    train_a, validation_a = split_conversations(
        rows, validation_fraction=0.25, seed=7
    )
    train_b, validation_b = split_conversations(
        rows, validation_fraction=0.25, seed=7
    )
    assert [item.conversation_id for item in train_a] == [
        item.conversation_id for item in train_b
    ]
    assert [item.conversation_id for item in validation_a] == [
        item.conversation_id for item in validation_b
    ]
    assert {item.conversation_id for item in train_a}.isdisjoint(
        item.conversation_id for item in validation_a
    )
    assert {item.label for item in validation_a} == {
        SafetyLabel.BENIGN,
        SafetyLabel.HARMFUL,
    }


def test_split_rejects_tiny_and_bad_fraction_inputs():
    one = [normalize_record(_row(0, "benign"))]
    with pytest.raises(ValueError, match="At least two"):
        split_conversations(one)
    with pytest.raises(ValueError, match="between zero and one"):
        split_conversations(
            [normalize_record(_row(i, "benign")) for i in range(2)],
            validation_fraction=1.0,
        )


def test_domain_disjointness_is_enforced_explicitly():
    train = [normalize_record(_row(0, "benign", domain="social"))]
    validation = [normalize_record(_row(1, "harmful", domain="ethics"))]
    validate_domain_disjointness(train, validation)
    with pytest.raises(ValueError, match="overlap"):
        validate_domain_disjointness(train, [normalize_record(_row(2, "benign"))])


def test_prefpo_sample_round_trip_keeps_domain_category_and_label():
    record = normalize_record(_row(42, "harmful", category="violence"))
    sample = as_prefpo_sample(record)
    restored = from_prefpo_samples([sample])[0]
    assert restored.label is SafetyLabel.HARMFUL
    assert restored.domain == "social"
    assert restored.category == "violence"
    assert restored.conversation.transcript == record.conversation.transcript


def test_normalize_rejects_missing_label_or_transcript():
    with pytest.raises(ValueError, match="no binary safety label"):
        normalize_record({"id": "no-label", "text": "Only text"})
    with pytest.raises(ValueError, match="no turns"):
        normalize_record({"id": "no-turns", "label": "benign"})


def test_unsupported_format_and_negative_limit_are_rejected(tmp_path):
    path = tmp_path / "data.bin"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_conversations(path)
    with pytest.raises(ValueError, match="non-negative"):
        load_conversations(tmp_path / "anything.jsonl", limit=-1)
