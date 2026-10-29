"""Binary safety-classification metrics and deterministic uncertainty estimates."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from prefpo.lattice.models import SafetyLabel


@dataclass(frozen=True)
class ClassificationMetrics:
    """Confusion counts and derived metrics for harmful-as-positive labels."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float
    support: int

    @property
    def false_positive_rate(self) -> float:
        denominator = self.false_positive + self.true_negative
        return self.false_positive / denominator if denominator else 0.0

    @property
    def false_negative_rate(self) -> float:
        denominator = self.false_negative + self.true_positive
        return self.false_negative / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate and stratified evaluation results with row-level predictions."""

    overall: ClassificationMetrics
    by_domain: dict[str, ClassificationMetrics] = field(default_factory=dict)
    by_category: dict[str, ClassificationMetrics] = field(default_factory=dict)
    prediction_rows: tuple[dict[str, object], ...] = ()
    bootstrap_f1_interval: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": self.overall.to_dict(),
            "by_domain": {key: value.to_dict() for key, value in self.by_domain.items()},
            "by_category": {key: value.to_dict() for key, value in self.by_category.items()},
            "prediction_rows": list(self.prediction_rows),
            "bootstrap_f1_interval": self.bootstrap_f1_interval,
        }


def _label(value: SafetyLabel | str | int | bool) -> int:
    return int(SafetyLabel.parse(value))


def binary_metrics(
    truth: Sequence[SafetyLabel | str | int | bool],
    predictions: Sequence[SafetyLabel | str | int | bool],
) -> ClassificationMetrics:
    """Compute binary metrics using zero-valued denominators as zero."""
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have equal lengths")
    actual = [_label(value) for value in truth]
    predicted = [_label(value) for value in predictions]
    tp = sum(y == 1 and p == 1 for y, p in zip(actual, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(actual, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(actual, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    support = len(actual)
    return ClassificationMetrics(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=(tp + tn) / support if support else 0.0,
        specificity=specificity,
        support=support,
    )


def _bootstrap_f1(
    truth: Sequence[SafetyLabel | str | int | bool],
    predictions: Sequence[SafetyLabel | str | int | bool],
    *,
    repetitions: int,
    confidence: float,
    seed: int,
) -> tuple[float, float] | None:
    if repetitions <= 0 or len(truth) < 2:
        return None
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        indices = [rng.randrange(len(truth)) for _ in truth]
        estimates.append(
            binary_metrics(
                [truth[index] for index in indices],
                [predictions[index] for index in indices],
            ).f1
        )
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, math.floor(tail * repetitions))
    upper_index = min(repetitions - 1, math.ceil((1.0 - tail) * repetitions) - 1)
    return estimates[lower_index], estimates[upper_index]


def evaluate_predictions(
    truth: Sequence[SafetyLabel | str | int | bool],
    predictions: Sequence[SafetyLabel | str | int | bool],
    *,
    domains: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    conversation_ids: Sequence[str] | None = None,
    bootstrap_repetitions: int = 0,
    bootstrap_confidence: float = 0.95,
    seed: int = 42,
) -> EvaluationReport:
    """Compute overall, slice, row-level, and optional bootstrap metrics."""
    n = len(truth)
    if len(predictions) != n:
        raise ValueError("truth and predictions must have equal lengths")
    for name, values in (
        ("domains", domains),
        ("categories", categories),
        ("conversation_ids", conversation_ids),
    ):
        if values is not None and len(values) != n:
            raise ValueError(f"{name} must have the same length as truth")

    overall = binary_metrics(truth, predictions)

    def slices(values: Sequence[str] | None) -> dict[str, ClassificationMetrics]:
        if values is None:
            return {}
        indices_by_key: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(values):
            indices_by_key[value].append(index)
        return {
            key: binary_metrics(
                [truth[index] for index in indices],
                [predictions[index] for index in indices],
            )
            for key, indices in sorted(indices_by_key.items())
        }

    actual = [_label(value) for value in truth]
    predicted = [_label(value) for value in predictions]
    rows = tuple(
        {
            "index": index,
            "conversation_id": conversation_ids[index] if conversation_ids else str(index),
            "truth": actual[index],
            "prediction": predicted[index],
            "correct": actual[index] == predicted[index],
            "domain": domains[index] if domains else "unknown",
            "category": categories[index] if categories else "general",
        }
        for index in range(n)
    )
    interval = _bootstrap_f1(
        truth,
        predictions,
        repetitions=bootstrap_repetitions,
        confidence=bootstrap_confidence,
        seed=seed,
    )
    return EvaluationReport(
        overall=overall,
        by_domain=slices(domains),
        by_category=slices(categories),
        prediction_rows=rows,
        bootstrap_f1_interval=interval,
    )


def macro_f1(
    truth: Iterable[SafetyLabel | str | int | bool],
    predictions: Iterable[SafetyLabel | str | int | bool],
) -> float:
    """Return unweighted F1 across benign and harmful classes."""
    actual = [_label(value) for value in truth]
    predicted = [_label(value) for value in predictions]
    if len(actual) != len(predicted):
        raise ValueError("truth and predictions must have equal lengths")
    class_scores: list[float] = []
    for class_id in (0, 1):
        class_truth = [int(value == class_id) for value in actual]
        class_pred = [int(value == class_id) for value in predicted]
        class_scores.append(binary_metrics(class_truth, class_pred).f1)
    return sum(class_scores) / 2
