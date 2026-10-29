"""Classification metric and confidence interval checks."""

import pytest

from prefpo.lattice.metrics import (
    binary_metrics,
    evaluate_predictions,
    macro_f1,
)


def test_binary_metrics_compute_confusion_and_rates():
    metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (metrics.true_positive, metrics.false_positive) == (1, 1)
    assert (metrics.false_negative, metrics.true_negative) == (1, 1)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.specificity == pytest.approx(0.5)
    assert metrics.false_positive_rate == pytest.approx(0.5)


def test_binary_metrics_define_empty_denominators_as_zero():
    metrics = binary_metrics([], [])
    assert metrics.support == 0
    assert metrics.precision == metrics.recall == metrics.f1 == 0.0
    only_benign = binary_metrics([0, 0], [0, 0])
    assert only_benign.specificity == 1.0
    assert only_benign.recall == 0.0


def test_binary_metrics_reject_length_mismatch_and_invalid_labels():
    with pytest.raises(ValueError, match="equal lengths"):
        binary_metrics([0], [0, 1])
    with pytest.raises(ValueError, match="Unsupported"):
        binary_metrics(["unknown"], ["benign"])


def test_evaluation_report_has_domain_category_slices_and_predictions():
    report = evaluate_predictions(
        [1, 0, 1],
        [1, 1, 0],
        domains=["social", "social", "ethics"],
        categories=["violence", "general", "privacy"],
        conversation_ids=["a", "b", "c"],
    )
    assert report.overall.support == 3
    assert set(report.by_domain) == {"social", "ethics"}
    assert report.by_category["general"].false_positive == 1
    assert report.prediction_rows[2]["conversation_id"] == "c"


def test_bootstrap_interval_is_seeded_and_bounded():
    args = ([1, 1, 0, 0], [1, 0, 1, 0])
    first = evaluate_predictions(*args, bootstrap_repetitions=200, seed=3)
    second = evaluate_predictions(*args, bootstrap_repetitions=200, seed=3)
    assert first.bootstrap_f1_interval == second.bootstrap_f1_interval
    low, high = first.bootstrap_f1_interval
    assert 0 <= low <= high <= 1


def test_evaluation_rejects_bad_slice_lengths_and_confidence():
    with pytest.raises(ValueError, match="domains"):
        evaluate_predictions([1], [1], domains=[])
    with pytest.raises(ValueError, match="confidence"):
        evaluate_predictions([1, 0], [1, 0], bootstrap_repetitions=10, bootstrap_confidence=1)


def test_macro_f1_includes_both_classes():
    assert macro_f1([0, 0, 1], [0, 1, 1]) == pytest.approx((2 / 3 + 2 / 3) / 2)
