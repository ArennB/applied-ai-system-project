"""Tests for the trained task model (pawpal_model)."""

import pytest

from pawpal_model import (
    MINUTE_BUCKETS,
    TARGETS,
    TaskModel,
    evaluate,
    load_examples,
    parse_quick_add,
)
from pawpal_system import Priority


@pytest.fixture(scope="module")
def model() -> TaskModel:
    return TaskModel.train()


def test_dataset_is_well_formed():
    inputs, labels = load_examples()
    assert len(inputs) >= 200
    assert set(labels["priority"]) == {"low", "medium", "high"}
    assert set(labels["frequency"]) == {"once", "daily", "weekly"}
    assert {int(m) for m in labels["minutes"]} <= set(MINUTE_BUCKETS)
    assert len(set(inputs)) == len(inputs), "duplicate rows would leak between train and test"


@pytest.mark.parametrize(
    "task, species, priority, frequency",
    [
        ("give Luna her kidney pill", "cat", Priority.HIGH, "daily"),
        ("insulin injection", "dog", Priority.HIGH, "daily"),
        ("vet appointment", "dog", Priority.HIGH, "once"),
        ("feed the fish", "other", Priority.HIGH, "daily"),
        ("scoop litter", "cat", Priority.HIGH, "daily"),
    ],
)
def test_clear_cut_tasks(model, task, species, priority, frequency):
    prediction = model.predict(task, species)
    assert (prediction.priority, prediction.frequency) == (priority, frequency)


def test_grooming_is_not_high_priority(model):
    assert model.predict("brush the dog", "dog").priority is not Priority.HIGH


def test_medication_is_short(model):
    assert model.predict("give thyroid tablet", "cat").minutes <= 10


def test_prediction_shape_and_determinism(model):
    first = model.predict("evening walk", "dog")
    assert first == model.predict("evening walk", "dog")
    assert first.minutes in MINUTE_BUCKETS
    assert set(first.confidence) == set(TARGETS)
    assert all(0 < value <= 1 for value in first.confidence.values())


def test_model_beats_majority_baseline_on_every_target():
    results = evaluate()
    for target in TARGETS:
        assert results[target]["cv_mean"] > results[target]["cv_baseline"], target
        assert results[target]["holdout_accuracy"] > results[target]["holdout_baseline"], target
    assert results["minutes"]["within_one_bucket"] >= 0.6


def test_confidence_tracks_accuracy():
    """Confident predictions must be right more often, or the 70% bar means nothing."""
    from pawpal_agent import SECOND_OPINION_CONFIDENCE
    from pawpal_model import CONFIDENCE_BAR

    assert CONFIDENCE_BAR == SECOND_OPINION_CONFIDENCE
    results = evaluate()
    for target in TARGETS:
        (sure_ok, sure_n), (unsure_ok, unsure_n) = results[target]["sure"], results[target]["unsure"]
        assert sure_n and unsure_n, target
        assert sure_ok / sure_n > unsure_ok / unsure_n, target


@pytest.mark.parametrize(
    "text, title, minutes, start",
    [
        ("walk Mochi for 30 min at 7pm", "Walk Mochi", 30, 19 * 60),
        ("vet at 10:00", "Vet", None, 600),
        ("half an hour of play", "Play", 30, None),
        ("long hike 1.5 hours", "Long hike", 90, None),
        ("brush for an hour", "Brush", 60, None),
        ("kidney meds at 8am", "Kidney meds", None, 480),
        ("feed fish", "Feed fish", None, None),
        ("nap 500 minutes", "Nap", None, None),  # out of range: left to the model
        ("meds at 25:00", "Meds at 25:00", None, None),  # not a time: kept in the title
    ],
)
def test_parse_quick_add(text, title, minutes, start):
    parsed = parse_quick_add(text)
    assert (parsed.title, parsed.minutes, parsed.fixed_start) == (title, minutes, start)
