"""PawPal+ task model: a small classifier trained to label pet care tasks.

Given a task as someone would type it ("give Luna her kidney pill") and the
pet's species, the model predicts three things:

    priority   - low / medium / high
    minutes    - a duration bucket: 5, 10, 15, 20, 30, 45, 60 or 90
    frequency  - once / daily / weekly

It is trained on data/task_dataset.csv, a hand-labelled set of pet care tasks
whose labels follow the care guide in knowledge/. Each target gets its own
scikit-learn pipeline: word and character TF-IDF features into a logistic
regression. Character n-grams let it cope with typos and word forms the
dataset never saw ("walkies", "meds", "vaccs").

The app uses it in two places:
    * Quick add - fills in priority, duration and frequency from one line.
    * A second opinion on the Gemini agent - flags tasks where the trained
      model confidently disagrees with the priority the agent chose.

Training takes well under a second, so the app trains at startup instead of
loading a pickled model; that sidesteps scikit-learn version mismatches.

Run `python -m pawpal_model` to evaluate it and write data/model_report.md.
"""

import csv
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from pawpal_logging import get_logger
from pawpal_system import Priority, parse_time

log = get_logger("model")

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATASET = DATA_DIR / "task_dataset.csv"
DEFAULT_REPORT = DATA_DIR / "model_report.md"

TARGETS = ("priority", "minutes", "frequency")
MINUTE_BUCKETS = (5, 10, 15, 20, 30, 45, 60, 90)
RANDOM_STATE = 42
# Matches pawpal_agent.SECOND_OPINION_CONFIDENCE (not imported, to keep this
# module free of the agent); evaluate() reports accuracy on each side of it.
CONFIDENCE_BAR = 0.7


# --- Data ------------------------------------------------------------------------


def model_input(task: str, species: str) -> str:
    """The text the model sees: species first, so "cat" and "dog" are features."""
    return f"{species.strip().lower()} {task.strip().lower()}"


def load_examples(path: str | Path = DEFAULT_DATASET) -> tuple[list[str], dict[str, list[str]]]:
    """Read the dataset into model inputs and one label list per target."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    inputs = [model_input(row["task"], row["species"]) for row in rows]
    labels = {target: [row[target].strip() for row in rows] for target in TARGETS}
    return inputs, labels


def build_pipeline() -> Pipeline:
    """Word 1-2 grams plus character 3-5 grams, into a logistic regression.

    class_weight="balanced" stops the model from simply favouring the most
    common label ("high", "5", "daily"), which the dataset leans towards.
    """
    features = FeatureUnion(
        [
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ]
    )
    classifier = LogisticRegression(
        C=10.0, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
    )
    return Pipeline([("features", features), ("classifier", classifier)])


# --- The model ---------------------------------------------------------------------


@dataclass
class TaskPrediction:
    """The model's labels for one task, each with how sure it is (0 to 1)."""

    priority: Priority
    minutes: int
    frequency: str
    confidence: dict[str, float]


class TaskModel:
    """Three trained pipelines, one per target, behind a single predict()."""

    def __init__(self, pipelines: dict[str, Pipeline]):
        self.pipelines = pipelines

    @classmethod
    def train(cls, path: str | Path = DEFAULT_DATASET) -> "TaskModel":
        """Fit one pipeline per target on the whole dataset."""
        started = time.perf_counter()
        inputs, labels = load_examples(path)
        pipelines = {}
        for target in TARGETS:
            pipeline = build_pipeline()
            pipeline.fit(inputs, labels[target])
            pipelines[target] = pipeline
        log.info("task model trained on %d examples in %.2fs", len(inputs), time.perf_counter() - started)
        return cls(pipelines)

    def predict(self, task: str, species: str) -> TaskPrediction:
        """Label one task. Deterministic for a given dataset."""
        text = [model_input(task, species)]
        labels: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for target, pipeline in self.pipelines.items():
            probabilities = pipeline.predict_proba(text)[0]
            best = probabilities.argmax()
            labels[target] = pipeline.classes_[best]
            confidence[target] = round(float(probabilities[best]), 3)
        log.debug("predict %r (%s): %s %s", task, species, labels, confidence)
        return TaskPrediction(
            priority=Priority[labels["priority"].upper()],
            minutes=int(labels["minutes"]),
            frequency=labels["frequency"],
            confidence=confidence,
        )


# --- Quick add parsing -------------------------------------------------------------

_HOURS = re.compile(r"\b(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", re.I)
_AN_HOUR = re.compile(r"\b(?:for\s+)?(half\s+an?\s+hour|half\s+hour|an\s+hour)\b", re.I)
_MINUTES = re.compile(r"\b(?:for\s+)?(\d+)\s*(?:m|min|mins|minute|minutes)\b", re.I)
_AT_TIME = re.compile(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)


@dataclass
class QuickAdd:
    """What could be read straight from a quick-add line, before the model."""

    title: str
    minutes: int | None  # an explicit duration, if the text gave one
    fixed_start: int | None  # minutes since midnight, if the text gave a time


def parse_quick_add(text: str) -> QuickAdd:
    """Pull an explicit duration and start time out of a one-line task.

    "walk Mochi for 30 min at 7pm" -> title "Walk Mochi", 30 minutes, 19:00.
    Anything the text does not state is left as None for the model to fill.
    Durations outside 1-240 minutes and impossible times are ignored.
    """
    remaining = text.strip()
    minutes: int | None = None

    for pattern, to_minutes in (
        (_HOURS, lambda m: round(float(m.group(1)) * 60)),
        (_AN_HOUR, lambda m: 30 if m.group(1).lower().startswith("half") else 60),
        (_MINUTES, lambda m: int(m.group(1))),
    ):
        match = pattern.search(remaining)
        if match and minutes is None:
            value = to_minutes(match)
            if 1 <= value <= 240:
                minutes = value
            remaining = remaining[: match.start()] + remaining[match.end() :]

    fixed_start: int | None = None
    match = _AT_TIME.search(remaining)
    if match:
        hours, mins, meridiem = int(match.group(1)), match.group(2) or "00", match.group(3)
        if meridiem:
            if not 1 <= hours <= 12:
                hours = -1  # "at 15pm" is not a time
            else:
                hours = hours % 12 + (12 if meridiem.lower() == "pm" else 0)
        fixed_start = parse_time(f"{hours}:{mins}") if hours >= 0 else None
        if fixed_start is not None or meridiem:
            remaining = remaining[: match.start()] + remaining[match.end() :]

    title = re.sub(r"\s+", " ", remaining).strip(" ,.-")
    # "half an hour of play" leaves "of play"; drop the dangling connector.
    title = re.sub(r"^(?:of|for|with)\s+", "", title, flags=re.I)
    title = title[:1].upper() + title[1:] if title else text.strip()
    return QuickAdd(title, minutes, fixed_start)


# --- Evaluation ----------------------------------------------------------------------


def evaluate(path: str | Path = DEFAULT_DATASET) -> dict:
    """How well the model generalises to tasks it was not trained on.

    Two measurements per target:
      * hold-out: train on 80%, score on the other 20% (split stratified by
        priority), against a majority-class baseline trained the same way;
      * 5-fold cross-validation accuracy (mean and spread), with the same
        baseline, which uses every row as test data exactly once.
    For minutes it also reports mean absolute error and how often the
    prediction is within one bucket of the truth, since 15 vs 20 minutes is a
    much smaller miss than 5 vs 60.
    """
    inputs, labels = load_examples(path)
    indices = list(range(len(inputs)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=labels["priority"]
    )
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results: dict = {"rows": len(inputs), "train_rows": len(train_idx), "test_rows": len(test_idx)}
    for target in TARGETS:
        y = labels[target]
        x_train, y_train = [inputs[i] for i in train_idx], [y[i] for i in train_idx]
        x_test, y_test = [inputs[i] for i in test_idx], [y[i] for i in test_idx]

        model = build_pipeline().fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        predicted = list(model.classes_[probabilities.argmax(axis=1)])
        confidence = probabilities.max(axis=1)
        correct = [p == t for p, t in zip(predicted, y_test)]
        baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)

        # Does the model's confidence mean anything? Split the hold-out
        # predictions at the bar the app uses before a second opinion is shown.
        sure = [ok for ok, c in zip(correct, confidence) if c >= CONFIDENCE_BAR]
        unsure = [ok for ok, c in zip(correct, confidence) if c < CONFIDENCE_BAR]
        entry = {
            "mean_confidence": float(confidence.mean()),
            "sure": (sum(sure), len(sure)),
            "unsure": (sum(unsure), len(unsure)),
            "holdout_accuracy": accuracy_score(y_test, predicted),
            "holdout_baseline": accuracy_score(y_test, baseline.predict(x_test)),
            "majority_label": Counter(y_train).most_common(1)[0][0],
            "report": classification_report(y_test, predicted, zero_division=0),
        }
        cv = cross_val_score(build_pipeline(), inputs, y, cv=folds)
        cv_baseline = cross_val_score(DummyClassifier(strategy="most_frequent"), inputs, y, cv=folds)
        entry.update(
            cv_mean=float(cv.mean()), cv_std=float(cv.std()), cv_baseline=float(cv_baseline.mean())
        )
        if target == "minutes":
            truth = [int(v) for v in y_test]
            guess = [int(v) for v in predicted]
            position = {bucket: i for i, bucket in enumerate(MINUTE_BUCKETS)}
            entry["mae_minutes"] = sum(abs(t - g) for t, g in zip(truth, guess)) / len(truth)
            entry["within_one_bucket"] = sum(
                abs(position[t] - position[g]) <= 1 for t, g in zip(truth, guess)
            ) / len(truth)
        results[target] = entry
    return results


def format_report(results: dict) -> str:
    """The evaluation as markdown, for data/model_report.md and the terminal."""
    lines = [
        "# PawPal+ task model: evaluation",
        "",
        f"Dataset: `data/task_dataset.csv`, {results['rows']} labelled tasks."
        f" Hold-out split: {results['train_rows']} train / {results['test_rows']} test"
        " (stratified by priority). Cross-validation: 5 folds over all rows.",
        "",
        "| Target | Hold-out accuracy | Majority baseline | 5-fold CV accuracy | CV baseline |",
        "|--------|-------------------|-------------------|--------------------|-------------|",
    ]
    for target in TARGETS:
        r = results[target]
        lines.append(
            f"| {target} | {r['holdout_accuracy']:.0%} | {r['holdout_baseline']:.0%}"
            f" (always \"{r['majority_label']}\") | {r['cv_mean']:.0%} ± {r['cv_std']:.0%}"
            f" | {r['cv_baseline']:.0%} |"
        )
    minutes = results["minutes"]
    lines += [
        "",
        f"Duration on the hold-out set: mean absolute error {minutes['mae_minutes']:.1f} min;"
        f" {minutes['within_one_bucket']:.0%} of predictions within one bucket"
        f" ({', '.join(map(str, MINUTE_BUCKETS))} min).",
        "",
        "## Confidence vs accuracy (hold-out set)",
        "",
        f"Each prediction comes with a confidence (the probability of the chosen label)."
        f" If confidence is meaningful, predictions at or above {CONFIDENCE_BAR:.0%}"
        " (the bar the app uses before the model second-guesses the agent) should be"
        " right more often than the rest.",
        "",
        f"| Target | Mean confidence | Accuracy when ≥ {CONFIDENCE_BAR:.0%} sure"
        f" | Accuracy when < {CONFIDENCE_BAR:.0%} sure |",
        "|--------|-----------------|------------------|------------------|",
    ]
    for target in TARGETS:
        r = results[target]
        (sure_ok, sure_n), (unsure_ok, unsure_n) = r["sure"], r["unsure"]

        def rate(ok: int, n: int) -> str:
            return f"{ok / n:.0%} ({ok}/{n})" if n else "n/a (0)"

        lines.append(
            f"| {target} | {r['mean_confidence']:.2f} | {rate(sure_ok, sure_n)}"
            f" | {rate(unsure_ok, unsure_n)} |"
        )
    lines += [
        "",
        "## Per-class results (hold-out set)",
    ]
    for target in TARGETS:
        lines += ["", f"### {target}", "", "```", results[target]["report"].rstrip(), "```"]
    lines += [
        "",
        "## Caveats",
        "",
        "- The dataset is small and its labels were written with AI assistance,"
        " following the care guide in `knowledge/`. The model learns whatever the"
        " labels say, so review them before trusting it on new kinds of task.",
        "- With ~44 test rows, one more or fewer correct prediction moves hold-out"
        " accuracy by about 2 points; the cross-validation figure is the steadier one.",
        "",
        "Regenerate with `python -m pawpal_model`.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    report = format_report(evaluate())
    DEFAULT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {DEFAULT_REPORT.relative_to(Path.cwd()) if DEFAULT_REPORT.is_relative_to(Path.cwd()) else DEFAULT_REPORT}")
