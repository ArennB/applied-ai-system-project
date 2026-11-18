"""Tests for the care-guide retrieval in pawpal_rag."""

import pytest

from pawpal_rag import KnowledgeBase, Passage, split_sections, tokenize


@pytest.fixture(scope="module")
def guide() -> KnowledgeBase:
    """The real knowledge/ folder shipped with the app."""
    return KnowledgeBase.from_directory()


def test_split_sections_makes_one_passage_per_heading():
    markdown = "# Title\n\nintro ignored\n\n## First\nalpha beta\n\n## Second\ngamma\n"
    passages = split_sections("x.md", markdown)
    assert [(p.heading, p.text) for p in passages] == [("First", "alpha beta"), ("Second", "gamma")]
    assert all(p.source == "x.md" for p in passages)


def test_split_sections_skips_empty_sections():
    assert split_sections("x.md", "## Empty\n\n## Full\ntext") == [Passage("x.md", "Full", "text")]


def test_tokenize_folds_plurals_and_drops_stopwords():
    assert tokenize("The huskies need walks") == ["husky", "walk"]


def test_readme_is_not_indexed(guide):
    assert all(p.source != "README.md" for p in guide.passages)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("senior cat kidney medication", "senior_pets.md › Senior cat kidney disease"),
        ("husky exercise", "dog_exercise.md › High-energy breeds"),
        ("vet appointment", "vet_visits.md › Planning a vet appointment"),
        ("litter box", "cat_care.md › Litter box"),
        ("rabbit hay", "small_animals.md › Rabbits"),
    ],
)
def test_relevant_passage_is_retrieved(guide, query, expected):
    # The agent sees the top three passages, so that is the bar that matters.
    assert expected in [p.cite() for p in guide.search(query, k=3)]


def test_unrelated_query_returns_nothing(guide):
    assert guide.search("quantum chromodynamics") == []


def test_empty_query_returns_nothing(guide):
    assert guide.search("") == []
    assert guide.search("the and of") == []  # stopwords only


def test_search_respects_k_and_orders_by_score(guide):
    results = guide.search("daily feeding walk play", k=5)
    assert len(results) == 5
    assert [p.score for p in results] == sorted((p.score for p in results), reverse=True)


def test_search_is_deterministic(guide):
    assert guide.search("puppy training") == guide.search("puppy training")


def test_search_does_not_mutate_stored_passages(guide):
    guide.search("cat grooming")
    assert all(p.score == 0.0 for p in guide.passages)
