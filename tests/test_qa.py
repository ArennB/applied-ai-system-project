"""Tests for Ask the care guide (pawpal_qa): retrieve, gate, generate, verify."""

import pytest

from pawpal_qa import NOT_COVERED, CareGuideQA, build_prompt, cited_numbers
from pawpal_rag import KnowledgeBase


@pytest.fixture(scope="module")
def guide() -> KnowledgeBase:
    return KnowledgeBase.from_directory()


class FakeTextLLM:
    """Returns a fixed reply and records every prompt it was sent."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


@pytest.mark.parametrize(
    "question",
    [
        "what is the capital of France?",
        "can my dog drive a car?",  # shares only the generic word "dog"
        "is chocolate bad for dogs?",
        "my cat is sneezing",
        "",
    ],
)
def test_off_topic_questions_are_not_covered_without_calling_the_model(guide, question):
    llm = FakeTextLLM("should never be used [1]")
    result = CareGuideQA(guide, llm).ask(question)
    assert result.status == "not_covered"
    assert llm.prompts == []


@pytest.mark.parametrize(
    "question, expected_source",
    [
        ("How often should I scoop the litter box?", "cat_care.md › Litter box"),
        ("How often should I bathe my dog?", "grooming.md › Bathing dogs"),
        ("How do I train my dog to sit?", "enrichment.md › Training sessions"),
        ("How long should a puppy walk?", "puppy_kitten.md › Puppy exercise"),
        ("What do rabbits eat?", "small_animals.md › Rabbits"),
        ("How much exercise does a husky need?", "dog_exercise.md › High-energy breeds"),
    ],
)
def test_on_topic_questions_retrieve_the_right_passage(guide, question, expected_source):
    result = CareGuideQA(guide).ask(question)
    assert result.status == "retrieval_only"
    assert expected_source in [p.cite() for p in result.passages]


def test_answer_keeps_only_the_passages_it_cites(guide):
    llm = FakeTextLLM("Scoop at least once a day [1]. Wash the box weekly [1].")
    result = CareGuideQA(guide, llm).ask("How often should I scoop the litter box?")
    assert result.status == "answered"
    assert [p.cite() for p in result.citations] == ["cat_care.md › Litter box"]
    assert result.answer.startswith("Scoop at least once a day")


def test_prompt_contains_numbered_passages_and_question(guide):
    llm = FakeTextLLM("ok [1]")
    CareGuideQA(guide, llm).ask("How often should I bathe my dog?")
    prompt = llm.prompts[0]
    assert "[1] (grooming.md › Bathing dogs)" in prompt
    assert "every 4 to 8 weeks" in prompt
    assert prompt.endswith("Question: How often should I bathe my dog?")


def test_model_saying_not_covered_is_respected(guide):
    result = CareGuideQA(guide, FakeTextLLM(NOT_COVERED)).ask("Can my dog eat grapes?")
    assert result.status == "not_covered"
    assert result.passages  # retrieval found feeding passages; the model judged them


def test_answer_without_citations_is_flagged_ungrounded(guide):
    result = CareGuideQA(guide, FakeTextLLM("Bathe your dog every week.")).ask(
        "How often should I bathe my dog?"
    )
    assert result.status == "ungrounded"
    assert result.citations == []


def test_citing_a_passage_that_was_not_given_does_not_count(guide):
    result = CareGuideQA(guide, FakeTextLLM("Daily [9].")).ask("How often should I bathe my dog?")
    assert result.status == "ungrounded"


def test_cited_numbers_are_distinct_in_order_and_in_range():
    assert cited_numbers("a [2] b [1][2] c [7] d [0]", 3) == [2, 1]


def test_build_prompt_numbers_from_one(guide):
    passages = guide.search("litter box", k=2)
    assert build_prompt("q", passages).startswith("Care guide passages:\n\n[1] (")
