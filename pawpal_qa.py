"""PawPal+ "Ask the care guide": retrieval-augmented question answering.

Unlike the planning agent, which decides for itself when to search, this
pipeline always retrieves first and then constrains the model to what it found:

    1. retrieve  - KnowledgeBase.search() for the best-matching passages
    2. gate      - if nothing matches well enough, answer "not covered" without
                   calling the model at all
    3. generate  - give the model only those passages, numbered, and require a
                   [n] citation for every claim
    4. verify    - parse the citations; an answer that cites nothing, or that
                   states a number none of its passages contain, is marked
                   ungrounded rather than presented as fact

Guardrails (pawpal_guardrails) wrap both ends: the question is cleaned,
length-capped and screened for prompt injection before retrieval, and the
answer is cleaned, length-capped and given a vet reminder if it is about health.

Without an API key, steps 3 and 4 are skipped and the passages themselves are
returned, so the feature still does something useful offline.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from pawpal_guardrails import (
    MAX_ANSWER_CHARS,
    MAX_QUESTION_CHARS,
    add_vet_note,
    check_input,
    clean_model_text,
    unsupported_numbers,
)
from pawpal_logging import get_logger, preview
from pawpal_rag import KnowledgeBase, Passage, tokenize

log = get_logger("qa")

# Below this TF-IDF score a passage is too weak a match to put in front of the
# model at all.
MIN_SCORE = 0.1
TOP_K = 4
NOT_COVERED = "NOT_COVERED"

# Words that appear all over the guide and say nothing about what is being
# asked. Score alone cannot gate on topic: "can my dog drive a car?" scores
# well on the word "dog" by itself. So a passage must also share at least one
# word outside this set with the question. Kept in tokenize()'s stemmed form.
GENERIC_TERMS = frozenset(
    tokenize(
        """
        dog dogs cat cats pet pets animal animals puppy kitten
        often much many long best good bad need needs get make
        like know help day days time
        """
    )
)

SYSTEM_PROMPT = f"""\
You answer pet care questions for PawPal+ using ONLY the numbered passages from
its care guide that are given with each question.

Rules:
- Use only facts stated in the passages. Do not add outside knowledge.
- After each sentence that uses a passage, cite it like [1] or [2][3].
- Keep it short: two to four sentences, plain language.
- If the passages do not answer the question, reply with exactly {NOT_COVERED}
  and nothing else.
- For anything involving illness, injury or medication doses, add that the
  owner should follow their vet's advice.
"""


class TextLLM(Protocol):
    """Anything that can turn a system prompt and a user prompt into text."""

    def generate(self, system: str, prompt: str) -> str: ...


@dataclass
class QAResult:
    """The outcome of one question.

    status is one of:
        "answered"       - a grounded answer citing at least one passage
        "not_covered"    - the guide has nothing on this (retrieval or model said so)
        "ungrounded"     - the model answered but cited nothing; shown with a warning
        "retrieval_only" - no model available; passages are the answer
        "blocked"        - refused by an input guardrail; answer says why

    warnings explains why an answer was marked ungrounded, if it was.
    """

    question: str
    status: str
    answer: str = ""
    citations: list[Passage] = field(default_factory=list)  # passages the answer cites
    passages: list[Passage] = field(default_factory=list)  # everything retrieved
    warnings: list[str] = field(default_factory=list)


class CareGuideQA:
    """Retrieve, gate, generate and verify, over the PawPal+ care guide."""

    def __init__(self, knowledge: KnowledgeBase, llm: TextLLM | None = None, min_score: float = MIN_SCORE):
        self.knowledge = knowledge
        self.llm = llm
        self.min_score = min_score

    def ask(self, question: str) -> QAResult:
        """Answer from the guide, or say plainly that the guide does not cover it.

        LLMError from the model propagates to the caller.
        """
        check = check_input(question, MAX_QUESTION_CHARS, "question")
        question = check.text
        if not question:
            return QAResult(question, "not_covered")
        if not check.allowed:
            return QAResult(question, "blocked", answer=check.reason)

        topic = set(tokenize(question)) - GENERIC_TERMS
        retrieved = self.knowledge.search(question, k=TOP_K)
        passages = [
            passage
            for passage in retrieved
            if passage.score >= self.min_score
            and topic & set(tokenize(f"{passage.heading} {passage.text}"))
        ]
        log.info(
            "question %r: retrieved %s, kept %d",
            preview(question), [f"{p.cite()}={p.score}" for p in retrieved], len(passages),
        )
        if not passages:
            log.info("question %r: not covered (gate)", preview(question))
            return QAResult(question, "not_covered")
        if self.llm is None:
            return QAResult(question, "retrieval_only", passages=passages)

        started = time.perf_counter()
        try:
            reply = self.llm.generate(SYSTEM_PROMPT, build_prompt(question, passages)).strip()
        except Exception as exc:
            log.error("question %r: model call failed: %s", preview(question), exc)
            raise
        log.info("question %r: model replied in %.2fs", preview(question), time.perf_counter() - started)
        if not reply or NOT_COVERED in reply:
            log.info("question %r: not covered (model)", preview(question))
            return QAResult(question, "not_covered", passages=passages)

        reply = add_vet_note(question, clean_model_text(reply, MAX_ANSWER_CHARS))
        cited = cited_numbers(reply, len(passages))
        if not cited:
            log.warning("question %r: answer cites nothing", preview(question))
            return QAResult(
                question, "ungrounded", answer=reply, passages=passages,
                warnings=["This answer doesn't cite the care guide, so it may not come from it."],
            )
        citations = [passages[n - 1] for n in cited]
        invented = unsupported_numbers(reply, [p.text for p in citations])
        if invented:
            log.warning("question %r: numbers not in cited passages: %s", preview(question), invented)
            return QAResult(
                question, "ungrounded", answer=reply, citations=citations, passages=passages,
                warnings=[
                    f"This answer mentions {', '.join(invented)}, which the passages it cites"
                    " don't say. Check it against them before relying on it."
                ],
            )
        log.info("question %r: answered citing %s", preview(question), [p.cite() for p in citations])
        return QAResult(question, "answered", answer=reply, citations=citations, passages=passages)


def build_prompt(question: str, passages: list[Passage]) -> str:
    """The user prompt: numbered passages, then the question."""
    numbered = "\n\n".join(
        f"[{n}] ({passage.cite()})\n{passage.text}" for n, passage in enumerate(passages, start=1)
    )
    return f"Care guide passages:\n\n{numbered}\n\nQuestion: {question}"


def cited_numbers(answer: str, passage_count: int) -> list[int]:
    """The distinct passage numbers cited as [n], in order of first use.

    Numbers outside 1..passage_count are ignored, so a model citing a passage
    it was never given does not count as grounded.
    """
    seen: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", answer):
        number = int(match.group(1))
        if 1 <= number <= passage_count and number not in seen:
            seen.append(number)
    return seen
