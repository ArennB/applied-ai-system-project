"""PawPal+ guardrails: checks on what goes into the AI and what comes out.

The tools already validate every value the planning agent can set (see
PlannerTools in pawpal_agent). This module covers the rest:

    input   check_input()        clean the text, cap its length, refuse prompt
                                 injection, and flag possible emergencies
    output  clean_model_text()   strip control characters, cap the length
            unsupported_numbers() numbers in an answer that the guide never says
            add_vet_note()       a vet reminder on any health answer, enforced
                                 here rather than trusted to the prompt
    usage   RateLimiter          a cap on AI calls per session

Everything here is plain Python with no model involved, so it is deterministic
and cannot be talked out of its rules. Every guardrail that fires is logged to
pawpal.guardrails.
"""

import re
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Callable

from pawpal_logging import get_logger, preview

log = get_logger("guardrails")

# Length caps. A day plan needs a paragraph, a question a sentence, a quick-add
# line a few words; anything far longer is a paste accident or an attack.
MAX_REQUEST_CHARS = 2000
MAX_QUESTION_CHARS = 300
MAX_QUICK_ADD_CHARS = 120
MAX_EXPLANATION_CHARS = 1500
MAX_ANSWER_CHARS = 1200

# Text that tries to talk to the model rather than describe pets. Phrased
# narrowly so normal requests ("ignore the cat today, she's at the vet") pass.
_INJECTION = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,30}\b(?:previous|prior|above|earlier|system|your|all)\b[^.\n]{0,20}\b(?:instructions?|rules|prompts?|directions|guidelines)\b",
        r"\b(?:reveal|show|print|repeat|output|leak)\b[^.\n]{0,20}\b(?:system\s+prompt|your\s+(?:instructions|prompt|rules))\b",
        r"\bsystem\s+prompt\b",
        r"\byou\s+are\s+now\s+(?:a|an|in)\b",
        r"\b(?:developer|god|jailbreak)\s+mode\b",
        r"</?\s*(?:system|instructions?|assistant)\s*>",
        r"\bact\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:a|an)\s+(?:different|new|unrestricted)\b",
    )
]

# Signs the owner may be describing an emergency. These do not block anything:
# the app shows a "call your vet now" banner above whatever PawPal says, because
# a planner or a short care guide is the wrong tool in that moment.
_URGENT = re.compile(
    r"\b(?:poison(?:ed|ing)?|toxic|toxin|antifreeze|xylitol|rat\s+bait|chocolate|grapes?|raisins?"
    r"|lilies|not\s+breathing|(?:can'?t|cannot|trouble|struggling\s+to)\s+breath(?:e|ing)?"
    r"|seizures?|seizing|convuls\w*|collaps\w*|unconscious|unresponsive"
    r"|bleeding\s+(?:heavily|badly|a\s+lot)|hit\s+by\s+a\s+car|bloat(?:ed)?"
    r"|swallowed|choking|heat\s?stroke|(?:can'?t|cannot|straining\s+to)\s+(?:pee|urinate))\b",
    re.I,
)

URGENT_NOTICE = (
    "This sounds like it could be an emergency. If your pet may have eaten something"
    " toxic, is struggling to breathe, has collapsed or is badly hurt, call your vet or"
    " an emergency animal clinic now. Don't wait on an app."
)

VET_NOTE = "Follow your vet's advice for your own pet."
_HEALTH = re.compile(
    r"\b(?:medic\w*|meds|pills?|tablets?|dose|dosage|insulin|ill|illness|sick|disease|injur\w*"
    r"|pain|vomit\w*|diarrh\w*|kidney|diabet\w*|arthritis|infection|symptoms?|vaccin\w*)\b",
    re.I,
)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩]")


@dataclass
class InputCheck:
    """The verdict on one piece of user text.

    text is the cleaned text to use from here on. When allowed is False,
    reason is a message for the user. urgent is independent of allowed.
    """

    text: str
    allowed: bool
    reason: str = ""
    urgent: bool = False


class GuardrailError(ValueError):
    """User input was refused. str(error) is safe to show the user."""


def clean_text(text: str) -> str:
    """Normalise Unicode, drop control and invisible direction characters,
    and squeeze runs of blank lines and spaces."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _CONTROL.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def looks_like_injection(text: str) -> bool:
    """True if the text tries to instruct the model rather than describe pets."""
    return any(pattern.search(text) for pattern in _INJECTION)


def is_urgent(text: str) -> bool:
    """True if the text mentions signs of a possible pet emergency."""
    return bool(_URGENT.search(text))


def check_input(text: str, max_chars: int, kind: str = "request", screen_injection: bool = True) -> InputCheck:
    """Clean user text and decide whether it may go on to the AI.

    Args:
        text: what the user typed.
        max_chars: the longest text accepted, after cleaning.
        kind: what it is, for messages and logs ("request", "question", ...).
        screen_injection: refuse text that tries to instruct the model. Off for
            input that never reaches a language model, such as quick add.
    """
    cleaned = clean_text(text)
    urgent = is_urgent(cleaned)
    if urgent:
        log.warning("urgent %s flagged: %r", kind, preview(cleaned))
    if not cleaned:
        return InputCheck(cleaned, False, f"Type a {kind} first.", urgent)
    if len(cleaned) > max_chars:
        log.warning("%s refused: %d chars (limit %d)", kind, len(cleaned), max_chars)
        return InputCheck(
            cleaned,
            False,
            f"That {kind} is {len(cleaned)} characters; please keep it under {max_chars}.",
            urgent,
        )
    if screen_injection and looks_like_injection(cleaned):
        log.warning("%s refused as prompt injection: %r", kind, preview(cleaned))
        return InputCheck(
            cleaned,
            False,
            "That reads like instructions for the AI rather than something about your"
            " pets, so PawPal won't send it. Describe your pets and your day instead.",
            urgent,
        )
    return InputCheck(cleaned, True, urgent=urgent)


def require_input(text: str, max_chars: int, kind: str = "request") -> str:
    """check_input() for library code: the cleaned text, or GuardrailError."""
    check = check_input(text, max_chars, kind)
    if not check.allowed:
        raise GuardrailError(check.reason)
    return check.text


# --- Output -----------------------------------------------------------------------


def clean_model_text(text: str, max_chars: int) -> str:
    """Model output made safe to display: cleaned and cut to max_chars.

    A cut ends at the last full sentence that fits, so the owner never sees a
    sentence (or a dose) chopped in half.
    """
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars]
    end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("! "), cut.rfind("? "))
    shortened = cut[: end + 1] if end > max_chars // 2 else cut.rstrip() + "…"
    log.warning("model text cut from %d to %d chars", len(cleaned), len(shortened))
    return shortened


_NUMBER = re.compile(r"(?<![\[\d.])\d+(?:\.\d+)?(?![\]\d])")


def numbers_in(text: str) -> set[str]:
    """The numbers written in the text, ignoring [n] citation markers.

    Normalised so "8", "08" and "8.0" count as the same number.
    """
    return {format(float(match.group()), "g") for match in _NUMBER.finditer(text)}


def unsupported_numbers(answer: str, sources: list[str]) -> list[str]:
    """Numbers in the answer that appear in none of the source texts.

    A grounded answer can rephrase the guide but should not invent a figure:
    a duration, a frequency or, worst of all, a dose. Returned sorted so the
    result is stable.
    """
    known: set[str] = set()
    for source in sources:
        known |= numbers_in(source)
    return sorted(numbers_in(answer) - known, key=float)


def mentions_health(text: str) -> bool:
    """True if the text is about medication, illness or injury."""
    return bool(_HEALTH.search(text))


def add_vet_note(question: str, answer: str) -> str:
    """The answer, with a vet reminder added if it touches on health and has none."""
    if (mentions_health(question) or mentions_health(answer)) and not re.search(r"\bvets?\b", answer, re.I):
        log.info("vet note added to answer")
        return f"{answer.rstrip()} {VET_NOTE}"
    return answer


# --- Usage ---------------------------------------------------------------------------


class RateLimiter:
    """At most max_calls in any rolling window of per_seconds.

    One lives in each Streamlit session, which keeps a stuck key, a held-down
    button or a script from running up the Gemini bill.
    """

    def __init__(self, max_calls: int = 10, per_seconds: float = 300, clock: Callable[[], float] = time.monotonic):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self._clock = clock
        self._calls: deque[float] = deque()

    def _expire(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= self.per_seconds:
            self._calls.popleft()

    def retry_after(self) -> float:
        """Seconds until another call is allowed; 0 if one is allowed now."""
        now = self._clock()
        self._expire(now)
        if len(self._calls) < self.max_calls:
            return 0.0
        return self.per_seconds - (now - self._calls[0])

    def try_acquire(self, what: str = "AI call") -> bool:
        """Record a call and return True, or return False if over the limit."""
        if self.retry_after() > 0:
            log.warning("rate limit: %s refused (%d per %ds)", what, self.max_calls, self.per_seconds)
            return False
        self._calls.append(self._clock())
        return True
