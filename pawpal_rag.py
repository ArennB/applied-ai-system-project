"""PawPal+ retrieval layer: a small TF-IDF search over the care guide.

The planning agent calls KnowledgeBase.search() before it invents durations or
priorities, so its tasks are grounded in the curated guide in knowledge/ rather
than in whatever the model happens to remember.

Each markdown file is split on its "## " headings, and each section becomes one
Passage. Ranking is plain TF-IDF with cosine similarity, written out by hand:
the guide is a few dozen passages, so there is no need for a vector database or
an embedding model, and the results are deterministic, which keeps the tests
exact.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# Words too common to say anything about which passage is relevant. Kept short
# on purpose: domain words such as "daily" or "minutes" stay in, because IDF
# already discounts them when they appear everywhere.
STOPWORDS = frozenset(
    """
    a about after an and are as at be before but by can do does for from has have
    how i if in into is it its my need needs of on or our should so than that the
    their them then there these they this to up us was we what when which who will
    with would you your
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers in the text, minus stopwords.

    Word forms are folded together with a few suffix rules: plurals ("walks" ->
    "walk", "huskies" -> "husky"), "-ing" ("training" -> "train", "bathing" ->
    "bath") and a final "e" ("bathe" -> "bath"). It is a deliberately crude
    stemmer, but the same rules run on the guide and the query, so they always
    agree, and predictability matters more here than linguistic accuracy.
    """
    tokens = []
    for token in _TOKEN.findall(text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("e"):
            token = token[:-1]
        tokens.append(token)
    return tokens


@dataclass
class Passage:
    """One section of the care guide, and how well it matched a query."""

    source: str  # filename, e.g. "senior_pets.md"
    heading: str  # the "## " heading, e.g. "Senior cat kidney disease"
    text: str
    score: float = 0.0

    def cite(self) -> str:
        """A short label for display, e.g. "senior_pets.md › Senior exercise"."""
        return f"{self.source} › {self.heading}"


def split_sections(source: str, markdown: str) -> list[Passage]:
    """Split one markdown file into a Passage per "## " heading.

    Text before the first "## " heading (the "# " title) is not a passage on its
    own, but the title is prefixed onto every section's indexed text by the
    caller so that, say, "Cat daily care" helps a query about cats match the
    "Feeding cats" section.
    """
    passages: list[Passage] = []
    heading: str | None = None
    lines: list[str] = []

    def flush() -> None:
        body = " ".join(line.strip() for line in lines if line.strip())
        if heading is not None and body:
            passages.append(Passage(source, heading, body))

    for line in markdown.splitlines():
        if line.startswith("## "):
            flush()
            heading, lines = line[3:].strip(), []
        elif not line.startswith("# "):
            lines.append(line)
    flush()
    return passages


def _title(markdown: str) -> str:
    """The file's "# " title, or "" if it has none."""
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


class KnowledgeBase:
    """A searchable set of care-guide passages.

    Build one with from_directory() for the real guide, or pass passages
    directly in tests.
    """

    def __init__(self, passages: list[Passage], titles: dict[str, str] | None = None):
        self.passages = passages
        titles = titles or {}
        # Index each passage on its file title, heading and body together.
        documents = [
            tokenize(f"{titles.get(p.source, '')} {p.heading} {p.text}") for p in passages
        ]
        document_frequency = Counter(term for doc in documents for term in set(doc))
        count = len(documents)
        # Smoothed IDF: a term in every passage still gets a small positive weight.
        self._idf = {
            term: math.log((1 + count) / (1 + freq)) + 1
            for term, freq in document_frequency.items()
        }
        self._vectors = [self._vectorize(doc) for doc in documents]

    @classmethod
    def from_directory(cls, path: str | Path = DEFAULT_KNOWLEDGE_DIR) -> "KnowledgeBase":
        """Load every .md file in the folder (except README.md) as passages."""
        passages: list[Passage] = []
        titles: dict[str, str] = {}
        for file in sorted(Path(path).glob("*.md")):
            if file.name.lower() == "readme.md":
                continue
            markdown = file.read_text(encoding="utf-8")
            titles[file.name] = _title(markdown)
            passages.extend(split_sections(file.name, markdown))
        return cls(passages, titles)

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        """A unit-length TF-IDF vector for the tokens, as a sparse dict."""
        counts = Counter(tokens)
        vector = {
            term: count * self._idf[term] for term, count in counts.items() if term in self._idf
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        return {term: weight / norm for term, weight in vector.items()} if norm else {}

    def search(self, query: str, k: int = 3) -> list[Passage]:
        """The k passages most similar to the query, best first.

        Passages sharing no terms with the query are left out, so a query about
        something the guide does not cover returns an empty list rather than k
        irrelevant passages. Ties keep the guide's file order, because sorted()
        is stable.

        Returns:
            Copies of the matching passages with their score filled in; the
            stored passages are not modified.
        """
        query_vector = self._vectorize(tokenize(query))
        if not query_vector:
            return []
        scored = []
        for passage, vector in zip(self.passages, self._vectors):
            score = sum(weight * vector.get(term, 0.0) for term, weight in query_vector.items())
            if score > 0:
                scored.append(Passage(passage.source, passage.heading, passage.text, round(score, 4)))
        scored.sort(key=lambda passage: passage.score, reverse=True)
        return scored[:k]
