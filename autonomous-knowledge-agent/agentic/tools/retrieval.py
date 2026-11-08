"""Retrieval used by RAG (knowledge base) and long-term memory search.

``HybridRetriever`` ranks documents with a dependency-free TF-IDF cosine
similarity (title and tags are up-weighted). If an embeddings model is given,
the final score is ``0.25 * TF-IDF + 0.75 * embedding cosine`` (blend calibrated on
labelled queries; see tests/test_live_llm.py::test_live_hybrid_retrieval_calibration).
See ``agentic/design/README.md`` -> "RAG" for the full description.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_STOPWORDS = set(
    """a an the and or but if then else of to in on at for with from by as is are was were be been being
    i me my we our you your he she it its they them their this that these those what which who whom how
    can could should would will do does did done have has had not no yes so very just about into over
    please hi hello hey thanks thank want need get got any some there here when where why also too
    am im i'm it's don't cant can't won't""".split()
)
_TOKEN = re.compile(r"[a-zà-ú0-9]+")


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall((text or "").lower()):
        if tok in _STOPWORDS or len(tok) < 2:
            continue
        # light stemming so "reservations"/"reserve"/"reserved" overlap
        for suffix in ("ations", "ation", "ings", "ing", "ed", "es", "s"):
            if tok.endswith(suffix) and len(tok) - len(suffix) >= 4:
                tok = tok[: -len(suffix)]
                break
        if tok.endswith("e") and len(tok) > 4:  # "charge"/"charged" -> "charg"
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    tags: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def weighted_text(self) -> str:
        # title x3 and tags x2 so that headline topics dominate the ranking
        return " ".join([self.title] * 3 + [self.tags] * 2 + [self.content])


# Minimum score for an article to count as relevant, per scoring mode.
# Calibrated in tests/test_retrieval.py (lexical) and tests/test_live_llm.py (hybrid).
LEXICAL_THRESHOLD = 0.15
HYBRID_THRESHOLD = 0.20
HYBRID_ALPHA = 0.25  # weight of TF-IDF in hybrid mode (embeddings get 0.75)


def embedding_text(doc: "Document") -> str:
    return f"{doc.title}\n{doc.tags}\n{doc.content}"


class HybridRetriever:
    def __init__(
        self,
        documents: list[Document],
        embeddings=None,
        alpha: float = HYBRID_ALPHA,
        doc_embeddings: list[list[float]] | None = None,
    ):
        self.documents = documents
        self.embeddings = embeddings
        self.alpha = alpha
        tokenized = [tokenize(d.weighted_text) for d in documents]
        n = max(len(documents), 1)
        df = Counter(tok for toks in tokenized for tok in set(toks))
        self.idf = {tok: math.log((1 + n) / (1 + c)) + 1.0 for tok, c in df.items()}
        self.vectors = [self._vectorize(toks) for toks in tokenized]
        self.doc_embeddings = doc_embeddings
        if embeddings is not None and documents and doc_embeddings is None:
            self.doc_embeddings = embeddings.embed_documents([embedding_text(d) for d in documents])

    @property
    def mode(self) -> str:
        return "hybrid" if self.doc_embeddings is not None and self.embeddings is not None else "lexical"

    @property
    def threshold(self) -> float:
        return HYBRID_THRESHOLD if self.mode == "hybrid" else LEXICAL_THRESHOLD

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {tok: (1 + math.log(c)) * self.idf.get(tok, 0.0) for tok, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {tok: v / norm for tok, v in vec.items()}

    @staticmethod
    def _cosine_sparse(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(k, 0.0) for k, v in a.items())

    @staticmethod
    def _cosine_dense(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def search(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        if not self.documents:
            return []
        qvec = self._vectorize(tokenize(query))
        scores = [self._cosine_sparse(qvec, dvec) for dvec in self.vectors]
        if self.doc_embeddings is not None and self.embeddings is not None:
            qemb = self.embeddings.embed_query(query)
            scores = [
                self.alpha * s + (1 - self.alpha) * self._cosine_dense(qemb, demb)
                for s, demb in zip(scores, self.doc_embeddings)
            ]
        ranked = sorted(zip(self.documents, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc, round(score, 4)) for doc, score in ranked[:k]]
