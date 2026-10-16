"""
Paragraph importance scoring.

Three cheap, deterministic signals combined into one score per paragraph:

  - centrality:   windowed TextRank (TF-IDF cosine graph + PageRank power
                  iteration). Windowed, not global, so cost stays O(n) in
                  document length rather than O(n^2) — a 600-paragraph
                  book and a 6000-paragraph book both stay fast.
  - definitional: regex match for "X is defined as / refers to / means".
  - positional:   the paragraph immediately following a heading.

No embeddings, no LLM call, no heavy NLP dependency (no sklearn/networkx/
gensim) — pure-Python TF-IDF + power iteration, so this stays cheap
enough to run on every document without its own latency budget.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from app.document_model import Block, Document, HeadingBlock, ParagraphBlock
from app.importance_model import DocumentImportance, ImportantPassage, ImportanceSignal

MIN_IMPORTANCE_SCORE = 0.7
WINDOW = 40          # only compare a paragraph's TF-IDF vector against
                      # paragraphs within this many positions of it —
                      # keeps the similarity graph sparse and cost linear
PAGERANK_DAMPING = 0.85
PAGERANK_ITERS = 30
STOPWORDS = frozenset("""
a an the of to in on for and or is are was were be been being this that
these those it its as with by from at but not no so if than then which
who whom whose what when where why how can could will would shall should
may might must do does did have has had i you he she we they them his
her our your their
""".split())

_WORD = re.compile(r"[a-zA-Z]{2,}")
_DEFINITIONAL = re.compile(
    r"\b(is|are|was|were)\s+(?:also\s+)?(defined\s+as|known\s+as|referred\s+to\s+as|called)\b"
    r"|\b(we|they)\s+call\s+(this|these|it|them)\b"
    r"|\brefers\s+to\b"
    r"|\bmeans\s+(that|doing|having|making)\b"
    r"|\bis\s+a\s+term\s+(for|used)\b"
    r"|\b(consists|composed|made\s+up)\s+of\b"
    r"|\b(divided|classified|grouped|broken\s+down)\s+into\b"
    r"|\bthe\s+(purpose|goal|role|definition|meaning|key|essence)\s+of\b"
    r"|\b(in\s+other\s+words|as\s+opposed\s+to|in\s+contrast\s+to)\b"
    r"|\bthere\s+are\s+(two|three|four|five|several)\s+(main|key|primary|major|different|basic|core)\b",
    re.I,
)


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in STOPWORDS]


def _tf(tokens: list[str]) -> Counter:
    return Counter(tokens)


def _tfidf_vectors(paragraphs: list[ParagraphBlock]) -> list[Counter]:
    n = len(paragraphs)
    token_lists = [_tokens(p.text) for p in paragraphs]
    has_substantive = any(len(t) >= 4 for t in token_lists)

    df: Counter = Counter()
    for tokens in token_lists:
        df.update(set(tokens))

    idf = {w: math.log((n + 1) / (count + 1)) + 1.0 for w, count in df.items()}
    vectors: list[Counter] = []
    for tokens in token_lists:
        if has_substantive and len(tokens) < 4:
            vectors.append(Counter())
            continue
        tf = _tf(tokens)
        vectors.append(Counter({w: (1.0 + math.log(cnt)) * idf[w] for w, cnt in tf.items()}))
    return vectors


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _textrank(paragraphs: list[ParagraphBlock]) -> dict[str, float]:
    n = len(paragraphs)
    if n == 0:
        return {}
    if n == 1:
        return {paragraphs[0].id: 1.0}

    vectors = _tfidf_vectors(paragraphs)

    # sparse windowed adjacency: edge (i, j) only within WINDOW of each other
    edges: dict[int, dict[int, float]] = {i: {} for i in range(n)}
    for i in range(n):
        if not vectors[i]:
            continue
        for j in range(i + 1, min(i + 1 + WINDOW, n)):
            if not vectors[j]:
                continue
            sim = _cosine(vectors[i], vectors[j])
            if sim > 0:
                edges[i][j] = sim
                edges.setdefault(j, {})[i] = sim

    scores = [1.0 / n if vectors[i] else 0.0 for i in range(n)]
    for _ in range(PAGERANK_ITERS):
        new_scores = [(1 - PAGERANK_DAMPING) / n if vectors[i] else 0.0 for i in range(n)]
        for i in range(n):
            neighbors = edges.get(i, {})
            total_weight = sum(neighbors.values())
            if total_weight == 0:
                continue
            contribution = PAGERANK_DAMPING * scores[i] / total_weight
            for j, w in neighbors.items():
                new_scores[j] += contribution * w
        scores = new_scores

    half = WINDOW // 2
    result: dict[str, float] = {}
    for i in range(n):
        if not vectors[i]:
            result[paragraphs[i].id] = 0.0
            continue
        local_max = max(scores[max(0, i - half) : min(n, i + half + 1)]) or 1.0
        result[paragraphs[i].id] = scores[i] / local_max
    return result


def _positional_ids(blocks: list[Block]) -> set[str]:
    ids = set()
    prev_was_heading = False
    for b in blocks:
        if isinstance(b, HeadingBlock):
            prev_was_heading = True
            continue
        if isinstance(b, ParagraphBlock):
            if prev_was_heading:
                ids.add(b.id)
            prev_was_heading = False
    return ids


def score_document(document: Document) -> DocumentImportance:
    paragraphs = [b for b in document.blocks if isinstance(b, ParagraphBlock)]
    centrality = _textrank(paragraphs)
    positional_ids = _positional_ids(document.blocks)

    selected: list[tuple[float, ImportantPassage]] = []
    for p in paragraphs:
        if len(p.text.split()) < 4 or len(p.text) < 20:
            continue
        signals: list[ImportanceSignal] = []
        c = centrality.get(p.id, 0.0)
        if c > 0:
            signals.append(ImportanceSignal.CENTRALITY)

        d_boost = 0.0
        if _DEFINITIONAL.search(p.text):
            signals.append(ImportanceSignal.DEFINITIONAL)
            d_boost = 0.3

        p_boost = 0.0
        if p.id in positional_ids:
            signals.append(ImportanceSignal.POSITIONAL)
            p_boost = 0.15

        score = round(min(1.0, 0.7 * c + d_boost + p_boost), 4)
        if score > MIN_IMPORTANCE_SCORE:
            selected.append((score, ImportantPassage(block_id=p.id, message=p.text, signals=signals)))

    selected.sort(key=lambda item: item[0], reverse=True)

    return DocumentImportance(
        document_id=document.document_id,
        extraction_version=document.extraction_version,
        passages=[passage for _, passage in selected],
    )
