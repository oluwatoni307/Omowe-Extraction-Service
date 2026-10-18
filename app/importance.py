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
from app.importance_model import Concept, DocumentImportance, ImportantPassage, ImportanceSignal

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


_DEF_SPLIT = re.compile(
    r"\b(?:is|are|was|were)\s+(?:also\s+)?(?:defined as|known as|referred to as|called|a term for)\s+",
    re.I,
)


_PRONOUNS_AND_NOISE = frozenset("""
this that these those it its they them their which what who whom where when why
how here there we our you he she his her such another one all both each every
some any none
""".split())

_STRUCTURAL_HEADINGS = re.compile(
    r"\b(introduction|conclusion|references|chapter|part|abstract|arxiv|author|google|"
    r"university|copyright|permission|background|related\s+work|methodology|methods|"
    r"materials\s+and\s+methods|experiments|experimental\s+setup|discussion|results|"
    r"evaluation|overview|implementation\s+details|future\s+work|acknowledgements|"
    r"acknowledgments|appendix|table|figure)\b",
    re.I,
)

_PREFIX_NOISE = re.compile(
    r"^(?:in\s+summary|in\s+addition|formally|specifically|moreover|furthermore|"
    r"for\s+example|note\s+that|clearly|here|thus|hence|therefore|first|second|third),?\s*",
    re.I,
)


def _clean_term_candidate(term_raw: str) -> str | None:
    # Strip leading numbering like "1. ", "3.2 "
    term = re.sub(r"^\d+(?:\.\d+)*\s*[:.\-–—]?\s*", "", term_raw.strip())
    # Strip common discourse transition phrases
    term = _PREFIX_NOISE.sub("", term).strip(" \t\n\"'•-–—*:,")
    # Strip leading articles
    term = re.sub(r"^(?:the|a|an)\s+", "", term, flags=re.I).strip()
    if not term:
        return None
    words = term.split()
    if not (1 <= len(words) <= 4 and len(term) < 40):
        return None
    # Reject pronouns / demonstratives
    lower_words = [w.lower() for w in words]
    if lower_words[0] in _PRONOUNS_AND_NOISE:
        return None
    # Must have at least one substantive word (len >= 3 and not in stopwords)
    substantive = [w for w in lower_words if len(w) >= 3 and w not in STOPWORDS]
    if not substantive:
        return None
    # Reject if it matches structural paper headings
    if _STRUCTURAL_HEADINGS.search(term):
        return None
    return term


def _extract_concepts(paragraphs: list[ParagraphBlock], blocks: list[Block], max_concepts: int = 8) -> list[Concept]:
    concepts: list[Concept] = []
    seen_terms: set[str] = set()

    for p in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", p.text.strip())
        for s in sentences:
            match = _DEF_SPLIT.search(s)
            if match:
                term_part = s[:match.start()].strip(" \t\n\"'•-–—*")
                def_part = s[match.end():].strip(" \t\n\"'").rstrip(".!?")
                clean_term = _clean_term_candidate(term_part)
                if clean_term and len(def_part) > 15:
                    clean_def = def_part[0].upper() + def_part[1:] + "." if def_part else "Key concept."
                    if clean_term.lower() not in seen_terms:
                        seen_terms.add(clean_term.lower())
                        concepts.append(Concept(term=clean_term, definition=clean_def))
                        if len(concepts) >= max_concepts:
                            return concepts

    prev_heading = None
    for b in blocks:
        if isinstance(b, HeadingBlock):
            prev_heading = b.text.strip()
        elif isinstance(b, ParagraphBlock) and prev_heading:
            clean_term = _clean_term_candidate(prev_heading)
            def_words = b.text.strip().split()
            if clean_term and len(def_words) >= 6 and clean_term.lower() not in seen_terms:
                summary_sent = re.split(r"(?<=[.!?])\s+", b.text.strip())[0]
                if len(summary_sent) > 25:
                    seen_terms.add(clean_term.lower())
                    concepts.append(Concept(term=clean_term, definition=summary_sent))
                    if len(concepts) >= max_concepts:
                        return concepts
            prev_heading = None

    return concepts


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
    concepts = _extract_concepts(paragraphs, document.blocks)

    return DocumentImportance(
        document_id=document.document_id,
        extraction_version=document.extraction_version,
        passages=[passage for _, passage in selected],
        concepts=concepts,
    )
