"""
Core extraction engine.

PDF (bytes) -> Document (pydantic, validated).

Heuristic, not ML-based — deliberately, per the scoping decisions this
was built against: born-digital, prose-dominant documents only (the
eligibility gate in eligibility.py enforces that before this runs).

No files are written to disk. Image/formula crops are returned as
base64-encoded PNGs embedded directly on the block (ImageBlock.asset_url
/ CodeBlock.asset_url carry a data URI) — the server is stateless by
design; the client owns persisting assets locally.
"""
from __future__ import annotations

import base64
import re
import statistics
from collections import Counter, defaultdict
from io import BytesIO

import pymupdf as fitz

from app.document_model import (
    Block,
    CodeBlock,
    CodeKind,
    Document,
    DocumentMeta,
    HeadingBlock,
    ImageBlock,
    ParagraphBlock,
    TableBlock,
)

EXTRACTION_VERSION = "heuristic-v1"

# ----------------------------------------------------------------- patterns
BULLET = re.compile(r"^\s*([•▪◦●■\-–—*]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)])\s+")
NUM_HEAD = re.compile(r"^(\d+(?:\.\d+){0,3}|[IVX]{1,6}|[A-Za-z])[.)]\s+\S")
NOISE_LINE = re.compile(r"^[\d\s/.\-–—•]+$")
PAGE_NUM = re.compile(r"^(page\s*)?\d{1,4}(\s*/\s*\d{1,4})?$", re.I)
TABLE_CAP = re.compile(r"^(table|tab\.)\s*[\dIVXivx]+", re.I)
FIG_CAP = re.compile(r"^(fig\.?|figure)\s*\d+", re.I)
BOLD_RE = re.compile(r"bold|black|heavy|semibold|demi", re.I)
MONO_RE = re.compile(r"mono|courier|consol|menlo|inconsolata|typewriter|cmtt|source ?code|fira ?code|jetbrains", re.I)
MATH_FONT_RE = re.compile(r"^(?:[A-Z]{6}\+)?(CM(MI|SY|EX|BSY|MIB)\d*|MSAM|MSBM|Cambria ?Math|STIX|MTMI|MTSY|MTEX|Euclid)", re.I)


# ------------------------------------------------------------------ helpers
def _area(r):
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _overlap_frac(r, outer):
    x0, y0 = max(r[0], outer[0]), max(r[1], outer[1])
    x1, y1 = min(r[2], outer[2]), min(r[3], outer[3])
    return _area((x0, y0, x1, y1)) / max(_area(r), 1e-6)


def _union(rects):
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))


def _clean(s):
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


def _join_lines(a, b):
    if a.endswith(("\u2010", "\u00ad")):
        return a[:-1] + b
    if a.endswith("-") and re.match(r"[a-z]", b) and re.search(r"[A-Za-z]{2}-$", a):
        return a + b
    return a + " " + b


def _png_data_uri(pix: "fitz.Pixmap") -> str:
    buf = BytesIO(pix.tobytes("png"))
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------- line extraction
def _line_text(spans):
    out, prev = "", None
    for s in spans:
        t = s["text"]
        if not t:
            continue
        if prev is not None:
            gap = s["bbox"][0] - prev["bbox"][2]
            if (gap > 0.25 * max(s["size"], 1) and not out.endswith((" ", "\t", "\xa0"))
                    and not t.startswith((" ", "\t", "\xa0"))):
                out += " "
        out += t
        prev = s
    return _clean(out)


def _extract_page_lines(page, pno):
    lines, images = [], []
    d = page.get_text("dict")
    for bi, b in enumerate(d["blocks"]):
        if b["type"] == 1:
            images.append(tuple(b["bbox"]))
            continue
        for ln in b.get("lines", []):
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = _line_text(ln["spans"])
            if not text:
                continue
            total = sum(len(s["text"].strip()) for s in spans)
            sizes, bold, mono, math = Counter(), 0, 0, 0
            for s in spans:
                n = len(s["text"].strip())
                sizes[round(s["size"], 1)] += n
                f, fl = s["font"], s["flags"]
                if fl & 16 or BOLD_RE.search(f):
                    bold += n
                if fl & 8 or MONO_RE.search(f):
                    mono += n
                if MATH_FONT_RE.match(f):
                    math += n
            size = max(sizes.items(), key=lambda kv: (kv[1], kv[0]))[0]
            lines.append(dict(page=pno, bbox=tuple(ln["bbox"]), text=text, size=size,
                              bold=bold / total > 0.7, mono=mono / total > 0.7,
                              math=math / total, bid=(pno, bi)))
    return lines, images


def _body_stats(all_lines):
    c = Counter()
    for l in all_lines:
        c[round(l["size"] * 2) / 2] += len(l["text"])
    body = c.most_common(1)[0][0] if c else 11.0
    by_block = defaultdict(list)
    for l in all_lines:
        by_block[l["bid"]].append(l)
    deltas = []
    for ls in by_block.values():
        ls.sort(key=lambda l: l["bbox"][1])
        for a, b in zip(ls, ls[1:]):
            if abs(a["size"] - body) < 0.6 and abs(b["size"] - body) < 0.6:
                dl = b["bbox"][1] - a["bbox"][1]
                if 0.8 * body < dl < 3 * body:
                    deltas.append(dl)
    pitch = statistics.median(deltas) if deltas else 1.25 * body
    tot = sum(len(l["text"]) for l in all_lines) or 1
    mono_frac = sum(len(l["text"]) for l in all_lines if l["mono"]) / tot
    return body, pitch, mono_frac


def _find_furniture(pages_lines, sizes, npages):
    if npages < 3:
        return set(), set()
    sig = Counter()
    for lines, (W, H) in zip(pages_lines, sizes):
        seen = set()
        for l in lines:
            y0, y1 = l["bbox"][1], l["bbox"][3]
            if (y0 < 0.09 * H or y1 > 0.91 * H) and len(l["text"]) < 120:
                k = (re.sub(r"\d+", "#", l["text"].lower()), round(y0 / 6))
                if k not in seen:
                    seen.add(k)
                    sig[k] += 1
    min_rep = 2 if npages >= 4 else 3
    repeated_sigs = {k for k, v in sig.items() if v >= min_rep}
    rep_by_y = Counter()
    for (_, yb), v in sig.items():
        if v >= min_rep:
            rep_by_y[yb] += v
    thr = max(3, int(0.3 * npages))
    furniture_y = {yb for yb, cnt in rep_by_y.items() if cnt >= thr}
    return repeated_sigs, furniture_y


def _is_heading(l, body):
    t, n = l["text"], len(l["text"])
    if n < 2 or n > 110 or NOISE_LINE.match(t):
        return False
    diff = abs(l["size"] - body) / body >= 0.10
    if l["size"] >= body * 1.15:
        return not t.endswith((",", ";"))
    if NUM_HEAD.match(t) and (diff or l["bold"]) and not t.endswith((".", ",")):
        return True
    if l["bold"] and n <= 80 and not t.endswith((".", ",", ";")) and len(t) >= 3:
        return True
    letters = [c for c in t if c.isalpha()]
    if t.isupper() and len(letters) >= 3 and diff and n <= 70:
        return True
    return False


def _two_column(items, W):
    lines = [i for i in items if i.get("kind") == "line"]
    if len(lines) < 8:
        return False
    mid, gut = W / 2, 0.015 * W
    left = right = cross = 0
    for l in lines:
        x0, _, x1, _ = l["bbox"]
        if x0 < mid - gut and x1 > mid + gut:
            cross += 1
        elif (x0 + x1) / 2 < mid:
            left += 1
        else:
            right += 1
    n = len(lines)
    return min(left, right) >= 0.2 * n and cross <= 0.4 * n


def _order_items(items, W, two):
    if not two:
        items.sort(key=lambda i: (i["bbox"][1], i["bbox"][0]))
        for i in items:
            i["grp"] = (0, None)
        return items
    mid, gut = W / 2, 0.015 * W

    def cls(i):
        x0, _, x1, _ = i["bbox"]
        if x0 < mid - gut and x1 > mid + gut:
            return "C"
        return 0 if (x0 + x1) / 2 < mid else 1

    cross = sorted([i for i in items if cls(i) == "C"], key=lambda i: i["bbox"][1])
    cross_cy = [(c["bbox"][1] + c["bbox"][3]) / 2 for c in cross]
    keyed = []
    for i in items:
        c = cls(i)
        cy = (i["bbox"][1] + i["bbox"][3]) / 2
        if c == "C":
            j = cross.index(i)
            i["grp"] = (j, "C")
            keyed.append(((j, 1, 0, i["bbox"][1]), i))
        else:
            band = sum(1 for v in cross_cy if v < cy)
            i["grp"] = (band, c)
            keyed.append(((band, 0, c, i["bbox"][1]), i))
    keyed.sort(key=lambda kv: kv[0])
    return [i for _, i in keyed]


def _process_page(page, pno, lines, imgs, ctx) -> list[Block]:
    W, H = page.rect.width, page.rect.height
    body, pitch = ctx["body"], ctx["pitch"]

    keep = []
    for l in lines:
        y0, y1 = l["bbox"][1], l["bbox"][3]
        margin = y0 < 0.09 * H or y1 > 0.91 * H
        yb = round(y0 / 6)
        k = (re.sub(r"\d+", "#", l["text"].lower()), yb)
        if margin and len(l["text"]) < 120 and (
            k in ctx["furniture"] or yb in ctx["furniture_y"] or PAGE_NUM.match(l["text"])
        ):
            continue
        keep.append(l)
    lines = keep

    raw_items: list[dict] = []

    def absorb(rect, frac=0.6):
        nonlocal lines
        inside = [l for l in lines if _overlap_frac(l["bbox"], rect) >= frac]
        lines = [l for l in lines if l not in inside]
        return inside

    def find_caption(rect, pattern):
        nonlocal lines
        for l in lines:
            if not pattern.match(l["text"]):
                continue
            above = rect[1] - l["bbox"][3]
            below = l["bbox"][1] - rect[3]
            horiz = min(rect[2], l["bbox"][2]) - max(rect[0], l["bbox"][0])
            if horiz > 0 and (-4 <= above <= 45 or -4 <= below <= 45):
                lines = [x for x in lines if x is not l]
                return l["text"]
        return None

    table_boxes = []
    try:
        found = page.find_tables().tables
    except Exception:
        found = []
    for t in found:
        rows = [[_clean(c) for c in r] for r in t.extract()]
        rows = [r for r in rows if any(r)]
        ncols = max((len(r) for r in rows), default=0)
        cells = [c for r in rows for c in r]
        if len(rows) < 2 or ncols < 2 or sum(1 for c in cells if c) < 0.3 * len(cells):
            continue
        bbox = tuple(t.bbox)
        absorb(bbox, 0.5)
        table_boxes.append(bbox)
        raw_items.append(dict(kind="table", bbox=bbox, rows=rows,
                              caption=find_caption(bbox, TABLE_CAP)))

    for bb in imgs:
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w < 30 or h < 30 or _area(bb) >= 0.75 * W * H:
            continue
        if any(_overlap_frac(bb, tb) > 0.5 for tb in table_boxes):
            continue
        pix = page.get_pixmap(clip=fitz.Rect(*bb), dpi=150)
        absorb(bb, 0.8)
        raw_items.append(dict(kind="image", bbox=bb, asset=_png_data_uri(pix),
                              caption=find_caption(bb, FIG_CAP)))

    if hasattr(page, "cluster_drawings"):
        try:
            clusters = page.cluster_drawings()
        except Exception:
            clusters = []
        for r in clusters:
            bb = tuple(r)
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            if w < 80 or h < 30 or _area(bb) > 0.6 * W * H:
                continue
            if any(_overlap_frac(bb, tb) > 0.5 or _overlap_frac(tb, bb) > 0.5 for tb in table_boxes):
                continue
            if any(_overlap_frac(bb, it["bbox"]) > 0.5 for it in raw_items if it["kind"] == "image"):
                continue
            inside = [l for l in lines if _overlap_frac(l["bbox"], bb) >= 0.8]
            chars = sum(len(l["text"]) for l in inside)
            med = statistics.median([len(l["text"]) for l in inside]) if inside else 0
            if chars > 400 or (inside and med >= 30):
                continue
            pix = page.get_pixmap(clip=fitz.Rect(*bb), dpi=150)
            absorb(bb, 0.8)
            raw_items.append(dict(kind="image", bbox=bb, asset=_png_data_uri(pix),
                                  caption=find_caption(bb, FIG_CAP)))

    for l in lines:
        l["kind"] = "line"
    items = raw_items + lines
    if not items:
        return []
    two = _two_column(items, W)
    items = _order_items(items, W, two)
    widths = [i["bbox"][2] - i["bbox"][0] for i in items if i.get("kind") == "line"]
    colw = statistics.quantiles(widths, n=10)[-1] if len(widths) >= 10 else (max(widths) if widths else W)

    blocks: list[Block] = []
    cur = None

    def classify(l):
        if not ctx["mono_doc"] and l["mono"] and l["size"] <= body * 1.25:
            return "code"
        if l["math"] >= 0.5:
            return "formula"
        if _is_heading(l, body):
            return "heading"
        return "text"

    def can_merge(cur, l, k):
        last = cur["lines"][-1]
        if cur["k"] != k or l["grp"] != last["grp"]:
            return False
        pt = l["bbox"][1] - last["bbox"][1]
        s = l["size"]
        if pt <= 0:
            return False
        if k == "heading":
            return (abs(s - last["size"]) < 0.6 and l["bold"] == last["bold"]
                    and pt <= 1.6 * s and not NUM_HEAD.match(l["text"]))
        if k in ("code", "formula"):
            return pt <= 1.8 * s
        if abs(s - last["size"]) >= 0.6 or BULLET.match(l["text"]):
            return False
        if pt > 1.35 * pitch * (s / body):
            return False
        if l["bbox"][0] - last["bbox"][0] > 0.9 * s:
            return False
        last_w = last["bbox"][2] - last["bbox"][0]
        if last["text"].endswith((".", "?", "!", ":")) and last_w < 0.8 * colw:
            return False
        return True

    def flush():
        nonlocal cur
        if not cur:
            return
        ls, k = cur["lines"], cur["k"]
        if k == "heading":
            text = ls[0]["text"]
            for l in ls[1:]:
                text = _join_lines(text, l["text"])
            blocks.append(HeadingBlock(page=pno, text=text))
        elif k == "code":
            x0 = min(l["bbox"][0] for l in ls)
            out = []
            for l in ls:
                cw = 0.6 * l["size"]
                out.append(" " * int(round((l["bbox"][0] - x0) / cw)) + l["text"])
            blocks.append(CodeBlock(page=pno, kind=CodeKind.CODE, text="\n".join(out)))
        elif k == "formula":
            bb = _union([l["bbox"] for l in ls])
            pad = (bb[0] - 3, bb[1] - 3, bb[2] + 3, bb[3] + 3)
            pix = page.get_pixmap(clip=fitz.Rect(*pad), dpi=150)
            blocks.append(CodeBlock(page=pno, kind=CodeKind.FORMULA,
                                    text=" ".join(l["text"] for l in ls),
                                    asset_url=_png_data_uri(pix)))
        else:
            text = ls[0]["text"]
            for l in ls[1:]:
                text = _join_lines(text, l["text"])
            if not NOISE_LINE.match(text):
                blocks.append(ParagraphBlock(page=pno, text=text))
        cur = None

    for it in items:
        if it.get("kind") == "line":
            k = classify(it)
            if cur and can_merge(cur, it, k):
                cur["lines"].append(it)
            else:
                flush()
                cur = dict(k=k, lines=[it])
        else:
            flush()
            if it["kind"] == "table":
                blocks.append(TableBlock(page=pno, rows=it["rows"], caption=it["caption"]))
            else:
                blocks.append(ImageBlock(page=pno, asset_url=it["asset"], caption=it["caption"]))
    flush()
    return blocks


def _extract_from_fitz(doc: fitz.Document, source_filename: str) -> Document:
    pages = list(range(len(doc)))

    raw, sizes = {}, []
    for p in pages:
        raw[p] = _extract_page_lines(doc[p], p + 1)
        sizes.append((doc[p].rect.width, doc[p].rect.height))
    all_lines = [l for p in pages for l in raw[p][0]]
    body, pitch, mono_frac = _body_stats(all_lines)
    furniture, furniture_y = _find_furniture([raw[p][0] for p in pages], sizes, len(pages))

    ctx = dict(
        body=body,
        pitch=pitch,
        mono_doc=mono_frac > 0.6,
        furniture=furniture,
        furniture_y=furniture_y,
    )

    blocks: list[Block] = []
    for p in pages:
        lines, imgs = raw[p]
        blocks += _process_page(doc[p], p + 1, lines, imgs, ctx)

    meta = DocumentMeta(
        title=(doc.metadata or {}).get("title") or None,
        source_filename=source_filename,
        page_count=len(doc),
        body_font_size=body,
    )
    return Document(extraction_version=EXTRACTION_VERSION, meta=meta, blocks=blocks)


def extract_document(pdf: bytes | fitz.Document, source_filename: str) -> Document:
    """Run heuristic extraction over an already-eligible PDF and return a
    validated Document. Caller is responsible for having already checked
    eligibility.check_eligibility() — this function does not re-check."""
    if isinstance(pdf, fitz.Document):
        return _extract_from_fitz(pdf, source_filename)
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        return _extract_from_fitz(doc, source_filename)
