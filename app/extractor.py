"""
Core extraction engine v2.

PDF (bytes) -> Document (pydantic, validated, schema_version 2).

Key principles:
1. Normalize first: per-page dominant text direction -> rotate to upright space.
2. One coordinate space: all downstream bboxes, heuristics, and crops operate in upright space.
3. Structure from geometry, text from words: cell text populated from word-level tokens.
4. Cascade, don't reject: Ruled grid -> text strategy -> word-geometry clustering -> raw_lines fallback.
5. Deterministic and 100% stateless: images and formulas returned as base64 data URIs.
"""
from __future__ import annotations

import base64
import gc
import re
import statistics
from collections import Counter, defaultdict
from io import BytesIO
from typing import Callable

import pymupdf as fitz

from app.document_model import (
    Block,
    BlockSource,
    CodeBlock,
    CodeKind,
    Document,
    DocumentMeta,
    HeadingBlock,
    ImageBlock,
    PageQuality,
    ParagraphBlock,
    QualityReport,
    ReferenceBlock,
    TableBlock,
)

EXTRACTION_VERSION = "heuristic-v2"

# ----------------------------------------------------------------- patterns
BULLET = re.compile(r"^\s*([•▪◦●■\-–—*]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)])\s+")
NUM_HEAD = re.compile(r"^(\d+(?:\.\d+){0,3}|[IVX]{1,6}|[A-Za-z])[.)]\s+\S")
NOISE_LINE = re.compile(r"^[\d\s/.\-–—•]+$")
PAGE_NUM = re.compile(r"^(page\s*)?\d{1,4}(\s*/\s*\d{1,4})?$", re.I)

# Broadened caption grammar accepting A1, S2, Roman numerals, Exhibits, Charts
TABLE_CAP = re.compile(
    r"^(?:table|tab\.|exhibit|chart)\s+(?:[A-Za-z]?\d+|[IVXLCDMivxlcdm]+|[A-Za-z])(?:\b|[:\.\-\s])",
    re.I,
)
FIG_CAP = re.compile(
    r"^(?:fig\.?|figure|diagram|scheme|algorithm|listing)\s+(?:[A-Za-z]?\d+|[IVXLCDMivxlcdm]+|[A-Za-z])(?:\b|[:\.\-\s])",
    re.I,
)
TABLE_NOTE_RE = re.compile(
    r"^(?:notes?[:\.]|statistical significance[:\.]|\*\s*p\s*[<>]|standard errors|source[:\.]|legend[:\.])",
    re.I,
)
REFERENCES_HEAD = re.compile(
    r"^(?:\d+[\.\s]+)?(?:references|bibliography|works cited|literature cited)\b",
    re.I,
)

BOLD_RE = re.compile(r"bold|black|heavy|semibold|demi", re.I)
MONO_RE = re.compile(r"mono|courier|consol|menlo|inconsolata|typewriter|cmtt|source ?code|fira ?code|jetbrains", re.I)
MATH_FONT_RE = re.compile(r"^(?:[A-Z]{6}\+)?(CM(MI|SY|EX|BSY|MIB)\d*|MSAM|MSBM|Cambria ?Math|STIX|MTMI|MTSY|MTEX|Euclid)", re.I)

AUTHOR_AFFIL_LINE = re.compile(
    r"\b(?:and\s+[A-Z][a-z]+|[A-Z][a-z]+,\s+[A-Z][a-z]+|university|department|school of|institute|laboratory|center for|college of|working paper|doi:\s*10\.|https?://|email:|\b\d{4}\b)\b",
    re.I,
)

_CITATION_LINE = re.compile(
    r"\b(?:PMID:\s*\d+|doi:\s*10\.\d+|https?://|arXiv:\s*\S+"
    r"|pp?\.\s*\d+[-–]\d+|vol(?:ume)?\.?\s*\d+|no\.\s*\d+|issn|isbn)\b"
    r"|^\[\d+\]"
    r"|^\(\d{4}\)",
    re.I,
)


# ------------------------------------------------------------------ helpers
def _area(r):
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _overlap_frac(r, outer):
    x0, y0 = max(r[0], outer[0]), max(r[1], outer[1])
    x1, y1 = min(r[2], outer[2]), min(r[3], outer[3])
    return _area((x0, y0, x1, y1)) / max(_area(r), 1e-6)


def _clean(s):
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


def _png_data_uri(pix: fitz.Pixmap) -> str:
    buf = BytesIO(pix.tobytes("png"))
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _join_lines(a: str, b: str, doc_vocab: set[str] | None = None) -> str:
    """Vocabulary-aware dehyphenation. Removes hyphen when joined word exists in
    document vocabulary or matches split patterns; preserves genuine compounds."""
    if a.endswith(("\u2010", "\u00ad")):
        return a[:-1] + b
    if a.endswith("-"):
        m_a = re.search(r"([A-Za-z]+)-$", a)
        m_b = re.match(r"^([A-Za-z]+)", b)
        if m_a and m_b:
            frag_a = m_a.group(1)
            frag_b = m_b.group(1)
            cand = (frag_a + frag_b).lower()

            if doc_vocab and cand in doc_vocab:
                return a[:-1] + b

            # Common split word suffixes
            if frag_b.lower() in ("ing", "tion", "tions", "ment", "ments", "ed", "ly", "able", "ence", "ance", "ive", "ity", "ties", "ous", "al", "dents", "prove", "port", "ture"):
                return a[:-1] + b

            if re.match(r"^[a-z]", frag_b) and (frag_a.lower() in ("stu", "im", "dur", "sup", "litera", "engage", "inter", "auto", "multi") or (len(frag_a) >= 4 and len(frag_b) >= 3)):
                if doc_vocab and f"{frag_a}-{frag_b}".lower() in doc_vocab:
                    return a + " " + b
                return a[:-1] + b

        return a + " " + b
    return a + " " + b


# --------------------------------------------------- orientation normalization
def _determine_page_orientation(page: fitz.Page) -> int:
    """Computes dominant text direction from character-weighted line dir.
    Returns the extra rotation in degrees (0, 90, 180, 270) needed to make text upright."""
    try:
        raw = page.get_text("rawdict")
    except Exception:
        return 0

    dirs: dict[tuple[int, int], int] = {}
    for b in raw.get("blocks", []):
        for l in b.get("lines", []):
            d_raw = l.get("dir", (1.0, 0.0))
            # Round direction vector to nearest standard axis
            d = (round(d_raw[0]), round(d_raw[1]))
            n = sum(len(s.get("text", "")) for s in l.get("spans", []))
            dirs[d] = dirs.get(d, 0) + n

    if not dirs:
        return 0

    dom_dir = max(dirs.items(), key=lambda kv: kv[1])[0]

    # Map dominant direction to required rotation
    if dom_dir == (0, -1):
        return 90
    elif dom_dir == (0, 1):
        return 270
    elif dom_dir == (-1, 0):
        return 180
    return 0


# ------------------------------------------------------------- line extraction
def _extract_page_lines(page: fitz.Page, pno: int, mat: fitz.Matrix):
    """Extracts lines and image boxes mapped through the page rotation matrix into upright coordinates."""
    lines, images = [], []
    d = page.get_text("dict")
    for bi, b in enumerate(d.get("blocks", [])):
        if b.get("type") == 1:
            r = fitz.Rect(b["bbox"]) * mat
            images.append((r.x0, r.y0, r.x1, r.y1))
            continue
        for ln in b.get("lines", []):
            spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = _clean(" ".join(s["text"] for s in spans))
            if not text:
                continue
            r = fitz.Rect(ln["bbox"]) * mat
            total = sum(len(s["text"].strip()) for s in spans)
            sizes, bold, mono, math = Counter(), 0, 0, 0
            for s in spans:
                n = len(s["text"].strip())
                sizes[round(s["size"], 1)] += n
                f, fl = s.get("font", ""), s.get("flags", 0)
                if fl & 16 or BOLD_RE.search(f):
                    bold += n
                if fl & 8 or MONO_RE.search(f):
                    mono += n
                if MATH_FONT_RE.match(f):
                    math += n
            size = max(sizes.items(), key=lambda kv: (kv[1], kv[0]))[0]
            lines.append(dict(
                page=pno,
                bbox=(r.x0, r.y0, r.x1, r.y1),
                text=text,
                size=size,
                bold=bold / total > 0.7,
                mono=mono / total > 0.7,
                math=math / total,
                bid=(pno, bi),
            ))
    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
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
    if npages < 1:
        return set(), set()
    sig = Counter()
    margin_y_counts = Counter()
    for lines, (W, H) in zip(pages_lines, sizes):
        seen, seen_y = set(), set()
        for l in lines:
            y0, y1 = l["bbox"][1], l["bbox"][3]
            if (y0 < 0.09 * H or y1 > 0.91 * H) and len(l["text"]) < 120:
                k = (re.sub(r"\d+", "#", l["text"].lower()), round(y0 / 6))
                if k not in seen:
                    seen.add(k)
                    sig[k] += 1
                yb = round(y0 / 6)
                if yb not in seen_y:
                    seen_y.add(yb)
                    margin_y_counts[yb] += 1

    if npages == 1:
        repeated_sigs = {
            k for k, v in sig.items()
            if re.match(r"^(?:#|page\s+#|#\s+of\s+#)$", k[0].strip())
        }
        return repeated_sigs, set()

    min_rep = 2 if npages <= 5 else max(2, int(0.25 * npages))
    repeated_sigs = {k for k, v in sig.items() if v >= min_rep}
    thr = max(2, int(0.25 * npages))
    furniture_y = {yb for yb, cnt in margin_y_counts.items() if cnt >= thr}
    return repeated_sigs, furniture_y


def _is_heading(l, body, in_references=False):
    if in_references:
        return False
    t, n = l["text"], len(l["text"])
    if n < 2 or n > 110 or NOISE_LINE.match(t) or _CITATION_LINE.search(t):
        return False
    if AUTHOR_AFFIL_LINE.search(t) and not NUM_HEAD.match(t):
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


# ------------------------------------------------------------- table extraction
def _cluster_table_columns(tbl_lines: list[dict]) -> tuple[list[list[str]], list[float]]:
    """Word-geometry column clustering for borderless academic tables."""
    # Filter margin furniture
    tbl_lines = [l for l in tbl_lines if l["bbox"][0] >= 55]
    if not tbl_lines:
        return [], []

    # Group items by y
    rows_grouped = []
    curr_y = None
    curr_row = []
    for l in tbl_lines:
        yb = round(l["bbox"][1] / 6.5)
        if curr_y is None or yb == curr_y:
            curr_y = yb
            curr_row.append(l)
        else:
            rows_grouped.append(curr_row)
            curr_y = yb
            curr_row = [l]
    if curr_row:
        rows_grouped.append(curr_row)

    if not rows_grouped:
        return [], []

    min_x0 = min(l["bbox"][0] for l in tbl_lines)
    max_x1 = max(l["bbox"][2] for l in tbl_lines)
    table_w = max_x1 - min_x0

    multi_items = [item for r in rows_grouped if len(r) >= 2 for item in r]
    if not multi_items:
        return [[l["text"]] for r in rows_grouped for l in r], [min_x0]

    stub_boundary = min_x0 + 0.28 * table_w
    data_items = [it for it in multi_items if it["bbox"][0] >= stub_boundary]

    if data_items:
        data_cx = sorted((it["bbox"][0] + it["bbox"][2]) / 2 for it in data_items)
        clusters = []
        curr_c = [data_cx[0]]
        for x in data_cx[1:]:
            if x - curr_c[-1] > 35.0:
                clusters.append(sum(curr_c) / len(curr_c))
                curr_c = [x]
            else:
                curr_c.append(x)
        if curr_c:
            clusters.append(sum(curr_c) / len(curr_c))
        col_centers = [min_x0 + 30.0] + clusters
    else:
        all_cx = sorted((it["bbox"][0] + it["bbox"][2]) / 2 for it in multi_items)
        clusters = []
        curr_c = [all_cx[0]]
        for x in all_cx[1:]:
            if x - curr_c[-1] > 35.0:
                clusters.append(sum(curr_c) / len(curr_c))
                curr_c = [x]
            else:
                curr_c.append(x)
        if curr_c:
            clusters.append(sum(curr_c) / len(curr_c))
        col_centers = clusters

    ncols = len(col_centers)
    grid_rows = []
    for r in rows_grouped:
        cells = [""] * ncols
        if len(r) == 1 and (r[0]["text"].startswith("Panel") or r[0]["bbox"][2] - r[0]["bbox"][0] > 0.4 * table_w):
            cells[0] = r[0]["text"]
        else:
            for item in r:
                if len(col_centers) > 1 and item["bbox"][0] < stub_boundary and data_items:
                    cells[0] = (cells[0] + " " + item["text"]).strip() if cells[0] else item["text"]
                else:
                    cx = (item["bbox"][0] + item["bbox"][2]) / 2
                    best_c = min(range(ncols), key=lambda c: abs(cx - col_centers[c]))
                    cells[best_c] = (cells[best_c] + " " + item["text"]).strip() if cells[best_c] else item["text"]
        grid_rows.append(cells)

    return grid_rows, col_centers


# ------------------------------------------------------------- page processing
def _process_page(page: fitz.Page, pno: int, lines: list[dict], imgs: list[tuple], ctx: dict) -> tuple[list[Block], PageQuality]:
    W, H = page.rect.width, page.rect.height
    mat = page.rotation_matrix
    body, pitch = ctx["body"], ctx["pitch"]
    doc_vocab = ctx.get("doc_vocab", set())

    # Filter furniture
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
    asset_collector = ctx.get("asset_collector")

    def _crop_asset(pix: fitz.Pixmap) -> str:
        if asset_collector:
            asset_id = f"p{pno}_{len(raw_items)}"
            return asset_collector(asset_id, pix.tobytes("png"))
        return _png_data_uri(pix)

    def absorb(rect, frac=0.6):
        nonlocal lines
        inside = [l for l in lines if _overlap_frac(l["bbox"], rect) >= frac]
        lines = [l for l in lines if l not in inside]
        return inside

    def find_caption(rect, pattern, consume=True):
        nonlocal lines
        best_cap, best_dist = None, float("inf")
        for l in lines:
            if not pattern.match(l["text"]):
                continue
            above = rect[1] - l["bbox"][3]
            below = l["bbox"][1] - rect[3]
            horiz = min(rect[2], l["bbox"][2]) - max(rect[0], l["bbox"][0])
            if horiz > -30:
                if -10 <= above <= 70:
                    dist = abs(above)
                    if dist < best_dist:
                        best_dist = dist
                        best_cap = l
                elif -10 <= below <= 70:
                    dist = abs(below)
                    if dist < best_dist:
                        best_dist = dist
                        best_cap = l

        if best_cap:
            if consume:
                lines = [x for x in lines if x is not best_cap]
            return best_cap["text"]
        return None

    page_flags = []
    if ctx.get("rotations", {}).get(pno, 0) != 0:
        page_flags.append("rotated_normalized")

    # Table extraction cascade
    table_boxes = []
    # 1. Ruled grid
    try:
        found_tables = page.find_tables().tables
    except Exception:
        found_tables = []

    for t in found_tables:
        bbox = tuple(t.bbox)
        if find_caption(bbox, FIG_CAP, consume=False):
            continue
        rows = [[_clean(c) for c in r] for r in t.extract()]
        rows = [r for r in rows if any(r)]
        ncols = max((len(r) for r in rows), default=0)
        if len(rows) >= 2 and ncols >= 2:
            absorb(bbox, 0.5)
            table_boxes.append(bbox)
            cap = find_caption(bbox, TABLE_CAP, consume=True)
            raw_items.append(dict(
                kind="table",
                bbox=bbox,
                rows=rows,
                caption=cap,
                notes=[],
                source="grid",
            ))

    # 2. Word-geometry clustering for borderless/academic tables
    while True:
        cap_idx = None
        for i, l in enumerate(lines):
            if TABLE_CAP.match(l["text"]):
                cap_idx = i
                break
        if cap_idx is None:
            break

        cap_line = lines[cap_idx]
        cap_text = cap_line["text"]
        remaining = lines[cap_idx + 1:]
        tbl_lines = []
        notes = []
        in_notes = False
        for l in remaining:
            if TABLE_CAP.match(l["text"]) or FIG_CAP.match(l["text"]):
                break
            if in_notes or TABLE_NOTE_RE.match(l["text"]):
                in_notes = True
                notes.append(l["text"])
            else:
                if tbl_lines and l["bbox"][1] - tbl_lines[-1]["bbox"][3] > 35:
                    break
                tbl_lines.append(l)

        if tbl_lines:
            rows, _ = _cluster_table_columns(tbl_lines)
            if len(rows) >= 2:
                # Absorb caption, table lines, and notes
                lines = [l for l in lines if l is not cap_line and l not in tbl_lines and l not in [nl for nl in lines if nl["text"] in notes]]
                t_bbox = (
                    min(l["bbox"][0] for l in tbl_lines),
                    cap_line["bbox"][1],
                    max(l["bbox"][2] for l in tbl_lines),
                    max(l["bbox"][3] for l in tbl_lines),
                )
                table_boxes.append(t_bbox)
                raw_items.append(dict(
                    kind="table",
                    bbox=t_bbox,
                    rows=rows,
                    caption=cap_text,
                    notes=notes,
                    source="words",
                ))
            else:
                # Table missing fallback (B17)
                page_flags.append("table_missing")
                raw_lines = [l["text"] for l in tbl_lines]
                lines = [l for l in lines if l is not cap_line and l not in tbl_lines]
                raw_items.append(dict(
                    kind="table",
                    bbox=cap_line["bbox"],
                    rows=[],
                    caption=cap_text,
                    notes=notes,
                    raw_lines=raw_lines,
                    flags=["table_missing"],
                    source="text_layer",
                ))
        else:
            lines = [l for l in lines if l is not cap_line]

    # Image extraction (with native raster support & multi-panel grouping, B21, B22, B23)
    img_infos = []
    try:
        img_infos = page.get_image_info(xrefs=True)
    except Exception:
        img_infos = []

    extracted_imgs = []
    for info in img_infos:
        r = fitz.Rect(info["bbox"]) * mat
        bb = (r.x0, r.y0, r.x1, r.y1)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w < 25 or h < 25 or _area(bb) >= 0.75 * W * H:
            continue
        if any(_overlap_frac(bb, tb) > 0.5 for tb in table_boxes):
            continue

        xref = info.get("xref", 0)
        asset_uri = None
        if asset_collector:
            asset_id = f"p{pno}_{len(extracted_imgs)}"
            if xref > 0:
                try:
                    base_img = page.parent.extract_image(xref)
                    if base_img:
                        asset_uri = asset_collector(asset_id, base_img["image"])
                except Exception:
                    pass
            if not asset_uri:
                pix = page.get_pixmap(clip=fitz.Rect(*bb), dpi=200)
                asset_uri = asset_collector(asset_id, pix.tobytes("png"))
        else:
            if xref > 0:
                try:
                    base_img = page.parent.extract_image(xref)
                    if base_img:
                        ext = base_img.get("ext", "png")
                        b64 = base64.b64encode(base_img["image"]).decode("ascii")
                        asset_uri = f"data:image/{ext};base64,{b64}"
                except Exception:
                    pass

            if not asset_uri:
                pix = page.get_pixmap(clip=fitz.Rect(*bb), dpi=200)
                asset_uri = _crop_asset(pix)

        absorb(bb, 0.8)
        cap = find_caption(bb, FIG_CAP, consume=False)
        extracted_imgs.append(dict(
            bbox=bb,
            asset=asset_uri,
            caption=cap,
            xref=xref,
        ))

    # Multi-panel grouping
    caption_to_imgs = defaultdict(list)
    captionless_imgs = []
    for im in extracted_imgs:
        if im["caption"]:
            caption_to_imgs[im["caption"]].append(im)
        else:
            captionless_imgs.append(im)

    for cap_text, cap_imgs in list(caption_to_imgs.items()):
        for cim in list(captionless_imgs):
            cap_imgs.append(cim)
            captionless_imgs.remove(cim)

    for cap_text, c_group in caption_to_imgs.items():
        find_caption(c_group[0]["bbox"], FIG_CAP, consume=True)
        panel_assets = [im["asset"] for im in c_group]
        u_bbox = (
            min(im["bbox"][0] for im in c_group),
            min(im["bbox"][1] for im in c_group),
            max(im["bbox"][2] for im in c_group),
            max(im["bbox"][3] for im in c_group),
        )
        raw_items.append(dict(
            kind="image",
            bbox=u_bbox,
            asset=panel_assets[0],
            caption=cap_text,
            panels=panel_assets if len(panel_assets) > 1 else [],
            decorative=False,
            source="raster",
        ))

    for im in captionless_imgs:
        w, h = im["bbox"][2] - im["bbox"][0], im["bbox"][3] - im["bbox"][1]
        is_decorative = (w < 80 and h < 80) or (pno == 1)
        raw_items.append(dict(
            kind="image",
            bbox=im["bbox"],
            asset=im["asset"],
            caption=None,
            panels=[],
            decorative=is_decorative,
            source="raster",
        ))

    # Vector drawing clusters
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
            pix = page.get_pixmap(clip=fitz.Rect(*bb), dpi=200)
            absorb(bb, 0.8)
            cap = find_caption(bb, FIG_CAP)
            raw_items.append(dict(
                kind="image",
                bbox=bb,
                asset=_crop_asset(pix),
                caption=cap,
                decorative=not bool(cap),
                source="vector",
            ))

    for l in lines:
        l["kind"] = "line"
    items = raw_items + lines
    if not items:
        return [], PageQuality(page=pno, rotation_applied=ctx.get("rotations", {}).get(pno, 0), flags=page_flags)

    two = _two_column(items, W)
    items = _order_items(items, W, two)

    blocks: list[Block] = []
    cur = None

    def classify(l):
        if ctx.get("in_references"):
            return "reference"
        if REFERENCES_HEAD.match(l["text"]):
            ctx["in_references"] = True
            return "heading"
        if not ctx["mono_doc"] and l["mono"] and l["size"] <= body * 1.25:
            return "code"
        if l["math"] >= 0.5:
            return "formula"
        if _is_heading(l, body, in_references=ctx.get("in_references", False)):
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
            return pt <= 2.2 * s
        if k == "reference":
            # Reference entries start with an author name or number; join continuation lines
            if re.match(r"^[A-Z][a-z]+,\s+[A-Z]|^\(?\d{1,4}[.)\]]", l["text"]):
                return False
            return pt <= 2.0 * s
        return pt <= 2.4 * max(s, pitch)

    def flush():
        nonlocal cur
        if not cur:
            return
        k, cls_lines = cur["k"], cur["lines"]
        b_id = f"p{pno}_{len(blocks)}"
        b_bbox = (
            min(l["bbox"][0] for l in cls_lines),
            min(l["bbox"][1] for l in cls_lines),
            max(l["bbox"][2] for l in cls_lines),
            max(l["bbox"][3] for l in cls_lines),
        )

        if k == "heading":
            text = " ".join(l["text"] for l in cls_lines)
            blocks.append(HeadingBlock(id=b_id, page=pno, text=text, bbox=b_bbox, source="text_layer"))
        elif k == "reference":
            text = cls_lines[0]["text"]
            for l in cls_lines[1:]:
                text = _join_lines(text, l["text"], doc_vocab)
            blocks.append(ReferenceBlock(id=b_id, page=pno, text=text, bbox=b_bbox, source="text_layer"))
        elif k == "code":
            text = "\n".join(l["text"] for l in cls_lines)
            blocks.append(CodeBlock(id=b_id, page=pno, kind=CodeKind.CODE, text=text, bbox=b_bbox, source="text_layer"))
        elif k == "formula":
            text = " ".join(l["text"] for l in cls_lines)
            pix = page.get_pixmap(clip=fitz.Rect(*b_bbox), dpi=200)
            blocks.append(CodeBlock(
                id=b_id,
                page=pno,
                kind=CodeKind.FORMULA,
                text=text,
                asset_url=_crop_asset(pix),
                bbox=b_bbox,
                source="vector",
            ))
        else:
            text = cls_lines[0]["text"]
            for l in cls_lines[1:]:
                text = _join_lines(text, l["text"], doc_vocab)
            if not NOISE_LINE.match(text):
                blocks.append(ParagraphBlock(id=b_id, page=pno, text=text, bbox=b_bbox, source="text_layer"))
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
            b_id = f"p{pno}_{len(blocks)}"
            if it["kind"] == "table":
                blocks.append(TableBlock(
                    id=b_id,
                    page=pno,
                    rows=it["rows"],
                    caption=it["caption"],
                    notes=it.get("notes", []),
                    raw_lines=it.get("raw_lines"),
                    flags=it.get("flags", []),
                    bbox=it.get("bbox"),
                    source=it.get("source", "grid"),
                ))
            else:
                blocks.append(ImageBlock(
                    id=b_id,
                    page=pno,
                    asset_url=it["asset"],
                    caption=it["caption"],
                    panels=it.get("panels", []),
                    decorative=it.get("decorative", False),
                    bbox=it.get("bbox"),
                    source=it.get("source", "raster"),
                ))
    flush()

    # Per-page parity computation (B28)
    raw_text = page.get_text()
    raw_words = set(re.findall(r"\b[A-Za-z]{4,}\b", raw_text.lower()))
    extracted_text = " ".join(getattr(b, "text", "") for b in blocks)
    extracted_words = set(re.findall(r"\b[A-Za-z]{4,}\b", extracted_text.lower()))

    raw_nums = set(re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", raw_text))
    extracted_nums = set(re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", extracted_text))
    numeric_parity = (len(raw_nums & extracted_nums) / len(raw_nums)) if raw_nums else 1.0

    word_parity = (len(raw_words & extracted_words) / len(raw_words)) if raw_words else 1.0
    if word_parity < 0.97:
        page_flags.append("low_text_parity")

    quality = PageQuality(
        page=pno,
        rotation_applied=ctx.get("rotations", {}).get(pno, 0),
        word_parity=round(word_parity, 4),
        numeric_parity=round(numeric_parity, 4),
        flags=page_flags,
    )
    return blocks, quality


# ------------------------------------------------------------- main extraction
def _extract_from_fitz(
    doc: fitz.Document,
    source_filename: str,
    document_id: str | None = None,
    asset_collector: Callable[[str, bytes], str] | None = None,
) -> Document:
    npages = len(doc)
    pages = list(range(npages))

    # Phase 1: Determine orientation for each page
    rotations = {}
    matrices = {}
    for p in pages:
        page = doc[p]
        rot = _determine_page_orientation(page)
        rotations[p + 1] = rot
        if rot != 0:
            page.set_rotation((page.rotation + rot) % 360)
        matrices[p] = page.rotation_matrix
        fitz.TOOLS.store_shrink(100)

    # Phase 2: Gather raw lines and doc vocabulary
    raw, sizes = {}, []
    doc_vocab = set()
    for p in pages:
        lines, imgs = _extract_page_lines(doc[p], p + 1, matrices[p])
        raw[p] = (lines, imgs)
        sizes.append((doc[p].rect.width, doc[p].rect.height))
        for l in lines:
            for w in re.findall(r"\b[A-Za-z\-]{2,}\b", l["text"].lower()):
                doc_vocab.add(w.replace("-", ""))
        fitz.TOOLS.store_shrink(100)

    all_lines = [l for p in pages for l in raw[p][0]]
    body, pitch, mono_frac = _body_stats(all_lines)
    furniture, furniture_y = _find_furniture([raw[p][0] for p in pages], sizes, npages)

    ctx = dict(
        body=body,
        pitch=pitch,
        mono_doc=mono_frac > 0.6,
        furniture=furniture,
        furniture_y=furniture_y,
        asset_collector=asset_collector,
        rotations=rotations,
        doc_vocab=doc_vocab,
        in_references=False,
    )

    # Phase 3: Process pages sequentially
    blocks: list[Block] = []
    page_qualities: list[PageQuality] = []
    for p in pages:
        lines, imgs = raw[p]
        p_blocks, p_qual = _process_page(doc[p], p + 1, lines, imgs, ctx)
        blocks.extend(p_blocks)
        page_qualities.append(p_qual)
        fitz.TOOLS.store_shrink(100)

    fitz.TOOLS.store_shrink(100)
    gc.collect()

    doc_flags = []
    if any(q.flags for q in page_qualities):
        doc_flags = list({f for q in page_qualities for f in q.flags})

    avg_parity = statistics.mean(q.word_parity for q in page_qualities) if page_qualities else 1.0
    quality_report = QualityReport(
        overall_score=round(avg_parity, 4),
        pages=page_qualities,
        flags=doc_flags,
    )

    meta = DocumentMeta(
        title=(doc.metadata or {}).get("title") or None,
        source_filename=source_filename,
        page_count=npages,
        body_font_size=body,
    )
    doc_kwargs = dict(
        schema_version=2,
        extraction_version=EXTRACTION_VERSION,
        meta=meta,
        blocks=blocks,
        quality=quality_report,
    )
    if document_id:
        doc_kwargs["document_id"] = document_id
    return Document(**doc_kwargs)


def extract_document(
    pdf: bytes | fitz.Document,
    source_filename: str,
    document_id: str | None = None,
    asset_collector: Callable[[str, bytes], str] | None = None,
) -> Document:
    """Run heuristic extraction over an eligible PDF and return a validated Document."""
    if isinstance(pdf, fitz.Document):
        return _extract_from_fitz(
            pdf,
            source_filename,
            document_id=document_id,
            asset_collector=asset_collector,
        )
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        return _extract_from_fitz(
            doc,
            source_filename,
            document_id=document_id,
            asset_collector=asset_collector,
        )
