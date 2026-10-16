"""
Run with: pytest tests/ -v   (from the repo root, after `pip install -r requirements.txt pytest`)
"""
import io
from pathlib import Path

import pymupdf as fitz
import pytest

from app.document_model import (
    Block,
    CodeBlock,
    Document,
    HeadingBlock,
    ImageBlock,
    ParagraphBlock,
    TableBlock,
)
from app.eligibility import check_eligibility
from app.extractor import extract_document

FIXTURES = Path(__file__).parent / "fixtures"


def _open(name: str) -> fitz.Document:
    return fitz.open(FIXTURES / name)


def test_eligible_document_passes_gate():
    result = check_eligibility(_open("test.pdf"))
    assert result.eligible is True
    assert result.reasons == []


def test_scanned_pdf_rejected_no_text_layer():
    # build a PDF with an image but no text layer, in-memory
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(fitz.Rect(0, 0, 400, 400), pixmap=pix)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    result = check_eligibility(fitz.open(stream=buf.getvalue(), filetype="pdf"))
    assert result.eligible is False
    assert "no_text_layer" in [r.value for r in result.reasons]


def test_scanned_pdf_with_ocr_text_is_rejected():
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
        pix.set_rect(pix.irect, (200, 200, 200))
        page.insert_image(page.rect, pixmap=pix)
        page.insert_text((72, 72), "OCR text copied from the scanned page.")

    result = check_eligibility(doc)

    assert result.eligible is False
    assert "scanned_document" in [reason.value for reason in result.reasons]


def test_short_extractable_text_passes_gate():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A")

    result = check_eligibility(doc)

    assert result.eligible is True
    assert result.reasons == []


def test_text_after_first_25_pages_passes_gate():
    doc = fitz.open()
    for _ in range(26):
        doc.new_page()
    doc[25].insert_text((72, 72), "Text on a later page.")

    result = check_eligibility(doc)

    assert result.eligible is True
    assert result.reasons == []


def test_text_rich_pdf_with_varied_fonts_and_vector_art_passes_gate():
    doc = fitz.open()
    page = doc.new_page()
    fonts = ["helv", "hebo", "heit", "hebi", "cour"]
    for index, font in enumerate(fonts):
        page.insert_text(
            (36, 45 + index * 45),
            "Readable text passage for eligibility testing. " * 2,
            fontsize=8 + index * 4,
            fontname=font,
        )
    page.draw_rect(fitz.Rect(10, 200, 580, 790), color=(0, 0, 0))

    result = check_eligibility(doc)

    assert result.eligible is True
    assert result.reasons == []


def test_extraction_produces_all_expected_block_types():
    with open(FIXTURES / "test.pdf", "rb") as f:
        pdf_bytes = f.read()

    document = extract_document(pdf_bytes, source_filename="test.pdf")

    assert document.extraction_version
    assert document.meta.page_count == 4
    assert len(document.blocks) > 10

    by_type = {}
    for b in document.blocks:
        by_type.setdefault(type(b), []).append(b)

    assert HeadingBlock in by_type, "should detect at least one heading"
    assert ParagraphBlock in by_type, "should detect body paragraphs"
    assert TableBlock in by_type, "should detect the ruled table on page 2"
    assert ImageBlock in by_type, "should detect the raster image and/or diagram on page 3"
    assert CodeBlock in by_type, "should detect the monospaced code block on page 2"


def test_table_cells_have_no_none_values():
    with open(FIXTURES / "test.pdf", "rb") as f:
        pdf_bytes = f.read()
    document = extract_document(pdf_bytes, source_filename="test.pdf")
    tables = [b for b in document.blocks if isinstance(b, TableBlock)]
    assert tables, "fixture should contain a table"
    for t in tables:
        for row in t.rows:
            assert all(cell is not None for cell in row)


def test_image_assets_are_base64_data_uris():
    with open(FIXTURES / "test.pdf", "rb") as f:
        pdf_bytes = f.read()
    document = extract_document(pdf_bytes, source_filename="test.pdf")
    images = [b for b in document.blocks if isinstance(b, ImageBlock)]
    assert images
    for img in images:
        assert img.asset_url.startswith("data:image/png;base64,")


def test_blocks_are_in_non_decreasing_page_order():
    with open(FIXTURES / "test.pdf", "rb") as f:
        pdf_bytes = f.read()
    document = extract_document(pdf_bytes, source_filename="test.pdf")
    pages = [b.page for b in document.blocks]
    assert pages == sorted(pages)


def test_furniture_is_stripped():
    # the fixture PDF has a repeated running header + page numbers on every page
    with open(FIXTURES / "test.pdf", "rb") as f:
        pdf_bytes = f.read()
    document = extract_document(pdf_bytes, source_filename="test.pdf")
    texts = [b.text for b in document.blocks if isinstance(b, (ParagraphBlock, HeadingBlock))]
    assert not any(t.strip().isdigit() for t in texts), "bare page numbers should be dropped"
    assert not any("Jev Engineering Working Note" in t for t in texts), "running header should be dropped"


def test_document_round_trips_through_json():
    with open(FIXTURES / "test.pdf", "rb") as f:
        pdf_bytes = f.read()
    document = extract_document(pdf_bytes, source_filename="test.pdf")
    from app.document_model import Document
    round_tripped = Document.model_validate_json(document.model_dump_json())
    assert round_tripped == document


# ------------------------------------------------------------ importance

def test_importance_returns_only_passages_above_cutoff(monkeypatch):
    import app.importance as importance_module
    from app.document_model import DocumentMeta
    from app.importance import MIN_IMPORTANCE_SCORE, score_document
    from app.importance_model import ImportanceSignal

    doc = Document(
        extraction_version="test",
        meta=DocumentMeta(source_filename="x.pdf", page_count=1),
        blocks=[
            HeadingBlock(page=1, text="Basics"),
            ParagraphBlock(page=1, text="Opening paragraph after the heading."),
            ParagraphBlock(page=1, text="A pointer refers to a memory address."),
            HeadingBlock(page=1, text="More"),
            ParagraphBlock(page=1, text="A second useful paragraph."),
            ParagraphBlock(page=1, text="A plain paragraph."),
        ],
    )
    paragraphs = [block for block in doc.blocks if isinstance(block, ParagraphBlock)]
    monkeypatch.setattr(
        importance_module,
        "_textrank",
        lambda items: {
            paragraphs[0].id: 0.0,
            paragraphs[1].id: 0.6,
            paragraphs[2].id: 0.9,
            paragraphs[3].id: 1.0,
        },
    )

    importance = score_document(doc)

    assert MIN_IMPORTANCE_SCORE == 0.7
    assert [passage.block_id for passage in importance.passages] == [paragraphs[2].id, paragraphs[1].id]
    assert [passage.message for passage in importance.passages] == [paragraphs[2].text, paragraphs[1].text]
    assert ImportanceSignal.POSITIONAL in importance.passages[0].signals
    assert ImportanceSignal.DEFINITIONAL in importance.passages[1].signals
    assert all("score" not in passage.model_dump() for passage in importance.passages)


def test_fixture_importance_returns_a_filtered_subset():
    from app.importance import score_document
    with open(FIXTURES / "test.pdf", "rb") as f:
        pdf_bytes = f.read()
    document = extract_document(pdf_bytes, source_filename="test.pdf")
    importance = score_document(document)

    paragraph_ids = {b.id for b in document.blocks if isinstance(b, ParagraphBlock)}
    message_ids = {p.block_id for p in importance.passages}
    assert message_ids < paragraph_ids
    assert importance.passages
    paragraphs_by_id = {b.id: b.text for b in document.blocks if isinstance(b, ParagraphBlock)}
    assert all(p.message == paragraphs_by_id[p.block_id] for p in importance.passages)


def test_definitional_signal_fires_on_definition_sentences(monkeypatch):
    import app.importance as importance_module
    from app.document_model import DocumentMeta
    from app.importance import score_document

    doc = Document(
        extraction_version="test",
        meta=DocumentMeta(source_filename="x.pdf", page_count=1),
        blocks=[
            ParagraphBlock(page=1, text="A pointer refers to a memory address where a value is stored."),
            HeadingBlock(page=1, text="Weather"),
            ParagraphBlock(page=1, text="The weather that day was unusually cold and windy."),
        ],
    )
    paragraphs = [block for block in doc.blocks if isinstance(block, ParagraphBlock)]
    monkeypatch.setattr(
        importance_module,
        "_textrank",
        lambda items: {paragraphs[0].id: 0.6, paragraphs[1].id: 0.9},
    )
    importance = score_document(doc)
    by_id = {passage.block_id: passage for passage in importance.passages}
    def_para, plain_para = paragraphs

    from app.importance_model import ImportanceSignal
    assert ImportanceSignal.DEFINITIONAL in by_id[def_para.id].signals
    assert ImportanceSignal.DEFINITIONAL not in by_id[plain_para.id].signals


def test_importance_handles_empty_document_without_crashing():
    from app.document_model import DocumentMeta
    from app.importance import score_document

    doc = Document(
        extraction_version="test",
        meta=DocumentMeta(source_filename="x.pdf", page_count=1),
        blocks=[HeadingBlock(page=1, text="Just a heading, no paragraphs")],
    )
    importance = score_document(doc)
    assert importance.passages == []


def test_chapter_level_running_headers_are_stripped():
    doc = fitz.open()
    for pno in range(1, 9):
        page = doc.new_page()
        section_header = "1.1 Overview" if pno <= 2 else "1.2 Deep Dive"
        page.insert_text((72, 24), section_header, fontsize=9)
        page.insert_text((72, 120), f"Body paragraph content on page {pno} with normal text.", fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)

    extracted = extract_document(buf.getvalue(), source_filename="running_headers.pdf")
    texts = [b.text for b in extracted.blocks if isinstance(b, (ParagraphBlock, HeadingBlock))]
    assert not any("1.1 Overview" in t for t in texts)
    assert not any("1.2 Deep Dive" in t for t in texts)
    assert len(texts) == 8


def test_everyday_definitional_and_structural_patterns_detected():
    from app.document_model import DocumentMeta
    from app.importance import score_document
    from app.importance_model import ImportanceSignal

    doc = Document(
        extraction_version="test",
        meta=DocumentMeta(source_filename="book.pdf", page_count=1),
        blocks=[
            HeadingBlock(page=1, text="Shaping Work"),
            ParagraphBlock(
                page=1,
                text="We call this the appetite. You can think of the appetite as a fixed time budget for a standard team.",
            ),
            ParagraphBlock(
                page=1,
                text="The pitch consists of five core ingredients that frame the problem and the bounded solution.",
            ),
        ],
    )
    importance = score_document(doc)
    by_id = {p.block_id: p for p in importance.passages}
    assert len(importance.passages) == 2
    assert ImportanceSignal.DEFINITIONAL in by_id[doc.blocks[1].id].signals
    assert ImportanceSignal.POSITIONAL in by_id[doc.blocks[1].id].signals
    assert ImportanceSignal.DEFINITIONAL in by_id[doc.blocks[2].id].signals


def test_positional_signal_survives_intervening_image_under_heading(monkeypatch):
    import app.importance as importance_module
    from app.document_model import DocumentMeta
    from app.importance import score_document
    from app.importance_model import ImportanceSignal

    doc = Document(
        extraction_version="test",
        meta=DocumentMeta(source_filename="x.pdf", page_count=1),
        blocks=[
            HeadingBlock(page=1, text="Architecture"),
            ImageBlock(page=1, asset_url="data:image/png;base64,iVBORw0KGgo="),
            ParagraphBlock(page=1, text="The system processes requests through a stateless pipeline."),
        ],
    )
    para = doc.blocks[2]
    monkeypatch.setattr(importance_module, "_textrank", lambda items: {para.id: 0.9})
    importance = score_document(doc)
    assert len(importance.passages) == 1
    assert ImportanceSignal.POSITIONAL in importance.passages[0].signals

