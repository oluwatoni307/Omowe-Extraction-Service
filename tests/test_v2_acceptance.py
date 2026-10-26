"""
Acceptance test suite for Extraction Service v2.
"""
from pathlib import Path
import pytest
import pymupdf as fitz

from app.document_model import (
    BlockType,
    HeadingBlock,
    ParagraphBlock,
    ReferenceBlock,
    TableBlock,
    ImageBlock,
)
from app.extractor import extract_document
from app.importance import score_document

SYNTHETIC_DIR = Path(__file__).resolve().parent / "fixtures" / "synthetic"
CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


def test_s1_sideways_0_minus1():
    pdf_path = SYNTHETIC_DIR / "s1_sideways_0_minus1.pdf"
    doc = extract_document(pdf_path.read_bytes(), "s1.pdf")
    
    # Table should be found
    tables = [b for b in doc.blocks if b.type == BlockType.TABLE]
    assert len(tables) >= 1, f"Expected table block on sideways page, got {len(tables)}"
    t = tables[0]
    assert len(t.rows) >= 2, f"Expected >= 2 rows, got {len(t.rows)}"
    # Verify table caption is attached
    assert t.caption is not None and "Table 1" in t.caption
    # Check that quality reports rotation
    if doc.quality:
        assert doc.quality.pages[0].rotation_applied != 0


def test_s2_sideways_0_1():
    pdf_path = SYNTHETIC_DIR / "s2_sideways_0_1.pdf"
    doc = extract_document(pdf_path.read_bytes(), "s2.pdf")
    
    tables = [b for b in doc.blocks if b.type == BlockType.TABLE]
    assert len(tables) >= 1, f"Expected table block on reverse sideways page, got {len(tables)}"
    t = tables[0]
    assert len(t.rows) >= 2
    assert t.caption is not None and "Table 2" in t.caption


def test_s3_upside_down():
    pdf_path = SYNTHETIC_DIR / "s3_upside_down.pdf"
    doc = extract_document(pdf_path.read_bytes(), "s3.pdf")
    
    # Upside down heading and text should be extracted upright
    texts = [getattr(b, "text", "") for b in doc.blocks]
    full_text = " ".join(texts)
    assert "Upside Down Heading" in full_text
    assert "printed upside down" in full_text


def test_s4_appendix_captions():
    pdf_path = SYNTHETIC_DIR / "s4_appendix_captions.pdf"
    doc = extract_document(pdf_path.read_bytes(), "s4.pdf")
    
    captions = [b.caption for b in doc.blocks if getattr(b, "caption", None)]
    captions_text = " ".join(captions)
    assert "Table A1" in captions_text
    assert "Figure S2" in captions_text
    assert "Exhibit IV" in captions_text


def test_s5_hyphenation():
    pdf_path = SYNTHETIC_DIR / "s5_hyphenation.pdf"
    doc = extract_document(pdf_path.read_bytes(), "s5.pdf")
    
    paragraphs = [b for b in doc.blocks if b.type in (BlockType.PARAGRAPH, BlockType.HEADING)]
    all_text = " ".join(b.text for b in paragraphs)
    
    # Broken words should be joined without hyphens:
    assert "students" in all_text
    assert "stu-dents" not in all_text
    assert "literature" in all_text
    assert "litera-ture" not in all_text
    
    # Genuine compounds should remain intact:
    assert "computer-assisted" in all_text
    assert "three-correct-in-a-row" in all_text


def test_s6_references():
    pdf_path = SYNTHETIC_DIR / "s6_references.pdf"
    doc = extract_document(pdf_path.read_bytes(), "s6.pdf")
    
    # Citations after 'References' must NOT be HeadingBlock
    headings = [b for b in doc.blocks if b.type == BlockType.HEADING]
    heading_texts = [h.text for h in headings]
    
    assert "References" in heading_texts or "6. Conclusion" in heading_texts
    # Authors must NOT be in headings
    assert not any("Angrist" in h for h in heading_texts)
    assert not any("Card" in h for h in heading_texts)
    assert not any("Duflo" in h for h in heading_texts)
    
    # Citations should be ReferenceBlock or role='reference'
    ref_blocks = [b for b in doc.blocks if b.type == BlockType.REFERENCE or getattr(b, "role", None) == "reference"]
    assert len(ref_blocks) >= 3, f"Expected 3 reference blocks, got {len(ref_blocks)}"


def test_g2_table_4():
    """Acceptance test G2: Table 4 on page 37 of ai26-1552.pdf."""
    pdf_path = CORPUS_DIR / "ai26-1552.pdf"
    if not pdf_path.exists():
        pytest.skip("ai26-1552.pdf not present in corpus")
        
    doc = extract_document(pdf_path.read_bytes(), "ai26-1552.pdf")
    p37_tables = [b for b in doc.blocks if b.page == 37 and b.type == BlockType.TABLE]
    assert len(p37_tables) >= 1, "Expected Table 4 on page 37"
    
    t4 = p37_tables[0]
    assert len(t4.rows) == 19, f"Expected 19 rows, got {len(t4.rows)}"
    assert len(t4.rows[0]) == 5, f"Expected 5 columns, got {len(t4.rows[0])}"
    
    # Check header
    header = t4.rows[0]
    assert "Questions Completed" in header[1]
    assert "Number Correct" in header[2]
    assert "3 Correct in a Row" in header[3]
    assert "Exercise Time" in header[4]
    
    # Check row values
    row_texts = [" ".join(r) for r in t4.rows]
    full_table_text = "\n".join(row_texts)
    
    # Assert Mastery row values
    assert "4.615***" in full_table_text
    assert "1.232***" in full_table_text
    assert "0.287***" in full_table_text
    assert "6.692***" in full_table_text
    assert "(0.255)" in full_table_text
    assert "(0.095)" in full_table_text
    assert "(0.016)" in full_table_text
    assert "(0.444)" in full_table_text
    
    # Assert AI row values
    assert "-1.042***" in full_table_text
    assert "-0.161*" in full_table_text
    assert "-0.021" in full_table_text
    assert "1.643***" in full_table_text
    
    # Assert Observations
    assert "6,997" in full_table_text
    
    # Assert notes attached
    assert len(t4.notes) >= 1


def test_g1_reference_paper():
    """Acceptance test G1: Full 73-page reference working paper ai26-1552.pdf."""
    pdf_path = CORPUS_DIR / "ai26-1552.pdf"
    if not pdf_path.exists():
        pytest.skip("ai26-1552.pdf not present in corpus")
        
    doc = extract_document(pdf_path.read_bytes(), "ai26-1552.pdf")
    imp = score_document(doc)
    
    # 1. 15 sideways pages normalized
    assert doc.quality is not None
    rotated_pages = [p for p in doc.quality.pages if p.rotation_applied != 0]
    assert len(rotated_pages) == 15, f"Expected 15 rotated pages, got {len(rotated_pages)}"
    
    # 2. Table blocks found across the document
    tables = [b for b in doc.blocks if b.type == BlockType.TABLE]
    assert len(tables) >= 20, f"Expected >= 20 tables, got {len(tables)}"
    
    # 3. Headings in References section (pages 28-32)
    headings_in_ref = [b for b in doc.blocks if 28 <= b.page <= 32 and b.type == BlockType.HEADING]
    # Exactly the 'References' heading itself, no citations misclassified as headings
    assert len(headings_in_ref) == 1 and headings_in_ref[0].text == "References"
    
    # 4. Zero important passages in References section (pages 28-32)
    imp_in_ref = [p for p in imp.passages if 28 <= doc.get(p.block_id).page <= 32]
    assert len(imp_in_ref) == 0, f"Expected 0 importance passages in references, got {len(imp_in_ref)}"

    # 5. Figure checks: Figure 1 native high-res, Figure A1 has 4 panels, Figure A2 has 3 panels, page 1 logo decorative
    fig1 = [b for b in doc.blocks if b.page == 36 and b.type == BlockType.IMAGE]
    assert len(fig1) >= 1 and "Figure 1" in (fig1[0].caption or "")
    
    fig_a1 = [b for b in doc.blocks if b.page == 46 and b.type == BlockType.IMAGE and b.caption and "Figure A1" in b.caption]
    assert len(fig_a1) >= 1, "Expected Figure A1 on page 46"
    assert len(fig_a1[0].panels) == 4, f"Expected 4 panels for Figure A1, got {len(fig_a1[0].panels)}"

    fig_a2 = [b for b in doc.blocks if b.page == 47 and b.type == BlockType.IMAGE and b.caption and "Figure A2" in b.caption]
    assert len(fig_a2) >= 1, "Expected Figure A2 on page 47"
    assert len(fig_a2[0].panels) == 3, f"Expected 3 panels for Figure A2, got {len(fig_a2[0].panels)}"

    p1_imgs = [b for b in doc.blocks if b.page == 1 and b.type == BlockType.IMAGE]
    assert len(p1_imgs) >= 1 and p1_imgs[0].decorative is True, "Expected page 1 logo marked decorative"

