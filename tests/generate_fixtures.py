"""
Synthetic PDF fixture generator for Extraction Service v2 acceptance tests.
Generates S1 - S6 (and helper for S7).
"""
import sys
from pathlib import Path
import pymupdf as fitz

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "synthetic"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_s1_sideways_0_minus1():
    """S1: portrait page with a sideways-drawn borderless table, rotation flag 0, text direction (0,-1)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    
    # In PyMuPDF, rotate=90 produces line dir (0, -1)
    # Title at x=100
    page.insert_text((100, 200), "Table 1: Sideways Experimental Results", rotate=90, fontsize=12)
    
    # Header row at x=140
    page.insert_text((140, 200), "Treatment", rotate=90, fontsize=10)
    page.insert_text((140, 320), "Baseline", rotate=90, fontsize=10)
    page.insert_text((140, 420), "Score", rotate=90, fontsize=10)
    page.insert_text((140, 500), "p-value", rotate=90, fontsize=10)
    
    # Row 1 at x=180
    page.insert_text((180, 200), "Control Group", rotate=90, fontsize=10)
    page.insert_text((180, 320), "12.45", rotate=90, fontsize=10)
    page.insert_text((180, 420), "15.30", rotate=90, fontsize=10)
    page.insert_text((180, 500), "0.042*", rotate=90, fontsize=10)
    
    # Row 2 at x=220
    page.insert_text((220, 200), "Active AI Tutor", rotate=90, fontsize=10)
    page.insert_text((220, 320), "12.50", rotate=90, fontsize=10)
    page.insert_text((220, 420), "19.85", rotate=90, fontsize=10)
    page.insert_text((220, 500), "0.001***", rotate=90, fontsize=10)
    
    out_path = FIXTURES_DIR / "s1_sideways_0_minus1.pdf"
    doc.save(str(out_path))
    doc.close()
    return out_path


def generate_s2_sideways_0_1():
    """S2: the same with direction (0,1). In PyMuPDF, rotate=270 gives dir (0, 1)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    
    page.insert_text((500, 200), "Table 2: Reverse Sideways Table", rotate=270, fontsize=12)
    
    page.insert_text((460, 200), "Cohort", rotate=270, fontsize=10)
    page.insert_text((460, 320), "Enrollment", rotate=270, fontsize=10)
    page.insert_text((460, 440), "Retention Rate", rotate=270, fontsize=10)
    
    page.insert_text((420, 200), "Fall 2025", rotate=270, fontsize=10)
    page.insert_text((420, 320), "1,240", rotate=270, fontsize=10)
    page.insert_text((420, 440), "88.4%", rotate=270, fontsize=10)
    
    page.insert_text((380, 200), "Spring 2026", rotate=270, fontsize=10)
    page.insert_text((380, 320), "1,310", rotate=270, fontsize=10)
    page.insert_text((380, 440), "92.1%", rotate=270, fontsize=10)
    
    out_path = FIXTURES_DIR / "s2_sideways_0_1.pdf"
    doc.save(str(out_path))
    doc.close()
    return out_path


def generate_s3_upside_down():
    """S3: an upside-down page. In PyMuPDF, rotate=180 gives dir (-1, 0)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    
    page.insert_text((500, 700), "Upside Down Heading", rotate=180, fontsize=16)
    page.insert_text((500, 650), "This entire paragraph was printed upside down on the page.", rotate=180, fontsize=11)
    page.insert_text((500, 630), "Normalization should recognize the dominant 180 orientation and flip it.", rotate=180, fontsize=11)
    
    out_path = FIXTURES_DIR / "s3_upside_down.pdf"
    doc.save(str(out_path))
    doc.close()
    return out_path


def generate_s4_appendix_captions():
    """S4: appendix captions (Table A1, Figure S2, Exhibit IV, Fig. 3)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    
    y = 50
    page.insert_text((50, y), "Appendix A: Supplementary Analysis", fontsize=16)
    
    y += 50
    page.insert_text((50, y), "Table A1: Summary Statistics for Appendix Cohort", fontsize=11)
    y += 20
    # draw simple grid
    page.draw_rect(fitz.Rect(50, y, 400, y + 60))
    page.draw_line(fitz.Point(50, y + 20), fitz.Point(400, y + 20))
    page.draw_line(fitz.Point(200, y), fitz.Point(200, y + 60))
    page.insert_text((60, y + 15), "Metric", fontsize=10)
    page.insert_text((210, y + 15), "Value", fontsize=10)
    page.insert_text((60, y + 45), "Observations", fontsize=10)
    page.insert_text((210, y + 45), "5,432", fontsize=10)
    
    y += 100
    page.insert_text((50, y), "Figure S2: Robustness checks across subsamples.", fontsize=11)
    y += 20
    page.draw_rect(fitz.Rect(50, y, 250, y + 100), color=(0.5, 0.5, 0.5), fill=(0.9, 0.9, 0.9))
    
    y += 140
    page.insert_text((50, y), "Exhibit IV: Timeline of Experimental Interventions", fontsize=11)
    y += 20
    page.draw_rect(fitz.Rect(50, y, 400, y + 50))
    page.insert_text((60, y + 30), "Phase 1: Diagnostic Assessment", fontsize=10)
    
    out_path = FIXTURES_DIR / "s4_appendix_captions.pdf"
    doc.save(str(out_path))
    doc.close()
    return out_path


def generate_s5_hyphenation():
    """S5: paragraphs with line-end hyphens, both broken words and true compounds."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    
    page.insert_text((50, 50), "Hyphenation Test Cases", fontsize=14)
    
    # We want line 1 to end with 'stu-' and line 2 to start with 'dents'
    # And we also have the unhyphenated word 'students' later in the document vocabulary!
    page.insert_text((50, 90), "In this randomized trial, the participating stu-", fontsize=11)
    page.insert_text((50, 105), "dents showed remarkable progress throughout the semester.", fontsize=11)
    
    # Another broken word: 'litera-' / 'ture' with 'literature' later
    page.insert_text((50, 130), "Our findings contribute to the broader litera-", fontsize=11)
    page.insert_text((50, 145), "ture on educational interventions and student learning.", fontsize=11)
    
    # Genuine compound words that must NOT be joined into one word:
    page.insert_text((50, 170), "We developed a state-of-the-art computer-assisted learning platform.", fontsize=11)
    page.insert_text((50, 190), "The mastery threshold required three-correct-in-a-row before proceeding.", fontsize=11)
    
    # Natural occurrence of the vocabulary words so dehyphenation dictionary knows they are single words:
    page.insert_text((50, 220), "All students benefited from the literature on active study methods.", fontsize=11)
    
    out_path = FIXTURES_DIR / "s5_hyphenation.pdf"
    doc.save(str(out_path))
    doc.close()
    return out_path


def generate_s6_references():
    """S6: a reference list under a References heading."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    
    page.insert_text((50, 50), "6. Conclusion", fontsize=14)
    page.insert_text((50, 75), "In conclusion, our results demonstrate clear gains.", fontsize=11)
    
    page.insert_text((50, 120), "References", fontsize=14)
    
    ref1_a = "Angrist, Joshua D., and Jorn-Steffen Pischke, Mostly Harmless Econometrics: An Empiricist's"
    ref1_b = "Companion, Princeton University Press, 2009."
    page.insert_text((50, 150), ref1_a, fontsize=10)
    page.insert_text((70, 165), ref1_b, fontsize=10)
    
    ref2_a = "Card, David, 'The Causal Effect of Education on Earnings,' Handbook of Labor Economics,"
    ref2_b = "1999, 3, 1801-1863."
    page.insert_text((50, 195), ref2_a, fontsize=10)
    page.insert_text((70, 210), ref2_b, fontsize=10)
    
    ref3_a = "Duflo, Esther, 'Schooling and Labor Market Consequences of School Construction in Indonesia,'"
    ref3_b = "American Economic Review, 2001, 91 (4), 795-813."
    page.insert_text((50, 240), ref3_a, fontsize=10)
    page.insert_text((70, 255), ref3_b, fontsize=10)
    
    out_path = FIXTURES_DIR / "s6_references.pdf"
    doc.save(str(out_path))
    doc.close()
    return out_path


def generate_all():
    print("Generating synthetic test fixtures...")
    generate_s1_sideways_0_minus1()
    generate_s2_sideways_0_1()
    generate_s3_upside_down()
    generate_s4_appendix_captions()
    generate_s5_hyphenation()
    generate_s6_references()
    print("Done! Fixtures written to:", FIXTURES_DIR)


if __name__ == "__main__":
    generate_all()
