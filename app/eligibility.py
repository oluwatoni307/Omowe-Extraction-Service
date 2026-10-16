"""
Pre-flight eligibility gate.

single-voice documents (academic papers, plain-typeset non-fiction
Runs BEFORE block extraction. Checks whether sampled pages contain
enough extractable text to process. Font variety and visual layout do
not determine eligibility; extraction attempts to preserve reading
sequence after the document passes this gate.

This checks for extractable text and rejects documents whose pages are
mostly full-page raster images, including scans with an OCR text overlay.
It does not guarantee complete or correctly ordered extraction.
"""
from __future__ import annotations

import pymupdf as fitz

from app.document_model import EligibilityResult, RejectionReason

# A page with an image covering most of its area is likely a scanned page.
FULL_PAGE_IMAGE_COVERAGE = 0.7
SCANNED_PAGE_FRACTION = 0.5


def check_eligibility(doc: fitz.Document) -> EligibilityResult:
    if not _has_text_layer(doc):
        return EligibilityResult(
            eligible=False,
            reasons=[RejectionReason.NO_TEXT_LAYER],
        )
    if _is_mostly_scanned(doc):
        return EligibilityResult(
            eligible=False,
            reasons=[RejectionReason.SCANNED_DOCUMENT],
        )
    return EligibilityResult(eligible=True)


def _has_text_layer(doc: fitz.Document) -> bool:
    for page in doc:
        if page.get_text().strip():
            return True
    return False


def _is_mostly_scanned(doc: fitz.Document) -> bool:
    scanned_pages = 0
    for page in doc:
        page_area = page.rect.width * page.rect.height
        has_full_page_image = any(
            rect.width * rect.height >= FULL_PAGE_IMAGE_COVERAGE * page_area
            for image in page.get_images(full=True)
            for rect in page.get_image_rects(image[0])
        )
        scanned_pages += has_full_page_image
    return scanned_pages / len(doc) >= SCANNED_PAGE_FRACTION


