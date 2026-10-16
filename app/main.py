"""
Extraction service — the only entry point.

    POST /extract   (multipart file upload, a PDF)
        -> 200 { "eligible": true, "document": Document }
        -> 200 { "eligible": false, "reasons": [...] }
        -> 400 on a non-PDF / unreadable file
        -> 500 on an unexpected extraction failure

No auth, no persistence, no other endpoints. Stateless by design —
this service holds nothing between requests. See README.md.
"""
from __future__ import annotations

import logging

import pymupdf as fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.document_model import Document, RejectionReason
from app.eligibility import check_eligibility
from app.extractor import extract_document
from app.importance import score_document
from app.importance_model import DocumentImportance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extraction_service")

app = FastAPI(title="PDF Extraction Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractResponse(BaseModel):
    eligible: bool
    document: Document | None = None
    importance: DocumentImportance | None = None
    reasons: list[RejectionReason] | None = None


@app.get("/")
def root():
    return {
        "service": "PDF Extraction Service",
        "status": "ok",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB upload limit


@app.post("/extract", response_model=ExtractResponse)
def extract(file: UploadFile = File(...)):
    # Deliberately `def`, not `async def`: everything below (PyMuPDF parsing,
    # table/image detection, pixmap rendering) is synchronous, CPU-bound work.
    # FastAPI runs sync path-operation functions in a worker thread pool
    # automatically — an `async def` here would block the whole event loop,
    # including /health, for the entire extraction (measured: ~10s for a
    # 120-page book with no images; longer with images present).
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    pdf_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")

    try:
        fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open PDF: {e}") from e

    with fitz_doc:
        if fitz_doc.is_encrypted:
            raise HTTPException(status_code=400, detail="Encrypted/password-protected PDFs are not supported")

        try:
            eligibility = check_eligibility(fitz_doc)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not open PDF: {e}") from e

        if not eligibility.eligible:
            logger.info("rejected %s: %s", file.filename, eligibility.reasons)
            return ExtractResponse(eligible=False, reasons=eligibility.reasons)

        try:
            document = extract_document(fitz_doc, source_filename=file.filename or "unknown.pdf")
            importance = score_document(document)
        except Exception as e:
            logger.exception("extraction failed for %s", file.filename)
            raise HTTPException(status_code=500, detail=f"Extraction failed: {e}") from e

    logger.info("extracted %s: %d blocks", file.filename, len(document.blocks))
    return ExtractResponse(eligible=True, document=document, importance=importance)
