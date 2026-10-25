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
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import pymupdf as fitz
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.document_model import Document, QualityReport, RejectionReason
from app.eligibility import check_eligibility
from app.extractor import extract_document
from app.importance import score_document
from app.importance_model import DocumentImportance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extraction_service")

app = FastAPI(title="PDF Extraction Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractResponse(BaseModel):
    eligible: bool
    document: Document | None = None
    importance: DocumentImportance | None = None
    quality: QualityReport | None = None
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


ASSET_DIR = Path(tempfile.gettempdir()) / "omowe_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)
MAX_CACHE_BYTES = 100 * 1024 * 1024  # 100 MB max disk cache
ASSET_TTL_SECONDS = 3600  # 1 hour TTL


def _evict_old_assets() -> None:
    """Evicts asset directories older than ASSET_TTL_SECONDS or if cache exceeds MAX_CACHE_BYTES."""
    if not ASSET_DIR.exists():
        return
    now = time.time()
    try:
        doc_dirs = [d for d in ASSET_DIR.iterdir() if d.is_dir()]
        total_bytes = 0
        dir_stats = []

        for d in doc_dirs:
            png_files = list(d.glob("*.png"))
            d_size = sum(f.stat().st_size for f in png_files if f.is_file())
            mtime = d.stat().st_mtime
            # Evict if older than TTL
            if now - mtime > ASSET_TTL_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
            else:
                total_bytes += d_size
                dir_stats.append((mtime, d_size, d))

        # If still over size budget, remove oldest directories
        if total_bytes > MAX_CACHE_BYTES:
            dir_stats.sort(key=lambda x: x[0])  # oldest first
            for mtime, d_size, d in dir_stats:
                shutil.rmtree(d, ignore_errors=True)
                total_bytes -= d_size
                if total_bytes <= MAX_CACHE_BYTES:
                    break
    except Exception as e:
        logger.warning("Error during asset cache eviction: %s", e)


def store_asset(doc_id: str, asset_id: str, data: bytes) -> str:
    _evict_old_assets()
    target_dir = ASSET_DIR / doc_id
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{asset_id}.png"
    file_path.write_bytes(data)
    return f"/assets/{doc_id}/{asset_id}"


def get_asset(doc_id: str, asset_id: str) -> bytes | None:
    file_path = ASSET_DIR / doc_id / f"{asset_id}.png"
    if not file_path.is_file():
        return None
    # Check TTL
    if time.time() - file_path.stat().st_mtime > ASSET_TTL_SECONDS:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    try:
        return file_path.read_bytes()
    except OSError:
        return None


def clear_asset_cache() -> None:
    if ASSET_DIR.exists():
        for d in ASSET_DIR.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
            elif d.is_file():
                try:
                    d.unlink(missing_ok=True)
                except OSError:
                    pass


@app.get("/assets/{doc_id}/{asset_id}")
def get_asset_endpoint(doc_id: str, asset_id: str):
    data = get_asset(doc_id, asset_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return Response(content=data, media_type="image/png")


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
            doc_id = uuid.uuid4().hex[:12]

            def asset_collector(asset_id: str, png_bytes: bytes) -> str:
                return store_asset(doc_id, asset_id, png_bytes)

            document = extract_document(
                fitz_doc,
                source_filename=file.filename or "unknown.pdf",
                document_id=doc_id,
                asset_collector=asset_collector,
            )
            importance = score_document(document)
        except Exception as e:
            logger.exception("extraction failed for %s", file.filename)
            raise HTTPException(status_code=500, detail=f"Extraction failed: {e}") from e

    logger.info("extracted %s: %d blocks", file.filename, len(document.blocks))
    return ExtractResponse(eligible=True, document=document, importance=importance, quality=document.quality)
