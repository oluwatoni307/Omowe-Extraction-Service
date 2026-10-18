from pathlib import Path

from fastapi.testclient import TestClient

from app.main import MAX_UPLOAD_BYTES, app

FIXTURE = Path(__file__).parent / "fixtures" / "test.pdf"


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "PDF Extraction Service"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_accepts_pdf_at_upload_limit():
    assert MAX_UPLOAD_BYTES == 20 * 1024 * 1024
    pdf_bytes = FIXTURE.read_bytes()
    assert len(pdf_bytes) < MAX_UPLOAD_BYTES
    pdf_bytes += b"\0" * (MAX_UPLOAD_BYTES - len(pdf_bytes))

    with TestClient(app) as client:
        response = client.post(
            "/extract",
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["eligible"] is True


def test_extract_rejects_upload_one_byte_over_limit():
    pdf_bytes = b"x" * (MAX_UPLOAD_BYTES + 1)

    with TestClient(app) as client:
        response = client.post(
            "/extract",
            files={"file": ("oversized.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "File exceeds 20 MB limit"


def test_get_asset_not_found():
    with TestClient(app) as client:
        response = client.get("/assets/doc123/asset456")
    assert response.status_code == 404
    assert response.json()["detail"] == "Asset not found"


def test_store_and_get_asset():
    import pymupdf as fitz
    from app.main import clear_asset_cache, store_asset

    clear_asset_cache()
    # Generate real valid PNG bytes using PyMuPDF
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), False)
    pix.set_rect(pix.irect, (200, 100, 50))
    real_png = pix.tobytes("png")
    assert real_png.startswith(b"\x89PNG\r\n\x1a\n")

    url = store_asset("docA", "assetB", real_png)
    assert url == "/assets/docA/assetB"

    with TestClient(app) as client:
        response = client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == real_png


def test_asset_storage_eviction_bounded_size():
    import os
    import time
    from app.main import ASSET_DIR, clear_asset_cache, get_asset, store_asset, _evict_old_assets

    clear_asset_cache()
    dummy_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024

    # Store multiple assets
    store_asset("doc1", "img1", dummy_png)
    store_asset("doc2", "img2", dummy_png)
    assert get_asset("doc1", "img1") is not None
    assert get_asset("doc2", "img2") is not None

    # Test TTL eviction: artificially age doc1
    doc1_dir = ASSET_DIR / "doc1"
    old_time = time.time() - 4000  # older than 3600s
    os.utime(doc1_dir, (old_time, old_time))
    for f in doc1_dir.glob("*.png"):
        os.utime(f, (old_time, old_time))

    _evict_old_assets()
    assert get_asset("doc1", "img1") is None
    assert get_asset("doc2", "img2") is not None
    clear_asset_cache()


def test_extract_creates_ephemeral_assets():
    import io
    import pymupdf as fitz
    from app.main import clear_asset_cache

    clear_asset_cache()

    # Build a small eligible PDF in memory containing an image and text
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Text on page with image.")
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 80, 80), False)
    pix.set_rect(pix.irect, (100, 150, 200))
    page.insert_image(fitz.Rect(100, 100, 200, 200), pixmap=pix)

    buf = io.BytesIO()
    doc.save(buf)
    pdf_bytes = buf.getvalue()

    with TestClient(app) as client:
        response = client.post(
            "/extract",
            files={"file": ("has_image.pdf", pdf_bytes, "application/pdf")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["eligible"] is True
    doc_data = data["document"]
    assert doc_data is not None

    image_blocks = [b for b in doc_data["blocks"] if b["type"] == "image"]
    if image_blocks:
        asset_url = image_blocks[0]["asset_url"]
        assert asset_url.startswith("/assets/"), f"expected ephemeral URL, got {asset_url}"
        with TestClient(app) as client:
            asset_res = client.get(asset_url)
        assert asset_res.status_code == 200
        assert asset_res.headers["content-type"] == "image/png"
        assert asset_res.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_extract_real_ground_truth_pdf():
    from pathlib import Path
    corpus_pdf = Path(__file__).resolve().parent.parent.parent / "under-20mb" / "Attention-Is-All-You-Need.pdf"
    if not corpus_pdf.exists():
        return

    pdf_bytes = corpus_pdf.read_bytes()
    with TestClient(app) as client:
        response = client.post(
            "/extract",
            files={"file": ("Attention-Is-All-You-Need.pdf", pdf_bytes, "application/pdf")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["eligible"] is True
    doc = data["document"]
    assert doc is not None
    assert len(doc["blocks"]) > 100

    # Ensure payload is light (< 150 KB) without inline base64 bloat
    payload_size_kb = len(response.content) / 1024.0
    assert payload_size_kb < 150.0, f"Payload should be < 150 KB, got {payload_size_kb:.1f} KB"

    # Verify concepts are substantive and free of pronouns or noise headings
    importance = data.get("importance")
    assert importance is not None
    concepts = importance.get("concepts", [])
    terms = [c["term"].lower() for c in concepts]
    assert "this" not in terms
    assert "it" not in terms
    assert "background" not in terms
    assert "related work" not in terms

    # Verify ephemeral assets resolve to valid PNGs
    image_blocks = [b for b in doc["blocks"] if b["type"] == "image"]
    for img in image_blocks[:3]:
        asset_url = img["asset_url"]
        assert asset_url.startswith("/assets/")
        with TestClient(app) as client:
            asset_res = client.get(asset_url)
        assert asset_res.status_code == 200
        assert asset_res.content.startswith(b"\x89PNG\r\n\x1a\n")