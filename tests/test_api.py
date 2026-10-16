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