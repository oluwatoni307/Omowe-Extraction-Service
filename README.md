# PDF Extraction Service

A lightweight, stateless Python microservice that extracts born-digital PDFs into validated, structured `Document` models and identifies key passage importance.

---

## 1. Overview & Architectural Role

```
   [ Client / Processing Service ]
                 │
                 │ POST /extract (Multipart PDF <= 20MB)
                 ▼
     ┌───────────────────────┐
     │  FastAPI Application  │
     │  (Extraction Service) │
     ├───────────────────────┤
     │ • Eligibility Check   │ (Rejects scans & image-only PDFs)
     │ • PyMuPDF Extractor   │ (Headings, paragraphs, tables, images, code/math)
     │ • Passage Importance  │ (Heuristic scoring for definitions & key concepts)
     └───────────────────────┘
                 │
                 │ JSON { eligible: true, document: Document, importance: ... }
                 ▼
```

### Key Principles
- **100% Stateless:** Nothing is persisted to disk. Image and formula assets are cropped and returned as base64 data URIs directly within blocks (`ImageBlock.asset_url`, `CodeBlock.asset_url`). The client or downstream processing service persists assets locally.
- **AGPL-3.0 Isolation Boundary:** PyMuPDF is AGPL-3.0. Isolating all PDF parsing into this standalone microservice prevents AGPL-3.0 infection of your downstream processing service, mobile client, and business logic.
- **CPU-Bound Thread Pool Execution:** The `/extract` route handler is deliberately declared as synchronous (`def extract` rather than `async def`). FastAPI automatically offloads synchronous handlers to Starlette's worker thread pool, preventing CPU-heavy PDF parsing from blocking the ASGI event loop or stalling `/health` probes.

---

## 2. Server Architecture: Pure Uvicorn

FastAPI is an ASGI framework and requires an ASGI server to execute. We use **pure Uvicorn** with multi-process workers:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-2}
```

- **Local Development:** Runs with `--reload` for immediate live code updating.
- **Production (Render / Docker):** Runs with `--workers 2` (or configured via `$WORKERS`) and dynamically binds to Render's assigned `$PORT`. Render automatically manages container health and HTTPS termination.

---

## 3. API Specification

### `GET /`
Returns service metadata and navigation links.
```bash
curl http://localhost:8000/
```
```json
{
  "service": "PDF Extraction Service",
  "status": "ok",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

### `GET /health`
Liveness and readiness probe for container orchestrators (Kubernetes, Cloud Run, AWS ECS).
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok"
}
```

### `POST /extract`
Uploads and extracts a PDF document.
- **Content-Type:** `multipart/form-data`
- **Field Name:** `file`
- **Max File Size:** `20 MB`

#### Example Request:
```bash
curl -X POST "http://localhost:8000/extract" \
     -H "accept: application/json" \
     -F "file=@sample_book.pdf"
```

#### Successful Response (`200 OK - Eligible`):
```json
{
  "eligible": true,
  "document": {
    "schema_version": 1,
    "extraction_version": "v1",
    "document_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "meta": {
      "title": "Computer Systems: A Programmer's Perspective",
      "source_filename": "sample_book.pdf",
      "page_count": 142,
      "body_font_size": 10.5
    },
    "blocks": [
      {
        "type": "heading",
        "id": "h_p1_0",
        "page": 1,
        "text": "1 A Tour of Computer Systems",
        "level": 1
      },
      {
        "type": "paragraph",
        "id": "p_p1_1",
        "page": 1,
        "text": "A computer system consists of hardware and systems software..."
      }
    ]
  },
  "importance": {
    "schema_version": 1,
    "document_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "paragraph_scores": [
      {
        "block_id": "p_p1_1",
        "score": 0.88,
        "signals": ["centrality", "definitional"]
      }
    ]
  },
  "reasons": null
}
```

#### Rejection Response (`200 OK - Ineligible / Scanned`):
```json
{
  "eligible": false,
  "document": null,
  "importance": null,
  "reasons": ["scanned_document"]
}
```

#### Error Codes:
- `400 Bad Request`: Non-PDF file, empty file, password-protected/encrypted PDF, or corrupt binary.
- `413 Content Too Large`: PDF file exceeds the 20 MB size limit.
- `500 Internal Server Error`: Unexpected internal extraction failure.

---

## 4. Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | `8000` | Port for the HTTP server to bind (automatically populated by Cloud Run, Render, Heroku, Railway). |
| `WORKERS` | `2` | Number of Gunicorn worker processes (adjust based on CPU cores, e.g. `2 * CPU + 1`). |
| `PYTHONUNBUFFERED` | `1` | Ensures Python logs are flushed to standard output immediately. |

---

## 5. Local Setup & Testing

### 1. Install Dependencies
```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Run Locally (Development)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI Swagger docs will be accessible at: `http://localhost:8000/docs`

### 3. Run Test Suite
```bash
pytest tests/ -v
```

---

## 6. Docker & Container Deployment

### Local Docker Build & Run
```bash
# Build the container
docker build -t extraction-service .

# Run container mapping port 8000
docker run -p 8000:8000 extraction-service
```

Verify the running container:
```bash
curl http://localhost:8000/health
```

---

## 7. Cloud Deployment: Render

Render provides managed container hosting with automated zero-downtime deploys, automated SSL certificates, and integrated DDoS protection.

### Method 1: Render Blueprint (`render.yaml` - Recommended)
1. Push this repository to GitHub or GitLab.
2. Log into [Render Dashboard](https://dashboard.render.com).
3. Click **New +** -> **Blueprint**.
4. Select your `extraction_service` repository.
5. Render reads `render.yaml` and deploys the service automatically.

### Method 2: Manual Web Service
1. In Render Dashboard, click **New +** -> **Web Service**.
2. Connect your repository.
3. Configure settings:
   - **Name:** `pdf-extraction-service`
   - **Language / Runtime:** `Docker`
   - **Instance Type:** `Starter` (or `Free` for testing)
   - **Health Check Path:** `/health`
4. Click **Create Web Service**.

Once deployed, Render assigns an HTTPS domain (e.g. `https://pdf-extraction-service-xxxx.onrender.com`).
Verify by visiting:
```bash
curl https://<your-render-subdomain>.onrender.com/health
```

---

### Alternative Deployment Options

<details>
<summary>Google Cloud Run</summary>

```bash
gcloud run deploy pdf-extraction-service \
  --source . \
  --region us-central1 \
  --platform managed \
  --memory 1Gi \
  --cpu 1 \
  --allow-unauthenticated
```
</details>

<details>
<summary>Railway</summary>

```bash
railway init
railway up
```
</details>

---

## 8. License

PyMuPDF is licensed under GNU AGPL-3.0. This repository is published as an isolated microservice in accordance with the AGPL-3.0 requirements.
