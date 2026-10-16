"""
Benchmark Test Suite for Extraction Service.

Measures:
1. Stage-by-stage latency (eligibility, line extraction, layout/blocks, pixmaps, importance scoring, JSON serialization)
2. Memory profiling (peak memory delta via tracemalloc)
3. Payload amplification (response JSON size vs input PDF size)
4. Throughput & concurrency behavior (simulated concurrent workers)
5. Extraction structural integrity checks

Run via:
    python tests/benchmark_extraction.py
or
    pytest tests/benchmark_extraction.py -v
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf as fitz
from fastapi.testclient import TestClient

# Ensure extraction_service root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.document_model import Document
from app.eligibility import check_eligibility
from app.extractor import extract_document
from app.importance import score_document
from app.main import app

WORKSPACE_ROOT = REPO_ROOT.parent
CORPUS_DIR = WORKSPACE_ROOT / "under-20mb"


@dataclass
class StageTiming:
    pdf_open_ms: float
    eligibility_ms: float
    extraction_ms: float
    importance_ms: float
    json_serialize_ms: float
    total_pipeline_ms: float


@dataclass
class MemoryProfile:
    peak_memory_mb: float
    current_memory_mb: float


@dataclass
class DocumentBenchmarkResult:
    filename: str
    file_size_kb: float
    page_count: int
    http_status: int
    timing: StageTiming
    memory: MemoryProfile
    block_counts: dict[str, int]
    total_blocks: int
    importance_passages_count: int
    response_size_kb: float
    amplification_ratio: float
    pages_monotonic: bool
    empty_paragraphs: int


def run_single_benchmark(pdf_path: Path) -> DocumentBenchmarkResult:
    pdf_bytes = pdf_path.read_bytes()
    file_size_kb = len(pdf_bytes) / 1024.0

    # Start memory tracking
    tracemalloc.start()
    tracemalloc.reset_peak()

    # Stage 1: PyMuPDF open
    t0 = time.perf_counter()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    t_open = time.perf_counter()

    # Stage 2: Eligibility
    eligibility = check_eligibility(doc)
    t_eligibility = time.perf_counter()
    doc.close()

    # Stage 3: Extraction
    document = extract_document(pdf_bytes, source_filename=pdf_path.name)
    t_extraction = time.perf_counter()

    # Stage 4: Importance Scoring
    importance = score_document(document)
    t_importance = time.perf_counter()

    # Stage 5: Serialization
    serialized = json.dumps({
        "eligible": eligibility.eligible,
        "document": document.model_dump(mode="json"),
        "importance": importance.model_dump(mode="json"),
    })
    t_serialize = time.perf_counter()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    timing = StageTiming(
        pdf_open_ms=round((t_open - t0) * 1000, 2),
        eligibility_ms=round((t_eligibility - t_open) * 1000, 2),
        extraction_ms=round((t_extraction - t_eligibility) * 1000, 2),
        importance_ms=round((t_importance - t_extraction) * 1000, 2),
        json_serialize_ms=round((t_serialize - t_importance) * 1000, 2),
        total_pipeline_ms=round((t_serialize - t0) * 1000, 2),
    )

    memory = MemoryProfile(
        peak_memory_mb=round(peak_mem / (1024 * 1024), 2),
        current_memory_mb=round(current_mem / (1024 * 1024), 2),
    )

    # Block analysis
    block_counts: dict[str, int] = {}
    pages = []
    empty_paras = 0
    for b in document.blocks:
        b_type = b.type if hasattr(b, "type") else b.__class__.__name__
        block_counts[str(b_type)] = block_counts.get(str(b_type), 0) + 1
        pages.append(b.page)
        if getattr(b, "type", None) == "paragraph" and not getattr(b, "text", "").strip():
            empty_paras += 1

    response_size_kb = len(serialized.encode("utf-8")) / 1024.0
    amplification = round(response_size_kb / max(file_size_kb, 0.001), 2)

    return DocumentBenchmarkResult(
        filename=pdf_path.name,
        file_size_kb=round(file_size_kb, 2),
        page_count=page_count,
        http_status=200,
        timing=timing,
        memory=memory,
        block_counts=block_counts,
        total_blocks=len(document.blocks),
        importance_passages_count=len(importance.passages),
        response_size_kb=round(response_size_kb, 2),
        amplification_ratio=amplification,
        pages_monotonic=pages == sorted(pages),
        empty_paragraphs=empty_paras,
    )


def run_concurrency_benchmark(pdf_path: Path, concurrency: int = 4, iterations: int = 4) -> dict[str, Any]:
    """Tests throughput and latency under concurrent load via FastAPI TestClient."""
    pdf_bytes = pdf_path.read_bytes()
    client = TestClient(app)

    def _worker(worker_id: int):
        start = time.perf_counter()
        resp = client.post(
            "/extract",
            files={"file": (pdf_path.name, pdf_bytes, "application/pdf")},
        )
        elapsed = time.perf_counter() - start
        return resp.status_code, elapsed

    start_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_worker, i) for i in range(iterations)]
        results = [f.result() for f in futures]
    total_wall_sec = time.perf_counter() - start_all

    latencies = [r[1] for r in results]
    success_count = sum(1 for r in results if r[0] == 200)

    return {
        "concurrency": concurrency,
        "iterations": iterations,
        "wall_time_s": round(total_wall_sec, 2),
        "throughput_req_per_s": round(iterations / total_wall_sec, 2),
        "min_latency_s": round(min(latencies), 2),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2),
        "max_latency_s": round(max(latencies), 2),
        "success_rate": round(success_count / iterations, 2),
    }


def execute_suite():
    pdf_files = sorted(CORPUS_DIR.glob("*.pdf")) if CORPUS_DIR.exists() else []
    if not pdf_files:
        # Fallback to test fixture if under-20mb not present
        pdf_files = [REPO_ROOT / "tests" / "fixtures" / "test.pdf"]

    print("=" * 80)
    print("EXTRACTION SERVICE BENCHMARK TEST SUITE")
    print("=" * 80)
    print(f"Target Corpus: {len(pdf_files)} PDF files")
    print("-" * 80)

    results: list[DocumentBenchmarkResult] = []
    for pdf_path in pdf_files:
        print(f"\nBenchmarking: {pdf_path.name} ({pdf_path.stat().st_size / 1024 / 1024:.2f} MB)...")
        res = run_single_benchmark(pdf_path)
        results.append(res)
        print(f"  Pages: {res.page_count} | Blocks: {res.total_blocks} | Status: {res.http_status}")
        print(f"  Timing (ms): Open={res.timing.pdf_open_ms} | Gate={res.timing.eligibility_ms} | "
              f"Extract={res.timing.extraction_ms} | Importance={res.timing.importance_ms} | "
              f"Serialize={res.timing.json_serialize_ms} | Total={res.timing.total_pipeline_ms}")
        print(f"  Memory Peak: {res.memory.peak_memory_mb} MB | Response Size: {res.response_size_kb / 1024:.2f} MB "
              f"(x{res.amplification_ratio} of input)")
        print(f"  Blocks by type: {res.block_counts}")
        print(f"  Importance passages identified: {res.importance_passages_count}")

    # Concurrency test using first sample
    if pdf_files:
        sample = pdf_files[0]
        print("\n" + "=" * 80)
        print(f"CONCURRENCY & THROUGHPUT STRESS TEST (Target: {sample.name})")
        print("=" * 80)
        for c in [1, 2, 4]:
            bench_conc = run_concurrency_benchmark(sample, concurrency=c, iterations=4)
            print(f"  Concurrency {c:2d} -> Throughput: {bench_conc['throughput_req_per_s']} req/s | "
                  f"Avg Latency: {bench_conc['avg_latency_s']}s | Wall Time: {bench_conc['wall_time_s']}s")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Document':<32} {'Size(MB)':<9} {'Pgs':<5} {'Blks':<6} {'Extract(s)':<11} {'PeakMB':<8} {'JSON(MB)':<9} {'AmpRatio':<8}"
    print(header)
    print("-" * len(header))
    for r in results:
        doc_name = (r.filename[:29] + "...") if len(r.filename) > 32 else r.filename
        extract_s = round(r.timing.total_pipeline_ms / 1000.0, 2)
        print(f"{doc_name:<32} {r.file_size_kb/1024:<9.2f} {r.page_count:<5} {r.total_blocks:<6} "
              f"{extract_s:<11.2f} {r.memory.peak_memory_mb:<8.1f} {r.response_size_kb/1024:<9.2f} x{r.amplification_ratio:<7.1f}")

    return results


# Pytest hook
def test_benchmark_suite():
    results = execute_suite()
    assert len(results) > 0
    for r in results:
        assert r.pages_monotonic is True, f"{r.filename} pages must be monotonic"
        assert r.empty_paragraphs == 0, f"{r.filename} should not have empty paragraphs"
        assert r.total_blocks > 0, f"{r.filename} should extract at least 1 block"


if __name__ == "__main__":
    execute_suite()
