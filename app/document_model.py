"""
Document model — the single source of truth passed between every service
in the pipeline (extraction -> segmentation -> blob-ing -> transform -> client).

Design rules this file enforces, not just documents:
  1. Blocks are the only place raw content lives. Everything built after
     extraction (chapters, blobs, pieces) REFERENCES block ids — it never
     copies or re-derives block content. One source of truth, no drift.
  2. Table / Image / Code blocks are structurally incapable of being sent
     through a "rewrite this text" step, because their content isn't a
     plain string the way Paragraph/Heading's is (rows, an asset path,
     raw text tied to a `kind`) — the type system reflects the contract
     already agreed on: only Paragraph/Heading content is ever edited.
  3. `extraction_version` is mandatory on every Document. Nothing
     downstream is allowed to be cached against a Document without it —
     this is what lets extraction improve later without silently serving
     stale segmentation/transform output.
  4. Every boundary (chapter, blob) is explicit about how it was decided
     (heading vs. similarity-dip vs. size-forced), because a heading
     boundary is authorial truth and a similarity-dip boundary is a guess
     the LLM chunking step is still allowed to second-guess.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# ══════════════════════════════════════════════════════════════════════
# PRIMITIVES
# ══════════════════════════════════════════════════════════════════════

def _new_id() -> str:
    return uuid4().hex[:12]


# ══════════════════════════════════════════════════════════════════════
# BLOCKS — the only place content lives
# ══════════════════════════════════════════════════════════════════════

class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    REFERENCE = "reference"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"


class BlockSource(str, Enum):
    TEXT_LAYER = "text_layer"
    GRID = "grid"
    WORDS = "words"
    RASTER = "raster"
    VECTOR = "vector"


class _BlockBase(BaseModel):
    id: str = Field(default_factory=_new_id)
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float] | None = None
    source: BlockSource | str | None = None
    flags: list[str] = Field(default_factory=list)


class HeadingBlock(_BlockBase):
    type: Literal[BlockType.HEADING] = BlockType.HEADING
    text: str
    level: int = 1


class ParagraphBlock(_BlockBase):
    type: Literal[BlockType.PARAGRAPH] = BlockType.PARAGRAPH
    text: str
    role: Literal["body", "reference"] = "body"


class ReferenceBlock(_BlockBase):
    type: Literal[BlockType.REFERENCE] = BlockType.REFERENCE
    text: str


class TableBlock(_BlockBase):
    type: Literal[BlockType.TABLE] = BlockType.TABLE
    rows: list[list[str]] = Field(default_factory=list)
    header_rows: int = Field(default=1, ge=0)
    caption: str | None = None
    notes: list[str] = Field(default_factory=list)
    raw_lines: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_empty_cells(cls, data):
        # PyMuPDF's table.extract() returns None for empty cells, not "".
        # A missing cell is a legitimate sparse-table case, not malformed
        # input — normalize here so every producer doesn't have to remember.
        if isinstance(data, dict) and "rows" in data and data["rows"] is not None:
            data = dict(data)
            data["rows"] = [["" if c is None else c for c in row] for row in data["rows"]]
        return data

    @model_validator(mode="after")
    def _shape_ok(self) -> "TableBlock":
        if len(self.rows) < 1 and not self.raw_lines:
            raise ValueError("table must have at least one row or raw_lines")
        return self


class ImageBlock(_BlockBase):
    type: Literal[BlockType.IMAGE] = BlockType.IMAGE
    asset_url: str = ""     # hosted URL, local/temp path, or base64 data URI
    caption: str | None = None
    alt_text: str | None = None   # optional, filled later (accessibility pass)
    panels: list[str] = Field(default_factory=list)
    decorative: bool = False


class CodeKind(str, Enum):
    CODE = "code"
    FORMULA = "formula"


class CodeBlock(_BlockBase):
    type: Literal[BlockType.CODE] = BlockType.CODE
    kind: CodeKind
    # for kind=code: the monospaced text, indentation preserved.
    # for kind=formula: raw glyph text extracted from the PDF — unreliable,
    # kept only for search/debugging. Rendering ALWAYS uses asset_url.
    text: str
    language: str | None = None          # kind=code only; None = unknown
    asset_url: str | None = None         # kind=formula: cropped PNG, required
                                          # kind=code: never set
    alt_text: str | None = None          # kind=formula only: enrichment description,
                                          # same role as ImageBlock.alt_text — see
                                          # transform.py::enrich_block. Symmetric field
                                          # name so both asset-carrying block types are
                                          # handled identically wherever alt_text is read.

    @model_validator(mode="after")
    def _formula_needs_asset(self) -> "CodeBlock":
        if self.kind == CodeKind.FORMULA and not self.asset_url:
            raise ValueError("formula blocks must carry a cropped-image asset_url")
        return self


# Discriminated union — this is what makes "only paragraph/heading content
# is editable text" a type-level fact rather than a convention someone can
# forget. Any code that wants a block's plain text has to narrow the type
# first; there's no `.text` on Table/Image that could be accidentally fed
# to an LLM the way a stray dict field could.
Block = Annotated[
    Union[HeadingBlock, ParagraphBlock, ReferenceBlock, TableBlock, ImageBlock, CodeBlock],
    Field(discriminator="type"),
]


# ══════════════════════════════════════════════════════════════════════
# QUALITY REPORT (schema_version 2)
# ══════════════════════════════════════════════════════════════════════

class PageQuality(BaseModel):
    page: int
    rotation_applied: int = 0
    word_parity: float = 1.0
    numeric_parity: float = 1.0
    flags: list[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    overall_score: float = 1.0
    pages: list[PageQuality] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# DOCUMENT — output of the extraction service (Stage 1)
# ══════════════════════════════════════════════════════════════════════

class DocumentMeta(BaseModel):
    title: str | None = None
    source_filename: str
    page_count: int = Field(ge=1)
    body_font_size: float | None = None   # diagnostic, from extraction stats


class Document(BaseModel):
    """The full output of extraction. Immutable once produced — everything
    downstream (Section, Blob, ProcessedPiece) references block ids into
    this object, never copies block content out of it."""
    schema_version: Literal[1, 2] = 2
    extraction_version: str          # e.g. "pdf_to_blocks@2026-09-20-a1b2c3"
                                       # bump whenever extraction logic changes;
                                       # every cache downstream keys off this
    document_id: str = Field(default_factory=_new_id)
    meta: DocumentMeta
    blocks: list[Block]
    quality: QualityReport | None = None

    @model_validator(mode="after")
    def _ids_unique_and_ordered_by_page(self) -> "Document":
        ids = [b.id for b in self.blocks]
        if len(ids) != len(set(ids)):
            raise ValueError("block ids must be unique within a Document")
        pages = [b.page for b in self.blocks]
        if pages != sorted(pages):
            raise ValueError("blocks must be in reading order (non-decreasing page)")
        return self

    def get(self, block_id: str) -> Block:
        # small helper every downstream service needs; keeps lookups
        # consistent instead of each service rolling its own dict
        for b in self.blocks:
            if b.id == block_id:
                return b
        raise KeyError(f"no block {block_id!r} in document {self.document_id}")


# ══════════════════════════════════════════════════════════════════════
# ELIGIBILITY — output of the pre-flight gate, runs before extraction
# is trusted enough to feed the rest of the pipeline
# ══════════════════════════════════════════════════════════════════════

class RejectionReason(str, Enum):
    NO_TEXT_LAYER = "no_text_layer"
    SCANNED_DOCUMENT = "scanned_document"


class EligibilityResult(BaseModel):
    eligible: bool
    reasons: list[RejectionReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reasons_iff_ineligible(self) -> "EligibilityResult":
        if self.eligible and self.reasons:
            raise ValueError("an eligible document cannot carry rejection reasons")
        if not self.eligible and not self.reasons:
            raise ValueError("an ineligible document must state at least one reason")
        return self


# ══════════════════════════════════════════════════════════════════════
# SECTIONS — output of segmentation (Stage 2). References into Document.
# ══════════════════════════════════════════════════════════════════════

class SectionKind(str, Enum):
    FRONT_MATTER = "front_matter"
    CHAPTER = "chapter"
    BACK_MATTER = "back_matter"


class Section(BaseModel):
    """A slice of the document, by block id — never by copied content.
    front_matter/back_matter sections exist so the transform stage can
    refuse to process them (mirrors the old `run_pipeline` type-check
    that raised on non-Chapter input — enforced by SectionKind here
    instead of by isinstance)."""
    id: str = Field(default_factory=_new_id)
    kind: SectionKind
    title: str | None = None       # heading text of the first block, if any
    block_ids: list[str]           # ordered, references Document.blocks

    @model_validator(mode="after")
    def _nonempty(self) -> "Section":
        if not self.block_ids:
            raise ValueError("a Section must reference at least one block")
        return self


class SegmentedDocument(BaseModel):
    document_id: str               # matches Document.document_id
    extraction_version: str        # copied at segmentation time; if this
                                    # doesn't match the live Document's, the
                                    # segmentation is stale and must be redone
    sections: list[Section]


# ══════════════════════════════════════════════════════════════════════
# BLOBS — output of blob-ization (cheap NLP pass over a Chapter section)
# ══════════════════════════════════════════════════════════════════════

class BoundaryKind(str, Enum):
    HEADING = "heading"         # authorial — hard wall, never reconsidered
    SIMILARITY_DIP = "similarity_dip"  # TextTiling-style guess — advisory,
                                        # the downstream LLM chunker may
                                        # still choose to cut elsewhere nearby
    SIZE_FORCED = "size_forced"        # no boundary found in budget; blunt cut


class Blob(BaseModel):
    """A batch of several reading-pieces' worth of content, sized so the
    downstream LLM chunker has room to choose graceful cut points inside
    it. Sent to the client as soon as ready; the client requests the next
    blob's transform once the reader is most of the way through this one."""
    id: str = Field(default_factory=_new_id)
    section_id: str                 # the Chapter Section this blob belongs to
    sequence: int = Field(ge=0)     # order within the section
    block_ids: list[str]
    start_boundary: BoundaryKind
    end_boundary: BoundaryKind

    @model_validator(mode="after")
    def _nonempty(self) -> "Blob":
        if not self.block_ids:
            raise ValueError("a Blob must reference at least one block")
        return self


# ══════════════════════════════════════════════════════════════════════
# PROCESSED PIECES — output of the transform stage (Stage 4), cached
# per-blob-derived-piece forever. This is the ONLY model in the whole
# contract that carries LLM-authored text, and it's additive — it never
# replaces the Document's blocks, it sits alongside them.
# ══════════════════════════════════════════════════════════════════════

class ProcessedPiece(BaseModel):
    id: str = Field(default_factory=_new_id)
    blob_id: str
    sequence: int = Field(ge=0)         # order within the blob
    source_block_ids: list[str]         # paragraph/heading blocks this piece
                                         # was edited from — for audit/debug
    title: str
    edited_markdown: str                # content_master output, non-text
                                         # blocks already spliced back in by
                                         # the reassemble step
    explanation_markdown: str | None = None

    @model_validator(mode="after")
    def _has_content(self) -> "ProcessedPiece":
        if not self.edited_markdown.strip():
            raise ValueError("a ProcessedPiece cannot have empty content")
        return self


class ChapterProcessingStatus(str, Enum):
    PENDING = "pending"       # requested, in flight (lock held)
    READY = "ready"           # all pieces cached, safe to serve
    FAILED = "failed"         # whole-chapter atomicity: any piece failing
                               # marks the whole chapter failed, not partial


class ProcessedChapter(BaseModel):
    section_id: str
    extraction_version: str    # must match Document; mismatch => reprocess
    status: ChapterProcessingStatus
    pieces: list[ProcessedPiece] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ready_implies_pieces(self) -> "ProcessedChapter":
        if self.status == ChapterProcessingStatus.READY and not self.pieces:
            raise ValueError("a READY chapter must have at least one piece")
        return self
