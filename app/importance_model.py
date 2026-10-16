"""
Paragraph importance scoring — separate concern from extraction, additive
to the Document contract, never embedded inside it.

References Document.blocks by id, same pattern as everything else in
document_model.py. Downstream features (quizzes, flashcards, practice)
consume this independently; nothing here knows or cares what they do
with it.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ImportanceSignal(str, Enum):
    CENTRALITY = "centrality"       # TextRank score within its local window
    DEFINITIONAL = "definitional"   # matched a definition-style pattern
    POSITIONAL = "positional"       # opening paragraph after a heading


class ImportantPassage(BaseModel):
    block_id: str                    # references a ParagraphBlock in Document.blocks
    message: str
    signals: list[ImportanceSignal] = Field(default_factory=list)


class DocumentImportance(BaseModel):
    document_id: str
    extraction_version: str          # same staleness contract as everything else
    passages: list[ImportantPassage]
