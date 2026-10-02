from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STAGE_ORDER = (
    "chunk",
    "extract",
    "verify",
    "resolve",
    "graph",
    "leiden",
    "report",
    "embed",
)

StageName = Literal[
    "chunk",
    "extract",
    "verify",
    "resolve",
    "graph",
    "leiden",
    "report",
    "embed",
]

StageStatus = Literal["pending", "running", "waiting_llm", "done", "stale", "failed"]


class TextUnit(BaseModel):
    id: str
    chapter: str
    chapter_num: int
    position: int
    text: str
    token_count: int


class ExtractedEntity(BaseModel):
    name: str
    type: str = "concept"
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    type: str = "related"
    description: str = ""
    quote: str = ""
    confidence: float = 0.5


class ExtractPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class VerifyVerdict(BaseModel):
    relationship_key: str
    accepted: bool
    reason: str = ""
    quote: str = ""


class VerifyPayload(BaseModel):
    verdicts: list[VerifyVerdict] = Field(default_factory=list)


class ResolveEntity(BaseModel):
    canonical_name: str
    type: str = "concept"
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class RejectedMerge(BaseModel):
    names: list[str] = Field(default_factory=list)
    reason: str = ""


class ResolvePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: list[ResolveEntity]
    rejected_merges: list[RejectedMerge] = Field(default_factory=list)


class CommunityReportIn(BaseModel):
    community: int
    title: str
    summary: str
    findings: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class ReportPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reports: list[CommunityReportIn]


class Citation(BaseModel):
    chunk_id: str
    quote: str
    chapter: str = ""


class RetrievalPacket(BaseModel):
    mode: str
    question: str
    chunk_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    community_ids: list[str] = Field(default_factory=list)
    context: str
    citations: list[Citation] = Field(default_factory=list)


class AnswerResult(BaseModel):
    answer: str
    packet: RetrievalPacket
