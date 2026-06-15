from pydantic import BaseModel, Field
from fastapi_users import schemas as fu_schemas
import uuid


class UserRead(fu_schemas.BaseUser[uuid.UUID]):
    pass


class UserCreate(fu_schemas.BaseUserCreate):
    pass


class UserUpdate(fu_schemas.BaseUserUpdate):
    pass


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    use_rag: bool = True
    debug: bool = False


class RetrievedChunk(BaseModel):
    id: str
    title: str
    content: str
    score: float


class EvaluationResult(BaseModel):
    score: str
    reason: str


class ChatResponse(BaseModel):
    user_message: str
    bot_reply: str
    intent: str
    session_id: str
    rag_used: bool
    retrieved_chunks: list[RetrievedChunk] | None = None
    evaluation: EvaluationResult | None = None
    retry_count: int | None = None
    retry_happened: bool | None = None


class ResetResponse(BaseModel):
    message: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    request_id: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5)
    comments: str | None = None


class FeedbackSummaryResponse(BaseModel):
    total_feedback: int
    average_rating: float
    rating_counts: dict[str, int]


class UploadedDocumentChunk(BaseModel):
    id: str
    title: str
    content: str


class UploadPDFResponse(BaseModel):
    message: str
    filename: str
    document_id: str
    total_characters: int
    total_chunks: int
    chunks: list[UploadedDocumentChunk]


class DocumentItem(BaseModel):
    doc_id: str
    filename: str
    uploaded_at: str


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentItem]


class DocumentDeleteResponse(BaseModel):
    message: str
    doc_id: str
