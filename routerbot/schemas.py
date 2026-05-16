from pydantic import BaseModel, Field


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


class BuildKnowledgeBaseResponse(BaseModel):
    message: str
    total_documents: int
    processed_documents: int
    skipped_documents: int
    total_chunks: int
    knowledge_file: str


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