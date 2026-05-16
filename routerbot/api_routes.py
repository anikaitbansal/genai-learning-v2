import logging
from fastapi import APIRouter, HTTPException, Request, Form, UploadFile, File
from schemas import ChatRequest, ChatResponse, ResetResponse, ResetRequest, FeedbackRequest, FeedbackSummaryResponse, BuildKnowledgeBaseResponse, UploadPDFResponse
from dependencies import get_memory, build_chat_service, reload_retriever, get_retriever
from feedback_manager import FeedbackManager
from build_knowledge_base import build_knowledge_base
from pdf_ingestion import ingest_pdf_file

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/", tags=["general"])
def home():
    return {"message": "chatbot is running."}



@router.get("/health", tags=["general"])
def health():
    return {"status": "healthy"}



@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True, tags=["chat"])
def chat(request: ChatRequest, http_request: Request):
    request_id = http_request.state.request_id

    try:
        logger.info(
            "[request_id=%s] endpoint=/chat stage=request_received session_id=%s message_length=%s use_rag=%s debug=%s",
            request_id,
            request.session_id,
            len(request.message.strip()),
            request.use_rag,
            request.debug
        )      

        memory = get_memory(request.session_id)
        service = build_chat_service(memory)
        logger.info(
            "[request_id=%s] endpoint=/chat stage=service_call_start session_id=%s",
            request_id,
            request.session_id
        )
        
        result = service.process_message(
            request.message,
            request.session_id,
            request_id,
            use_rag=request.use_rag,
            debug=request.debug
        )
        

        logger.info(
            "[request_id=%s] endpoint=/chat stage=service_call_done session_id=%s intent=%s rag_used=%s response_length=%s",
            request_id,
            request.session_id,
            result["intent"],
            result["rag_used"],
            len(result["bot_reply"])
        )

        logger.info(
            "[request_id=%s] endpoint=/chat stage=response_sent session_id=%s",
            request_id,
            request.session_id
        )
        return result
    
    except ValueError as error:
        logger.exception("[request_id=%s] endpoint=/chat stage=validation_error", request_id)
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        logger.exception("[request_id=%s] endpoint=/chat stage=unexpected_error", request_id)
        raise HTTPException(status_code=500, detail=str(error))



@router.post("/chat-form", response_model=ChatResponse, response_model_exclude_none=True,  tags=["chat"])
def chat_form(
    http_request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
    use_rag: bool = Form(True),
    debug: bool = Form(False)
):
    request_id = http_request.state.request_id

    try:
        logger.info(
            "[request_id=%s] endpoint=/chat-form stage=request_received session_id=%s message_length=%s use_rag=%s debug=%s",
            request_id,
            session_id,
            len(message.strip()),
            use_rag,
            debug
        )

        memory = get_memory(session_id)
        service = build_chat_service(memory)
        logger.info(
            "[request_id=%s] endpoint=/chat-form stage=service_call_start session_id=%s",
            request_id,
            session_id
        )

        result = service.process_message(
            message,
            session_id,
            request_id,
            use_rag=use_rag,
            debug=debug
        )


        logger.info(
            "[request_id=%s] endpoint=/chat-form stage=service_call_done session_id=%s intent=%s rag_used=%s response_length=%s",
            request_id,
            session_id,
            result["intent"],
            result["rag_used"],
            len(result["bot_reply"])
        )

        logger.info(
            "[request_id=%s] endpoint=/chat-form stage=response_sent session_id=%s",
            request_id,
            session_id
        )
        return result

    except ValueError as error:
        logger.exception("[request_id=%s] endpoint=/chat-form stage=validation_error", request_id)
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        logger.exception("[request_id=%s] endpoint=/chat-form stage=unexpected_error", request_id)
        raise HTTPException(status_code=500, detail=str(error))



@router.post("/reset", response_model=ResetResponse, tags=["chat"])
def reset( request: ResetRequest, http_request: Request):
    request_id = http_request.state.request_id

    try:
        logger.info("[request_id=%s] /reset called for session: %s", request_id, request.session_id)
    
        memory = get_memory(request.session_id)
        memory.clear()

        logger.info(
            "[request_id=%s] Memory cleared successfully for session: %s",
            request_id,
            request.session_id
        )

        return {"message": "Memory cleared.",
                "session_id": request.session_id,
                }
    
    except Exception as error:
        logger.exception(
            "[request_id=%s] Unexpected error in /reset",
            request_id
        )
        raise HTTPException(status_code=500, detail=str(error))
    


@router.post("/feedback", tags=["feedback"])
def submit_feedback(request: FeedbackRequest, http_request: Request):
    request_id = http_request.state.request_id

    try:
        logger.info(
            "[request_id=%s] Received feedback for session: %s (target_request_id=%s)",
            request_id,
            request.session_id,
            request.request_id
        )

        feedback_manager = FeedbackManager()

        entry = feedback_manager.create_feedback_entry(
            session_id=request.session_id,
            request_id=request.request_id,
            rating=request.rating,
            comments=request.comments
        )
        feedback_manager.save_feedback(entry)

        logger.info(
            "[request_id=%s] Feedback saved successfully for target_request_id=%s",
            request_id,
            request.request_id
        )

        return {"message": "Feedback submitted successfully.",
                }    
    except Exception as error:
        logger.exception(
            "[request_id=%s] Error while storing feedback",
            request_id
        )
        raise HTTPException(status_code=500, detail=str(error))
    


@router.get("/feedback/summary", response_model=FeedbackSummaryResponse, tags=["feedback"])
def feedback_summary(http_request: Request):
    request_id = http_request.state.request_id

    try:
        logger.info(
            "[request_id=%s] Fetching feedback summary",
            request_id
        )

        feedback_manager = FeedbackManager()
        summary = feedback_manager.get_summary()

        logger.info(
            "[request_id=%s] Feedback summary generated successfully",
            request_id
        )

        return summary

    except Exception as error:
        logger.exception(
            "[request_id=%s] Error while generating feedback summary",
            request_id
        )
        raise HTTPException(status_code=500, detail=str(error))
    


@router.post("/knowledge-base/rebuild", response_model=BuildKnowledgeBaseResponse, tags=["rag"])
def rebuild_knowledge_base(http_request: Request):
    request_id = http_request.state.request_id

    try:
        logger.info(
            "[request_id=%s] Rebuilding knowledge base",
            request_id
        )

        result = build_knowledge_base()
        reload_retriever()

        logger.info(
            "[request_id=%s] Knowledge base rebuilt successfully | documents=%s | chunks=%s | file=%s",
            request_id,
            result["total_documents"],
            result["total_chunks"],
            result["knowledge_file"]
        )

        return result

    except Exception as error:
        logger.exception(
            "[request_id=%s] Error while rebuilding knowledge base: %s",
            request_id,
            str(error)
        )
        raise HTTPException(status_code=500, detail=str(error))
    


@router.post("/upload-pdf", response_model=UploadPDFResponse, tags=["documents"])
async def upload_pdf(file: UploadFile = File(...), http_request: Request = None):
    request_id = http_request.state.request_id

    try:
        logger.info(
            "[request_id=%s] endpoint=/upload-pdf stage=request_received filename=%s content_type=%s",
            request_id,
            file.filename,
            file.content_type
        )

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        file.file.seek(0)

        retriever = get_retriever()

        result = ingest_pdf_file(file.file, file.filename, retriever)

        reload_retriever()

        logger.info(
            "[request_id=%s] endpoint=/upload-pdf stage=processing_done filename=%s total_characters=%s total_chunks=%s added_chunks=%s",
            request_id,
            file.filename,
            result["total_characters"],
            result["total_chunks"],
            result["added_chunks"]
        )

        return {
            "message": "PDF uploaded and processed successfully.",
            "filename": file.filename,
            "document_id": result["document"]["doc_id"],
            "total_characters": result["total_characters"],
            "total_chunks": result["total_chunks"],
            "chunks": result["chunks"]
        }

    except HTTPException:
        raise

    except ValueError as error:
        logger.exception(
            "[request_id=%s] endpoint=/upload-pdf stage=validation_error",
            request_id
        )
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        logger.exception(
            "[request_id=%s] endpoint=/upload-pdf stage=unexpected_error",
            request_id
        )
        raise HTTPException(status_code=500, detail=str(error))
    


@router.get("/debug/feedback", tags=["debug"])
def get_all_feedback(http_request: Request):
    request_id = http_request.state.request_id

    try:
        from app_database import get_connection

        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM feedback")
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "session_id": row[1],
                    "request_id": row[2],
                    "rating": row[3],
                    "comments": row[4],
                    "timestamp": row[5]
                })

        return {
            "total_rows": len(results),
            "data": results
        }

    except Exception as error:
        logger.exception("[request_id=%s] Error fetching debug feedback", request_id)
        raise HTTPException(status_code=500, detail=str(error))
    


@router.get("/debug/chat-logs", tags=["debug"])
def get_chat_logs(http_request: Request):
    request_id = http_request.state.request_id

    try:
        from app_database import get_connection

        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM chat_logs ORDER BY id DESC")
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "session_id": row[1],
                    "request_id": row[2],
                    "user_message": row[3],
                    "bot_reply": row[4],
                    "intent": row[5],
                    "rag_used": bool(row[6])
                })

        return {
            "total_rows": len(results),
            "data": results
        }

    except Exception as error:
        logger.exception("[request_id=%s] Error fetching chat logs", request_id)
        raise HTTPException(status_code=500, detail=str(error))
    
    