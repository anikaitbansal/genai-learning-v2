import logging
from fastapi import APIRouter, HTTPException, Request, Form, UploadFile, File
from schemas import (
    ChatRequest,
    ChatResponse,
    ResetResponse,
    ResetRequest,
    FeedbackRequest,
    FeedbackSummaryResponse,
    UploadPDFResponse,
)
from dependencies import get_memory, build_chat_service
from feedback_manager import FeedbackManager
from pdf_ingestion import ingest_pdf_file
from auth import current_active_user
from models import User
from fastapi import Depends

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", tags=["general"])
def home():
    return {"message": "chatbot is running."}


@router.get("/health", tags=["general"])
def health():
    return {"status": "healthy"}


@router.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
    tags=["Chat"],
)
def chat(
    request: ChatRequest,
    http_request: Request,
    user: User = Depends(current_active_user),
):
    request_id = http_request.state.request_id
    try:

        logger.info(
            f"Request ID: {request_id} - endpoint = /chat stage = request_received "
            f"user_id: {user.id} Session: {request.session_id} message_length: {len(request.message.strip())} "
            f"use_rag: {request.use_rag} debug: {request.debug}"
        )

        memory = get_memory(request.session_id, user.id)
        service = build_chat_service(memory, user.id)

        logger.info(
            f"Request ID: {request_id} - endpoint = /chat stage = service_call_start "
            f"Session: {request.session_id}"
        )

        result = service.process_message(
            request.message,
            request.session_id,
            request_id,
            user.id,
            use_rag=request.use_rag,
            debug=request.debug,
        )

        logger.info(
            f"Request ID: {request_id} - endpoint = /chat stage = service_call_done "
            f"Session: {request.session_id} intent: {result['intent']} "
            f"rag_used: {result['rag_used']} response_length: {len(result['bot_reply'])}"
        )

        logger.info(
            f"Request ID: {request_id} - endpoint = /chat stage = response_sent "
            f"Session: {request.session_id}"
        )

        return result

    except PermissionError as pe:
        logger.warning(
            f"Request ID: {request_id} - endpoint = /chat stage = permission_denied user_id: {user.id} session_id: {request.session_id}"
        )
        raise HTTPException(status_code=403, detail=str(pe))

    except ValueError as ve:
        logger.exception(
            f"Request ID: {request_id} - endpoint = /chat stage = validation_error"
        )
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as error:
        logger.exception(
            f"Request ID: {request_id} - endpoint = /chat stage = unexpected_error"
        )
        raise HTTPException(status_code=500, detail=str(error))


@router.post(
    "/chat-form",
    response_model=ChatResponse,
    response_model_exclude_none=True,
    tags=["Chat"],
)
def chat_form(
    http_request: Request,
    user_input: str = Form(...),
    session_id: str = Form(...),
    use_rag: bool = Form(True),
    debug: bool = Form(False),
    user: User = Depends(current_active_user),
):
    request_id = http_request.state.request_id

    try:
        logger.info(
            f"Request ID: {request_id} - endpoint = /chat-form stage = request_received "
            f"user_id: {user.id} Session: {session_id} message_length: {len(user_input.strip())} "
            f"use_rag: {use_rag} debug: {debug}"
        )

        memory = get_memory(session_id, user.id)
        service = build_chat_service(memory, user.id)

        logger.info(
            f"Request ID: {request_id} - endpoint = /chat-form stage = service_call_start "
            f"Session: {session_id}"
        )

        result = service.process_message(
            user_input, session_id, request_id, user.id, use_rag=use_rag, debug=debug
        )

        logger.info(
            f"Request ID: {request_id} - endpoint = /chat-form stage = service_call_done "
            f"Session: {session_id} intent: {result['intent']} "
            f"rag_used: {result['rag_used']} response_length: {len(result['bot_reply'])}"
        )

        logger.info(
            f"Request ID: {request_id} - endpoint = /chat-form stage = response_sent "
            f"Session: {session_id}"
        )

        return result

    except PermissionError as pe:
        logger.warning(
            f"Request ID: {request_id} - endpoint = /chat-form stage = permission_denied user_id: {user.id} session_id: {session_id}"
        )
        raise HTTPException(status_code=403, detail=str(pe))

    except ValueError as ve:
        logger.exception(
            f"Request ID: {request_id} - endpoint = /chat-form stage = validation_error"
        )
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as error:
        logger.exception(
            f"Request ID: {request_id} - endpoint = /chat-form stage = unexpected_error"
        )
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/reset", response_model=ResetResponse, tags=["Chat"])
def reset(
    request: ResetRequest,
    http_request: Request,
    user: User = Depends(current_active_user),
):
    request_id = http_request.state.request_id

    try:
        logger.info(
            f"Request ID: {request_id} - /reset called for user_id: {user.id} session: {request.session_id}"
        )

        memory = get_memory(request.session_id, user.id)
        memory.clear()

        logger.info(
            f"Request ID: {request_id} - Memory cleared successfully for session: {request.session_id}"
        )

        return {
            "message": "Chat memory cleared successfully",
            "session_id": request.session_id,
        }

    except PermissionError as pe:
        logger.warning(
            f"Request ID: {request_id} - endpoint = /reset stage = permission_denied user_id: {user.id} session_id: {request.session_id}"
        )
        raise HTTPException(status_code=403, detail=str(pe))

    except Exception as error:
        logger.exception(f"Request ID: {request_id} - Unexpected error in /reset")
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/feedback", tags=["Feedback"])
def submit_feedback(
    request: FeedbackRequest,
    http_request: Request,
    user: User = Depends(current_active_user),
):

    request_id = http_request.state.request_id

    try:
        logger.info(
            f"Request ID: {request_id} - Feedback received for user_id: {user.id} session: {request.session_id} (request: {request.request_id})"
        )

        feedback_manager = FeedbackManager()

        entry = feedback_manager.create_feedback_entry(
            session_id=request.session_id,
            request_id=request.request_id,
            rating=request.rating,
            comments=request.comments,
        )

        feedback_manager.save_feedback(entry, user.id)

        logger.info(
            f"Request ID: {request_id} - Feedback saved successfully for target request: {request.request_id}"
        )

        return {
            "message": "Feedback submitted successfully",
        }

    except PermissionError as pe:
        logger.warning(
            f"Request ID: {request_id} - endpoint = /feedback stage = permission_denied user_id: {user.id} session_id: {request.session_id}"
        )
        raise HTTPException(status_code=403, detail=str(pe))

    except Exception as error:
        logger.exception(f"Request ID: {request_id} - Error while saving feedback")
        raise HTTPException(status_code=500, detail=str(error))


@router.get(
    "/feedback/summary", response_model=FeedbackSummaryResponse, tags=["Feedback"]
)
def feedback_summary(http_request: Request, user: User = Depends(current_active_user)):
    request_id = http_request.state.request_id

    try:
        logger.info(
            f"Request ID: {request_id} - Fetching feedback summary for user_id: {user.id}"
        )

        feedback_manager = FeedbackManager()
        summary = feedback_manager.get_summary(user.id)

        logger.info(
            f"Request ID: {request_id} - Feedback summary generated successfully"
        )

        return summary

    except Exception as error:
        logger.exception(
            f"Request ID: {request_id} - Error while generating feedback summary"
        )
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/upload-pdf", response_model=UploadPDFResponse, tags=["Documents"])
async def upload_pdf(
    http_request: Request,
    file: UploadFile = File(...),
    user: User = Depends(current_active_user),
):

    request_id = http_request.state.request_id

    try:
        logger.info(
            f"Request ID: {request_id} - endpoint = /upload-pdf stage = request_received user_id: {user.id} filename: {file.filename} content_type: {file.content_type}"
        )

        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        file.file.seek(0)

        result = ingest_pdf_file(file.file, file.filename, str(user.id))

        logger.info(
            f"Request ID: {request_id} - endpoint = /upload-pdf stage = processing_done "
            f"user_id: {user.id} filename: {file.filename} "
            f"total_characters: {result['total_characters']} "
            f"total_chunks: {result['total_chunks']} "
            f"added_chunks: {result['added_chunks']}"
        )

        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "document_id": result["document"]["doc_id"],
            "total_characters": result["total_characters"],
            "total_chunks": result["total_chunks"],
            "chunks": result["chunks"],
        }

    except HTTPException:
        raise

    except ValueError as ve:
        logger.exception(
            f"Request ID: {request_id} - endpoint = /upload-pdf stage = validation_error"
        )
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as error:
        logger.exception(
            f"Request ID: {request_id} - endpoint = /upload-pdf stage = unexpected_error"
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
                results.append(
                    {
                        "id": row[0],
                        "session_id": row[1],
                        "request_id": row[2],
                        "rating": row[3],
                        "comments": row[4],
                        "timestamp": row[5],
                    }
                )

        return {"total_rows": len(results), "data": results}

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
                results.append(
                    {
                        "id": row[0],
                        "session_id": row[1],
                        "request_id": row[2],
                        "user_message": row[3],
                        "bot_reply": row[4],
                        "intent": row[5],
                        "rag_used": bool(row[6]),
                    }
                )

        return {"total_rows": len(results), "data": results}

    except Exception as error:
        logger.exception("[request_id=%s] Error fetching chat logs", request_id)
        raise HTTPException(status_code=500, detail=str(error))
