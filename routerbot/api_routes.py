import logging
from fastapi import APIRouter, HTTPException, Request, Form, UploadFile, File
from schemas import (
    ChatRequest,
    ChatResponse,
    ResetRequest,
    ResetResponse,
    FeedbackRequest,
    FeedbackSummaryResponse,
    UploadPDFResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
)
from dependencies import get_memory, build_chat_service
from feedback_manager import FeedbackManager
from pdf_ingestion import prepare_pdf_chunks
from qdrant_store import upsert_chunks, delete_chunks_by_doc_id
from qdrant_reconcile import reconcile_orphan_vectors
from auth import current_active_user, fastapi_users
from models import User
from fastapi import Depends
from document_repository import DocumentRepository

logger = logging.getLogger(__name__)

router = APIRouter()

current_superuser = fastapi_users.current_user(active=True, superuser=True)


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
            f"use_rag: {request.use_rag} debug: {request.debug} selected_doc_ids: {request.selected_doc_ids}"
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
            selected_doc_ids=request.selected_doc_ids,
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

        result = prepare_pdf_chunks(file.file, file.filename)
        doc_id = result["document"]["doc_id"]

        document_repository = DocumentRepository()
        document_repository.save_document(
            filename=file.filename, doc_id=doc_id, user_id=user.id
        )

        logger.info(
            f"Request ID: {request_id} - endpoint = /upload-pdf stage = postgres_saved "
            f"user_id: {user.id} doc_id: {doc_id}"
        )

        try:
            upsert_result = upsert_chunks(result["chunks"], str(user.id))
        except Exception as qdrant_error:
            document_repository.delete_document(doc_id, user.id)
            logger.exception(
                f"Request ID: {request_id} - endpoint = /upload-pdf stage = qdrant_upsert_failed_rolled_back "
                f"user_id: {user.id} doc_id: {doc_id}"
            )
            raise HTTPException(
                status_code=500, detail="Failed to store document vectors."
            ) from qdrant_error

        logger.info(
            f"Request ID: {request_id} - endpoint = /upload-pdf stage = processing_done "
            f"user_id: {user.id} filename: {file.filename} "
            f"total_characters: {result['total_characters']} "
            f"total_chunks: {result['total_chunks']} "
            f"added_chunks: {upsert_result['added_chunks']}"
        )

        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "document_id": doc_id,
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


@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents(http_request: Request, user: User = Depends(current_active_user)):
    request_id = http_request.state.request_id

    try:
        logger.info(
            f"Request ID: {request_id} - endpoint = /documents stage = list_start user_id: {user.id}"
        )

        document_repository = DocumentRepository()
        documents = document_repository.list_documents(user.id)

        logger.info(
            f"Request ID: {request_id} - endpoint = /documents stage = list_done user_id: {user.id} count: {len(documents)}"
        )

        return {"total": len(documents), "documents": documents}

    except Exception as error:
        logger.exception(
            f"Request ID: {request_id} - endpoint = /documents stage = unexpected_error"
        )
        raise HTTPException(status_code=500, detail=str(error))


@router.delete(
    "/documents/{doc_id}", response_model=DocumentDeleteResponse, tags=["Documents"]
)
def delete_document(
    doc_id: str, http_request: Request, user: User = Depends(current_active_user)
):
    request_id = http_request.state.request_id

    try:
        logger.info(
            f"Request ID: {request_id} - endpoint = /documents/{doc_id} stage = delete_start user_id: {user.id}"
        )

        document_repository = DocumentRepository()
        found = document_repository.delete_document(doc_id, user.id)

        if not found:
            logger.warning(
                f"Request ID: {request_id} - endpoint = /documents/{doc_id} stage = not_found user_id: {user.id}"
            )
            raise HTTPException(status_code=404, detail="Document not found.")

        delete_chunks_by_doc_id(doc_id, str(user.id))

        logger.info(
            f"Request ID: {request_id} - endpoint = /documents/{doc_id} stage = delete_done user_id: {user.id}"
        )

        return {"message": "Document deleted successfully", "doc_id": doc_id}

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            f"Request ID: {request_id} - endpoint = /documents/{doc_id} stage = unexpected_error"
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


@router.post("/admin/reconcile-vectors", tags=["Admin"])
def reconcile_vectors(
    http_request: Request,
    dry_run: bool = True,
    user: User = Depends(current_superuser),
):
    request_id = http_request.state.request_id

    try:
        logger.info(
            f"Request ID: {request_id} - endpoint = /admin/reconcile-vectors stage = start "
            f"user_id: {user.id} dry_run: {dry_run}"
        )

        summary = reconcile_orphan_vectors(dry_run=dry_run)

        logger.info(
            f"Request ID: {request_id} - endpoint = /admin/reconcile-vectors stage = done "
            f"user_id: {user.id} dry_run: {dry_run} "
            f"orphans_found: {summary['orphans_found']} deleted: {summary['deleted']}"
        )

        return summary

    except Exception as error:
        logger.exception(
            f"Request ID: {request_id} - endpoint = /admin/reconcile-vectors stage = unexpected_error"
        )
        raise HTTPException(status_code=500, detail=str(error))
