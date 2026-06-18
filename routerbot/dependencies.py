import logging
from langchain_core.retrievers import BaseRetriever
from retriever import build_compression_retriever
from chat_service import ChatService
from memory_manager import MemoryManager

from uuid import UUID

logger = logging.getLogger(__name__)

debug_mode = False


def get_memory(session_id: str, user_id: UUID):
    return MemoryManager(session_id, user_id)


def get_retriever(
    user_id: UUID, selected_doc_ids: list[str] | None = None
) -> BaseRetriever:
    return build_compression_retriever(str(user_id), selected_doc_ids)


def build_chat_service(memory, user_id: UUID):
    return ChatService(
        memory, user_id=user_id, retriever_builder=get_retriever, debug=debug_mode
    )
