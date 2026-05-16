from memory_manager import MemoryManager
from chat_service import ChatService
from retriever import FAISSRetriever
import logging


DEBUG_MODE = False
logger = logging.getLogger(__name__)

_retriever = None

def get_memory(session_id: str):
    return MemoryManager(session_id)



def initialize_retriever():
    global _retriever
    if _retriever is None:
        logger.info("Initializing shared FAISS retriever...")
        _retriever = FAISSRetriever()
        logger.info("Shared FAISS retriever initialized successfully.")
    return _retriever



def get_retriever():

    if _retriever is None:
        raise RuntimeError("Retriever has not been initialized.")

    return _retriever



def reload_retriever():
    global _retriever
    logger.info("Reloading shared FAISS retriever...")
    _retriever = FAISSRetriever()
    logger.info("Shared FAISS retriever reloaded successfully.")
    return _retriever


def build_chat_service(memory):
    return ChatService(memory, retriever = get_retriever(), debug=DEBUG_MODE)
    