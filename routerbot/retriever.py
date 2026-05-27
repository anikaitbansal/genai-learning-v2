import logging
from typing import List
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from config import MODEL_NAME
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from qdrant_store import retrieve_from_qdrant
from config import RAG_TOP_K
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class QdrantRetriever(BaseRetriever):
    user_id: str
    top_k: int = RAG_TOP_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        logger.info(
            f"retriever_stage = retrieve_start user_id: {self.user_id} "
            f"query_length: {len(query.strip())} top_k: {self.top_k}"
        )

        chunks = retrieve_from_qdrant(query, self.user_id, self.top_k)

        documents = [
            Document(
                page_content=chunk["content"],
                metadata={
                    "id": chunk["id"],
                    "title": chunk["title"],
                },
            )
            for chunk in chunks
        ]

        logger.info(
            f"retriever_stage = retrieve_done user_id: {self.user_id} "
            f"returned: {len(documents)}"
        )

        return documents

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        documents = self._get_relevant_documents(
            query, run_manager=None  # type: ignore
        )
        return [
            {
                "id": doc.metadata.get("id", ""),
                "title": doc.metadata.get("title", ""),
                "content": doc.page_content,
                "score": 0.0,
            }
            for doc in documents[:top_k]
        ]


MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are an AI assistant. Your task is to generate exactly 3 different "
        "rephrasings of the user's question to help retrieve relevant documents "
        "from a vector database.\n\n"
        "Strict output rules:\n"
        "- Output EXACTLY 3 lines\n"
        "- One rephrased question per line\n"
        "- No numbering, no bullets, no preamble, no explanations\n"
        "- No blank lines\n"
        "- Each line must be a complete standalone question\n\n"
        "Original question: {question}\n\n"
        "Three rephrased questions:"
    ),
)


def build_multi_query_retriever(user_id: str) -> MultiQueryRetriever:
    base_retriever = QdrantRetriever(user_id=user_id)

    query_generation_llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.0,
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=query_generation_llm,
        prompt=MULTI_QUERY_PROMPT,
        include_original=True,
    )

    logger.info(
        f"retriever_stage = multi_query_built user_id: {user_id} "
        f"base: QdrantRetriever llm: {MODEL_NAME}"
    )

    return multi_query_retriever
