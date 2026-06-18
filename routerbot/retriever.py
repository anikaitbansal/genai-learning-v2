import logging
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from qdrant_store import fetch_all_chunks_for_user
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank
from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from qdrant_store import retrieve_from_qdrant
from config import (
    RAG_TOP_K,
    MODEL_NAME,
    MULTI_QUERY_PROMPT_TEMPLATE,
    COHERE_RERANK_MODEL,
)

logger = logging.getLogger(__name__)


class QdrantRetriever(BaseRetriever):
    user_id: str
    selected_doc_ids: list[str] | None = None
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

        chunks = retrieve_from_qdrant(
            query, self.user_id, self.top_k, self.selected_doc_ids
        )

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


def retrieve_as_dicts(
    retriever: BaseRetriever, query: str, top_k: int = RAG_TOP_K
) -> list[dict]:
    documents = retriever.invoke(query)

    return [
        {
            "id": doc.metadata.get("id", ""),
            "title": doc.metadata.get("title", ""),
            "content": doc.page_content,
            "score": doc.metadata.get("relevance_score", 0.0),
        }
        for doc in documents[:top_k]
    ]


def build_bm25_retriever(
    user_id: str, selected_doc_ids: list[str] | None = None
) -> BM25Retriever:
    logger.info(f"bm25_stage = build_start user_id: {user_id}")

    chunks = fetch_all_chunks_for_user(user_id, selected_doc_ids)

    if not chunks:
        logger.warning(f"bm25_stage = build_empty_corpus user_id: {user_id}")
        documents = [Document(page_content="", metadata={"id": "empty", "title": ""})]
    else:
        documents = [
            Document(
                page_content=chunk["content"],
                metadata={"id": chunk["id"], "title": chunk["title"]},
            )
            for chunk in chunks
        ]

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = RAG_TOP_K

    logger.info(
        f"bm25_stage = build_done user_id: {user_id} "
        f"document_count: {len(documents)} k: {RAG_TOP_K}"
    )

    return bm25_retriever


def build_ensemble_retriever(
    user_id: str, selected_doc_ids: list[str] | None = None
) -> EnsembleRetriever:
    logger.info(f"ensemble_stage = build_start user_id: {user_id}")

    bm25_retriever = build_bm25_retriever(user_id, selected_doc_ids)
    qdrant_retriever = QdrantRetriever(
        user_id=user_id, selected_doc_ids=selected_doc_ids
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, qdrant_retriever], weights=[0.5, 0.5]
    )

    logger.info(
        f"ensemble_stage = build_done user_id: {user_id} "
        f"retrievers: [bm25, qdrant] weights: [0.5, 0.5]"
    )

    return ensemble_retriever


MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=MULTI_QUERY_PROMPT_TEMPLATE,
)


def build_hybrid_retriever(
    user_id: str, selected_doc_ids: list[str] | None = None
) -> MultiQueryRetriever:
    logger.info(f"hybrid_stage = build_start user_id: {user_id}")

    ensemble_retriever = build_ensemble_retriever(user_id, selected_doc_ids)

    query_generation_llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.0,
    )

    hybrid_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=query_generation_llm,
        prompt=MULTI_QUERY_PROMPT,
        include_original=True,
    )

    logger.info(
        f"hybrid_stage = build_done user_id: {user_id} "
        f"architecture: MultiQuery -> Ensemble(BM25 + Qdrant) llm: {MODEL_NAME}"
    )

    return hybrid_retriever


def build_compression_retriever(
    user_id: str, selected_doc_ids: list[str] | None = None
) -> ContextualCompressionRetriever:
    logger.info(f"compression_stage = build_start user_id: {user_id}")

    base_retriever = build_hybrid_retriever(user_id, selected_doc_ids)

    reranker = CohereRerank(
        model=COHERE_RERANK_MODEL,
        top_n=RAG_TOP_K,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker,
    )

    logger.info(
        f"compression_stage = build_done user_id: {user_id} "
        f"architecture: MultiQuery -> Ensemble(BM25 + Qdrant) -> CohereRerank "
        f"rerank_model: {COHERE_RERANK_MODEL} top_n: {RAG_TOP_K}"
    )

    return compression_retriever
