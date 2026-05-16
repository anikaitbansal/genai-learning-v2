import json
import os
import logging
import faiss
import numpy as np
from config import RAG_METADATA_FILE, FAISS_INDEX_FILE, RAG_TOP_K, RAG_SIMILARITY_THRESHOLD
from embeddings_utils import embed_text

logger = logging.getLogger(__name__)


class FAISSRetriever:
    def __init__(self):
        self.metadata = self.load_metadata()
        self.index = self.load_index()

        logger.info(
            "FAISSRetriever initialized | metadata_count=%s | index_loaded=%s",
            len(self.metadata),
            self.index is not None
        )

    def load_metadata(self):
        if not os.path.exists(RAG_METADATA_FILE):
            logger.warning(f"Retriever metadata file not found: {RAG_METADATA_FILE}")
            return []

        with open(RAG_METADATA_FILE, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        logger.info(f"Retriever metadata loaded successfully | metadata_count={len(metadata)}")
        return metadata

    def load_index(self):
        if not os.path.exists(FAISS_INDEX_FILE):
            logger.warning(f"FAISS index file not found: {FAISS_INDEX_FILE}")
            return None

        index = faiss.read_index(FAISS_INDEX_FILE)
        logger.info("FAISS index loaded successfully")
        return index

    def save_metadata(self):
        with open(RAG_METADATA_FILE, "w", encoding="utf-8") as file:
            json.dump(self.metadata, file, indent=4)

        logger.info(f"Retriever metadata saved successfully | metadata_count={len(self.metadata)}")

    def save_index(self):
        if self.index is None:
            logger.warning("save_index skipped because FAISS index is None")
            return

        faiss.write_index(self.index, FAISS_INDEX_FILE)
        logger.info("FAISS index saved successfully")

    def add_chunks(self, chunks: list[dict]):
        logger.info(f"retriever_stage=add_chunks_start chunk_count={len(chunks)}")

        if not chunks:
            logger.info("retriever_stage=add_chunks_skipped reason=no_chunks")
            return {
                "added_chunks": 0
            }

        embedding_vectors = []
        valid_chunks = []

        for chunk in chunks:
            chunk_content = chunk.get("content", "").strip()

            if not chunk_content:
                logger.info("retriever_stage=chunk_skipped reason=empty_content")
                continue

            embedding = embed_text(chunk_content)
            embedding_vectors.append(embedding)
            valid_chunks.append(chunk)

        if not embedding_vectors:
            logger.info("retriever_stage=add_chunks_skipped reason=no_valid_embeddings")
            return {
                "added_chunks": 0
            }

        embedding_matrix = np.array(embedding_vectors, dtype="float32")
        faiss.normalize_L2(embedding_matrix)

        if self.index is None:
            vector_dimension = embedding_matrix.shape[1]
            self.index = faiss.IndexFlatIP(vector_dimension)
            logger.info(f"retriever_stage=index_created vector_dimension={vector_dimension}")

        self.index.add(embedding_matrix)
        self.metadata.extend(valid_chunks)

        self.save_index()
        self.save_metadata()

        logger.info(
            f"retriever_stage=add_chunks_done added_chunks={len(valid_chunks)} metadata_count={len(self.metadata)}"
        )

        return {
            "added_chunks": len(valid_chunks)
        }

    def retrieve(self, query, top_k=RAG_TOP_K):
        logger.info(
            "retriever_stage=retrieve_start query_length=%s top_k=%s similarity_threshold=%s",
            len(query.strip()),
            top_k,
            RAG_SIMILARITY_THRESHOLD
        )

        if not self.metadata or self.index is None:
            logger.warning(
                "retriever_stage=retrieve_skipped reason=missing_metadata_or_index metadata_count=%s index_loaded=%s",
                len(self.metadata),
                self.index is not None
            )
            return []

        query_embedding = embed_text(query)
        query_vector = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_vector)

        search_k = min(max(top_k * 3, top_k), len(self.metadata))
        distances, indices = self.index.search(query_vector, search_k)

        logger.info(
            "retriever_stage=faiss_search_done raw_scores=%s raw_indices=%s",
            distances[0].tolist(),
            indices[0].tolist()
        )

        results = []
        seen_contents = set()

        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                logger.info(
                    "retriever_stage=chunk_skipped reason=invalid_index idx=%s score=%s",
                    idx,
                    round(float(score), 4)
                )
                continue

            similarity_score = float(score)

            if similarity_score < RAG_SIMILARITY_THRESHOLD:
                logger.info(
                    "retriever_stage=chunk_skipped reason=below_threshold chunk_id=%s score=%s threshold=%s",
                    self.metadata[idx]["id"],
                    round(similarity_score, 4),
                    RAG_SIMILARITY_THRESHOLD
                )
                continue

            chunk = self.metadata[idx]
            chunk_content = chunk["content"].strip()

            if not chunk_content:
                logger.info(
                    "retriever_stage=chunk_skipped reason=empty_content chunk_id=%s",
                    chunk["id"]
                )
                continue

            normalized_content = " ".join(chunk_content.split()).lower()

            if normalized_content in seen_contents:
                logger.info(
                    "retriever_stage=chunk_skipped reason=duplicate_content chunk_id=%s",
                    chunk["id"]
                )
                continue

            seen_contents.add(normalized_content)

            results.append({
                "id": chunk["id"],
                "title": chunk["title"],
                "content": chunk["content"],
                "score": round(similarity_score, 4)
            })

            if len(results) >= top_k:
                break

        logger.info("retriever_stage=retrieve_done returned_count=%s", len(results))
        return results