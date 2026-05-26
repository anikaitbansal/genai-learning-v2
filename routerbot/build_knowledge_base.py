import json
import os
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    RAG_METADATA_FILE,
    FAISS_INDEX_FILE,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
)
from embeddings_utils import embed_text

DOCUMENTS_FILE = "knowledge_docs.json"


def load_documents():
    if not os.path.exists(DOCUMENTS_FILE):
        return []
    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def build_chunks_from_document(document):
    if not isinstance(document, dict):
        return []
    doc_id = document.get("doc_id")
    if not doc_id:
        return []
    content = document.get("content", "").strip()
    if not content:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP
    )
    text_chunks = splitter.split_text(content)

    chunks = []
    for index, chunk_text in enumerate(text_chunks, start=1):
        chunk = {
            "id": f"{doc_id}-chunk-{index}",
            "doc_id": doc_id,
            "title": document.get("title", "Untitled"),
            "content": chunk_text,
            "chunk_index": index,
        }
        chunks.append(chunk)
    return chunks


def build_knowledge_base():
    documents = load_documents()
    metadata = []
    embedding_vectors = []
    processed_docs = 0
    skipped_docs = 0

    for document in documents:
        document_chunks = build_chunks_from_document(document)
        if not document_chunks:
            skipped_docs += 1
            continue
        processed_docs += 1
        for chunk in document_chunks:
            embedding = embed_text(chunk["content"])
            metadata.append(chunk)
            embedding_vectors.append(embedding)

    with open(RAG_METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    if embedding_vectors:
        embedding_matrix = np.array(embedding_vectors, dtype="float32")
        faiss.normalize_L2(embedding_matrix)
        vector_dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatIP(vector_dimension)
        index.add(embedding_matrix)
        faiss.write_index(index, FAISS_INDEX_FILE)

    return {
        "message": "Knowledge base built successfully.",
        "total_documents": len(documents),
        "processed_documents": processed_docs,
        "skipped_documents": skipped_docs,
        "total_chunks": len(metadata),
        "knowledge_file": RAG_METADATA_FILE,
    }


if __name__ == "__main__":
    result = build_knowledge_base()
    print(result["message"])
    print(f"Total documents: {result['total_documents']}")
    print(f"Total chunks: {result['total_chunks']}")
    print(f"Knowledge file: {result['knowledge_file']}")
