from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP


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
