import uuid
from pypdf import PdfReader
from build_knowledge_base import build_chunks_from_document


def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    pages_text = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_text.append(page_text.strip())

    return "\n".join(pages_text).strip()


def ingest_pdf_file(file, original_filename: str, retriever) -> dict:
    extracted_text = extract_text_from_pdf(file)

    if not extracted_text:
        raise ValueError("No extractable text found in the uploaded PDF.")

    document = {
        "doc_id": str(uuid.uuid4()),
        "title": original_filename,
        "content": extracted_text
    }

    chunks = build_chunks_from_document(document)

    if not chunks:
        raise ValueError("No chunks could be created from the uploaded PDF.")

    retriever_result = retriever.add_chunks(chunks)

    return {
        "document": document,
        "chunks": chunks,
        "total_characters": len(extracted_text),
        "total_chunks": len(chunks),
        "added_chunks": retriever_result["added_chunks"]
    }