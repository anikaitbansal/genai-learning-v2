import uuid
from langchain_community.document_loaders import PyMuPDFLoader
from build_knowledge_base import build_chunks_from_document
import tempfile
import os


def prepare_pdf_chunks(file, original_filename: str) -> dict:
    pdf_bytes = file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()
    finally:
        os.remove(tmp_path)

    if not pages:
        raise ValueError("No extractable text found in the uploaded PDF.")

    extracted_text = "\n".join(
        page.page_content for page in pages if page.page_content.strip()
    )

    if not extracted_text.strip():
        raise ValueError("No extractable text found in the uploaded PDF.")

    document = {
        "doc_id": str(uuid.uuid4()),
        "title": original_filename,
        "content": extracted_text,
    }

    chunks = build_chunks_from_document(document)

    if not chunks:
        raise ValueError("No chunks could be created from the uploaded PDF.")

    return {
        "document": document,
        "chunks": chunks,
        "total_characters": len(extracted_text),
        "total_chunks": len(chunks),
    }
