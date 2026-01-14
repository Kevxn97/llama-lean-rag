import os
from pathlib import Path

from llama_parse import LlamaParse
from openai import OpenAI

from .config import config
from .db import document_exists, insert_document, insert_chunks


parser = LlamaParse(
    api_key=config.LLAMA_CLOUD_API_KEY,
    result_type="markdown",
    skip_diagonal_text=True,
    do_not_unroll_columns=False,
)
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


def parse_pdf(file_path: str) -> list[dict]:
    """Parse PDF using LlamaParse and return pages with text."""
    print(f"  Parsing with LlamaParse...")

    documents = parser.load_data(file_path)

    pages = []
    for i, doc in enumerate(documents):
        pages.append({
            "page_number": i + 1,
            "text": doc.text
        })

    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap

    return chunks


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts using OpenAI."""
    if not texts:
        return []

    response = openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts
    )

    return [item.embedding for item in response.data]


def ingest_pdf(file_path: str, force: bool = False):
    """Ingest a single PDF file into the database."""
    filename = os.path.basename(file_path)
    print(f"Processing: {filename}")

    if not force and document_exists(filename):
        print(f"  Already exists in database, skipping...")
        return

    pages = parse_pdf(file_path)
    print(f"  Parsed {len(pages)} page(s)")

    all_chunks = []
    for page in pages:
        text_chunks = chunk_text(
            page["text"],
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
        for chunk_text_content in text_chunks:
            all_chunks.append({
                "content": chunk_text_content,
                "page_number": page["page_number"]
            })

    print(f"  Created {len(all_chunks)} chunks")

    if not all_chunks:
        print(f"  No content found, skipping...")
        return

    print(f"  Generating embeddings...")
    texts = [c["content"] for c in all_chunks]
    embeddings = get_embeddings(texts)

    for i, emb in enumerate(embeddings):
        all_chunks[i]["embedding"] = emb

    doc_id = insert_document(filename)
    insert_chunks(doc_id, all_chunks)

    print(f"  Stored {len(all_chunks)} chunks in database")


def ingest_directory(directory: str, force: bool = False):
    """Ingest all PDF files from a directory."""
    path = Path(directory)

    if not path.exists():
        print(f"Directory not found: {directory}")
        return

    pdf_files = list(path.glob("*.pdf")) + list(path.glob("*.PDF"))

    if not pdf_files:
        print(f"No PDF files found in: {directory}")
        return

    print(f"Found {len(pdf_files)} PDF file(s)")
    if force:
        print("Force mode: re-ingesting all files")
    print()

    for pdf_file in pdf_files:
        ingest_pdf(str(pdf_file), force=force)
        print()

    print("Ingestion complete!")
