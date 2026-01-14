import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from .config import config


def get_connection():
    conn = psycopg2.connect(config.DATABASE_URL)
    register_vector(conn)
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Drop existing tables to recreate with correct schema
    cur.execute("DROP TABLE IF EXISTS chunks CASCADE")
    cur.execute("DROP TABLE IF EXISTS documents CASCADE")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            embedding vector({config.EMBEDDING_DIMS}),
            page_number INTEGER,
            chunk_index INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks USING hnsw (embedding vector_cosine_ops)
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized successfully.")


def insert_document(filename: str) -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT id FROM documents WHERE filename = %s",
        (filename,)
    )
    row = cur.fetchone()

    if row:
        doc_id = row[0]
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        conn.commit()
    else:
        cur.execute(
            "INSERT INTO documents (filename) VALUES (%s) RETURNING id",
            (filename,)
        )
        doc_id = cur.fetchone()[0]
        conn.commit()

    cur.close()
    conn.close()
    return doc_id


def document_exists(filename: str) -> bool:
    """Check if a document with the given filename already exists."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT id FROM documents WHERE filename = %s",
        (filename,)
    )
    exists = cur.fetchone() is not None

    cur.close()
    conn.close()
    return exists


def insert_chunks(document_id: int, chunks: list[dict]):
    conn = get_connection()
    cur = conn.cursor()

    data = [
        (
            document_id,
            chunk["content"],
            chunk["embedding"],
            chunk.get("page_number"),
            chunk.get("chunk_index", i)
        )
        for i, chunk in enumerate(chunks)
    ]

    execute_values(
        cur,
        """
        INSERT INTO chunks (document_id, content, embedding, page_number, chunk_index)
        VALUES %s
        """,
        data,
        template="(%s, %s, %s::vector, %s, %s)"
    )

    conn.commit()
    cur.close()
    conn.close()


def search_similar(query_embedding: list[float], top_k: int = 20) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            c.content,
            c.page_number,
            d.filename,
            1 - (c.embedding <=> %s::vector) as similarity
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k)
    )

    results = []
    for row in cur.fetchall():
        results.append({
            "content": row[0],
            "page_number": row[1],
            "filename": row[2],
            "similarity": row[3]
        })

    cur.close()
    conn.close()
    return results
