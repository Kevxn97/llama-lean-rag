from openai import OpenAI

from .config import config
from .db import search_similar


openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


def get_query_embedding(query: str) -> list[float]:
    """Get embedding for a query string."""
    response = openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=query
    )
    return response.data[0].embedding


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """Retrieve relevant chunks for a query."""
    if top_k is None:
        top_k = config.TOPK_VEC

    query_embedding = get_query_embedding(query)
    results = search_similar(query_embedding, top_k=top_k)

    return results


def format_context(results: list[dict], max_chunks: int = None) -> str:
    """Format retrieved chunks as context for the LLM."""
    if max_chunks is None:
        max_chunks = config.FINAL_EVIDENCE

    context_parts = []

    for i, result in enumerate(results[:max_chunks]):
        source = f"{result['filename']}"
        if result.get("page_number"):
            source += f", Seite {result['page_number']}"

        context_parts.append(
            f"[Quelle {i+1}: {source}]\n{result['content']}"
        )

    return "\n\n---\n\n".join(context_parts)
