from openai import OpenAI

from .config import config
from .retriever import retrieve, format_context


openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent fuer technische Datenblaetter.
Beantworte Fragen basierend auf den bereitgestellten Dokumenten.

Regeln:
- Antworte nur basierend auf den bereitgestellten Quellen
- Wenn die Information nicht in den Quellen steht, sage das klar
- Nenne die Quelle(n) deiner Antwort
- Antworte auf Deutsch, es sei denn, die Frage ist auf Englisch
- Sei praezise und technisch korrekt
"""


def chat_response(query: str, history: list[dict] | None = None, show_sources: bool = True) -> str:
    """Generate a response for a user query, keeping optional history."""
    history = history or []
    results = retrieve(query)

    if not results:
        return "Keine relevanten Informationen in den Dokumenten gefunden."

    context = format_context(results)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    messages.append(
        {
            "role": "user",
            "content": f"Kontext aus den Dokumenten:\n\n{context}\n\n---\n\nFrage: {query}",
        }
    )

    response = openai_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=0.1
    )

    answer = response.choices[0].message.content

    if show_sources:
        sources = set()
        for r in results[:config.FINAL_EVIDENCE]:
            source = r["filename"]
            if r.get("page_number"):
                source += f" (S. {r['page_number']})"
            sources.add(source)

        answer += f"\n\n---\nQuellen: {', '.join(sorted(sources))}"

    return answer


def chat_loop():
    """Interactive chat loop."""
    print("RAG Chat - Technische Datenblaetter")
    print("Tippe 'exit' oder 'quit' zum Beenden")
    print("-" * 40)
    print()

    history: list[dict] = []

    while True:
        try:
            query = input("Du: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAuf Wiedersehen!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Auf Wiedersehen!")
            break

        print()
        print("Assistent:", end=" ")

        try:
            response = chat_response(query, history=history)
            print(response)
            # keep short running history so follow-up questions have context
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"Fehler: {e}")

        print()
