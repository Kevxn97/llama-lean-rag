from openai import OpenAI

from .config import config
from .retriever import retrieve, format_context


openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

PROMPT_IDENTITY = """Du bist ein hilfreicher Assistent fuer technische Datenblaetter.
Beantworte Fragen ausschliesslich basierend auf den bereitgestellten Dokumentauszuegen."""

PROMPT_SOURCE_RULES = """Regeln zur Quellenbindung:
- Nutze nur Informationen aus dem bereitgestellten Kontext.
- Jede fachliche Aussage muss mindestens eine Kontextreferenz im Format [Quelle n] enthalten.
- Wenn eine Information (z. B. die Bedeutung einer Gruppe/Klasse) nicht im Kontext steht, schreibe explizit:
  "Nicht in den bereitgestellten Quellen enthalten."
- Erfinde keine Werte, Definitionen oder Norminhalte."""

PROMPT_RESPONSE_FORMAT = """Ausgabeformat (immer in genau dieser Reihenfolge, auch bei kurzen Fragen):
Kurzantwort:
- Eine direkte, knappe Antwort auf die Frage mit [Quelle n].

Einordnung und Bedeutung:
- Erklaere relevante Begriffe, Gruppen, Klassen oder Bezeichnungen und was sie bedeuten.
- Falls eine Bedeutung im Kontext fehlt: "Nicht in den bereitgestellten Quellen enthalten."

Bedingungen/Anwendungshinweise:
- Nenne Voraussetzungen, Grenzen, Abhaengigkeiten oder Risiken fuer die Anwendung.

Quellen:
- Liste die verwendeten [Quelle n] mit kurzer Angabe, wofuer sie genutzt wurden."""

PROMPT_LANGUAGE_RULES = """Sprachregel:
- Antworte auf Deutsch, es sei denn, die Frage ist auf Englisch.
- Sei praezise, technisch korrekt und klar strukturiert."""


def _build_system_prompt() -> str:
    """Build the system prompt from maintainable prompt blocks."""
    return "\n\n".join(
        [
            PROMPT_IDENTITY,
            PROMPT_SOURCE_RULES,
            PROMPT_RESPONSE_FORMAT,
            PROMPT_LANGUAGE_RULES,
        ]
    )


def _build_user_prompt(context: str, query: str) -> str:
    """Build a grounded user message with explicit citation requirements."""
    return (
        "Kontext aus den Dokumenten:\n\n"
        f"{context}\n\n"
        "---\n\n"
        "Nutze nur diesen Kontext. "
        "Jede fachliche Aussage muss mindestens eine [Quelle n] enthalten.\n"
        f"Frage: {query}"
    )


def _collect_sources(results: list[dict], max_sources: int) -> str:
    """Collect unique source labels from retrieval results."""
    sources = set()
    for result in results[:max_sources]:
        source = result["filename"]
        if result.get("page_number"):
            source += f" (S. {result['page_number']})"
        sources.add(source)
    return ", ".join(sorted(sources))


SYSTEM_PROMPT = _build_system_prompt()


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
            "content": _build_user_prompt(context=context, query=query),
        }
    )

    response = openai_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=0.1
    )

    answer = response.choices[0].message.content or "Keine Antwort vom Modell erhalten."

    if show_sources:
        sources = _collect_sources(results, config.FINAL_EVIDENCE)
        answer += f"\n\n---\nQuellen: {sources}"

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
