import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _install_dependency_stubs():
    """Install minimal stubs so tests can import the module tree offline."""
    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    fake_openai.OpenAI = FakeOpenAI
    sys.modules["openai"] = fake_openai

    fake_dotenv = types.ModuleType("dotenv")

    def _load_dotenv(*args, **kwargs):
        return None

    fake_dotenv.load_dotenv = _load_dotenv
    sys.modules["dotenv"] = fake_dotenv

    fake_psycopg2 = types.ModuleType("psycopg2")

    def _connect(*args, **kwargs):
        raise RuntimeError("psycopg2.connect should not be called in unit tests")

    fake_psycopg2.connect = _connect
    sys.modules["psycopg2"] = fake_psycopg2

    fake_psycopg2_extras = types.ModuleType("psycopg2.extras")

    def _execute_values(*args, **kwargs):
        return None

    fake_psycopg2_extras.execute_values = _execute_values
    sys.modules["psycopg2.extras"] = fake_psycopg2_extras

    fake_pgvector = types.ModuleType("pgvector")
    fake_pgvector_psycopg2 = types.ModuleType("pgvector.psycopg2")

    def _register_vector(*args, **kwargs):
        return None

    fake_pgvector_psycopg2.register_vector = _register_vector
    sys.modules["pgvector"] = fake_pgvector
    sys.modules["pgvector.psycopg2"] = fake_pgvector_psycopg2


_install_dependency_stubs()

chat = importlib.import_module("src.chat")


class ChatPromptTests(unittest.TestCase):
    def test_system_prompt_has_required_sections(self):
        prompt = chat._build_system_prompt()

        self.assertIn("Kurzantwort:", prompt)
        self.assertIn("Einordnung und Bedeutung:", prompt)
        self.assertIn("Bedingungen/Anwendungshinweise:", prompt)
        self.assertIn("Quellen:", prompt)
        self.assertIn("Nicht in den bereitgestellten Quellen enthalten.", prompt)
        self.assertIn("[Quelle n]", prompt)

    def test_build_user_prompt_includes_context_query_and_citation_rule(self):
        prompt = chat._build_user_prompt(
            context="[Quelle 1: TKB-01.pdf, Seite 4]\nBeispieltext",
            query="Welche Spachtelmasse fuer Mosaikparkett?",
        )

        self.assertIn("Kontext aus den Dokumenten:", prompt)
        self.assertIn("[Quelle 1: TKB-01.pdf, Seite 4]", prompt)
        self.assertIn("Jede fachliche Aussage muss mindestens eine [Quelle n] enthalten.", prompt)
        self.assertIn("Frage: Welche Spachtelmasse fuer Mosaikparkett?", prompt)

    def test_collect_sources_deduplicates_and_formats_pages(self):
        results = [
            {"filename": "B.pdf", "page_number": 2},
            {"filename": "A.pdf", "page_number": None},
            {"filename": "B.pdf", "page_number": 2},
        ]

        sources = chat._collect_sources(results, max_sources=8)
        self.assertEqual(sources, "A.pdf, B.pdf (S. 2)")

    def test_chat_response_returns_not_found_message_when_no_results(self):
        with patch("src.chat.retrieve", return_value=[]):
            response = chat.chat_response("Unbekannte Frage")

        self.assertEqual(response, "Keine relevanten Informationen in den Dokumenten gefunden.")

    def test_chat_response_uses_structured_messages_and_appends_source_footer(self):
        llm_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Kurzantwort:\n- Geeignete Gruppe laut Datenblatt [Quelle 1]."
                    )
                )
            ]
        )
        create_mock = Mock(return_value=llm_response)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        results = [{"filename": "TKB-01.pdf", "page_number": 3, "content": "Beispiel"}]

        with patch("src.chat.retrieve", return_value=results), patch(
            "src.chat.format_context",
            return_value="[Quelle 1: TKB-01.pdf, Seite 3]\nBeispiel",
        ), patch("src.chat.openai_client", fake_client):
            response = chat.chat_response("Welche Spachtelmasse fuer Mosaikparkett?")

        call_kwargs = create_mock.call_args.kwargs
        self.assertEqual(call_kwargs["model"], chat.config.LLM_MODEL)
        self.assertEqual(call_kwargs["temperature"], 0.1)
        self.assertIn("Einordnung und Bedeutung:", call_kwargs["messages"][0]["content"])
        self.assertIn("[Quelle 1: TKB-01.pdf, Seite 3]", call_kwargs["messages"][-1]["content"])
        self.assertIn("Frage: Welche Spachtelmasse fuer Mosaikparkett?", call_kwargs["messages"][-1]["content"])
        self.assertTrue(response.endswith("Quellen: TKB-01.pdf (S. 3)"))


if __name__ == "__main__":
    unittest.main()
