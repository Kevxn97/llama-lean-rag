# AGENTS

## Purpose
Repo working notes for Codex.

## Repo workflows (CLI)
- Initialize DB (drops/recreates tables): `python main.py init-db`
- Ingest PDFs from a directory: `python main.py ingest <path> [-f|--force]`
- Start chat loop: `python main.py chat`

## Environment
Required for normal operation:
- `DATABASE_URL` (Postgres; defaults to `postgresql://localhost:5432/pdf_rag`)
- `OPENAI_API_KEY`
- `LLAMA_CLOUD_API_KEY`

Optional overrides:
- `EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `EMBEDDING_DIMS` (default `1536`)
- `LLM_MODEL` (default `gpt-4o-mini`)
- `DATA_DIR` (default `./data`)
- `TOPK_VEC` (default `20`)
- `FINAL_EVIDENCE` (default `8`)

## Notes
- `init-db` requires the Postgres `vector` extension (created automatically on init).
- Ingest expects PDFs in the provided directory (case-insensitive `.pdf`).

## TODO
- Add test/lint commands once they exist in the repo.
