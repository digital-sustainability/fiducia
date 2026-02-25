# Fiducia (Public PoC)

Fiducia is a document-analysis chatbot PoC built with Chainlit, Haystack, and Qdrant. It supports retrieval-augmented chat over uploaded files, collection management, timeline extraction, and facts extraction.

## Scope and limitations

- This repository is a proof of concept, not a production-hardened product.
- Private datasets, infrastructure-specific deployment details, and secret material are intentionally excluded.
- OAuth is optional and can be enabled via `ENABLE_OAUTH=true`.

See `docs/REDACTIONS_AND_NON_PUBLIC_COMPONENTS.md` for what is intentionally not published.

## Tech stack

- Python 3.11+
- Chainlit UI (`src/frontend/app.py`)
- Haystack retrieval and generation pipeline (`src/backend/chatbot/pipeline.py`)
- Qdrant vector store
- Together and DeepInfra API-compatible model endpoints

## Prerequisites

- Python 3.11+
- `uv` installed
- A running Qdrant instance
- API keys for required providers
- OAuth provider configuration if running with auth enabled (`ENABLE_OAUTH=true`)
- OAuth public-key file path configured via `OAUTH_AUTHENTIK_PRIVATE_KEY_PATH` (store key outside repo)

## Quickstart (local)

1. Copy the environment template:
   - `cp .env.example .env`
2. Fill all required variables in `.env`.
3. Install dependencies:
   - `uv sync`
4. Run the app:
   - `uv run chainlit run src/frontend/app.py --host 0.0.0.0 --port 7855`
5. Open the app in your browser:
   - `http://localhost:7855`

## Quickstart (Docker)

1. Copy and configure environment:
   - `cp .env.example .env`
2. Start with Docker Compose:
   - `docker compose up --build`
3. Access the app:
   - `http://localhost:7855`

## Core workflows

- Upload and embed documents via the `upload_documents` action callback.
- Query across one collection or all collections through the chat interface.
- Generate timeline and facts summaries from indexed content.

## Documentation map

- `docs/ARCHITECTURE.md` - system components and data flow
- `docs/OPERATIONS.md` - runbook, setup, and operational behavior
- `docs/API_CONTRACTS.md` - Chainlit action callback contracts
- `docs/REDACTIONS_AND_NON_PUBLIC_COMPONENTS.md` - excluded/redacted elements
- `docs/show_facts_feature.md` - facts feature details

## Security and contribution

- Security reporting: `SECURITY.md`
- Contribution process: `CONTRIBUTING.md`
- License terms: `LICENSE`
