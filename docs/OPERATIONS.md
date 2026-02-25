# Operations

## Runtime modes

- Local mode: run via `uv run chainlit run ...`
- Container mode: run via `docker compose up --build`

## Required configuration

Set all required variables in `.env` based on `.env.example`:

- `OPENAI_API_KEY`
- `DEEPINFRA_API_KEY`
- `QDRANT_ENDPOINT`
- `QDRANT_API_KEY`
- `QDRANT_INDEX`
- `ENABLE_OAUTH` (`true` to enable OAuth flow, default `false`)
- OAuth variables if auth is enabled, including `OAUTH_AUTHENTIK_PRIVATE_KEY_PATH`

## Startup

1. Ensure Qdrant is reachable.
2. Ensure provider credentials are valid.
3. Start the app.
4. Verify login flow (if OAuth enabled) and UI readiness.

## Ingestion lifecycle

1. User uploads files through custom UI element.
2. Files are written under `DOCUMENT_BASE_PATH/<collection>/`.
3. A background embedding worker updates `.embedding_status.json`.
4. Vector store is updated with parsed and embedded chunks.

## Health checks

- UI accessible at configured host/port.
- Upload action accepts files and creates collection directory.
- `get_embedding_status` reports progress and completion.
- Retrieval answers include references for indexed collections.

## Common failure modes

- Missing required env vars -> startup/runtime `ValueError`.
- Missing OAuth key material -> auth provider initialization failure.
- Invalid Qdrant credentials/index -> indexing/retrieval failures.
- Provider quota/rate limits -> degraded generation or embedding.

## CORS and deployment note

Current Chainlit config allows broad origins in repo defaults. For internet-facing deployments, restrict `allow_origins` to known trusted origins.
