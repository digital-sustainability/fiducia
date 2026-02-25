# Contributing

## Development setup

1. Install Python 3.11+ and `uv`.
2. Copy `.env.example` to `.env` and configure required values.
3. Install dependencies with `uv sync`.
4. Run locally with `uv run chainlit run src/frontend/app.py --host 0.0.0.0 --port 7855`.

## Pull requests

- Keep changes focused and scoped.
- Add or update docs when behavior/configuration changes.
- Avoid committing secrets, private keys, or internal infrastructure identifiers.
- Prefer clear commit messages explaining intent.

## Code quality

- Keep imports at file top-level.
- Avoid adding environment-specific defaults in source.
- Prefer fail-closed behavior for required secrets and endpoints.

## Reporting issues

Use issues for bugs and feature requests. For security issues, follow `SECURITY.md`.
