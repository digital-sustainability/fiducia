# Redactions and Non-Public Components

This public PoC release intentionally excludes or sanitizes private operational artifacts.

## Excluded from repository

- Live secrets and credential-bearing `.env` files
- Private key files (for example OAuth PEM files)
- Private datasets and customer documents
- Local runtime artifact directories (`.files/`, generated status/db artifacts under data paths)

## Sanitized in public configuration

- Internal hostnames and private IP addresses
- Environment-specific deployment mounts and routing labels
- Internal-only default vector index names and endpoints

## Still required from deployers

- Valid provider credentials (Together, DeepInfra, Qdrant)
- Reachable Qdrant endpoint and a chosen index name
- OAuth provider setup and key material if auth is enabled
