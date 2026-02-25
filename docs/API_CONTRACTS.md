# API Contracts (Chainlit Action Callbacks)

This document describes the callback contracts implemented in `src/frontend/app.py`.

## `send_toast`

- Payload:
  - `message` (string, required)
  - `type` (string, required)
- Behavior: emits a UI toast.
- Return: none.

## `open_pdf_document`

- Payload:
  - `file_path` (string, required)
  - `filename` (string, optional)
  - `page` (number, optional, default `1`)
- Behavior: opens a PDF in Chainlit sidebar.
- Return: none.

## `delete_collection`

- Payload:
  - `collection_name` (string, required)
- Behavior: deletes all vectors/documents for a collection and associated facts cache.
- Return: none (toast-based feedback).

## `delete_file_from_collection`

- Payload:
  - `collection_name` (string, required)
  - `filename` (string, required)
- Behavior: deletes one file from a collection.
- Return: none (toast-based feedback).

## `upload_documents`

- Payload:
  - `collectionName` (string, required)
  - `files` (array, required)
    - `name` (string)
    - `relativePath` (string)
    - `content` (base64 string)
- Behavior:
  - writes files to collection directory
  - starts asynchronous embedding/indexing worker
  - persists worker status in `.embedding_status.json`
- Return:
  - success: `{ "success": true, "embedding_queued": true, "message": string }`
  - error: `{ "success": false, "error": string }`

## `get_embedding_status`

- Payload:
  - `collectionName` (string, required)
- Behavior: reads background status file and returns status object.
- Return:
  - success: `{ "success": true, "response": { ...status } }`
  - error: `{ "success": false, "error": string }`

## `refresh_available_collections`

- Payload: none.
- Behavior: reloads available collection list and refreshes settings.
- Return: `{ "success": true }` or `{ "success": false, "error": string }`

## `reload_vector_store`

- Payload: none.
- Behavior: reinitializes vector manager instance and refreshes settings.
- Return: `{ "success": true }` or `{ "success": false, "error": string }`
