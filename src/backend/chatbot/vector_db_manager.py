import functools
import os
import hashlib
from pathlib import Path
import time
from typing import Optional, List, Set, Dict, Any
from dataclasses import dataclass

import typer
from haystack import Pipeline, Document, component
from haystack.utils import Secret
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder, OpenAIDocumentEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from docling_haystack.converter import ExportType
from docling_haystack.converter import DoclingConverter
from docling.chunking import HybridChunker

from backend.utils import relative_project_path
from backend.chatbot.components import CustomCleaner, TokenEfficientContextualizer, FileMetadataInjector
from backend.chatbot.utils import model_warmup


app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Vector Database Management Tool - Create, update, and manage document embeddings",
    add_completion=False,
)


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@dataclass
class DocumentMetadata:
    """Metadata for tracking document state in vector store."""

    file_path: str
    file_hash: str
    chunk_count: int


class VectorStoreManager:
    """Manages incremental updates to the vector store and provides query utility functions."""

    def __init__(self, recreate_index: bool = False):
        self.document_store = QdrantDocumentStore(
            url=_require_env_var("QDRANT_ENDPOINT"),
            api_key=Secret.from_env_var("QDRANT_API_KEY"),
            index=_require_env_var("QDRANT_INDEX"),
            recreate_index=recreate_index,
            progress_bar=True,
            use_sparse_embeddings=True,
            embedding_dim=int(os.getenv("QDRANT_EMBEDDING_DIM", 1024)),
        )
        self.pipeline = create_pipeline(self.document_store)

    def get_available_collections(self) -> list[str]:
        """Get all available collections in the vector store."""
        try:
            # Get all unique collection names from the document store
            all_docs = self.document_store.filter_documents()
            collections = set()

            for doc in all_docs:
                if hasattr(doc, "payload") and "collection_name" in doc.payload:
                    collections.add(doc.payload["collection_name"])
                elif hasattr(doc, "meta") and "collection_name" in doc.meta:
                    # Fallback to meta for older documents
                    collections.add(doc.meta["collection_name"])

            return sorted(list(collections))
        except Exception as e:
            print(f"Error getting collections: {e}")
            return []

    def get_collection_stats(self) -> dict:
        """Get statistics about each collection."""
        try:
            all_docs = self.document_store.filter_documents()
            stats = {}

            for doc in all_docs:
                collection_name = None
                if hasattr(doc, "payload") and "collection_name" in doc.payload:
                    collection_name = doc.payload["collection_name"]
                elif hasattr(doc, "meta") and "collection_name" in doc.meta:
                    collection_name = doc.meta["collection_name"]

                if collection_name:
                    if collection_name not in stats:
                        stats[collection_name] = {"document_count": 0, "file_types": set(), "files": set()}

                    stats[collection_name]["document_count"] += 1

                    # Get file extension
                    if hasattr(doc, "payload") and "file_extension" in doc.payload:
                        if doc.payload["file_extension"]:
                            stats[collection_name]["file_types"].add(doc.payload["file_extension"])

                    # Get filename
                    if hasattr(doc, "payload") and "filename" in doc.payload:
                        stats[collection_name]["files"].add(doc.payload["filename"])
                    elif hasattr(doc, "meta") and "filename" in doc.meta:
                        stats[collection_name]["files"].add(doc.meta["filename"])

            # Convert sets to lists for JSON serialization
            for collection_name, data in stats.items():
                data["file_types"] = sorted(list(data["file_types"]))
                data["file_count"] = len(data["files"])
                data["files"] = sorted(list(data["files"]))

            return stats
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}

    def delete_collection(self, collection_name: str) -> bool:
        """Delete all documents from a specific collection."""
        try:
            # Create filter to match collection - using correct Haystack filter syntax
            collection_filter = {"field": "meta.collection_name", "operator": "==", "value": collection_name}

            # Get all documents from the collection first
            docs_to_delete = self.document_store.filter_documents(filters=collection_filter)

            if not docs_to_delete:
                print(f"No documents found in collection '{collection_name}'")
                return False

            # Delete documents by their IDs
            doc_ids = [doc.id for doc in docs_to_delete]
            self.document_store.delete_documents(document_ids=doc_ids)

            print(f"Successfully deleted {len(doc_ids)} documents from collection '{collection_name}'")
            return True

        except Exception as e:
            print(f"Error deleting collection '{collection_name}': {e}")
            return False

    def delete_file_from_collection(self, collection_name: str, filename: str) -> bool:
        """Delete all document chunks from a specific file within a collection."""
        try:
            # Create filter to match both collection and filename - using correct Haystack filter syntax
            file_filter = {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.collection_name", "operator": "==", "value": collection_name},
                    {"field": "meta.filename", "operator": "==", "value": filename},
                ],
            }

            # Get all document chunks from this specific file
            docs_to_delete = self.document_store.filter_documents(filters=file_filter)

            if not docs_to_delete:
                print(f"No documents found for file '{filename}' in collection '{collection_name}'")
                return False

            # Delete documents by their IDs
            doc_ids = [doc.id for doc in docs_to_delete]
            self.document_store.delete_documents(document_ids=doc_ids)

            print(
                f"Successfully deleted {len(doc_ids)} document chunks for file '{filename}' from collection '{collection_name}'"
            )
            return True

        except Exception as e:
            print(f"Error deleting file '{filename}' from collection '{collection_name}': {e}")
            return False

    def get_collection_timeline(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        Retrieves and sorts all timeline events for a given collection.

        :param collection_name: The name of the collection to retrieve the timeline for.
        :return: A list of chronologically sorted, unique timeline events.
        """
        # 1. Filter documents by collection name
        filters = {"field": "meta.collection_name", "operator": "==", "value": collection_name}
        docs_in_collection = self.document_store.filter_documents(filters=filters)

        all_events = []
        seen_events = set()  # To track and remove duplicate events

        # 2. Iterate through documents and extract timeline events
        for doc in docs_in_collection:
            timeline = doc.meta.get("timeline")
            if not timeline or not isinstance(timeline, dict):
                continue

            for date, events_on_date in timeline.items():
                for event in events_on_date:
                    # Create a unique identifier for the event to handle duplicates
                    event_identifier = f"{date}:{event.get('label')}"
                    if event_identifier not in seen_events:
                        event_data = {
                            "date": date,
                            "label": event.get("label"),
                            "event_type": event.get("event_type"),
                            "participants": event.get("participants", []),
                            "source_file": doc.meta.get("filename", "Unknown"),
                            "source_path": doc.meta.get("file_path", ""),
                            "collection_name": doc.meta.get("collection_name", collection_name),
                            "page_start": doc.meta.get("page_start", 1),
                            "page_end": doc.meta.get("page_end", 1),
                            "chunk_id": doc.id if hasattr(doc, "id") else None,
                        }
                        all_events.append(event_data)
                        seen_events.add(event_identifier)

        # 3. Sort events chronologically
        all_events.sort(key=lambda x: x["date"])

        return all_events

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _get_existing_document_metadata(self) -> Set[str]:
        """Get metadata of existing documents in the vector store."""
        try:
            # Query all documents to get their metadata
            all_docs = self.document_store.filter_documents()
            existing_files = set()
            for doc in all_docs:
                # Try to get file_path and file_hash from our injected metadata
                if hasattr(doc, "meta"):
                    file_path = doc.meta.get("file_path")
                    file_hash = doc.meta.get("file_hash")

                    if file_path and file_hash:
                        existing_files.add(f"{file_path}:{file_hash}")

                    # Also check for Docling metadata as fallback
                    elif (
                        "dl_meta" in doc.meta
                        and "meta" in doc.meta["dl_meta"]
                        and "origin" in doc.meta["dl_meta"]["meta"]
                    ):

                        origin = doc.meta["dl_meta"]["meta"]["origin"]
                        filename = origin.get("filename")
                        binary_hash = origin.get("binary_hash")

                        if filename and binary_hash:
                            existing_files.add(f"{filename}:{binary_hash}")

            return existing_files
        except Exception as e:
            print(f"Warning: Could not retrieve existing document metadata: {e}")
            return set()

    def _filter_new_files(self, file_paths: List[Path]) -> List[Path]:
        """Filter out files that are already embedded with the same hash."""
        existing_metadata = self._get_existing_document_metadata()
        new_files = []

        for file_path in file_paths:
            if file_path.is_file():
                try:
                    file_hash = self._calculate_file_hash(file_path)
                    # Check both absolute path and filename
                    file_key_abs = f"{str(file_path)}:{file_hash}"
                    file_key_name = f"{file_path.name}:{file_hash}"

                    if file_key_abs not in existing_metadata and file_key_name not in existing_metadata:
                        new_files.append(file_path)
                    else:
                        print(f"Skipping {file_path.name} (already embedded with same hash)")
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")

        return new_files

    def _delete_documents_by_file_path(self, file_path: str) -> int:
        """Delete all documents/chunks associated with a specific file path."""
        try:
            # Get all documents and filter manually since we need flexible matching
            all_docs = self.document_store.filter_documents()
            docs_to_delete = []

            file_path_obj = Path(file_path)

            for doc in all_docs:
                if hasattr(doc, "meta"):
                    # Check our injected file_path metadata
                    doc_file_path = doc.meta.get("file_path")
                    if doc_file_path:
                        doc_path_obj = Path(doc_file_path)
                        if (
                            str(doc_path_obj) == str(file_path_obj)
                            or doc_path_obj.name == file_path_obj.name
                            or str(doc_path_obj) == file_path
                            or doc_path_obj.name == file_path
                        ):
                            docs_to_delete.append(doc)
                            continue

                    # Check Docling metadata as fallback
                    if (
                        "dl_meta" in doc.meta
                        and "meta" in doc.meta["dl_meta"]
                        and "origin" in doc.meta["dl_meta"]["meta"]
                    ):

                        filename = doc.meta["dl_meta"]["meta"]["origin"].get("filename")
                        if filename:
                            filename_obj = Path(filename)
                            if (
                                str(filename_obj) == str(file_path_obj)
                                or filename_obj.name == file_path_obj.name
                                or str(filename_obj) == file_path
                                or filename_obj.name == file_path
                            ):
                                docs_to_delete.append(doc)

            if docs_to_delete:
                doc_ids = [doc.id for doc in docs_to_delete]
                self.document_store.delete_documents(doc_ids)
                print(f"Deleted {len(doc_ids)} chunks for file: {file_path}")
                return len(doc_ids)
            else:
                print(f"No documents found for file: {file_path}")
                return 0

        except Exception as e:
            print(f"Error deleting documents for {file_path}: {e}")
            return 0

    def _process_files(self, file_paths: List[Path], skip_existing: bool = True) -> None:
        """Process and embed a list of files."""
        if not file_paths:
            print("No files to process.")
            return

        if skip_existing:
            file_paths = self._filter_new_files(file_paths)
            if not file_paths:
                print("All files are already embedded and up to date.")
                return

        print(f"Processing {len(file_paths)} files...")

        # Process each file individually
        for file_path in file_paths:
            try:
                print(f"Embedding file: {file_path.name}")

                # Run pipeline for this specific file
                result = self.pipeline.run({"converter": {"paths": [file_path]}})

                # Update metadata for all generated documents
                if "writer" in result and "documents_written" in result["writer"]:
                    doc_count = result["writer"]["documents_written"]
                    print(f"Successfully embedded {doc_count} chunks from {file_path.name}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue


def create_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    assert "TEI_EMBEDDING_ENDPOINT" in os.environ, "TEI_EMBEDDING_ENDPOINT environment variable is not set!"

    pipeline = Pipeline()

    converter = DoclingConverter(
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer="Qwen/Qwen3-Embedding-0.6B", max_tokens=800),
    )

    cleaner = DocumentCleaner(ascii_only=False)
    custom_cleaner = CustomCleaner()
    metadata_injector = FileMetadataInjector(document_base_path=relative_project_path(os.getenv("DOCUMENT_BASE_PATH")))
    token_efficient_contextualizer = TokenEfficientContextualizer(
        api_base_url="https://api.together.xyz/v1",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        max_workers=int(os.getenv("CONTEXTUALIZER_MAX_WORKERS", "3")),
        batch_size=int(os.getenv("CONTEXTUALIZER_BATCH_SIZE", "5")),
        rate_limit_delay=float(os.getenv("CONTEXTUALIZER_RATE_LIMIT_DELAY", "0.1")),
        requests_per_minute=int(os.getenv("CONTEXTUALIZER_REQUESTS_PER_MINUTE", "80")),
    )

    sparse_embedder = FastembedSparseDocumentEmbedder(model="Qdrant/bm25", progress_bar=True)

    # Use this code instead if you want to use a local embedding model through a Hugging Face Text Embeddings Inference API:
    # dense_embedder = HuggingFaceAPIDocumentEmbedder(
    #     api_type="text_embeddings_inference",
    #     api_params={"url": os.getenv("TEI_EMBEDDING_ENDPOINT")},
    #     meta_fields_to_embed=["additional_embedding_info"],
    #     embedding_separator="\n\n",
    # )

    dense_embedder = OpenAIDocumentEmbedder(
        model="Qwen/Qwen3-Embedding-0.6B",
        api_base_url="https://api.deepinfra.com/v1/openai",
        dimensions=int(os.getenv("QDRANT_EMBEDDING_DIM", "1024")),
        api_key=Secret.from_env_var("DEEPINFRA_API_KEY"),
        meta_fields_to_embed=["additional_embedding_info"],
        embedding_separator="\n\n",
    )

    # Monkey patch the embedder; TODO: fix as soon as the bug is fixed in haystack
    dense_embedder.client.embeddings.create = functools.partial(
        dense_embedder.client.embeddings.create, encoding_format="float"
    )

    writer = DocumentWriter(document_store=document_store)

    pipeline.add_component("converter", converter)
    pipeline.add_component("cleaner", cleaner)
    pipeline.add_component("custom_cleaner", custom_cleaner)
    pipeline.add_component("metadata_injector", metadata_injector)
    pipeline.add_component("contextualizer", token_efficient_contextualizer)

    pipeline.add_component("sparse_embedder", sparse_embedder)
    pipeline.add_component("dense_embedder", dense_embedder)
    pipeline.add_component("writer", writer)

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "custom_cleaner")
    pipeline.connect("custom_cleaner", "metadata_injector")
    pipeline.connect("metadata_injector", "contextualizer")
    pipeline.connect("contextualizer", "sparse_embedder")
    pipeline.connect("sparse_embedder", "dense_embedder")

    pipeline.connect("dense_embedder", "writer")

    return pipeline


@app.command()
def create_vector_db(
    document_directory: Optional[str] = None,
    recreate: bool = True,
) -> None:
    """Create or recreate the vector database from all documents in a directory."""
    assert document_directory or os.environ.get("DOCUMENT_BASE_PATH"), "No document directory provided!"

    document_directory = (
        Path(document_directory) if document_directory else relative_project_path(os.environ.get("DOCUMENT_BASE_PATH"))
    )

    t0 = time.time()
    # model_warmup(model="Embedding")  # Use this for local embedding models
    t1 = time.time()
    print(f"Embedding model successfully warmed up in {t1 - t0:.2f} seconds.")

    manager = VectorStoreManager(recreate_index=recreate)

    # Get all files in directory
    all_files = [f for f in document_directory.rglob("*") if f.is_file() and not f.name.startswith(".")]

    print(f"Found {len(all_files)} files in {document_directory}")
    manager._process_files(all_files, skip_existing=not recreate)

    print("Vector database creation completed.")


@app.command()
def add_file(
    file_path: str,
    force: bool = typer.Option(False, "--force", "-f", help="Force re-embedding even if file already exists"),
) -> None:
    """Add a single file to the vector database."""
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        return

    if not file_path.is_file():
        print(f"Error: {file_path} is not a file.")
        return

    t0 = time.time()
    # model_warmup(model="Embedding")  # Use this for local embedding models
    t1 = time.time()
    print(f"Embedding model successfully warmed up in {t1 - t0:.2f} seconds.")

    manager = VectorStoreManager(recreate_index=False)

    # If force is True, delete existing embeddings first
    if force:
        manager._delete_documents_by_file_path(str(file_path))

    manager._process_files([file_path], skip_existing=not force)
    print(f"File {file_path.name} processing completed.")


@app.command()
def add_directory(
    directory_path: str,
    force: bool = typer.Option(False, "--force", "-f", help="Force re-embedding even if files already exist"),
) -> None:
    """Add all files from a directory to the vector database."""
    directory_path = Path(directory_path)

    if not directory_path.exists():
        print(f"Error: Directory {directory_path} does not exist.")
        return

    if not directory_path.is_dir():
        print(f"Error: {directory_path} is not a directory.")
        return

    t0 = time.time()
    # model_warmup(model="Embedding")  # Use this for local embedding models
    t1 = time.time()
    print(f"Embedding model successfully warmed up in {t1 - t0:.2f} seconds.")

    manager = VectorStoreManager(recreate_index=False)

    # Get all files in directory
    all_files = [f for f in directory_path.rglob("*") if f.is_file() and not f.name.startswith(".")]

    print(f"Found {len(all_files)} files in {directory_path}")

    # If force is True, delete existing embeddings first
    if force:
        for file_path in all_files:
            manager._delete_documents_by_file_path(str(file_path))

    manager._process_files(all_files, skip_existing=not force)
    print(f"Directory {directory_path} processing completed.")


@app.command()
def delete_file(file_path: str) -> None:
    """Delete embeddings for a specific file from the vector database."""
    manager = VectorStoreManager(recreate_index=False)
    deleted_count = manager._delete_documents_by_file_path(file_path)

    if deleted_count > 0:
        print(f"Successfully deleted {deleted_count} chunks for file: {file_path}")
    else:
        print(f"No embeddings found for file: {file_path}")


@app.command()
def delete_directory(directory_path: str) -> None:
    """Delete embeddings for all files in a directory from the vector database."""
    directory_path = Path(directory_path)

    if not directory_path.exists():
        print(f"Error: Directory {directory_path} does not exist.")
        return

    manager = VectorStoreManager(recreate_index=False)

    # Get all files in directory
    all_files = [f for f in directory_path.rglob("*") if f.is_file() and not f.name.startswith(".")]

    total_deleted = 0
    for file_path in all_files:
        deleted_count = manager._delete_documents_by_file_path(str(file_path))
        total_deleted += deleted_count

    print(f"Successfully deleted {total_deleted} chunks for {len(all_files)} files in directory: {directory_path}")


@app.command()
def update_directory(
    directory_path: Optional[str] = None,
) -> None:
    """Update the vector database by adding new/changed files and keeping existing ones."""
    assert directory_path or os.environ.get("DOCUMENT_BASE_PATH"), "No directory provided!"

    directory_path = (
        Path(directory_path) if directory_path else relative_project_path(os.environ.get("DOCUMENT_BASE_PATH"))
    )

    t0 = time.time()
    # model_warmup(model="Embedding")  # Use this for local embedding models
    t1 = time.time()
    print(f"Embedding model successfully warmed up in {t1 - t0:.2f} seconds.")

    manager = VectorStoreManager(recreate_index=False)

    # Get all files in directory
    all_files = [f for f in directory_path.rglob("*") if f.is_file() and not f.name.startswith(".")]

    print(f"Found {len(all_files)} files in {directory_path}")
    manager._process_files(all_files, skip_existing=True)

    print("Vector database update completed.")


@app.command()
def status() -> None:
    """Show the current status of the vector database."""
    try:
        manager = VectorStoreManager(recreate_index=False)

        # Get all documents
        all_docs = manager.document_store.filter_documents()

        if not all_docs:
            print("Vector database is empty.")
            return

        # Analyze documents by file
        file_stats = {}
        collection_stats = {}

        for doc in all_docs:
            if hasattr(doc, "meta"):
                # Prefer payload fields if available (new format), fall back to meta
                collection_name = None
                filename = None
                subdirectory = ""
                file_path = None
                file_hash = None

                if hasattr(doc, "payload"):
                    collection_name = doc.payload.get("collection_name")
                    filename = doc.payload.get("filename")
                    subdirectory = doc.payload.get("subdirectory", "")
                    file_hash = doc.payload.get("file_hash")
                    file_path = doc.meta.get("file_path")  # Still in meta

                # Fallback to meta fields for older documents
                if not collection_name:
                    collection_name = doc.meta.get("collection_name")
                if not filename:
                    filename = doc.meta.get("filename")
                if not subdirectory:
                    subdirectory = doc.meta.get("subdirectory", "")
                if not file_hash:
                    file_hash = doc.meta.get("file_hash")
                if not file_path:
                    file_path = doc.meta.get("file_path")

                # Final fallback to Docling metadata for very old documents
                if not file_path and "dl_meta" in doc.meta:
                    dl_meta = doc.meta["dl_meta"]
                    if "meta" in dl_meta and "origin" in dl_meta["meta"]:
                        origin = dl_meta["meta"]["origin"]
                        file_path = origin.get("filename")
                        if not file_hash:
                            file_hash = str(origin.get("binary_hash", "unknown"))
                        if not filename:
                            filename = os.path.basename(file_path) if file_path else "unknown"

                if file_path and collection_name:
                    key = (
                        f"{collection_name}/{subdirectory}/{filename}"
                        if subdirectory
                        else f"{collection_name}/{filename}"
                    )

                    if key not in file_stats:
                        file_stats[key] = {
                            "chunks": 0,
                            "hash": file_hash or "unknown",
                            "exists": Path(file_path).exists() if file_path else False,
                            "collection": collection_name or "unknown",
                            "filename": filename or "unknown",
                            "subdirectory": subdirectory,
                        }
                    file_stats[key]["chunks"] += 1

                    # Track collection stats
                    if collection_name not in collection_stats:
                        collection_stats[collection_name] = {"files": 0, "chunks": 0}
                    if file_stats[key]["chunks"] == 1:  # First chunk for this file
                        collection_stats[collection_name]["files"] += 1
                    collection_stats[collection_name]["chunks"] += 1

        print(f"Vector Database Status:")
        print(f"Total documents/chunks: {len(all_docs)}")
        print(f"Total files: {len(file_stats)}")
        print(f"Collections: {len(collection_stats)}")
        print()

        # Group by collection for better display
        collections = {}
        for key, stats in file_stats.items():
            collection = stats["collection"]
            if collection not in collections:
                collections[collection] = []
            collections[collection].append((key, stats))

        for collection_name, files in sorted(collections.items()):
            print(f"Collection: {collection_name}")
            for key, stats in sorted(files):
                status_indicator = "✓" if stats["exists"] else "✗"
                hash_display = stats["hash"][:8] + "..." if len(str(stats["hash"])) > 8 else str(stats["hash"])
                subdirectory = stats.get("subdirectory", "")
                display_path = f"{subdirectory}/{stats['filename']}" if subdirectory else stats["filename"]
                print(f"  {status_indicator} {display_path}: {stats['chunks']} chunks (hash: {hash_display})")
            print()

    except Exception as e:
        print(f"Error accessing vector database: {e}")


@app.command()
def get_timeline(
    collection_name: str = typer.Argument(..., help="The name of the collection to retrieve the timeline for.")
) -> None:
    """Retrieves and displays the chronological timeline for a specific collection."""
    print(f"Retrieving timeline for collection: '{collection_name}'...")
    manager = VectorStoreManager()
    timeline = manager.get_collection_timeline(collection_name)

    if not timeline:
        print("No timeline events found for this collection.")
        return

    print("\n--- Chronological Timeline ---")
    for event in timeline:
        participants = ", ".join(event["participants"]) if event["participants"] else "N/A"
        print(f"Date: {event['date']}")
        print(f"  Event: {event['label']}")
        print(f"  Type: {event['event_type']}")
        print(f"  Participants: {participants}")
        print("-" * 10)

    print(f"\nFound {len(timeline)} unique events.")


# Environment variables for TokenEfficientContextualizer parallelization:
# CONTEXTUALIZER_MAX_WORKERS=3          # Number of parallel workers (default: 3)
# CONTEXTUALIZER_BATCH_SIZE=5           # Chunks per batch (default: 5)
# CONTEXTUALIZER_RATE_LIMIT_DELAY=0.1   # Delay between requests in seconds (default: 0.1)
# CONTEXTUALIZER_REQUESTS_PER_MINUTE=80 # Max API calls per minute (default: 80)


if __name__ == "__main__":
    app()
