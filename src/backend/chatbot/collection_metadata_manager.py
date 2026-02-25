import os
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from haystack import Document
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils import Secret

from backend.utils import relative_project_path
from backend.chatbot.vector_db_manager import VectorStoreManager
from backend.chatbot.utils import get_prompt_builder


@dataclass
class CollectionFacts:
    """Container for collection facts data."""

    collection_name: str
    facts_summary: str


class CollectionMetadataManager:
    """Manages collection metadata including facts summaries stored in SQLite."""

    def __init__(self):
        # Get the document base path and create the SQLite database there
        doc_base_path = relative_project_path(os.getenv("DOCUMENT_BASE_PATH"))
        self.db_path = Path(doc_base_path) / "collection_metadata.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

        # Initialize LLM for fact extraction
        self.llm = OpenAIGenerator(
            api_base_url="https://api.together.xyz/v1",
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            generation_kwargs={"temperature": 0.1},
        )

    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    collection_name TEXT PRIMARY KEY,
                    facts_summary TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    def get_facts_for_collection(self, collection_name: str) -> Optional[str]:
        """
        Retrieve existing facts summary for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Facts summary if exists, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT facts_summary FROM facts WHERE collection_name = ?", (collection_name,))
            result = cursor.fetchone()
            return result[0] if result else None

    def save_facts_for_collection(self, collection_name: str, facts_summary: str) -> None:
        """
        Save facts summary for a collection.

        Args:
            collection_name: Name of the collection
            facts_summary: Generated facts summary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO facts (collection_name, facts_summary, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                (collection_name, facts_summary),
            )
            conn.commit()

    def delete_facts_for_collection(self, collection_name: str) -> bool:
        """
        Delete facts summary for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            True if facts were deleted, False if no facts existed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM facts WHERE collection_name = ?", (collection_name,))
            conn.commit()
            return cursor.rowcount > 0

    def invalidate_facts_for_collection(self, collection_name: str) -> None:
        """
        Invalidate (delete) facts for a collection when documents are modified.
        This should be called when new documents are added to force regeneration.

        Args:
            collection_name: Name of the collection
        """
        deleted = self.delete_facts_for_collection(collection_name)
        if deleted:
            print(f"Invalidated existing facts for collection '{collection_name}' due to document changes")

    def list_collections_with_facts(self) -> List[str]:
        """
        Get list of collections that have facts summaries.

        Returns:
            List of collection names with existing facts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT collection_name FROM facts ORDER BY collection_name")
            return [row[0] for row in cursor.fetchall()]

    def _get_chunks_grouped_by_file(
        self, vector_manager: VectorStoreManager, collection_name: str
    ) -> Dict[str, List[Document]]:
        """
        Retrieve all chunks for a collection, grouped by filename and ordered by chunk order.

        Args:
            vector_manager: VectorStoreManager instance
            collection_name: Name of the collection

        Returns:
            Dictionary mapping filename to list of ordered Document chunks
        """
        # Get all documents for the collection
        collection_filter = {"field": "meta.collection_name", "operator": "==", "value": collection_name}
        docs_in_collection = vector_manager.document_store.filter_documents(filters=collection_filter)

        if not docs_in_collection:
            print(f"No documents found for collection: {collection_name}")
            return {}

        # Group documents by filename
        files_chunks: Dict[str, List[Document]] = {}

        for doc in docs_in_collection:
            filename = doc.meta.get("filename", "unknown_file")
            if filename not in files_chunks:
                files_chunks[filename] = []
            files_chunks[filename].append(doc)

        # Sort chunks within each file by their order
        for filename in files_chunks:
            try:
                files_chunks[filename] = sorted(files_chunks[filename], key=lambda chunk: self._get_chunk_order(chunk))
            except Exception as e:
                print(f"Warning: Could not sort chunks for file {filename}: {e}")
                # Keep original order if sorting fails

        total_chunks = sum(len(chunks) for chunks in files_chunks.values())
        print(f"Found {len(files_chunks)} files with {total_chunks} total chunks in collection '{collection_name}'")
        return files_chunks

    def _get_chunk_order(self, chunk: Document) -> int:
        """
        Extract chunk order from document metadata.

        Args:
            chunk: Document chunk

        Returns:
            Chunk order as integer, defaults to 0 if not found
        """
        try:
            # Try to get from dl_meta first
            if (
                "dl_meta" in chunk.meta
                and "meta" in chunk.meta["dl_meta"]
                and "doc_items" in chunk.meta["dl_meta"]["meta"]
            ):
                doc_items = chunk.meta["dl_meta"]["meta"]["doc_items"]
                if doc_items and isinstance(doc_items, list) and len(doc_items) > 0:
                    self_ref = doc_items[0].get("self_ref", 0)
                    if isinstance(self_ref, (int, float)):
                        return int(self_ref)

            # Fallback: try to get chunk order from other metadata fields
            if "chunk_order" in chunk.meta:
                return int(chunk.meta["chunk_order"])
            elif "chunk_index" in chunk.meta:
                return int(chunk.meta["chunk_index"])
            elif "page" in chunk.meta:
                return int(chunk.meta["page"])

            # If no order information is found, return 0
            return 0

        except (KeyError, TypeError, ValueError, IndexError):
            # If any error occurs in extracting order, default to 0
            return 0

    def _extract_facts_from_chunk(self, current_summary: str, chunk_content: str, filename: str) -> str:
        """
        Use LLM to iteratively extend the facts summary with information from a new chunk.

        Args:
            current_summary: Current accumulated facts summary
            chunk_content: Content of the current chunk
            filename: Name of the source file

        Returns:
            Updated facts summary
        """
        if not current_summary.strip():
            # First chunk - create initial summary
            prompt_builder = get_prompt_builder("facts_extraction_initial.prompt")
            prompt = prompt_builder.run(filename=filename, chunk_content=chunk_content)["prompt"]
        else:
            # Subsequent chunks - extend existing summary
            prompt_builder = get_prompt_builder("facts_extraction_update.prompt")
            prompt = prompt_builder.run(
                current_summary=current_summary, filename=filename, chunk_content=chunk_content
            )["prompt"]

        response = self.llm.run(prompt=prompt, generation_kwargs={"temperature": 0.1, "max_tokens": 512})
        return response["replies"][0].strip()

    def _finalize_facts_summary(self, accumulated_summary: str, collection_name: str) -> str:
        """
        Make a final LLM call to create a well-rounded summary for the entire case.

        Args:
            accumulated_summary: The accumulated facts from all chunks
            collection_name: Name of the collection

        Returns:
            Finalized, well-rounded facts summary
        """
        prompt_builder = get_prompt_builder("facts_finalization.prompt")
        prompt = prompt_builder.run(accumulated_summary=accumulated_summary, collection_name=collection_name)["prompt"]

        response = self.llm.run(prompt=prompt)
        return response["replies"][0].strip()

    async def generate_facts_for_collection(
        self, vector_manager: VectorStoreManager, collection_name: str, progress_callback=None
    ) -> str:
        """
        Generate facts summary for a collection by processing all chunks.

        Args:
            vector_manager: VectorStoreManager instance
            collection_name: Name of the collection
            progress_callback: Optional async function to call with progress updates (expects dict with progress info)

        Returns:
            Generated facts summary
        """
        print(f"Generating facts for collection: {collection_name}")

        # Get all chunks grouped by file
        files_chunks = self._get_chunks_grouped_by_file(vector_manager, collection_name)

        if not files_chunks:
            return f"No documents found in collection '{collection_name}'."

        # Calculate total chunks for progress reporting
        total_chunks = sum(len(chunks) for chunks in files_chunks.values())
        processed_chunks = 0

        # Initialize empty facts summary
        facts_summary = ""

        # Process each file's chunks in order
        for file_idx, (filename, chunks) in enumerate(files_chunks.items(), 1):
            print(f"Processing {len(chunks)} chunks from file: {filename}")

            if progress_callback:
                await progress_callback(
                    {
                        "type": "file_start",
                        "filename": filename,
                        "file_index": file_idx,
                        "total_files": len(files_chunks),
                        "file_chunks": len(chunks),
                    }
                )

            for i, chunk in enumerate(chunks):
                chunk_content = chunk.content
                if chunk_content.strip():
                    print(f"  Processing chunk {i+1}/{len(chunks)}")
                    facts_summary = self._extract_facts_from_chunk(facts_summary, chunk_content, filename)
                    processed_chunks += 1

                    if progress_callback:
                        await progress_callback(
                            {
                                "type": "chunk_processed",
                                "filename": filename,
                                "chunk_index": i + 1,
                                "total_file_chunks": len(chunks),
                                "processed_chunks": processed_chunks,
                                "total_chunks": total_chunks,
                            }
                        )

        # Finalize the summary
        print("Finalizing facts summary...")
        if progress_callback:
            await progress_callback({"type": "finalizing"})

        facts_summary = self._finalize_facts_summary(facts_summary, collection_name)

        # Save to database
        self.save_facts_for_collection(collection_name, facts_summary)

        print(f"Facts generation completed for collection: {collection_name}")
        if progress_callback:
            await progress_callback({"type": "completed"})

        return facts_summary

    async def get_or_generate_facts(
        self, vector_manager: VectorStoreManager, collection_name: str, progress_callback=None
    ) -> str:
        """
        Get facts for a collection, generating them if they don't exist.

        Args:
            vector_manager: VectorStoreManager instance
            collection_name: Name of the collection
            progress_callback: Optional async function to call with progress updates (expects dict with progress info)

        Returns:
            Facts summary for the collection
        """
        # Check if facts already exist
        existing_facts = self.get_facts_for_collection(collection_name)

        if existing_facts:
            print(f"Retrieved existing facts for collection: {collection_name}")
            # No progress callback needed for existing facts since we return immediately
            return existing_facts

        # Generate new facts if they don't exist
        print(f"No existing facts found for collection: {collection_name}. Generating...")
        return await self.generate_facts_for_collection(vector_manager, collection_name, progress_callback)
