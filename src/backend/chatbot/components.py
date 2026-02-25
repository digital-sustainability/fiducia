import hashlib
import re
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from pathlib import Path
from typing import Any, List, Optional, Dict
from copy import copy

import requests
import ocrmypdf
from haystack import Document, Pipeline, component
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils import Secret
from docling.utils.utils import create_file_hash
from ptpython.repl import embed
from tqdm import tqdm
import chainlit as cl


from backend.chatbot.utils import (
    add_metadata_to_message,
    clean_text,
    create_binary_hash,
    extract_timeline,
    get_prompt_builder,
)


@component
class DebuggingTool:
    """
    A component to inspect the output of the previous component in an interactive shell
    """

    def __init__(self, previous_component: object):
        """
        Initialize the DebuggingTool component with an instance from the previous component.

        :param previous_component: The previous component in the pipeline
        """
        setattr(self.run.__func__, "_output_types_cache", getattr(previous_component.run, "_output_types_cache"))

    def run(self, inputs: Any):
        """
        A debugging tool to inspect the output of the previous component in an interactive shell.
        Usage: you can access the inputs of the previous component by referring to the `inputs` parameter.

        :param inputs: The inputs to the component
        """
        embed(globals(), locals())
        sys.exit()


@component
class DocumentPreprocessor:
    """
    Preprocesses the documents and saves them in a new directory. The preprocessing steps include:

    1. Use ocrmypdf to add a text layer to PDFs that do not have one and then copy it to the preprocessed_documents directory.
    """

    def __init__(self, skip_if_exists=False):
        """
        Initialize the DocumentPreprocessor component.

        :param skip_if_exists: If True, skip the preprocessing if the output file already exists.
        """
        self.skip_if_exists = skip_if_exists

    @component.output_types(sources=list[Path])
    def run(self, source_directory: PathLike, output_directory: PathLike):

        source_directory = Path(source_directory)
        output_directory = Path(output_directory)

        sources = list(source_directory.rglob("*"))

        # Preprocess PDF files
        for source in tqdm([s for s in sources if s.suffix == ".pdf"]):
            output_path = output_directory / source.relative_to(source_directory)

            if self.skip_if_exists and output_path.exists():
                print(f"Skipping {source} as the output file already exists.")
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                ocrmypdf.ocr(source, output_path, skip_text=True, color_conversion_strategy="RGB")
            except ocrmypdf.InputFileError as e:
                print(
                    f"Could not OCR {source}, it will be copied as is to the preprocessed documents directory. Error message: {e}"
                )
                output_path.write_bytes(source.read_bytes())

        # Copy other files to the preprocessed_documents directory
        for source in tqdm([s for s in sources if s.suffix != ".pdf"]):
            output_path = output_directory / source.relative_to(source_directory)
            if self.skip_if_exists and output_path.exists():
                print(f"Skipping {source} as the output file already exists.")
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(source.read_bytes())

        # TODO: do we want list the preprocessed files as the source or the original files?
        return {"sources": list(output_directory.rglob("*"))}


@component
class CustomCleaner:
    """
    A custom document cleaner based on the specific documents in the knowledge base.
    To adapt the cleaning behavior, change the `clean_text` function in the `utils.py` file.
    """

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):

        for doc in documents:
            doc.content = clean_text(doc.content)

        return {"documents": documents}


@component
class TokenEfficientContextualizer:
    """
    Processes a list of documents to generate document metadata (summaries, restrictions)
    and contextualizations for the chunks of the document.

    The metadata generation is done by a LLM that generates a summary and restrictions based
    on an initial excerpt of the document. The contextualization is done by a LLM that generates
    a context for each chunk based on the summary, the restrictions, and a chunk-context window around
    the current chunk.

    This version supports parallelization with rate limiting for faster processing.

    Performance Features:
    - Parallel processing of chunk batches using ThreadPoolExecutor
    - Configurable rate limiting to avoid API limits
    - Multiple LLM instances for concurrent requests
    - Batch processing to optimize throughput
    - Error handling with fallback contexts

    Rate Limiting:
    - requests_per_minute: Controls the maximum API calls per minute
    - rate_limit_delay: Additional delay between requests (seconds)
    - max_workers: Number of parallel threads (recommend 2-5 for most APIs)
    - batch_size: Number of chunks processed together (recommend 3-10)

    Example:
        # Conservative settings for rate-limited APIs
        contextualizer = TokenEfficientContextualizer(
            model="gpt-4",
            max_workers=2,
            batch_size=3,
            requests_per_minute=60
        )

        # Faster settings for higher-limit APIs
        contextualizer = TokenEfficientContextualizer(
            model="llama-70b",
            max_workers=5,
            batch_size=10,
            requests_per_minute=200
        )
    """

    def __init__(
        self,
        model: str,
        api_base_url: Optional[str] = None,
        num_initial_chunks: int = 10,
        chunk_context_window: int = 3,
        max_workers: int = 3,
        batch_size: int = 5,
        rate_limit_delay: float = 0.1,
        requests_per_minute: int = 100,
    ):
        self.chunk_context_window = chunk_context_window
        self.num_initial_chunks = num_initial_chunks
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.requests_per_minute = requests_per_minute
        self.min_request_interval = 60.0 / requests_per_minute  # seconds between requests

        metadata_prompt_builder = get_prompt_builder("contextual_retrieval/token_efficient_metadata.prompt")

        # Create multiple LLM instances for parallel processing
        self.llm_metadata = OpenAIGenerator(api_base_url=api_base_url, model=model)
        self.llm_contexts = [OpenAIGenerator(api_base_url=api_base_url, model=model) for _ in range(max_workers)]

        # Metadata generation pipeline
        self.metadata_generation_pipeline = Pipeline()
        self.metadata_generation_pipeline.add_component("metadata_generation_prompt", metadata_prompt_builder)
        self.metadata_generation_pipeline.add_component("llm", self.llm_metadata)
        self.metadata_generation_pipeline.connect("metadata_generation_prompt", "llm")

        # Contextualization generation pipelines (multiple for parallel processing)
        self.context_generation_pipelines = []
        for i, llm_context in enumerate(self.llm_contexts):
            pipeline = Pipeline()
            pipeline.add_component(
                "context_generation_prompt",
                get_prompt_builder("contextual_retrieval/token_efficient_contextualization.prompt"),
            )
            pipeline.add_component("llm", llm_context)
            pipeline.connect("context_generation_prompt", "llm")
            self.context_generation_pipelines.append(pipeline)

    def get_chunk_texts(self, doc_hash: int, chunks: list[Document]) -> list[str]:
        # If chunks overlap, there might be duplicated text in the resulting list
        document_chunks = sorted(
            [chunk for chunk in chunks if chunk.meta["dl_meta"]["meta"]["origin"]["binary_hash"] == doc_hash],
            key=lambda chunk: chunk.meta["dl_meta"]["meta"]["doc_items"][0]["self_ref"],
        )
        return [chunk.content for chunk in document_chunks]

    def _contextualize_chunk_batch(self, pipeline_idx: int, chunk_batch_data: list, last_request_time: list) -> list:
        """Process a batch of chunks with rate limiting."""
        results = []
        pipeline = self.context_generation_pipelines[pipeline_idx]

        for chunk_data in chunk_batch_data:
            chunk, context_window_excerpt, summary, restrictions = chunk_data

            # Rate limiting: ensure minimum time between requests
            current_time = time.time()
            time_since_last = current_time - last_request_time[0]
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)

            try:
                context_result = pipeline.run(
                    {
                        "context_generation_prompt": {
                            "excerpt": context_window_excerpt,
                            "summary": summary,
                            "restrictions": restrictions,
                        }
                    }
                )
                context = context_result["llm"]["replies"][0]
                results.append((chunk, context))

                # Update last request time
                last_request_time[0] = time.time()

                # Additional small delay to avoid overwhelming the API
                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                print(f"Error contextualizing chunk: {e}")
                # Use a fallback context
                results.append((chunk, f"Kontext konnte nicht generiert werden: {str(e)}"))

        return results

    def _process_document_chunks_parallel(
        self, document_chunks: list[Document], chunk_texts: list[str], summary: str, restrictions: str
    ) -> None:
        """Process chunks of a document in parallel with rate limiting."""

        # Prepare batch data for all chunks
        chunk_batch_data = []
        for idx, chunk in enumerate(document_chunks):
            texts = chunk_texts[
                max(0, idx - self.chunk_context_window) : min(idx + self.chunk_context_window + 1, len(chunk_texts))
            ]
            context_window_excerpt = " ".join(texts)
            chunk_batch_data.append((chunk, context_window_excerpt, summary, restrictions))

        # Split into batches to avoid overwhelming the API
        batches = [chunk_batch_data[i : i + self.batch_size] for i in range(0, len(chunk_batch_data), self.batch_size)]

        # Shared variable to track last request time for rate limiting
        last_request_time = [time.time()]

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
            future_to_batch = {
                executor.submit(
                    self._contextualize_chunk_batch,
                    i % len(self.context_generation_pipelines),
                    batch,
                    last_request_time,
                ): batch
                for i, batch in enumerate(batches)
            }

            # Collect results and update chunks
            all_results = []
            for future in tqdm(
                as_completed(future_to_batch),
                total=len(future_to_batch),
                desc="Processing chunk batches",
                unit="batch",
            ):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"Error processing batch: {e}")

        # Apply results to chunks
        for chunk, context in all_results:
            additional_embedding_info = f"<additional_info>\nZusammenfassung des gesamten Dokuments: {summary}\nZu beachtende Einschränkungen: {restrictions}\nSituierung des Textausschnitts im gesamten Dokument: {context}\n</additional_info>"

            chunk.meta["contextualization"] = {
                "summary": summary,
                "restrictions": restrictions,
                "context_in_document": context,
                "additional_embedding_info": additional_embedding_info,
            }

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        doc_hashes = set(chunk.meta["dl_meta"]["meta"]["origin"]["binary_hash"] for chunk in documents)

        print(f"Contextualizing {len(doc_hashes)} documents with up to {self.max_workers} parallel workers...")

        for doc_hash in tqdm(doc_hashes, desc="Processing documents", unit="document"):
            document_chunks = [
                chunk for chunk in documents if chunk.meta["dl_meta"]["meta"]["origin"]["binary_hash"] == doc_hash
            ]
            chunk_texts = self.get_chunk_texts(doc_hash, document_chunks)

            initial_excerpt = " ".join(chunk_texts[: self.num_initial_chunks])

            # Generate metadata (this is fast, so no need to parallelize)
            try:
                metadata_result = self.metadata_generation_pipeline.run(
                    {"metadata_generation_prompt": {"excerpt": initial_excerpt}}
                )
                matches = re.findall(r"Summary: (.*)\n+Restrictions: (.*)", metadata_result["llm"]["replies"][0])

                if len(matches) == 1 and len(matches[0]) == 2:
                    summary, restrictions = matches[0]
                else:
                    print(f"Warning: Could not parse LLM metadata response, using fallback")
                    summary = "Zusammenfassung konnte nicht generiert werden."
                    restrictions = "Keine spezifischen Einschränkungen identifiziert."

            except Exception as e:
                print(f"Error generating metadata for document: {e}")
                summary = "Zusammenfassung konnte nicht generiert werden."
                restrictions = "Keine spezifischen Einschränkungen identifiziert."

            # Process chunks in parallel with rate limiting
            self._process_document_chunks_parallel(document_chunks, chunk_texts, summary, restrictions)

        return {"documents": documents}


@component
class FollowUpQuestionExpander:
    """
    Classifies the user's query as a follow-up question or not. If it is a follow-up question,
    the query is expanded using the language model.
    """

    def __init__(self, model: str, api_base_url: Optional[str] = None):
        """
        Creates a FollowUpQuestionExpander component.

        :param model: The model to use for the OpenAIGenerator
        :param api_base_url: The base URL of the OpenAI-compatible API
        """
        follow_up_question_classification_prompt_builder = get_prompt_builder(
            "follow_up_question_classification.prompt"
        )
        follow_up_query_expansion_prompt_builder = get_prompt_builder("follow_up_query_expansion.prompt")
        llm_classification = OpenAIGenerator(api_base_url=api_base_url, model=model)
        llm_expansion = OpenAIGenerator(api_base_url=api_base_url, model=model)

        # Classification pipeline
        self.classification_pipeline = Pipeline()
        self.classification_pipeline.add_component(
            "follow_up_question_classification_prompt", follow_up_question_classification_prompt_builder
        )
        self.classification_pipeline.add_component("llm", llm_classification)
        self.classification_pipeline.connect("follow_up_question_classification_prompt", "llm")

        # Expansion pipeline
        self.expansion_pipeline = Pipeline()
        self.expansion_pipeline.add_component(
            "follow_up_query_expansion_prompt", follow_up_query_expansion_prompt_builder
        )
        self.expansion_pipeline.add_component("llm", llm_expansion)
        self.expansion_pipeline.connect("follow_up_query_expansion_prompt", "llm")

    def classify_follow_up_question(self, messages: list[ChatMessage]) -> bool:
        """
        Classifies the user's query as a follow-up question or not.

        :param messages: The previous messages in the conversation
        :return: True if the query is a follow-up question, otherwise False
        """
        last_message = messages[-1]
        assert last_message.role == ChatRole.USER, "The last message should be from the user"

        if len([m for m in messages if m.role == ChatRole.USER]) < 2:
            # Only one user message, no follow-up question
            return False

        classification_result = self.classification_pipeline.run(
            {"follow_up_question_classification_prompt": {"messages": messages}}
        )
        return classification_result["llm"]["replies"][0] == "follow_up"

    @cl.step(type="llm", name="Anfrage Umformulieren", show_input=False)
    def expand_query(self, messages: list[ChatMessage]) -> str:
        """
        Expands the user's query using the language model.

        :param messages: The previous messages in the conversation
        :return: The expanded query as a string
        """
        expansion_result = self.expansion_pipeline.run({"follow_up_query_expansion_prompt": {"messages": messages}})
        return expansion_result["llm"]["replies"][0]

    @component.output_types(message=ChatMessage)
    def run(self, messages: list[ChatMessage]):
        """
        Classifies the user's query as a follow-up question or not. If it is a follow-up question,
        the query is expanded using the language model. Otherwise, the original query is returned.

        :param messages: The previous messages in the conversation
        :return: The expanded query if it is a follow-up question, otherwise the original query
        """
        last_message = messages[-1]
        assert last_message.role == ChatRole.USER, "The last message should be from the user"

        metadata_to_add = {}
        metadata_to_add["original_message"] = last_message.text

        is_follow_up_question = self.classify_follow_up_question(messages)

        if not is_follow_up_question:
            # No follow-up question, return the last message
            metadata_to_add["is_follow_up_question"] = False
            return {"message": add_metadata_to_message(last_message, metadata_to_add)}

        # Get expanded query
        expanded_query = self.expand_query(messages)

        # Replace the last message with the expanded query and save the original message
        metadata_to_add["is_follow_up_question"] = True

        return {"message": ChatMessage.from_user(text=expanded_query, meta={**last_message.meta, **metadata_to_add})}


@component
class FileMetadataInjector:
    """Component to inject file metadata into documents for tracking with collection-aware paths."""

    def __init__(self, document_base_path: str = None):
        """
        Initialize the FileMetadataInjector.

        Args:
            document_base_path: Base path for documents (e.g., ./data/). If None, will be read from env.
        """
        from backend.utils import relative_project_path

        self.document_base_path = (
            Path(document_base_path)
            if document_base_path
            else relative_project_path(os.getenv("DOCUMENT_BASE_PATH", "./data/"))
        )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        """Add file path and hash metadata to documents for tracking with collection-relative paths."""
        for doc in documents:
            # Extract file path from Docling metadata
            if (
                hasattr(doc, "meta")
                and "dl_meta" in doc.meta
                and "meta" in doc.meta["dl_meta"]
                and "origin" in doc.meta["dl_meta"]["meta"]
                and "filename" in doc.meta["dl_meta"]["meta"]["origin"]
            ):

                original_filename = doc.meta["dl_meta"]["meta"]["origin"]["filename"]

                # Docling only stores the filename, not the relative file path, we need to search it
                candidate_files = list(self.document_base_path.rglob(original_filename))
                docling_hashes = [create_binary_hash(f) for f in candidate_files]

                try:
                    file_idx = docling_hashes.index(doc.meta["dl_meta"]["meta"]["origin"]["binary_hash"])
                    file_path = candidate_files[file_idx]
                except ValueError:
                    file_path = Path(original_filename)

                try:
                    # Get relative path from document base path
                    relative_to_base = file_path.relative_to(self.document_base_path)

                    # Extract collection name (first directory) and preserve subdirectory structure
                    path_parts = relative_to_base.parts
                    if len(path_parts) >= 1:
                        collection_name = path_parts[0]
                        filename = path_parts[-1]  # Last part is the filename
                        collection_relative_path = str(relative_to_base)

                        # Get subdirectory path (everything between collection and filename)
                        if len(path_parts) > 2:
                            subdirectory = str(Path(*path_parts[1:-1]))
                        else:
                            subdirectory = ""
                    else:
                        # Fallback for files directly in base path
                        collection_name = "default"
                        filename = file_path.name
                        collection_relative_path = filename
                        subdirectory = ""
                except ValueError:
                    # File is outside document base path, use original path
                    collection_name = "unknown"
                    filename = file_path.name
                    collection_relative_path = str(file_path)
                    subdirectory = ""

                # Save docling file hash in a flat manner
                file_hash = str(doc.meta["dl_meta"]["meta"]["origin"].get("binary_hash", "unknown"))
                doc_items = doc.meta["dl_meta"]["meta"]["doc_items"]
                provenances = set(prov["page_no"] for item in doc_items for prov in item["prov"])
                page_start = min(provenances) if provenances else 1
                page_end = max(provenances) if provenances else 1

                metadata = {
                    "file_hash": file_hash,
                    "file_path": str(file_path),
                    "collection_name": collection_name,
                    "filename": filename,
                    "subdirectory": subdirectory,
                    "collection_relative_path": collection_relative_path,
                    "document_base_path": str(self.document_base_path),
                    "page_start": page_start,
                    "page_end": page_end,
                }

                timeline, flat_fields = extract_timeline(doc.content)
                metadata.update(flat_fields)
                metadata["timeline"] = timeline

                self._add_metadata(doc, metadata)

        return {"documents": documents}

    def _add_metadata(self, doc: Document, metadata: Dict) -> None:
        """Add metadata to the document (both meta and payload for Qdrant)."""
        doc.meta.update(metadata)

        if not hasattr(doc, "payload"):
            doc.payload = {}

        doc.payload.update(metadata)


@component
class DeepInfraReranker:
    """
    Ranks documents based on their semantic similarity to the query using DeepInfra's reranker API.

    This component uses DeepInfra's hosted reranker models to rank documents by relevance to a query.

    ### Usage example

    ```python
    from haystack import Document
    from deepinfra_reranker import DeepInfraReranker

    ranker = DeepInfraReranker(
        model="Qwen/Qwen3-Reranker-4B",
        api_key=Secret.from_token("your_api_key")
    )
    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "City in Germany"
    result = ranker.run(query=query, documents=docs)
    docs = result["documents"]
    print(docs[0].content)
    ```
    """

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-Reranker-4B",
        api_key: Optional[Secret] = Secret.from_env_var(["DEEPINFRA_API_KEY"], strict=False),
        top_k: int = 10,
        query_prefix: str = "",
        document_prefix: str = "",
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        score_threshold: Optional[float] = None,
        api_base_url: str = "https://api.deepinfra.com/v1/inference",
    ):
        """
        Creates an instance of DeepInfraReranker.

        :param model:
            The DeepInfra reranker model to use (e.g., "Qwen/Qwen3-Reranker-4B").
        :param api_key:
            The API token for DeepInfra. Can be set via DEEPINFRA_api_key environment variable.
        :param top_k:
            The maximum number of documents to return per query.
        :param query_prefix:
            A string to add at the beginning of the query text before ranking.
        :param document_prefix:
            A string to add at the beginning of each document before ranking.
        :param meta_fields_to_embed:
            List of metadata fields to embed with the document.
        :param embedding_separator:
            Separator to concatenate metadata fields to the document.
        :param score_threshold:
            Use it to return documents with a score above this threshold only.
        :param api_base_url:
            The base URL for the DeepInfra API.

        :raises ValueError:
            If `top_k` is not > 0.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        self.model = model
        self.api_key = api_key
        self.top_k = top_k
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.score_threshold = score_threshold
        self.api_base_url = api_base_url

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def _prepare_documents(self, documents: List[Document]) -> List[str]:
        """
        Prepare documents by adding prefixes and metadata fields.

        :param documents: List of documents to prepare
        :return: List of prepared document strings
        """
        prepared_documents = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            prepared_doc = self.document_prefix + self.embedding_separator.join(
                meta_values_to_embed + [doc.content or ""]
            )
            prepared_documents.append(prepared_doc)
        return prepared_documents

    def _call_deepinfra_api(self, query: str, documents: List[str]) -> List[float]:
        """
        Call the DeepInfra reranker API.

        :param query: The query string
        :param documents: List of document strings
        :return: List of relevance scores
        :raises RuntimeError: If API call fails or returns invalid response
        """
        if not self.api_key:
            raise RuntimeError("API key is required but not provided")

        url = f"{self.api_base_url}/{self.model}"
        headers = {
            "Authorization": f"bearer {self.api_key.resolve_value()}",
            "Content-Type": "application/json",
        }
        payload = {
            "queries": [query] * len(documents),
            "documents": documents,
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            result = response.json()
            if "scores" not in result:
                raise RuntimeError(f"Invalid response from DeepInfra API: {result}")

            scores = result["scores"]
            if len(scores) != len(documents):
                raise RuntimeError(f"Expected {len(documents)} scores, but got {len(scores)} from DeepInfra API")

            return scores

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to call DeepInfra API: {e}")
        except KeyError as e:
            raise RuntimeError(f"Invalid response format from DeepInfra API: missing key {e}")

    @component.output_types(documents=List[Document])
    def run(
        self,
        *,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, List[Document]]:
        """
        Returns a list of documents ranked by their similarity to the given query.

        :param query:
            The input query to compare the documents to.
        :param documents:
            A list of documents to be ranked.
        :param top_k:
            The maximum number of documents to return.
        :param score_threshold:
            Use it to return documents only with a score above this threshold.
            If set, overrides the value set at initialization.
        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents closest to the query, sorted from most similar to least similar.

        :raises ValueError:
            If `top_k` is not > 0.
        :raises RuntimeError:
            If API token is not provided or API call fails.
        """
        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        # Prepare query and documents
        prepared_query = self.query_prefix + query
        prepared_documents = self._prepare_documents(documents)

        # Call DeepInfra API
        scores = self._call_deepinfra_api(prepared_query, prepared_documents)

        # Create ranked documents with scores
        ranked_docs = []
        for i, score in enumerate(scores):
            document = copy(documents[i])
            document.score = score
            ranked_docs.append(document)

        # Sort by score in descending order (highest similarity first)
        ranked_docs.sort(key=lambda doc: doc.score, reverse=True)

        # Apply score threshold if specified
        if score_threshold is not None:
            ranked_docs = [doc for doc in ranked_docs if doc.score >= score_threshold]

        return {"documents": ranked_docs[:top_k]}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return {
            "type": "deepinfra_reranker.DeepInfraReranker",
            "init_parameters": {
                "model": self.model,
                "api_key": self.api_key.to_dict() if self.api_key else None,
                "top_k": self.top_k,
                "query_prefix": self.query_prefix,
                "document_prefix": self.document_prefix,
                "meta_fields_to_embed": self.meta_fields_to_embed,
                "embedding_separator": self.embedding_separator,
                "score_threshold": self.score_threshold,
                "api_base_url": self.api_base_url,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeepInfraReranker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]
        if init_params.get("api_key"):
            init_params["api_key"] = Secret.from_dict(init_params["api_key"])
        return cls(**init_params)
