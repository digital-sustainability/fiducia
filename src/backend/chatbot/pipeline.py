import asyncio
import functools
import os
from typing import Optional

from dotenv import load_dotenv
from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.converters import OutputAdapter
from haystack.components.embedders import HuggingFaceAPITextEmbedder, OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.joiners.document_joiner import DocumentJoiner
from haystack.components.rankers import HuggingFaceTEIRanker
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever, QdrantSparseEmbeddingRetriever
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client.models import Filter, FieldCondition, MatchValue

from chainlit import Message as ChainlitMessage

from backend.utils import relative_project_path
from backend.chatbot.components import DeepInfraReranker, FollowUpQuestionExpander
from backend.chatbot.utils import (
    chainlit_to_haystack_messages,
    debug_component,
    get_prompt_builder,
)
from backend.chatbot.vector_db_manager import VectorStoreManager


load_dotenv(dotenv_path=relative_project_path(".env"))


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


class RetrievalAugmentedGenerationPipeline:
    """
    The main retrieval and generation pipeline.
    """

    def __init__(self) -> None:
        self._init_pipelines()
        self.current_streaming_message: ChainlitMessage | None = None
        self.vector_store_manager = VectorStoreManager()

    def _init_pipelines(self):
        document_store = QdrantDocumentStore(
            url=_require_env_var("QDRANT_ENDPOINT"),
            api_key=Secret.from_env_var("QDRANT_API_KEY"),
            index=_require_env_var("QDRANT_INDEX"),
            recreate_index=False,
            progress_bar=True,
            use_sparse_embeddings=True,
            embedding_dim=int(os.getenv("QDRANT_EMBEDDING_DIM", 1024)),
        )

        self.document_store = document_store

        follow_up_question_expander = FollowUpQuestionExpander(
            api_base_url="https://api.together.xyz/v1", model="meta-llama/Llama-3.3-70B-Instruct-Turbo"
        )
        rewritten_message_adapter = OutputAdapter(template="{{ message.text }}", output_type=str)
        original_message_adapter = OutputAdapter(template="{{ message.meta['original_message'] }}", output_type=str)

        # Use this code instead if you want to use a local embedding model through a Hugging Face Text Embeddings Inference API:
        # dense_embedder = HuggingFaceAPITextEmbedder(
        #     api_type="text_embeddings_inference",
        #     api_params={"url": os.getenv("TEI_EMBEDDING_ENDPOINT", "http://localhost:1111/")},
        # )

        dense_embedder = OpenAITextEmbedder(
            model="Qwen/Qwen3-Embedding-0.6B",
            dimensions=int(os.getenv("QDRANT_EMBEDDING_DIM", "1024")),
            api_base_url="https://api.deepinfra.com/v1/openai",
            api_key=Secret.from_env_var("DEEPINFRA_API_KEY"),
        )

        # Monkey patch the embedder; TODO: fix as soon as the bug is fixed in haystack
        dense_embedder.client.embeddings.create = functools.partial(
            dense_embedder.client.embeddings.create, encoding_format="float"
        )
        sparse_embedder = FastembedSparseTextEmbedder(model="Qdrant/bm25", progress_bar=True)

        dense_retriever = QdrantEmbeddingRetriever(document_store=document_store, top_k=30)
        sparse_retriever = QdrantSparseEmbeddingRetriever(document_store=document_store, top_k=10)

        joiner = DocumentJoiner(join_mode="concatenate")

        # Use this code instead if you want to use a local reranker model through a Hugging Face Text Embeddings Inference API:
        # reranker = HuggingFaceTEIRanker(
        #     url=os.getenv("TEI_RERANKER_ENDPOINT", "http://localhost:2222/"), top_k=5, timeout=120
        # )

        reranker = DeepInfraReranker(
            model="Qwen/Qwen3-Reranker-4B",
            api_key=Secret.from_env_var("DEEPINFRA_API_KEY"),
            top_k=5,
        )

        rag_prompt = get_prompt_builder("rag.prompt")
        llm_rag = OpenAIGenerator(
            api_base_url="https://api.together.xyz/v1",
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            system_prompt=get_prompt_builder("rag.system.prompt").run().get("prompt"),
            streaming_callback=self._streaming_callback,
        )

        pipeline_retrieval = Pipeline()
        pipeline_generation = Pipeline()

        pipeline_retrieval.add_component("follow_up_question_expander", follow_up_question_expander)
        pipeline_retrieval.add_component("rewritten_message_adapter", rewritten_message_adapter)
        pipeline_retrieval.add_component("original_message_adapter", original_message_adapter)
        pipeline_retrieval.add_component("sparse_embedder", sparse_embedder)
        pipeline_retrieval.add_component("sparse_retriever", sparse_retriever)
        pipeline_retrieval.add_component("dense_embedder", dense_embedder)
        pipeline_retrieval.add_component("dense_retriever", dense_retriever)
        pipeline_retrieval.add_component("joiner", joiner)
        pipeline_retrieval.add_component("reranker", reranker)

        pipeline_retrieval.connect("follow_up_question_expander", "rewritten_message_adapter")
        pipeline_retrieval.connect("follow_up_question_expander", "original_message_adapter")
        pipeline_retrieval.connect("rewritten_message_adapter", "dense_embedder")
        pipeline_retrieval.connect("rewritten_message_adapter", "sparse_embedder")
        pipeline_retrieval.connect("rewritten_message_adapter", "reranker.query")

        pipeline_retrieval.connect("sparse_embedder", "sparse_retriever")
        pipeline_retrieval.connect("dense_embedder.embedding", "dense_retriever.query_embedding")
        pipeline_retrieval.connect("dense_retriever", "joiner")
        pipeline_retrieval.connect("sparse_retriever", "joiner")
        pipeline_retrieval.connect("joiner", "reranker.documents")

        pipeline_generation.add_component("rag_prompt", rag_prompt)
        pipeline_generation.add_component("llm_rag", llm_rag)
        pipeline_generation.connect("rag_prompt", "llm_rag")

        self.pipeline_retrieval = pipeline_retrieval
        self.pipeline_generation = pipeline_generation

    async def query(
        self,
        streaming_message: ChainlitMessage,
        chat_history: list[ChainlitMessage],
        collection_name: Optional[str] = None,
    ) -> ChainlitMessage:
        self.current_streaming_message = streaming_message

        # Convert Chainlit messages to Haystack ChatMessage format
        messages = chainlit_to_haystack_messages(chat_history)
        results_retrieval = await self.retrieve(messages, collection_name=collection_name)

        intermediate_outputs = {
            "original_message": results_retrieval["original_message_adapter"]["output"],
            "rewritten_message": results_retrieval["rewritten_message_adapter"]["output"],
        }

        results_generation = self.pipeline_generation.run(
            {
                "rag_prompt": {
                    "documents": results_retrieval["reranker"]["documents"],
                    "question": results_retrieval["rewritten_message_adapter"]["output"],
                    "original_question": results_retrieval["original_message_adapter"]["output"],
                },
            },
            include_outputs_from={"llm_rag"},
        )

        # Check if the response contains valid replies.
        if "replies" in results_generation["llm_rag"]:
            streaming_message.content = results_generation["llm_rag"]["replies"][0]
            streaming_message.metadata = {
                "intermediate_outputs": intermediate_outputs,
                "sources": [
                    {
                        # Use collection-relative path if available, fallback to filename
                        "file": doc.meta.get("collection_relative_path")
                        or doc.meta["dl_meta"]["meta"]["origin"]["filename"],
                        "collection_name": doc.meta.get("collection_name", "unknown"),
                        "filename": doc.meta.get("filename")
                        or os.path.basename(doc.meta["dl_meta"]["meta"]["origin"]["filename"]),
                        "subdirectory": doc.meta.get("subdirectory", ""),  # New: subdirectory info
                        "pages": sorted(
                            {
                                provenance["page_no"]
                                for elem in doc.meta["dl_meta"]["meta"]["doc_items"]
                                for provenance in elem["prov"]
                            }
                        ),
                        "contextual_metadata": {
                            "document_summary": doc.meta["contextualization"]["summary"],
                            "document_restrictions": doc.meta["contextualization"]["restrictions"],
                            "chunk_context": doc.meta["contextualization"]["context_in_document"],
                        },
                        "content": doc.content,
                    }
                    for doc in results_retrieval["reranker"]["documents"]
                ],
            }
            return streaming_message
        else:
            raise Exception("No valid response or unexpected response format.")

    async def retrieve(self, messages: list[ChatMessage], collection_name: Optional[str] = None) -> dict[str, any]:
        data = {
            "follow_up_question_expander": {"messages": messages},
        }
        if collection_name:
            data["dense_retriever"] = {
                "filters": Filter(
                    must=FieldCondition(key="meta.collection_name", match=MatchValue(value=collection_name))
                ),
            }
            data["sparse_retriever"] = {
                "filters": Filter(
                    must=FieldCondition(key="meta.collection_name", match=MatchValue(value=collection_name))
                ),
            }

        return self.pipeline_retrieval.run(
            data=data,
            include_outputs_from={
                "rewritten_message_adapter",
                "original_message_adapter",
                "sparse_retriever",
                "dense_retriever",
                "reranker",
            },
        )

    def _streaming_callback(self, chunk: StreamingChunk):
        """
        Callback for streaming responses.
        This method is called for each chunk of the response.
        """
        asyncio.run(self.current_streaming_message.stream_token(chunk.content))

    def reload_vector_store(self):
        """
        Reload the vector store to pick up newly embedded documents.
        This method re-initializes the pipelines with a fresh document store connection.
        """
        print("Reloading vector store to pick up new embeddings...")
        self._init_pipelines()
        print("Vector store reloaded successfully.")
