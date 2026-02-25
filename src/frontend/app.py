from typing import Dict, Optional
import asyncio
import chainlit as cl
import os
import re
import logging
import base64
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from chainlit.input_widget import Select

from backend.chatbot.pipeline import RetrievalAugmentedGenerationPipeline
from backend.chatbot.auth.authentik_oauth_provider import AuthentikOAuthProvider
from backend.chatbot.auth.inject_custom_oauth_provider import add_custom_oauth_provider
from backend.chatbot.vector_db_manager import VectorStoreManager
from backend.chatbot.collection_metadata_manager import CollectionMetadataManager
from frontend.utils import load_translations, render_sources, run_warmup
from backend.utils import relative_project_path

logger = logging.getLogger("fiducia.frontend")


def _is_truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _configure_oauth_provider() -> None:
    if not _is_truthy(os.getenv("ENABLE_OAUTH", "false")):
        logger.info("OAuth is disabled (set ENABLE_OAUTH=true to enable).")
        return

    try:
        add_custom_oauth_provider(AuthentikOAuthProvider())
        logger.info("OAuth provider registered.")
    except Exception:
        logger.exception("Failed to register OAuth provider.")


_configure_oauth_provider()

commands = [
    {
        "id": "Add Collection",
        "icon": "file-plus",
        "description": "Add a new document collection",
        "button": True,
    },
    {
        "id": "Show Collections",
        "icon": "folder",
        "description": "Show available document collections with delete options",
        "button": True,
    },
    {
        "id": "Show Timeline",
        "icon": "square-chart-gantt",
        "description": "Show the timeline of events for a specific document collection",
        "button": True,
    },
    {
        "id": "Show Facts",
        "icon": "file-text",
        "description": "Extract and show key facts from a document collection",
        "button": True,
    },
]


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    _ = (token, raw_user_data)  # Avoid logging sensitive auth data.
    logger.info("OAuth callback triggered for provider '%s'.", provider_id)
    return default_user


@cl.on_chat_start
async def on_chat_start():
    await cl.context.emitter.set_commands(commands)
    # await run_warmup()
    manager = VectorStoreManager()
    metadata_manager = CollectionMetadataManager()
    cl.user_session.set("chat_history", [])
    cl.user_session.set("rag_pipeline", RetrievalAugmentedGenerationPipeline())
    cl.user_session.set("vector_store_manager", manager)
    cl.user_session.set("collection_metadata_manager", metadata_manager)
    cl.user_session.set("selected_collection", "Alle Sammlungen")
    cl.user_session.set("available_collections", manager.get_available_collections())
    await configure_settings()


async def configure_settings(selected_collection: str = "Alle Sammlungen") -> None:
    """
    Configure the settings menu.
    """
    manager: VectorStoreManager = cl.user_session.get("vector_store_manager")
    values = ["Alle Sammlungen"] + manager.get_available_collections()
    initial_index = values.index(selected_collection) if selected_collection in values else 0
    await cl.ChatSettings(
        [
            Select(
                id="collection",
                label="Dokumentensammlung",
                values=values,
                initial_index=initial_index,
            ),
        ]
    ).send()


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("selected_collection", settings.get("collection", "Alle Sammlungen"))


@cl.on_message
async def on_message(message: cl.Message):
    if message.command:
        await handle_command(message)
    else:
        await rag(message)


async def handle_command(message: cl.Message) -> None:
    """
    Handles user commands from the chat interface.
    """
    if message.command == "Add Collection":
        # This command is handled in the DirectoryUploader custom element
        await show_directory_uploader(message)
        return
    elif message.command == "Show Collections":
        await show_available_collections()
        return
    elif message.command == "Show Timeline":
        await show_timeline(message)
        return
    elif message.command == "Show Facts":
        await show_facts(message)
        return

    # If the command is not recognized, you can raise an error or ignore it
    raise Exception(f"Command '{message.command}' not found.")


@cl.action_callback("send_toast")
async def send_toast(action: cl.Action):
    assert "message" in action.payload, "Message is required for sending a toast."
    assert "type" in action.payload, "Type is required for sending a toast."
    await cl.context.emitter.send_toast(message=action.payload["message"], type=action.payload["type"])


@cl.action_callback("open_pdf_document")
async def open_pdf_document(action: cl.Action):
    """Open a PDF document in the sidebar at a specific page."""
    file_path = action.payload.get("file_path")
    filename = action.payload.get("filename", "Document")
    page = action.payload.get("page", 1)

    if not file_path:
        await cl.context.emitter.send_toast("No file path provided", type="error")
        return

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            await cl.context.emitter.send_toast(f"File not found: {filename}", type="error")
            return

        # Create a PDF element to display in the sidebar
        pdf_element = cl.Pdf(name=f"pdf_viewer_{hash(file_path)}", display="side", path=file_path, page=page)

        # Setting elements will open the sidebar
        await cl.ElementSidebar.set_elements([pdf_element])
        await cl.ElementSidebar.set_title("PDF Viewer")

    except Exception as e:
        logger.exception("Error opening PDF document.")
        await cl.context.emitter.send_toast(f"Failed to open PDF: {str(e)}", type="error")


async def rag(message: cl.Message):
    """
    Handles the RAG (Retrieval-Augmented Generation) pipeline query.
    This function is called when the user sends a message that is not a command.
    """
    rag_pipeline: RetrievalAugmentedGenerationPipeline = cl.user_session.get("rag_pipeline")
    chat_history: list[cl.Message] = cl.user_session.get("chat_history", [])

    collection = cl.user_session.get("selected_collection")
    if collection == "Alle Sammlungen":
        collection = None

    # Add the user message to the chat history
    chat_history.append(message)

    msg = cl.Message(content="")

    # Run the RAG pipeline with the user message
    response = await rag_pipeline.query(streaming_message=msg, chat_history=chat_history, collection_name=collection)

    # Add the response to the chat history
    chat_history.append(response)

    response = await render_sources(response, display="side")

    # Send a response back to the user
    await response.update()


async def show_directory_uploader(message: cl.Message):
    """
    Displays the DirectoryUploader custom element in the chat interface.
    This function is called when the user writes or clicks the "CreateCollection" command.
    """
    user_lang = cl.user_session.get("language", "de")

    # Now load_translations comes from frontend.utils
    translations = load_translations(user_lang)

    msg = cl.Message(
        content=translations.get("directory_uploader_instructions"),
        elements=[
            cl.CustomElement(
                name="DirectoryUploader",
                props={"userLanguage": user_lang, "translations": translations, "userMessage": message.content},
            ),
        ],
    )

    await msg.send()


async def show_timeline(message: cl.Message):
    """
    Asks for a collection name and displays its timeline.
    """
    available_collections = cl.user_session.get("available_collections", [])
    if not available_collections:
        await cl.Message(content="No collections found. Please add a collection first.").send()
        return

    # The collection name is the content of the message, after the command.
    collection_name = message.content.strip()

    # Collection name not provided, notify the user and list available collections.
    if not collection_name:
        await cl.Message(
            content=(
                "This command only works if you provide the name of the collection of which you want to see "
                f"the timeline. Available collections: `{'`, `'.join(available_collections)}`"
            ),
        ).send()
        return

    # Wrong collection name provided, notify the user.
    if collection_name not in available_collections:
        await cl.Message(
            content=(
                f"Collection '{collection_name}' not found. Available collections: "
                f"`{'`, `'.join(available_collections)}`"
            ),
        ).send()
        return

    try:
        manager: VectorStoreManager = cl.user_session.get("vector_store_manager")
        if not manager:
            await cl.context.emitter.send_toast("Vector Store Manager not available", type="error")
            return

        timeline_events = manager.get_collection_timeline(collection_name)

        await cl.Message(
            content=f"Timeline for **{collection_name}**:",
            elements=[
                cl.CustomElement(
                    name="TimelineViewer",
                    props={"timelineEvents": timeline_events, "collectionName": collection_name},
                )
            ],
        ).send()

    except Exception:
        logger.exception("Error retrieving timeline.")
        await cl.Message(content=f"❌ An error occurred while retrieving the timeline for '{collection_name}'.").send()


async def show_available_collections():
    """
    Display available collections with their statistics and management options using the CollectionViewer component.
    """
    try:
        manager: VectorStoreManager = cl.user_session.get("vector_store_manager")
        if not manager:
            await cl.context.emitter.send_toast("Vector Store Manager not available", type="error")
            return

        # Get collections and their stats
        stats = manager.get_collection_stats()
        collections = list(stats.keys())

        # Display using the custom CollectionViewer component
        await cl.Message(
            content="📚 **Document Collections**",
            elements=[
                cl.CustomElement(
                    name="CollectionViewer",
                    props={
                        "collections": collections,
                        "collectionStats": stats,
                    },
                ),
            ],
        ).send()

    except Exception as e:
        await cl.Message(content=f"❌ Error retrieving collections: {str(e)}").send()


async def show_facts(message: cl.Message):
    """
    Extract and show key facts from a document collection.
    """
    available_collections = cl.user_session.get("available_collections", [])
    if not available_collections:
        await cl.Message(content="No collections found. Please add a collection first.").send()
        return

    # The collection name is the content of the message, after the command.
    collection_name = message.content.strip()

    # Collection name not provided, notify the user and list available collections.
    if not collection_name:
        await cl.Message(
            content=(
                "This command only works if you provide the name of the collection from which you want to extract "
                f"facts. Available collections: `{'`, `'.join(available_collections)}`"
            ),
        ).send()
        return

    # Wrong collection name provided, notify the user.
    if collection_name not in available_collections:
        await cl.Message(
            content=(
                f"Collection '{collection_name}' not found. Available collections: "
                f"`{'`, `'.join(available_collections)}`"
            ),
        ).send()
        return

    try:
        vector_manager: VectorStoreManager = cl.user_session.get("vector_store_manager")
        metadata_manager: CollectionMetadataManager = cl.user_session.get("collection_metadata_manager")

        if not vector_manager:
            await cl.context.emitter.send_toast("Vector Store Manager not available", type="error")
            return

        if not metadata_manager:
            await cl.context.emitter.send_toast("Collection Metadata Manager not available", type="error")
            return

        # Check if facts already exist
        existing_facts = metadata_manager.get_facts_for_collection(collection_name)
        if existing_facts:
            await cl.Message(content=f"📋 **Facts Summary for {collection_name}**\n\n{existing_facts}").send()
            return

        # Get files and chunks info first to prepare TaskList
        files_chunks = metadata_manager._get_chunks_grouped_by_file(vector_manager, collection_name)

        if not files_chunks:
            await cl.Message(
                content=f"📭 **{collection_name}**\n\nNo documents found in collection '{collection_name}'."
            ).send()
            return

        total_chunks = sum(len(chunks) for chunks in files_chunks.values())
        total_files = len(files_chunks)

        # Show initial message with TaskList
        intro_msg = cl.Message(
            content=f"🔍 **Extracting facts from {collection_name}...**\n\nFound {total_files} files with {total_chunks} total chunks. Preparing task list...\n\n"
        )
        await intro_msg.send()

        # Small delay to separate intro message from TaskList visually
        await asyncio.sleep(0.3)

        # Create TaskList for progress tracking
        task_list = cl.TaskList()
        task_list.status = "🔧 Preparing extraction tasks..."

        # Create tasks for each file
        file_tasks = {}
        for i, (filename, chunks) in enumerate(files_chunks.items(), 1):
            # Use shorter, cleaner filename
            clean_filename = filename.replace(".pdf", "").replace("-", " ")
            if len(clean_filename) > 40:
                clean_filename = clean_filename[:37] + "..."

            task = cl.Task(title=f"📄 {clean_filename}", status=cl.TaskStatus.READY)
            file_tasks[filename] = task
            await task_list.add_task(task)

        # Add finalization task
        finalization_task = cl.Task(title="✨ Generating summary", status=cl.TaskStatus.READY)
        await task_list.add_task(finalization_task)

        # Send TaskList to UI immediately
        await task_list.send()
        logger.debug("Initial TaskList sent to UI")

        # Give the UI a moment to render the TaskList
        await asyncio.sleep(0.5)

        # Initialize progress tracking variables
        start_time = time.time()
        processed_chunks = 0
        current_file_task = None

        # Enhanced progress callback with TaskList updates
        async def update_progress(progress_info: dict):
            nonlocal processed_chunks, current_file_task, start_time

            try:
                progress_type = progress_info.get("type", "status")

                if progress_type == "file_start":
                    filename = progress_info["filename"]
                    file_index = progress_info["file_index"]

                    # Update previous file task to done if exists
                    if current_file_task:
                        current_file_task.status = cl.TaskStatus.DONE
                        logger.debug("Marking previous file task as DONE")

                    # Set current file task to running
                    current_file_task = file_tasks[filename]
                    current_file_task.status = cl.TaskStatus.RUNNING
                    logger.debug("Starting file '%s' and setting task to RUNNING.", filename)

                    # Calculate time estimates
                    elapsed_time = time.time() - start_time
                    if processed_chunks > 0:
                        avg_time_per_chunk = elapsed_time / processed_chunks
                        remaining_chunks = total_chunks - processed_chunks
                        eta_seconds = remaining_chunks * avg_time_per_chunk
                        eta = datetime.now() + timedelta(seconds=eta_seconds)
                        eta_str = f" (ETA: {eta.strftime('%H:%M:%S')})"
                    else:
                        eta_str = ""

                    # Update the TaskList status
                    if eta_str:
                        eta_display = eta_str.replace(" (ETA: ", "").replace(")", "")
                        task_list.status = f"📁 Analyzing file {file_index}/{total_files} │ ETA {eta_display}"
                    else:
                        task_list.status = f"📁 Analyzing file {file_index}/{total_files}"

                    logger.debug("File start - updating TaskList: %s", task_list.status)
                    await task_list.send()

                elif progress_type == "chunk_processed":
                    processed_chunks += 1
                    chunk_index = progress_info["chunk_index"]
                    total_file_chunks = progress_info["total_file_chunks"]
                    filename = progress_info["filename"]

                    # Update current file task title with chunk progress
                    if current_file_task:
                        progress_pct = int((processed_chunks / total_chunks) * 100)

                        # Create a progress bar visualization
                        bar_length = 10
                        filled_length = int(bar_length * processed_chunks / total_chunks)
                        progress_bar = "█" * filled_length + "░" * (bar_length - filled_length)

                        # Clean filename for display
                        clean_filename = filename.replace(".pdf", "").replace("-", " ")
                        if len(clean_filename) > 25:
                            clean_filename = clean_filename[:22] + "..."

                        new_title = f"📄 {clean_filename} │ {progress_bar} {progress_pct}%"
                        current_file_task.title = new_title
                        logger.debug("Updated task title.")

                    # Calculate time estimates and update less frequently (every 2 chunks or every 10%)
                    should_update = (
                        processed_chunks % 2 == 0  # Every 2 chunks
                        or processed_chunks == total_chunks  # Last chunk
                        or processed_chunks == 1  # First chunk
                    )

                    if should_update:
                        elapsed_time = time.time() - start_time
                        if processed_chunks > 0:
                            avg_time_per_chunk = elapsed_time / processed_chunks
                            remaining_chunks = total_chunks - processed_chunks
                            eta_seconds = remaining_chunks * avg_time_per_chunk
                            eta = datetime.now() + timedelta(seconds=eta_seconds)
                            eta_str = f" (ETA: {eta.strftime('%H:%M:%S')})"
                        else:
                            eta_str = ""

                        # Update task list status and send update
                        progress_pct = int((processed_chunks / total_chunks) * 100)

                        # Create elegant status with better formatting
                        if eta_str:
                            eta_display = eta_str.replace(" (ETA: ", "").replace(")", "")
                            new_status = f"⚡ Processing │ {processed_chunks}/{total_chunks} chunks ({progress_pct}%) │ ETA {eta_display}"
                        else:
                            new_status = f"⚡ Processing │ {processed_chunks}/{total_chunks} chunks ({progress_pct}%)"

                        task_list.status = new_status
                        logger.debug("Updating TaskList status.")

                        # Try a different approach: recreate and send TaskList with all current state
                        try:
                            # Send the updated TaskList
                            await task_list.send()
                            logger.debug("TaskList.send() completed successfully.")

                            # Force a small delay to allow UI to process
                            await asyncio.sleep(0.1)

                        except Exception as send_error:
                            logger.exception("Error sending TaskList.")

                elif progress_type == "finalizing":
                    # Mark last file as done
                    if current_file_task:
                        current_file_task.status = cl.TaskStatus.DONE

                    # Start finalization
                    finalization_task.status = cl.TaskStatus.RUNNING
                    task_list.status = "✨ Generating comprehensive summary..."
                    await task_list.send()

                elif progress_type == "completed":
                    # Mark finalization as done
                    finalization_task.status = cl.TaskStatus.DONE
                    task_list.status = "✅ Facts extraction completed!"
                    await task_list.send()

            except Exception as e:
                logger.exception("Error updating progress.")

        # Get or generate facts for the collection
        try:
            facts_summary = await metadata_manager.get_or_generate_facts(
                vector_manager, collection_name, progress_callback=update_progress
            )

            # Send final results
            await cl.Message(content=f"📋 **Facts Summary for {collection_name}**\n\n{facts_summary}").send()

        except Exception as fact_error:
            logger.exception("Error during fact generation.")
            # Mark current tasks as failed
            if current_file_task:
                current_file_task.status = cl.TaskStatus.FAILED
            if finalization_task:
                finalization_task.status = cl.TaskStatus.FAILED
            task_list.status = "❌ Failed"
            await task_list.send()

            await cl.Message(
                content=f"❌ **Error generating facts for {collection_name}**\n\nAn error occurred during fact extraction. Please try again or check the logs for details."
            ).send()
            return

    except Exception as e:
        logger.exception("Error extracting facts.")
        await cl.Message(
            content=f"❌ An error occurred while extracting facts for '{collection_name}': {str(e)}"
        ).send()


@cl.action_callback("delete_collection")
async def delete_collection(action: cl.Action):
    """Delete a collection and all its documents."""
    collection_name = action.payload.get("collection_name")

    if not collection_name:
        await cl.context.emitter.send_toast("No collection specified for deletion", type="error")
        return

    manager: VectorStoreManager = cl.user_session.get("vector_store_manager")
    metadata_manager: CollectionMetadataManager = cl.user_session.get("collection_metadata_manager")

    if not manager:
        await cl.context.emitter.send_toast("Vector Store Manager not available", type="error")
        return

    try:
        success = manager.delete_collection(collection_name)
        if success:
            # Also delete facts for this collection if metadata manager is available
            if metadata_manager:
                metadata_manager.delete_facts_for_collection(collection_name)

            await cl.context.emitter.send_toast(f"Collection '{collection_name}' deleted successfully", type="success")
            # Refresh the collections display
            await show_available_collections()
        else:
            await cl.context.emitter.send_toast(f"Failed to delete collection '{collection_name}'", type="error")
    except Exception as e:
        await cl.context.emitter.send_toast(f"Error deleting collection: {str(e)}", type="error")
    finally:
        # Reset the selected collection if it was the one deleted
        await configure_settings(cl.user_session.get("selected_collection", "Alle Sammlungen"))
        cl.user_session.set("available_collections", manager.get_available_collections())


@cl.action_callback("delete_file_from_collection")
async def delete_file_from_collection(action: cl.Action):
    """Delete a specific file from a collection."""
    collection_name = action.payload.get("collection_name")
    filename = action.payload.get("filename")

    if not collection_name or not filename:
        await cl.context.emitter.send_toast("Missing collection name or filename", type="error")
        return

    manager: VectorStoreManager = cl.user_session.get("vector_store_manager")
    if not manager:
        await cl.context.emitter.send_toast("Vector Store Manager not available", type="error")
        return

    try:
        success = manager.delete_file_from_collection(collection_name, filename)
        if success:
            await cl.context.emitter.send_toast(f"File '{filename}' deleted from '{collection_name}'", type="success")
            # Refresh the collections display
            await show_available_collections()
        else:
            await cl.context.emitter.send_toast(f"Failed to delete file '{filename}'", type="error")
    except Exception as e:
        await cl.context.emitter.send_toast(f"Error deleting file: {str(e)}", type="error")
    finally:
        # In case that this was the last file in the collection, reset the selected collection if it was the one deleted
        await configure_settings(cl.user_session.get("selected_collection", "Alle Sammlungen"))
        cl.user_session.set("available_collections", manager.get_available_collections())


@cl.action_callback("upload_documents")
async def upload_documents(action: cl.Action):
    """
    Handle document uploads directly through Chainlit, bypassing the FastAPI service.
    This processes files uploaded via the DirectoryUploader component.
    """
    try:
        payload = action.payload
        collection_name = payload.get("collectionName", "").strip()
        files_data = payload.get("files", [])

        if not collection_name:
            return {"success": False, "error": "Collection name cannot be empty"}

        if not files_data:
            return {"success": False, "error": "No files provided"}

        # Sanitize collection name (same logic as document management service)
        def sanitize_filename(name: str) -> str:
            sanitized_name = re.sub(r"[^\w\s-]", "", name)
            sanitized_name = sanitized_name.replace(" ", "-")
            sanitized_name = sanitized_name.strip("_-")
            return sanitized_name

        sanitized_collection_name = sanitize_filename(collection_name)
        if not sanitized_collection_name:
            return {"success": False, "error": "Invalid collection name after sanitization"}

        # Setup upload directory
        base_upload_directory = relative_project_path(os.getenv("DOCUMENT_BASE_PATH", "data"))
        collection_upload_directory = os.path.join(base_upload_directory, sanitized_collection_name)

        # Check if collection already exists
        if os.path.exists(collection_upload_directory):
            return {"success": False, "error": f"Collection '{collection_name}' already exists"}

        os.makedirs(collection_upload_directory, exist_ok=True)

        uploaded_count = 0

        # Process each file
        for file_data in files_data:
            try:
                filename = file_data.get("name", "")
                relative_path = file_data.get("relativePath", filename)
                content_base64 = file_data.get("content", "")

                if not filename or not content_base64:
                    continue

                # Decode base64 content
                try:
                    file_content = base64.b64decode(content_base64)
                except Exception:
                    logger.warning("Failed to decode uploaded file '%s'.", filename)
                    continue

                # Preserve directory structure, removing the top-level directory
                path_parts = relative_path.split(os.sep) if os.sep in relative_path else relative_path.split("/")

                if len(path_parts) > 1:
                    # Keep subdirectory structure: subdir/file.pdf -> subdir/file.pdf
                    collection_relative_path = os.sep.join(path_parts[1:])
                else:
                    # File was directly in selected directory: file.pdf -> file.pdf
                    collection_relative_path = path_parts[0]

                full_target_path = os.path.join(collection_upload_directory, collection_relative_path)

                # Security check: ensure target path is within collection directory
                full_target_path = os.path.normpath(full_target_path)
                collection_upload_directory_norm = os.path.normpath(collection_upload_directory)
                if not full_target_path.startswith(collection_upload_directory_norm + os.sep):
                    logger.warning("Rejected uploaded file with invalid relative path.")
                    continue

                # Create subdirectories if they don't exist
                os.makedirs(os.path.dirname(full_target_path), exist_ok=True)

                # Write file content
                with open(full_target_path, "wb") as buffer:
                    buffer.write(file_content)

                uploaded_count += 1
                logger.debug("Saved uploaded file for collection '%s'.", sanitized_collection_name)

            except Exception:
                logger.exception("Error processing uploaded file '%s'.", file_data.get("name", "unknown"))
                continue

        if uploaded_count == 0:
            # Clean up empty directory
            try:
                os.rmdir(collection_upload_directory)
            except:
                pass
            return {"success": False, "error": "No files were successfully uploaded"}

        # Start embedding process in background - completely fire-and-forget
        def start_background_embedding():
            """Start embedding in a completely separate process/thread"""

            def embedding_worker():
                try:
                    logger.info("Background embedding started for collection '%s'.", sanitized_collection_name)

                    # Import here to avoid circular imports
                    from backend.chatbot.vector_db_manager import VectorStoreManager

                    # Create a status file to track progress (more reliable than session)
                    status_file_path = os.path.join(collection_upload_directory, ".embedding_status.json")

                    def update_status_file(status_dict):
                        """Update the status file atomically"""
                        try:
                            # Write to a temporary file first, then move it (atomic operation)
                            temp_file = status_file_path + ".tmp"
                            with open(temp_file, "w") as f:
                                json.dump(status_dict, f)
                            os.rename(temp_file, status_file_path)
                            logger.debug("Updated embedding status file.")
                        except Exception:
                            logger.exception("Error updating embedding status file.")
                            # Fallback to direct write if atomic write fails
                            try:
                                with open(status_file_path, "w") as f:
                                    json.dump(status_dict, f)
                            except Exception:
                                logger.exception("Fallback write for embedding status failed.")

                    # Set initial status
                    initial_status = {
                        "status": "processing",
                        "message": "Starting document parsing and embedding...",
                        "start_time": time.time(),
                        "collection_name": sanitized_collection_name,
                    }
                    update_status_file(initial_status)

                    # Create vector store manager and process the directory
                    manager = VectorStoreManager(recreate_index=False)

                    # Get all files in the uploaded directory
                    directory_path = Path(collection_upload_directory)
                    all_files = [f for f in directory_path.rglob("*") if f.is_file() and not f.name.startswith(".")]

                    logger.info("Background embedding found %s files to process.", len(all_files))

                    # Update status with file count
                    processing_status = {
                        "status": "processing",
                        "message": f"Processing {len(all_files)} files (parsing + enrichment + embedding)...",
                        "start_time": time.time(),
                        "collection_name": sanitized_collection_name,
                        "total_files": len(all_files),
                    }
                    update_status_file(processing_status)

                    # Process files - this is the long-running operation
                    manager._process_files(all_files, skip_existing=False)

                    # Mark as completed - preserve original start_time
                    completion_status = {
                        "status": "completed",
                        "message": f"Successfully processed and embedded {len(all_files)} files",
                        "start_time": processing_status["start_time"],  # Preserve original start time
                        "completion_time": time.time(),
                        "collection_name": sanitized_collection_name,
                        "total_files": len(all_files),
                    }
                    update_status_file(completion_status)

                    logger.info("Background embedding completed for collection '%s'.", sanitized_collection_name)

                except Exception as e:
                    logger.exception("Background embedding failed for collection '%s'.", sanitized_collection_name)

                    # Update status file with error
                    error_status = {
                        "status": "error",
                        "message": f"Processing failed: {str(e)}",
                        "start_time": time.time(),
                        "collection_name": sanitized_collection_name,
                        "error": str(e),
                    }
                    update_status_file(error_status)

            # Start the worker thread (daemon so it doesn't prevent shutdown)
            worker_thread = threading.Thread(target=embedding_worker, daemon=True)
            worker_thread.start()
            logger.info("Started background embedding thread for collection '%s'.", sanitized_collection_name)

        # Start the background process immediately
        start_background_embedding()

        # Update available collections immediately with the new collection
        manager: VectorStoreManager = cl.user_session.get("vector_store_manager")
        if manager:
            cl.user_session.set("available_collections", manager.get_available_collections())

        return {
            "success": True,
            "embedding_queued": True,
            "message": f"Successfully uploaded {uploaded_count} documents to collection '{collection_name}'. Embedding process started.",
        }

    except Exception as e:
        logger.exception("Upload documents action failed.")
        return {"success": False, "error": f"Upload failed: {str(e)}"}


@cl.action_callback("get_embedding_status")
async def get_embedding_status(action: cl.Action):
    """
    Get the status of the embedding process for a collection.
    This reads from a status file instead of session to handle long-running processes.
    """
    try:
        payload = action.payload
        collection_name = payload.get("collectionName", "").strip()

        if not collection_name:
            return {"success": False, "error": "Collection name is required"}

        # Sanitize collection name to match the directory name
        def sanitize_filename(name: str) -> str:
            sanitized_name = re.sub(r"[^\w\s-]", "", name)
            sanitized_name = sanitized_name.replace(" ", "-")
            sanitized_name = sanitized_name.strip("_-")
            return sanitized_name

        sanitized_collection_name = sanitize_filename(collection_name)

        # Look for status file in the collection directory
        from backend.utils import relative_project_path

        base_upload_directory = relative_project_path(os.getenv("DOCUMENT_BASE_PATH", "data"))
        collection_upload_directory = os.path.join(base_upload_directory, sanitized_collection_name)
        status_file_path = os.path.join(collection_upload_directory, ".embedding_status.json")

        # Check if status file exists
        if not os.path.exists(status_file_path):
            return {"success": True, "response": {"status": "not_found", "message": "No embedding process found"}}

        # Read status from file
        try:
            with open(status_file_path, "r") as f:
                status = json.load(f)

            # Add time elapsed information
            if "start_time" in status:
                elapsed = time.time() - status["start_time"]
                status["elapsed_time"] = int(elapsed)
                status["elapsed_display"] = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

                # Update message with elapsed time for processing status
                if status["status"] == "processing":
                    base_message = status.get("message", "Processing...")
                    status["message"] = f"{base_message} (Running for {status['elapsed_display']})"

            return {"success": True, "response": status}

        except json.JSONDecodeError:
            return {"success": True, "response": {"status": "error", "message": "Status file corrupted"}}
        except Exception as e:
            return {"success": True, "response": {"status": "error", "message": f"Error reading status: {str(e)}"}}

    except Exception:
        logger.exception("get_embedding_status action failed.")
        return {"success": False, "error": "An unexpected error occurred while reading embedding status."}


@cl.action_callback("refresh_available_collections")
async def refresh_available_collections():
    """
    Refresh the list of available collections and update settings.
    """
    try:
        manager: VectorStoreManager = cl.user_session.get("vector_store_manager")
        if manager:
            available_collections = manager.get_available_collections()
            cl.user_session.set("available_collections", available_collections)

            # Update settings to reflect new collections
            current_collection = cl.user_session.get("selected_collection", "Alle Sammlungen")
            await configure_settings(current_collection)

        return {"success": True}
    except Exception as e:
        logger.exception("Refresh collections failed.")
        return {"success": False, "error": str(e)}


@cl.action_callback("reload_vector_store")
async def reload_vector_store():
    """
    Reload the vector store to pick up new collections.
    """
    try:
        manager: VectorStoreManager = cl.user_session.get("vector_store_manager")
        if manager:
            # Reinitialize the vector store manager to pick up new collections
            new_manager = VectorStoreManager()
            cl.user_session.set("vector_store_manager", new_manager)
            cl.user_session.set("available_collections", new_manager.get_available_collections())

            # Update settings
            current_collection = cl.user_session.get("selected_collection", "Alle Sammlungen")
            await configure_settings(current_collection)

        return {"success": True}
    except Exception as e:
        logger.exception("Reload vector store failed.")
        return {"success": False, "error": str(e)}
