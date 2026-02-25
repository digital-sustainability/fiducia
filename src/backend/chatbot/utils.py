import os
import re
import types
import json
import traceback
from os import PathLike
from pathlib import Path
from typing import Literal
import typing

import requests
from dotenv import load_dotenv
from pydantic import Field, ValidationError
from haystack import Document
from litellm import completion
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import ChatMessage, ChatRole
from docling.utils.utils import create_file_hash
from chainlit import Message as ChainlitMessage
from ptpython import embed

from backend.chatbot.schemas import LegalChunkAnalysis
from backend.utils import relative_project_path

Uint64 = typing.Annotated[int, Field(ge=0, le=(2**64 - 1))]

load_dotenv(dotenv_path=relative_project_path(".env"))


def extract_timeline(chunk_content: str) -> tuple[dict, dict]:
    """
    Extracts a structured timeline and flat metadata fields from a legal document chunk.

    This function first calls `extract_entities` to get a `LegalChunkAnalysis` object.
    It then processes this object to create two dictionaries:
    1. A timeline dictionary where keys are dates and values are the events that occurred on those dates.
    2. A flat_fields dictionary containing lists of all dates, event types, participants, and roles,
       suitable for metadata indexing and filtering.

    :param chunk_content: The content of the legal document chunk.
    :return: A tuple containing the timeline dictionary and the flat_fields dictionary.
    """
    analysis = extract_entities(chunk_content)

    timeline = {}
    flat_fields = {
        "mentioned_dates": set(),
        "event_types": set(),
        "participant_names": set(),
        "participant_roles": set(),
    }

    if not analysis or not analysis.events:
        # Convert sets to lists before returning
        return timeline, {k: list(v) for k, v in flat_fields.items()}

    # Create maps for quick lookups
    date_map = {d.id: d for d in analysis.dates}
    participant_map = {p.id: p for p in analysis.participants}

    for event in analysis.events:
        event_date_obj = date_map.get(event.date_id)
        if not event_date_obj:
            continue

        event_date_iso = event_date_obj.iso_start
        flat_fields["mentioned_dates"].add(event_date_iso)
        flat_fields["event_types"].add(event.event_type.value)

        # Resolve participant names for the event
        event_participants = []
        for p_id in event.participant_ids:
            participant = participant_map.get(p_id)
            if participant:
                event_participants.append(participant.name)
                flat_fields["participant_names"].add(participant.name)
                flat_fields["participant_roles"].add(participant.role.value)

        # Create the event entry for the timeline
        timeline_event = {
            "label": event.label,
            "event_type": event.event_type.value,
            "participants": event_participants,
        }

        # Add the event to the timeline under the correct date
        if event_date_iso not in timeline:
            timeline[event_date_iso] = []
        timeline[event_date_iso].append(timeline_event)

    # Convert sets to lists for JSON serialization
    final_flat_fields = {k: list(v) for k, v in flat_fields.items()}

    return timeline, final_flat_fields


def post_mortem_exception_hook(exc_type, exc_value, exc_traceback):
    print(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
    traceback.print_exception(exc_type, exc_value, exc_traceback)


def get_prompt_builder(prompt_path: PathLike) -> PromptBuilder:
    """
    Returns a PromptBuilder object from a file path relative to the prompts directory.

    :param prompt_path: The path to the prompt file relative to the prompts directory.
    :return: The PromptBuilder object
    """
    prompt_path = Path(__file__).parent / "prompts" / prompt_path

    assert prompt_path.is_file(), f"Invalid prompt path provided: {prompt_path}"

    with open(prompt_path, "r", encoding="utf-8") as prompt_file:
        template = prompt_file.read()

    return PromptBuilder(template=template, required_variables="*")


def clean_text(text: str) -> str:
    """
    Clean text by removing unwanted characters and replacing line breaks with spaces.

    :param text: The text to clean.
    :return: The cleaned text.
    """
    unwanted_characters = r"[^a-zA-Z0-9äöüÄÖÜß\-/:;.,!?\(\)\n\f ]"
    text = re.sub(r"-\n([A-Za-z]+)", r"\1", text)  # Replace separated words with full word
    text = re.sub(unwanted_characters, "", text)

    # Add your custom cleaning logic here

    return text


def debug_component(component, exit_after_debug=False):
    """
    Modify the `run` method of a component to include debugging before and after execution
    and optionally exit after debugging.

    Usage: `debug_component(component)` - once the component is run, a PtPython shell will be opened.
    Use Ctrl + d to exit the shell and continue execution. If you want to have a look at the inputs
    of the component, you can access them via kwargs, e.g. `kwargs["sources"]`.

    :param component: The component to modify.
    :param exit_after_debug: Whether to exit after debugging.
    """
    original_run = component.run

    def modified_run(self, *args, **kwargs):
        embed(globals(), locals())

        result = original_run(*args, **kwargs)

        embed(globals(), locals())

        if exit_after_debug:
            exit()

        return result

    # Dynamically replace the `run` method
    component.run = types.MethodType(modified_run, component)


def add_metadata_to_message(message: ChatMessage, metadata: dict) -> ChatMessage:
    """
    Add metadata to a ChatMessage object.

    :param message: The ChatMessage object to add metadata to.
    :param metadata: A dictionary containing the metadata to add.
    :return: The updated ChatMessage object with the added metadata.
    """
    if not isinstance(message, ChatMessage):
        raise TypeError("message must be an instance of ChatMessage")

    if message.role == ChatRole.ASSISTANT:
        method = ChatMessage.from_assistant
    elif message.role == ChatRole.USER:
        method = ChatMessage.from_user
    elif message.role == ChatRole.SYSTEM:
        method = ChatMessage.from_system
    else:
        raise ValueError("Invalid message role")

    return method(text=message.text, meta={**message.meta, **metadata})


def model_warmup(model: Literal["All", "Embedding", "Reranker"] = "All", timeout: int = 120) -> None:
    """
    Sends a health check request to the model to ensure it is warmed up.

    :param timeout: The maximum time to wait for the model to warm up (in seconds).
    :raises AssertionError: If the TEI_EMBEDDING_ENDPOINT or TEI_RERANKER_ENDPOINT environment variables are not set.
    :raises requests.RequestException: If the health check request fails.
    """
    if model in ["All", "Embedding"]:
        assert "TEI_EMBEDDING_ENDPOINT" in os.environ, "TEI_EMBEDDING_ENDPOINT environment variable is not set!"
        response = requests.get(os.getenv("TEI_EMBEDDING_ENDPOINT") + "health", timeout=timeout)
        response.raise_for_status()

    if model in ["All", "Reranker"]:
        assert "TEI_RERANKER_ENDPOINT" in os.environ, "TEI_RERANKER_ENDPOINT environment variable is not set!"
        response = requests.get(os.getenv("TEI_RERANKER_ENDPOINT") + "health", timeout=timeout)
        response.raise_for_status()


def chainlit_to_haystack_message(
    message: ChainlitMessage,
) -> ChatMessage:
    """
    Converts a Chainlit message to a Haystack ChatMessage.

    :param message: The Chainlit message to convert.
    :return: A Haystack ChatMessage object.
    """
    match message.type:
        case "user_message":
            create_msg = ChatMessage.from_user
        case "assistant_message":
            create_msg = ChatMessage.from_assistant
        case "system_message":
            create_msg = ChatMessage.from_system
        case _:
            raise ValueError(
                f"Unknown message type: {message.type}. Supported types are: user_message, assistant_message, system_message."
            )

    return create_msg(
        text=message.content,
        meta=message.metadata,
    )


def haystack_to_chainlit_message(
    message: ChatMessage,
) -> ChainlitMessage:
    """
    Converts a Haystack ChatMessage to a Chainlit message.

    :param message: The Haystack ChatMessage to convert.
    :return: A Chainlit Message object.
    """
    match message.role:
        case ChatRole.USER:
            msg_type = "user_message"
        case ChatRole.ASSISTANT:
            msg_type = "assistant_message"
        case ChatRole.SYSTEM:
            msg_type = "system_message"
        case _:
            raise ValueError(
                f"Unknown message role: {message.role}. Supported roles are: {ChatRole.USER}, {ChatRole.ASSISTANT}, {ChatRole.SYSTEM}."
            )
    return ChainlitMessage(
        content=message.text,
        metadata=message.meta,
        type=msg_type,
    )


def chainlit_to_haystack_messages(
    messages: list[ChainlitMessage],
) -> list[ChatMessage]:
    """
    Converts a list of Chainlit messages to a list of Haystack ChatMessages.

    :param messages: The list of Chainlit messages to convert.
    :return: A list of Haystack ChatMessage objects.
    """
    return [chainlit_to_haystack_message(message) for message in messages]


def haystack_to_chainlit_messages(
    messages: list[ChatMessage],
) -> list[ChainlitMessage]:
    """
    Converts a list of Haystack ChatMessages to a list of Chainlit messages.

    :param messages: The list of Haystack ChatMessages to convert.
    :return: A list of Chainlit Message objects.
    """
    return [haystack_to_chainlit_message(message) for message in messages]


def create_binary_hash(file: Path) -> int:
    """
    Creates a hash for a binary file. Same implementation as used in Docling,
    see Docling's DocumentOrigin class for more details.

    :param file: The path to the binary file.
    :return: The hash of the file as an integer.
    """
    if not file.is_file():
        raise FileNotFoundError(f"File not found: {file}")

    hash_value = create_file_hash(file)

    return Uint64(hash_value, 16) & 0xFFFFFFFFFFFFFFFF


def extract_entities(chunk_content: str) -> LegalChunkAnalysis:
    """
    Extracts entities from a legal document chunk.

    :param chunk_content: The content of the legal document chunk.
    :return: A LegalChunkAnalysis object containing the extracted entities.
    :raises ValidationError: If the response from the model does not match the expected schema.
    """
    system_message = get_prompt_builder("extract_entities.system.prompt").run()["prompt"]
    user_message = get_prompt_builder("extract_entities.user.prompt").run(chunk_content=chunk_content)["prompt"]
    response = completion(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=8192,
        response_format={"type": "json_object", "schema": LegalChunkAnalysis.model_json_schema()},
    )

    response_content = response.choices[0].message.content

    try:
        if isinstance(response_content, dict):
            return LegalChunkAnalysis.model_validate(response_content)
        else:
            # It's a string, try to parse it
            return LegalChunkAnalysis.model_validate_json(response_content)
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"Failed to parse LLM response into LegalChunkAnalysis. Error: {e}")
        print(f"Raw model response that caused the error: {response_content}")
        # Return an empty object if parsing fails, as it likely means no entities were found
        return LegalChunkAnalysis()
