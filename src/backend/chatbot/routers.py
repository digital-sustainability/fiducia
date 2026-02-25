from typing import List

from haystack import Document
from haystack.components.routers import ConditionalRouter


def get_contextualize_router():
    """
    Returns a ConditionalRouter that can be used to either redirect documents to a splitter if
    the documents are not chunked yet, or to redirect them to the embedding component if they are
    already chunked and contextualized. This router uses the unsafe flag, only use it if you trust
    the contents of the documents variable.
    """
    return ConditionalRouter(
        [
            {
                # Check if there is already a context in the document's metadata
                "condition": "{{ 'split_id' in documents[0].meta }}",
                "output": "{{documents}}",
                "output_name": "documents_with_splits",
                "output_type": List[Document],
            },
            {
                "condition": "{{ 'split_id' not in documents[0].meta }}",
                "output": "{{documents}}",
                "output_name": "documents_without_splits",
                "output_type": List[Document],
            },
        ],
        unsafe=True,  # Needed for returning a list of type Document
    )
