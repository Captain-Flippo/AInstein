"""Embeddings are used to convert text into vectors. 
This module provides a wrapper around the OpenAI functions to embed text.
We will use this to embed the user's question and the document documents, by using the `embed_query` and `embed_documents` methods.
As you will see, we will import the get_embedder function from this module in the frontend file."""

import logging
from omegaconf import OmegaConf
from langchain_openai import OpenAIEmbeddings
from enum import Enum

logger = logging.getLogger("AInstein")
config = OmegaConf.load("config.yaml")


class OpenAIEmbedderSelection(Enum):
    """A selection of OpenAI embedders to choose from."""
    
    LARGE = "text-embedding-3-large"
    SMALL = "text-embedding-3-small"  # Add other embedders as needed


def get_embedder(embedder: OpenAIEmbedderSelection.value) -> OpenAIEmbeddings:
    """Creates an OpenAIEmbeddings object.

    Args:
        embedder (OpenAIEmbedderSelection.value): Can be one of the OpenAIEmbedderSelection values.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object.
    """
    return OpenAIEmbeddings(
        model=embedder,
        api_key=config.api_key,
    )
    
