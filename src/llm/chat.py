"""Here, we define the chat generator that we will use to generate responses to user queries."""

import os
from enum import Enum
from logging import config
from langchain_openai import OpenAI
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
print(config)
# Load API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("API key for OpenAI is not set in environment variables.")

class OpenAIModelSelection(Enum):
    """A selection of OpenAI models to choose from."""
    
    GPT3 = "gpt-3.5-turbo"
    GPT4 = "gpt-4.0-turbo"

def get_chat_generator(model_name: OpenAIModelSelection) -> OpenAI:
    
        return OpenAI(
            model=model_name,
            api_key=API_KEY,
            temperature=float(config.llm.temperature),
            top_p=float(config.llm.top_p),
        )