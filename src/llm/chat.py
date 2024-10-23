"""Here, we define the chat generator that we will use to generate responses to user queries."""

import os
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI

load_dotenv()


def get_chat_generator(model_name: str) -> AzureOpenAI:

    load_dotenv()

    if model_name == "gpt-35-turbo-instruct":
        return AzureOpenAI(
            model_name=model_name,
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_GEN_INSTRUCT"),
            openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_type="azure",
        )

    else:
        raise ValueError(f"{model_name} is not a valid chat generator name")
