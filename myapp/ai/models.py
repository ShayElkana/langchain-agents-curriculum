import os

from langchain.chat_models import init_chat_model


def get_chat_model():
    """Single factory — model name is config, not scattered string literals."""
    return init_chat_model(
        os.getenv("MODEL", "openai:gpt-5-nano"),
        timeout=30,
        max_retries=2,
    )
