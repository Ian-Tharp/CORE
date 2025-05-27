import os
from functools import lru_cache
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

@lru_cache()
def get_4o_llm():
    return ChatOpenAI(
        model="gpt-4o",
        verbose=True,
        temperature=0.25,
        max_retries=3,
        streaming=True,
    )


@lru_cache()
def get_o1_llm():
    return ChatOpenAI(
        model="o1-preview-2024-09-12",
    )


@lru_cache()
def _get_openai_client() -> AsyncOpenAI:
    """Create and return an authenticated `AsyncOpenAI` client instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    return AsyncOpenAI(api_key=api_key)
