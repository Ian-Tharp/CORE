import os
from functools import lru_cache
from langchain_openai import ChatOpenAI
from openai import OpenAI


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
