from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Type, TypeVar

from backend.config import config

T = TypeVar("T", bound=BaseModel)


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """获取 DeepSeek LLM 实例"""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=temperature,
    )


def get_structured_llm(schema: Type[T], temperature: float = 0.3) -> ChatOpenAI:
    """获取带结构化输出的 LLM 实例"""
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=temperature,
    )
    return llm.with_structured_output(schema)
