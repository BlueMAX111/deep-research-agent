import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # DeepSeek
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    # Tavily
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Default settings
    DEFAULT_MAX_ITERATIONS: int = int(os.getenv("DEFAULT_MAX_ITERATIONS", "3"))
    DEFAULT_MAX_DETAIL_FETCHES: int = int(os.getenv("DEFAULT_MAX_DETAIL_FETCHES", "5"))
    DEFAULT_MODE: str = os.getenv("DEFAULT_MODE", "balanced")


config = Config()
