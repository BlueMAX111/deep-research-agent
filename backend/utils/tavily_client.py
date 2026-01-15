from typing import List, Optional
from tavily import TavilyClient as BaseTavilyClient

from backend.config import config
from backend.graph.state import RawSearchResult


class TavilyClient:
    """Tavily 搜索 API 封装"""

    def __init__(self):
        self.client = BaseTavilyClient(api_key=config.TAVILY_API_KEY)
        self._source_counter = 0

    def reset_counter(self):
        """重置来源计数器"""
        self._source_counter = 0

    def _generate_source_id(self) -> str:
        """生成唯一的来源 ID"""
        self._source_counter += 1
        return f"src_{self._source_counter}"

    def search_basic(
        self,
        queries: List[str],
        max_results: int = 5
    ) -> List[RawSearchResult]:
        """
        Basic 模式搜索 - 返回 snippet

        Args:
            queries: 搜索词列表
            max_results: 每个查询的最大结果数

        Returns:
            RawSearchResult 列表
        """
        results = []

        for query in queries:
            try:
                response = self.client.search(
                    query=query,
                    search_depth="basic",
                    max_results=max_results,
                    include_answer=False,
                )

                for item in response.get("results", []):
                    result: RawSearchResult = {
                        "id": self._generate_source_id(),
                        "query": query,
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),  # basic 模式下 content 是摘要
                        "content": None,
                        "score": item.get("score", 0.0),
                    }
                    results.append(result)

            except Exception as e:
                print(f"Search error for query '{query}': {e}")
                continue

        return results

    def search_advanced(
        self,
        urls: List[str],
        query: str = ""
    ) -> List[RawSearchResult]:
        """
        Advanced 模式 - 获取完整内容

        Args:
            urls: 需要深挖的 URL 列表
            query: 相关的搜索词（用于标记）

        Returns:
            RawSearchResult 列表（包含完整 content）
        """
        results = []

        for url in urls:
            try:
                # 使用 extract 方法获取完整内容
                response = self.client.extract(urls=[url])

                for item in response.get("results", []):
                    result: RawSearchResult = {
                        "id": self._generate_source_id(),
                        "query": query,
                        "title": item.get("title", ""),
                        "url": item.get("url", url),
                        "snippet": "",
                        "content": item.get("raw_content", ""),
                        "score": 1.0,  # 深挖的默认为高相关
                    }
                    results.append(result)

            except Exception as e:
                print(f"Extract error for URL '{url}': {e}")
                continue

        return results

    def search_with_context(
        self,
        query: str,
        max_results: int = 5
    ) -> dict:
        """
        带上下文的搜索（Tavily 的增强模式）

        Returns:
            包含 answer 和 results 的字典
        """
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
            )
            return response
        except Exception as e:
            print(f"Context search error: {e}")
            return {"answer": "", "results": []}
