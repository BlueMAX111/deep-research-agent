import json
import os
from datetime import datetime
from typing import List

from backend.graph.state import ResearchState, ProcessMessage, RawSearchResult, DetailTarget
from backend.utils import TavilyClient, logger

# 搜索结果保存目录
SEARCH_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "search_results")


# 全局 Tavily 客户端实例
_tavily_client = None


def get_tavily_client() -> TavilyClient:
    """获取或创建 Tavily 客户端"""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client


def save_search_results(results: List[RawSearchResult], iteration: int, mode: str = "basic"):
    """保存搜索结果到文件"""
    os.makedirs(SEARCH_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_iter{iteration}_{mode}.json"
    filepath = os.path.join(SEARCH_RESULTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return filepath


def searcher_basic_node(state: ResearchState) -> dict:
    """
    Searcher Basic 节点：执行基础搜索

    输入：current_queries
    输出：raw_results, messages, iteration (如果是 new_query 触发则 +1)
    """
    logger.log_node_start("searcher_basic")

    queries = state.get("current_queries", [])

    if not queries:
        logger.log_info("searcher", "没有搜索词，跳过")
        return {
            "messages": [
                ProcessMessage(
                    node="searcher",
                    type="warning",
                    content="没有搜索词，跳过搜索",
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                )
            ]
        }

    # 执行搜索
    logger.log_info("searcher", f"搜索 {len(queries)} 个关键词...")
    for q in queries[:3]:  # 只显示前3个
        logger.log_detail("searcher", "query", q[:50])

    client = get_tavily_client()
    results: List[RawSearchResult] = client.search_basic(queries, max_results=3)

    logger.log_info("searcher", f"获取到 {len(results)} 个结果")

    # 保存搜索结果到文件
    iteration = state.get("iteration", 1)
    filepath = save_search_results(results, iteration, "basic")
    logger.log_detail("searcher", "saved", os.path.basename(filepath))

    # 创建过程消息
    timestamp = datetime.now().strftime("%H:%M:%S")
    messages = [
        ProcessMessage(
            node="searcher",
            type="search",
            content=f"搜索关键词：{', '.join(queries)}",
            timestamp=timestamp,
        ),
        ProcessMessage(
            node="searcher",
            type="result",
            content=f"获取到 {len(results)} 个搜索结果",
            timestamp=timestamp,
        ),
    ]

    # 检查是否需要增加迭代计数
    # 如果是 Analyzer 触发的 new_query，则增加迭代
    analysis = state.get("analysis")
    should_increment = analysis is not None and analysis.get("decision") == "new_query"

    update = {
        "raw_results": results,
        "messages": messages,
    }

    if should_increment:
        update["iteration"] = state.get("iteration", 0) + 1

    logger.log_node_end("searcher_basic")
    return update


def searcher_advanced_node(state: ResearchState) -> dict:
    """
    Searcher Advanced 节点：深挖特定来源

    输入：pending_detail_targets（需要深挖的来源）
    输出：raw_results, messages, detail_fetches +1
    """
    logger.log_node_start("searcher_advanced")

    targets: List[DetailTarget] = state.get("pending_detail_targets", [])

    if not targets:
        logger.log_info("searcher", "没有需要深挖的目标")
        return {
            "messages": [
                ProcessMessage(
                    node="searcher",
                    type="warning",
                    content="没有需要深挖的目标",
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                )
            ]
        }

    # 从已有来源中找到对应的 URL
    sources = state.get("sources", [])
    source_map = {s["id"]: s for s in sources}

    urls_to_fetch = []
    for target in targets:
        source = source_map.get(target["source_id"])
        if source:
            urls_to_fetch.append(source["url"])

    if not urls_to_fetch:
        logger.log_info("searcher", "未找到需要深挖的 URL")
        return {
            "messages": [
                ProcessMessage(
                    node="searcher",
                    type="warning",
                    content="未找到需要深挖的 URL",
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                )
            ]
        }

    logger.log_info("searcher", f"深挖 {len(urls_to_fetch)} 个来源...")

    # 执行 Advanced 抓取
    client = get_tavily_client()
    results = client.search_advanced(urls_to_fetch, query=state.get("topic", ""))

    logger.log_info("searcher", f"获取到 {len(results)} 个完整内容")

    # 保存搜索结果到文件
    iteration = state.get("iteration", 1)
    filepath = save_search_results(results, iteration, "advanced")
    logger.log_detail("searcher", "saved", os.path.basename(filepath))

    # 创建过程消息
    timestamp = datetime.now().strftime("%H:%M:%S")
    messages = [
        ProcessMessage(
            node="searcher",
            type="deep_dive",
            content=f"深挖 {len(urls_to_fetch)} 个来源",
            timestamp=timestamp,
        ),
        ProcessMessage(
            node="searcher",
            type="result",
            content=f"获取到 {len(results)} 个完整内容",
            timestamp=timestamp,
        ),
    ]

    logger.log_node_end("searcher_advanced")
    return {
        "raw_results": results,
        "messages": messages,
        "detail_fetches": state.get("detail_fetches", 0) + 1,
        "pending_detail_targets": [],  # 清空待处理目标
    }
