import json
from datetime import datetime
from typing import List

from pydantic import BaseModel

from backend.graph.state import ResearchState, ProcessMessage
from backend.prompts import PLANNER_PROMPT
from backend.utils import get_llm, logger


class PlannerOutput(BaseModel):
    sub_queries: List[str]
    keywords: List[str]
    reasoning: str


def planner_node(state: ResearchState) -> dict:
    """
    Planner 节点：将用户问题拆解为可搜索的子问题

    输入：topic, mode
    输出：sub_queries, keywords, current_queries, messages
    """
    logger.log_node_start("planner")

    topic = state["topic"]
    mode = state["mode"]

    logger.log_info("planner", f"分析主题: {topic[:50]}...")

    # 构建 prompt
    prompt = PLANNER_PROMPT.format(topic=topic, mode=mode)

    # 调用 LLM
    llm = get_llm(temperature=0.7)
    response = llm.invoke(prompt)

    # 解析响应
    try:
        # 尝试从响应中提取 JSON
        content = response.content
        # 找到 JSON 部分
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")

        sub_queries = result.get("sub_queries", [])
        keywords = result.get("keywords", [])
        reasoning = result.get("reasoning", "")

    except (json.JSONDecodeError, ValueError) as e:
        # 兜底：使用简单的分割
        print(f"Planner parse error: {e}")
        sub_queries = [
            f"{topic} 基本概念",
            f"{topic} 最新进展",
            f"{topic} 应用场景",
        ]
        keywords = topic.split()
        reasoning = "使用默认拆分策略"

    # 终端日志
    logger.log_info("planner", f"生成 {len(sub_queries)} 个子问题")
    for q in sub_queries:
        logger.log_detail("planner", "子问题", q[:40])
    logger.log_info("planner", f"关键词: {', '.join(keywords[:5])}")
    logger.log_node_end("planner")

    # 创建过程消息
    timestamp = datetime.now().strftime("%H:%M:%S")
    messages = [
        ProcessMessage(
            node="planner",
            type="plan",
            content=f"已生成 {len(sub_queries)} 个子问题：{', '.join(sub_queries[:3])}...",
            timestamp=timestamp,
        ),
        ProcessMessage(
            node="planner",
            type="keywords",
            content=f"提取关键词：{', '.join(keywords[:5])}",
            timestamp=timestamp,
        ),
    ]

    return {
        "sub_queries": sub_queries,
        "keywords": keywords,
        "current_queries": sub_queries,  # 初始搜索词就是子问题
        "messages": messages,
    }
