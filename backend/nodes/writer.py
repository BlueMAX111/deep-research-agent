from datetime import datetime
from typing import List, Dict, AsyncIterator

from backend.graph.state import ResearchState, ProcessedSource
from backend.prompts import WRITER_PROMPT
from backend.utils import get_llm, logger


def format_sources_for_writer(sources: List[ProcessedSource], source_map: Dict[str, int]) -> str:
    """格式化来源供 Writer 使用（使用数字引用）"""
    if not sources:
        return "暂无来源"

    lines = []
    for source in sources:
        idx = source_map.get(source['id'], 0)
        lines.append(f"[{idx}]")
        lines.append(f"标题: {source['title']}")
        lines.append(f"URL: {source['url']}")
        lines.append(f"摘要: {source['summary']}")
        if source['key_points']:
            lines.append(f"要点:")
            for point in source['key_points']:
                lines.append(f"  - {point}")
        lines.append("")

    return '\n'.join(lines)


def format_findings(findings: List[str]) -> str:
    """格式化关键发现"""
    if not findings:
        return "暂无明确发现"
    return '\n'.join(f"- {f}" for f in findings)


async def writer_node_streaming(state: ResearchState) -> AsyncIterator[str]:
    """
    Writer 节点（流式版本）：逐 token 输出报告
    
    这是唯一的 writer 实现，用于真正的 LLM 流式输出
    """
    logger.log_node_start("writer")

    topic = state["topic"]
    sources = state.get("sources", [])
    all_findings = state.get("all_findings", [])

    # 过滤低相关度来源
    relevant_sources = [s for s in sources if s.get("relevance", 0) >= 0.4]

    # 如果没有足够来源，降低阈值
    if len(relevant_sources) < 3:
        relevant_sources = sorted(sources, key=lambda x: x.get("relevance", 0), reverse=True)[:5]

    logger.log_info("writer", f"整合 {len(relevant_sources)} 个来源")

    # 建立 src_id → 数字的映射
    source_map: Dict[str, int] = {}
    for i, source in enumerate(relevant_sources):
        source_map[source["id"]] = i + 1

    # 格式化输入（使用数字引用）
    sources_formatted = format_sources_for_writer(relevant_sources, source_map)
    key_findings = format_findings(all_findings)

    # 构建 prompt
    prompt = WRITER_PROMPT.format(
        topic=topic,
        sources_with_ids=sources_formatted,
        key_findings=key_findings,
    )

    # 调用 LLM 流式输出
    llm = get_llm(temperature=0.7)
    full_report = ""

    async for chunk in llm.astream(prompt):
        if hasattr(chunk, 'content') and chunk.content:
            full_report += chunk.content
            yield chunk.content

    # 生成参考来源部分
    references = generate_references(full_report, relevant_sources, source_map)
    if references:
        yield references

    logger.log_info("writer", f"报告生成完成 ({len(full_report)} 字符)")
    logger.log_node_end("writer")
    logger.log_complete(len(sources), state.get("iteration", 1))


def generate_references(report: str, sources: List[ProcessedSource], source_map: Dict[str, int]) -> str:
    """生成参考来源列表"""
    if "## 参考来源" in report or "## References" in report:
        return ""

    references = ["\n\n---\n\n## 参考来源\n\n"]
    for source in sources:
        idx = source_map.get(source["id"], 0)
        if idx > 0:
            references.append(f"[{idx}] {source['title']} - {source['url']}\n")

    return ''.join(references)
