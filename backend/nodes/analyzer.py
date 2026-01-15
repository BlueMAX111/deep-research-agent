import json
from datetime import datetime
from typing import List

from backend.graph.state import (
    ResearchState,
    ProcessMessage,
    AnalysisResult,
    DetailTarget,
    ProcessedSource,
)
from backend.prompts import get_analyzer_prompt
from backend.utils import get_llm, logger


def format_sources_summary(sources: List[ProcessedSource]) -> str:
    """格式化来源摘要供 Analyzer 阅读"""
    if not sources:
        return "暂无收集到的来源"

    lines = []
    for source in sources:
        lines.append(f"[{source['id']}] {source['title']}")
        lines.append(f"    URL: {source['url']}")
        lines.append(f"    摘要: {source['summary']}")
        if source['key_points']:
            lines.append(f"    要点: {'; '.join(source['key_points'][:3])}")
        lines.append(f"    相关度: {source['relevance']:.2f}")
        lines.append("")

    return '\n'.join(lines)


def analyzer_node(state: ResearchState) -> dict:
    """
    Analyzer 节点：评估信息充分度，决定下一步行动

    输入：sources, topic, mode, iteration, max_iterations, all_findings
    输出：analysis, current_queries/pending_detail_targets, all_findings, messages
    """
    logger.log_node_start("analyzer")

    sources = state.get("sources", [])
    topic = state["topic"]
    mode = state["mode"]
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 3)
    all_findings = state.get("all_findings", [])

    # 检查是否已达到最大迭代次数，直接结束不做分析
    if iteration >= max_iterations:
        logger.log_info("analyzer", f"已达最大迭代次数 ({max_iterations})，直接进入写作")
        logger.log_node_end("analyzer")

        analysis: AnalysisResult = {
            "decision": "sufficient",
            "reasoning": f"已达到最大迭代次数 {max_iterations}",
            "detail_targets": [],
            "new_queries": [],
            "query_type": "breadth",
            "current_coverage": 0.8,
            "key_findings": all_findings,
            "gaps": [],
        }

        timestamp = datetime.now().strftime("%H:%M:%S")
        return {
            "analysis": analysis,
            "messages": [
                ProcessMessage(
                    node="analyzer",
                    type="decision",
                    content=f"达到最大迭代次数，准备生成报告",
                    timestamp=timestamp,
                )
            ],
            "all_findings": all_findings,
        }

    logger.log_info("analyzer", f"分析 {len(sources)} 个来源 (迭代 {iteration}/{max_iterations})")

    # 格式化来源摘要
    sources_summary = format_sources_summary(sources)
    findings_str = '\n'.join(f"- {f}" for f in all_findings) if all_findings else "暂无"

    # 构建 prompt
    prompt = get_analyzer_prompt(
        topic=topic,
        mode=mode,
        iteration=iteration,
        max_iterations=max_iterations,
        sources_summary=sources_summary,
        all_findings=findings_str,
    )

    # 调用 LLM
    llm = get_llm(temperature=0.3)
    response = llm.invoke(prompt)

    # 解析响应
    try:
        content = response.content
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            parsed = json.loads(json_str)
        else:
            raise ValueError("No JSON found")

        analysis: AnalysisResult = {
            "decision": parsed.get("decision", "sufficient"),
            "reasoning": parsed.get("reasoning", ""),
            "detail_targets": [
                DetailTarget(source_id=t["source_id"], reason=t["reason"])
                for t in parsed.get("detail_targets", [])
            ],
            "new_queries": parsed.get("new_queries", []),
            "query_type": parsed.get("query_type", "breadth"),
            "current_coverage": parsed.get("current_coverage", 0.5),
            "key_findings": parsed.get("key_findings", []),
            "gaps": parsed.get("gaps", []),
        }

    except Exception as e:
        print(f"Analyzer parse error: {e}")
        # 兜底：如果迭代次数足够，就结束
        analysis: AnalysisResult = {
            "decision": "sufficient" if iteration >= max_iterations else "new_query",
            "reasoning": f"解析错误，使用默认决策: {str(e)}",
            "detail_targets": [],
            "new_queries": [f"{topic} 深入分析"] if iteration < max_iterations else [],
            "query_type": "depth",
            "current_coverage": 0.5,
            "key_findings": [],
            "gaps": [],
        }

    # 终端日志
    logger.log_info("analyzer", f"覆盖度: {analysis['current_coverage']:.0%}")
    logger.log_decision(analysis['decision'], analysis['reasoning'][:60])
    if analysis["decision"] == "new_query":
        for q in analysis["new_queries"][:2]:
            logger.log_detail("analyzer", "new", q[:40])
    logger.log_node_end("analyzer")

    # 创建过程消息
    timestamp = datetime.now().strftime("%H:%M:%S")
    messages = [
        ProcessMessage(
            node="analyzer",
            type="analysis",
            content=f"信息覆盖度: {analysis['current_coverage']:.0%}",
            timestamp=timestamp,
        ),
        ProcessMessage(
            node="analyzer",
            type="decision",
            content=f"决策: {analysis['decision']} - {analysis['reasoning'][:100]}",
            timestamp=timestamp,
        ),
    ]

    # 根据决策设置下一步输入
    update = {
        "analysis": analysis,
        "messages": messages,
        "all_findings": analysis["key_findings"],
    }

    if analysis["decision"] == "new_query":
        update["current_queries"] = analysis["new_queries"]
        messages.append(
            ProcessMessage(
                node="analyzer",
                type="new_query",
                content=f"新搜索词: {', '.join(analysis['new_queries'])}",
                timestamp=timestamp,
            )
        )
    elif analysis["decision"] == "need_detail":
        update["pending_detail_targets"] = analysis["detail_targets"]
        target_ids = [t["source_id"] for t in analysis["detail_targets"]]
        messages.append(
            ProcessMessage(
                node="analyzer",
                type="deep_dive",
                content=f"需要深挖: {', '.join(target_ids)}",
                timestamp=timestamp,
            )
        )

    return update
