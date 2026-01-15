import json
import os
from datetime import datetime
from typing import List

from backend.graph.state import ResearchState, ProcessMessage, ProcessedSource, RawSearchResult
from backend.prompts import SUMMARIZER_PROMPT
from backend.utils import get_llm, locate_relevant_segments, logger

# 摘要结果保存目录
SUMMARY_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "summary_results")


def save_summary_results(results: List[dict], iteration: int):
    """保存摘要处理结果到文件"""
    os.makedirs(SUMMARY_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_iter{iteration}_summary.json"
    filepath = os.path.join(SUMMARY_RESULTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return filepath


def summarizer_node(state: ResearchState) -> dict:
    """
    Summarizer 节点：将原始搜索结果处理成结构化摘要

    输入：raw_results, keywords, topic
    输出：sources, messages
    """
    logger.log_node_start("summarizer")

    raw_results: List[RawSearchResult] = state.get("raw_results", [])
    keywords = state.get("keywords", [])
    topic = state["topic"]
    iteration = state.get("iteration", 1)

    if not raw_results:
        logger.log_info("summarizer", "没有待处理的结果")
        logger.log_node_end("summarizer")
        return {
            "messages": [
                ProcessMessage(
                    node="summarizer",
                    type="warning",
                    content="没有待处理的搜索结果",
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                )
            ]
        }

    llm = get_llm(temperature=0.3)
    processed_sources: List[ProcessedSource] = []
    messages: List[ProcessMessage] = []
    # 用于保存到文件的详细记录
    summary_records: List[dict] = []

    timestamp = datetime.now().strftime("%H:%M:%S")
    messages.append(
        ProcessMessage(
            node="summarizer",
            type="processing",
            content=f"正在处理 {len(raw_results)} 个搜索结果...",
            timestamp=timestamp,
        )
    )

    for result in raw_results:
        # 确定要处理的内容
        original_content = result.get("content") or result.get("snippet", "")
        content_to_process = original_content

        if not content_to_process:
            continue

        # 记录是否使用了关键词定位
        used_keyword_locate = False
        located_content = None

        # 如果内容较长，使用关键词定位
        if len(content_to_process) > 1000:
            used_keyword_locate = True
            located_content = locate_relevant_segments(
                content_to_process,
                keywords,
                context_lines=2,
                max_segments=5
            )
            content_to_process = located_content

        # 构建 prompt
        prompt = SUMMARIZER_PROMPT.format(
            topic=topic,
            query=result["query"],
            source_id=result["id"],
            title=result["title"],
            url=result["url"],
            content=content_to_process[:2000],  # 限制长度
        )

        try:
            response = llm.invoke(prompt)
            content = response.content

            # 解析 JSON
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                parsed = json.loads(json_str)
            else:
                raise ValueError("No JSON found")

            processed: ProcessedSource = {
                "id": result["id"],
                "title": result["title"],
                "url": result["url"],
                "query": result["query"],
                "summary": parsed.get("summary", ""),
                "key_points": parsed.get("key_points", []),
                "relevance": parsed.get("relevance", 0.5),
                "raw_content": original_content,
            }

            # 构建详细记录
            record = {
                "id": result["id"],
                "title": result["title"],
                "url": result["url"],
                "query": result["query"],
                "original_content_length": len(original_content),
                "used_keyword_locate": used_keyword_locate,
                "located_content": located_content if used_keyword_locate else None,
                "llm_output": {
                    "summary": parsed.get("summary", ""),
                    "key_points": parsed.get("key_points", []),
                    "relevance": parsed.get("relevance", 0.5),
                },
                "kept": processed["relevance"] >= 0.3,
            }
            summary_records.append(record)

            # 只保留相关度较高的来源
            if processed["relevance"] >= 0.3:
                processed_sources.append(processed)
                # 终端输出关键要点和相关度
                logger.log_info("summarizer", f"[{result['id']}] {result['title'][:40]}...")
                logger.log_detail("summarizer", "相关度", f"{processed['relevance']:.2f}")
                for i, point in enumerate(processed["key_points"][:3], 1):
                    point_text = point[:50] + "..." if len(point) > 50 else point
                    logger.log_detail("summarizer", f"要点{i}", point_text)

        except Exception as e:
            print(f"Summarizer error for {result['id']}: {e}")
            # 兜底：使用原始内容
            processed: ProcessedSource = {
                "id": result["id"],
                "title": result["title"],
                "url": result["url"],
                "query": result["query"],
                "summary": content_to_process[:200] + "...",
                "key_points": [],
                "relevance": 0.5,
                "raw_content": original_content,
            }
            processed_sources.append(processed)

            # 记录错误情况
            record = {
                "id": result["id"],
                "title": result["title"],
                "url": result["url"],
                "query": result["query"],
                "original_content_length": len(original_content),
                "used_keyword_locate": used_keyword_locate,
                "located_content": located_content if used_keyword_locate else None,
                "llm_output": None,
                "error": str(e),
                "kept": True,
            }
            summary_records.append(record)

    # 保存详细记录到文件
    filepath = save_summary_results(summary_records, iteration)
    logger.log_info("summarizer", f"处理完成: {len(processed_sources)}/{len(raw_results)} 有效")
    logger.log_detail("summarizer", "saved", os.path.basename(filepath))
    logger.log_node_end("summarizer")

    messages.append(
        ProcessMessage(
            node="summarizer",
            type="complete",
            content=f"已处理 {len(processed_sources)} 个有效来源",
            timestamp=datetime.now().strftime("%H:%M:%S"),
        )
    )

    return {
        "sources": processed_sources,
        "messages": messages,
        "raw_results": [],  # 清空已处理的结果
    }
