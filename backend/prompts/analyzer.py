ANALYZER_BASE_PROMPT = """你是一个研究分析专家。你的任务是评估当前收集的信息是否足够，并决定下一步行动。

## 研究主题
{topic}

## 研究模式
{mode}

## 当前迭代
第 {iteration} 轮 / 最多 {max_iterations} 轮

## 已收集的来源摘要
{sources_summary}

## 累积的关键发现
{all_findings}

{mode_specific_instructions}

## 决策选项
1. **sufficient** - 信息已充分，可以撰写报告
2. **need_detail** - 某些来源需要深挖（获取完整内容）
3. **new_query** - 需要用新关键词搜索

## 输出格式
请按照以下 JSON 格式输出：
```json
{{
    "decision": "sufficient|need_detail|new_query",
    "reasoning": "决策理由...",
    "current_coverage": 0.75,
    "key_findings": ["发现1", "发现2"],
    "gaps": ["缺口1", "缺口2"],
    "detail_targets": [
        {{"source_id": "src_1", "reason": "需要深挖的原因"}}
    ],
    "new_queries": ["新搜索词1", "新搜索词2"],
    "query_type": "depth|breadth"
}}
```

注意：
- detail_targets 仅在 decision 为 "need_detail" 时填写
- new_queries 和 query_type 仅在 decision 为 "new_query" 时填写
- 其他情况下这些字段可以为空数组或空字符串
"""

DEPTH_MODE_INSTRUCTIONS = """
## 深度模式指导
你的目标是【深度挖掘】核心概念。

判断标准：
- 核心概念是否解释透彻？
- 技术细节是否充分？
- 原理机制是否清晰？

决策倾向：
- 如果某个来源提到了重要概念但细节不足 → need_detail
- 如果核心概念已经清晰 → sufficient
- 如果需要更深入的技术细节 → new_query（生成更具体的搜索词）
- 不主动扩展到其他方向
"""

BREADTH_MODE_INSTRUCTIONS = """
## 广度模式指导
你的目标是【广度覆盖】主题的各个方面。

判断标准：
- 是否覆盖了主题的主要方面？
- 是否有重要维度被遗漏？
- 各个方面是否都有基本了解？

决策倾向：
- 如果还有重要方面未涉及 → new_query（生成新方向的搜索词）
- 每个方向有概要信息即可，不要求 need_detail
- 覆盖面足够 → sufficient
"""

BALANCED_MODE_INSTRUCTIONS = """
## 平衡模式指导
你的目标是【先广后深】。

当前阶段判断：
- 前半程（iteration <= max_iterations/2）：优先广度覆盖
- 后半程（iteration > max_iterations/2）：对关键点深度挖掘

前半程策略：
- 识别主题的主要方面
- 确保基本覆盖
- 记录需要深入的关键点

后半程策略：
- 选择最重要的 1-2 个点深入
- 优先使用 need_detail 或深度 new_query
"""


def get_analyzer_prompt(
    topic: str,
    mode: str,
    iteration: int,
    max_iterations: int,
    sources_summary: str,
    all_findings: str
) -> str:
    """根据模式获取对应的 Analyzer Prompt"""

    mode_instructions = {
        "depth": DEPTH_MODE_INSTRUCTIONS,
        "breadth": BREADTH_MODE_INSTRUCTIONS,
        "balanced": BALANCED_MODE_INSTRUCTIONS,
    }

    return ANALYZER_BASE_PROMPT.format(
        topic=topic,
        mode=mode,
        iteration=iteration,
        max_iterations=max_iterations,
        sources_summary=sources_summary,
        all_findings=all_findings,
        mode_specific_instructions=mode_instructions.get(mode, BALANCED_MODE_INSTRUCTIONS)
    )
