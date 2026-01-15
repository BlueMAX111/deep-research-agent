from typing import TypedDict, List, Literal, Optional, Annotated
from operator import add


class RawSearchResult(TypedDict):
    """Tavily 返回的原始搜索结果"""
    id: str              # 唯一标识 "src_1", "src_2"
    query: str           # 产生此结果的搜索词
    title: str
    url: str
    snippet: str         # Basic 模式的摘要
    content: Optional[str]  # Advanced 模式的完整内容
    score: float         # 相关度评分


class ProcessedSource(TypedDict):
    """处理后的来源"""
    id: str
    title: str
    url: str
    query: str
    summary: str           # LLM 生成的摘要
    key_points: List[str]  # 提取的要点
    relevance: float       # 与主题的相关度 0-1
    raw_content: str       # 保留原文，供 Advanced 深挖


class DetailTarget(TypedDict):
    """需要深挖的目标"""
    source_id: str
    reason: str  # 为什么需要深挖


class AnalysisResult(TypedDict):
    """Analyzer 的输出"""
    decision: Literal["sufficient", "need_detail", "new_query"]
    reasoning: str  # 决策理由

    # decision = "need_detail" 时
    detail_targets: List[DetailTarget]

    # decision = "new_query" 时
    new_queries: List[str]
    query_type: Literal["depth", "breadth"]

    # 通用信息
    current_coverage: float      # 当前信息覆盖度 0-1
    key_findings: List[str]      # 目前的关键发现
    gaps: List[str]              # 信息缺口


class ProcessMessage(TypedDict):
    """过程消息（前端展示用）"""
    node: str           # 哪个节点产生的
    type: str           # 消息类型
    content: str        # 消息内容
    timestamp: str      # 时间戳


class ResearchState(TypedDict):
    """研究状态 - LangGraph 的核心状态定义"""

    # === 输入 ===
    topic: str                              # 用户原始问题
    mode: Literal["depth", "breadth", "balanced"]  # 研究模式

    # === 规划阶段 ===
    sub_queries: List[str]                  # Planner 生成的子问题
    keywords: List[str]                     # 提取的关键词

    # === 搜索阶段 ===
    current_queries: List[str]              # 当前轮次的搜索词
    pending_detail_targets: List[DetailTarget]  # 待深挖的目标
    raw_results: List[RawSearchResult]      # Searcher 返回的原始搜索结果

    # === 来源管理 ===
    sources: Annotated[List[ProcessedSource], add]  # 累积的所有来源

    # === 分析阶段 ===
    analysis: Optional[AnalysisResult]      # 最新的分析结果
    all_findings: Annotated[List[str], add]  # 累积的关键发现

    # === 控制变量 ===
    iteration: int                          # 当前迭代次数（仅 new_query 时 +1）
    max_iterations: int                     # 最大迭代次数
    detail_fetches: int                     # Advanced 抓取次数
    max_detail_fetches: int                 # Advanced 抓取上限

    # === 输出 ===
    report: str                             # 最终报告

    # === 追踪（前端展示用）===
    messages: Annotated[List[ProcessMessage], add]  # 过程消息流
