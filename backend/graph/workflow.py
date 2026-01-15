from langgraph.graph import StateGraph, END

from backend.graph.state import ResearchState
from backend.graph.edges import route_after_analyzer
from backend.nodes import (
    planner_node,
    searcher_basic_node,
    searcher_advanced_node,
    summarizer_node,
    analyzer_node,
)


def create_research_graph() -> StateGraph:
    """
    创建研究工作流图（不包含 writer 节点）

    工作流结构：
        Planner → Searcher(Basic) → Summarizer → Analyzer
                                                    ↓
                         ┌──────────────────────────┼──────────────────────────┐
                         ↓                          ↓                          ↓
                     sufficient               need_detail                 new_query
                         ↓                          ↓                          ↓
                        END               Searcher(Advanced)            Searcher(Basic)
                         ↑                          ↓                          ↓
                         │                     Summarizer                 Summarizer
                         │                          ↓                          ↓
                         │                      Analyzer                   Analyzer
                         │                          ↓                          ↓
                         └──────────────────────(循环...)──────────────────────┘
    
    注意：Writer 节点由流式 API 单独调用，以实现真正的 LLM 流式输出
    """

    # 创建状态图
    workflow = StateGraph(ResearchState)

    # 添加节点（不包含 writer）
    workflow.add_node("planner", planner_node)
    workflow.add_node("searcher_basic", searcher_basic_node)
    workflow.add_node("searcher_advanced", searcher_advanced_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("analyzer", analyzer_node)

    # 设置入口点
    workflow.set_entry_point("planner")

    # 添加边
    workflow.add_edge("planner", "searcher_basic")
    workflow.add_edge("searcher_basic", "summarizer")
    workflow.add_edge("searcher_advanced", "summarizer")
    workflow.add_edge("summarizer", "analyzer")

    # Analyzer → 条件路由（sufficient 时结束，由外部调用流式 writer）
    def route_without_writer(state):
        decision = route_after_analyzer(state)
        if decision == "writer":
            return END
        return decision
    
    workflow.add_conditional_edges(
        "analyzer",
        route_without_writer,
        {
            "searcher_basic": "searcher_basic",
            "searcher_advanced": "searcher_advanced",
            END: END,
        }
    )

    return workflow


def compile_research_graph():
    """编译研究工作流图"""
    workflow = create_research_graph()
    return workflow.compile()


# 创建可复用的编译后图实例
research_graph = None


def get_research_graph():
    """获取或创建编译后的图实例"""
    global research_graph
    if research_graph is None:
        research_graph = compile_research_graph()
    return research_graph
