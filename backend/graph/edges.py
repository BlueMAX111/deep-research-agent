from backend.graph.state import ResearchState


def route_after_analyzer(state: ResearchState) -> str:
    """
    Analyzer 之后的路由决策

    Returns:
        - "writer": 信息充分，生成报告
        - "searcher_advanced": 需要深挖某些来源
        - "searcher_basic": 需要用新关键词搜索
    """
    analysis = state.get("analysis")

    if analysis is None:
        return "writer"

    decision = analysis.get("decision", "sufficient")

    # 信息充分
    if decision == "sufficient":
        return "writer"

    # 需要深挖
    if decision == "need_detail":
        # 检查 Advanced 抓取次数限制
        detail_fetches = state.get("detail_fetches", 0)
        max_detail_fetches = state.get("max_detail_fetches", 5)

        if detail_fetches >= max_detail_fetches:
            print(f"Reached max detail fetches ({max_detail_fetches}), going to writer")
            return "writer"

        return "searcher_advanced"

    # 需要新搜索
    if decision == "new_query":
        # 检查迭代次数限制
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        if iteration >= max_iterations:
            print(f"Reached max iterations ({max_iterations}), going to writer")
            return "writer"

        return "searcher_basic"

    # 兜底
    return "writer"


def should_continue_after_planner(state: ResearchState) -> str:
    """
    Planner 之后的路由（总是去 searcher）
    """
    return "searcher_basic"
