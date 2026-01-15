"""
简洁的终端日志输出工具
"""
from datetime import datetime

# ANSI 颜色代码
class Colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    PINK = '\033[95m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    GRAY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# 节点颜色映射
NODE_COLORS = {
    'planner': Colors.PURPLE,
    'searcher': Colors.CYAN,
    'searcher_basic': Colors.CYAN,
    'searcher_advanced': Colors.CYAN,
    'summarizer': Colors.PINK,
    'analyzer': Colors.YELLOW,
    'writer': Colors.GREEN,
}


def log_node_start(node: str):
    """节点开始"""
    color = NODE_COLORS.get(node, Colors.GRAY)
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.GRAY}{timestamp}{Colors.RESET} {color}{Colors.BOLD}▶ {node.upper()}{Colors.RESET}")


def log_node_end(node: str):
    """节点结束"""
    color = NODE_COLORS.get(node, Colors.GRAY)
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.GRAY}{timestamp}{Colors.RESET} {color}✓ {node} 完成{Colors.RESET}")


def log_info(node: str, message: str):
    """普通信息"""
    color = NODE_COLORS.get(node, Colors.GRAY)
    timestamp = datetime.now().strftime("%H:%M:%S")
    # 截断过长消息
    if len(message) > 80:
        message = message[:77] + "..."
    print(f"{Colors.GRAY}{timestamp}{Colors.RESET} {color}│{Colors.RESET} {message}")


def log_detail(node: str, key: str, value):
    """详细信息（键值对）"""
    color = NODE_COLORS.get(node, Colors.GRAY)
    print(f"         {color}│{Colors.RESET}   {Colors.GRAY}{key}:{Colors.RESET} {value}")


def log_iteration(current: int, max_iter: int):
    """迭代信息"""
    print(f"\n{Colors.YELLOW}{'─' * 40}")
    print(f"  迭代 {current}/{max_iter}")
    print(f"{'─' * 40}{Colors.RESET}\n")


def log_decision(decision: str, reason: str = ""):
    """Analyzer 决策"""
    color = Colors.GREEN if decision == "sufficient" else Colors.YELLOW
    symbol = "✓" if decision == "sufficient" else "→"
    print(f"         {color}│{Colors.RESET}   决策: {color}{symbol} {decision}{Colors.RESET}")
    if reason and len(reason) > 60:
        reason = reason[:57] + "..."
    if reason:
        print(f"         {color}│{Colors.RESET}   原因: {reason}")


def log_separator():
    """分隔线"""
    print(f"{Colors.GRAY}{'─' * 50}{Colors.RESET}")


def log_start(topic: str, mode: str):
    """研究开始"""
    print(f"\n{Colors.BOLD}{'═' * 50}")
    print(f"  Deep Research Agent")
    print(f"{'═' * 50}{Colors.RESET}")
    print(f"  主题: {topic}")
    print(f"  模式: {mode}")
    print(f"{Colors.GRAY}{'─' * 50}{Colors.RESET}\n")


def log_complete(sources_count: int, iterations: int):
    """研究完成"""
    print(f"\n{Colors.GREEN}{'═' * 50}")
    print(f"  研究完成!")
    print(f"  来源数: {sources_count} | 迭代数: {iterations}")
    print(f"{'═' * 50}{Colors.RESET}\n")
