import re
from typing import List, Tuple

import jieba


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    从文本中提取关键词

    Args:
        text: 输入文本
        top_k: 返回的关键词数量

    Returns:
        关键词列表
    """
    # 使用 jieba 分词
    words = jieba.cut(text)

    # 停用词（简化版）
    stopwords = {
        "的", "是", "在", "了", "和", "与", "或", "等", "对", "中",
        "为", "有", "这", "个", "上", "下", "不", "也", "就", "都",
        "而", "及", "到", "以", "可以", "一个", "一种", "一些",
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "and",
        "or", "but", "if", "then", "else", "when", "where", "what",
        "which", "who", "whom", "this", "that", "these", "those",
        "it", "its", "of", "in", "on", "at", "to", "for", "with",
        "by", "from", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "under", "again",
        "further", "once", "here", "there", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very",
    }

    # 过滤并计数
    word_count = {}
    for word in words:
        word = word.strip().lower()
        if len(word) < 2:
            continue
        if word in stopwords:
            continue
        if re.match(r'^[\d\s\W]+$', word):
            continue
        word_count[word] = word_count.get(word, 0) + 1

    # 按频率排序
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    return [word for word, _ in sorted_words[:top_k]]


def locate_relevant_segments(
    content: str,
    keywords: List[str],
    context_lines: int = 2,
    max_segments: int = 5
) -> str:
    """
    根据关键词定位相关段落

    Args:
        content: 完整内容
        keywords: 关键词列表
        context_lines: 上下文行数
        max_segments: 最大段落数

    Returns:
        提取的相关段落文本
    """
    if not content or not keywords:
        return content[:1000] if content else ""

    # 按行分割
    lines = content.split('\n')
    line_scores = []

    # 计算每行的关键词命中分数
    for i, line in enumerate(lines):
        line_lower = line.lower()
        score = sum(1 for kw in keywords if kw.lower() in line_lower)
        if score > 0:
            line_scores.append((i, score))

    # 按分数排序
    line_scores.sort(key=lambda x: x[1], reverse=True)

    # 提取 top 段落及其上下文
    selected_lines = set()
    for line_idx, _ in line_scores[:max_segments]:
        start = max(0, line_idx - context_lines)
        end = min(len(lines), line_idx + context_lines + 1)
        for i in range(start, end):
            selected_lines.add(i)

    # 如果没有命中，返回开头和结尾
    if not selected_lines:
        head = '\n'.join(lines[:10])
        tail = '\n'.join(lines[-5:]) if len(lines) > 15 else ""
        return f"{head}\n...\n{tail}" if tail else head

    # 按顺序组合
    sorted_indices = sorted(selected_lines)
    segments = []
    prev_idx = -2

    for idx in sorted_indices:
        if idx > prev_idx + 1:
            segments.append("...")
        segments.append(lines[idx])
        prev_idx = idx

    return '\n'.join(segments)


def split_into_paragraphs(text: str) -> List[str]:
    """将文本分割成段落"""
    # 按双换行或多个换行分割
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    简单的文本相似度计算（基于词重叠）

    Returns:
        0-1 之间的相似度分数
    """
    words1 = set(jieba.cut(text1.lower()))
    words2 = set(jieba.cut(text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0
