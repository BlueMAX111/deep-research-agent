import json
import asyncio
from datetime import datetime
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import config
from backend.graph.workflow import get_research_graph
from backend.graph.state import ResearchState
from backend.utils import logger
from backend.nodes.writer import writer_node_streaming


app = FastAPI(
    title="Deep Research Agent API",
    description="AI 深度调研智能体 API",
    version="0.1.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    """研究请求"""
    topic: str
    mode: Literal["depth", "breadth", "balanced"] = "balanced"
    max_iterations: Optional[int] = None
    max_detail_fetches: Optional[int] = None


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "service": "Deep Research Agent",
        "version": "0.1.0",
    }


@app.get("/config")
async def get_config():
    """获取默认配置"""
    return {
        "max_iterations": config.DEFAULT_MAX_ITERATIONS,
        "max_detail_fetches": config.DEFAULT_MAX_DETAIL_FETCHES,
        "default_mode": config.DEFAULT_MODE,
    }


@app.post("/research/stream")
async def research_stream(request: ResearchRequest):
    """
    研究 API - 使用 SSE 实时推送进度和流式输出报告

    返回事件类型：
    - start: 研究开始
    - node_start: 节点开始执行
    - node_output: 节点输出
    - node_end: 节点执行完成
    - iteration: 迭代计数更新
    - report_start: 开始生成报告
    - report_chunk: 报告内容分块（LLM 逐 token 输出）
    - complete: 研究完成
    - error: 错误信息
    """

    async def event_generator():
        try:
            # 初始化状态
            initial_state: ResearchState = {
                "topic": request.topic,
                "mode": request.mode,
                "sub_queries": [],
                "keywords": [],
                "current_queries": [],
                "pending_detail_targets": [],
                "raw_results": [],
                "sources": [],
                "analysis": None,
                "all_findings": [],
                "iteration": 1,
                "max_iterations": request.max_iterations or config.DEFAULT_MAX_ITERATIONS,
                "detail_fetches": 0,
                "max_detail_fetches": request.max_detail_fetches or config.DEFAULT_MAX_DETAIL_FETCHES,
                "report": "",
                "messages": [],
            }

            # 终端日志
            logger.log_start(request.topic, request.mode)

            # 发送开始事件
            yield {
                "event": "start",
                "data": json.dumps({
                    "topic": request.topic,
                    "mode": request.mode,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }),
            }

            # 获取图实例（不包含 writer，由后续流式调用）
            graph = get_research_graph()

            # 使用 stream 方法获取中间状态
            current_node = None
            last_iteration = 1
            final_state = initial_state.copy()

            # 执行工作流直到 analyzer 决定 sufficient
            async for event in graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    # 节点开始
                    if node_name != current_node:
                        if current_node:
                            yield {
                                "event": "node_end",
                                "data": json.dumps({
                                    "node": current_node,
                                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                                }),
                            }

                        current_node = node_name
                        yield {
                            "event": "node_start",
                            "data": json.dumps({
                                "node": node_name,
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                            }),
                        }

                    # 更新最终状态
                    for key, value in node_output.items():
                        if key in ["sources", "all_findings", "messages"]:
                            # 累积列表类型
                            if key not in final_state:
                                final_state[key] = []
                            if isinstance(value, list):
                                final_state[key] = final_state.get(key, []) + value
                        else:
                            final_state[key] = value

                    # 处理节点输出中的消息
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        yield {
                            "event": "node_output",
                            "data": json.dumps({
                                "node": msg.get("node", node_name),
                                "type": msg.get("type", "info"),
                                "content": msg.get("content", ""),
                                "timestamp": msg.get("timestamp", datetime.now().strftime("%H:%M:%S")),
                            }),
                        }

                    # 检查迭代更新
                    new_iteration = node_output.get("iteration")
                    if new_iteration and new_iteration != last_iteration:
                        last_iteration = new_iteration
                        yield {
                            "event": "iteration",
                            "data": json.dumps({
                                "current": new_iteration,
                                "max": initial_state["max_iterations"],
                            }),
                        }

                # 小延迟，避免过快
                await asyncio.sleep(0.1)

            # 最后一个节点结束
            if current_node:
                yield {
                    "event": "node_end",
                    "data": json.dumps({
                        "node": current_node,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }),
                }

            # 工作流完成，现在开始流式生成报告
            yield {
                "event": "node_start",
                "data": json.dumps({
                    "node": "writer",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }),
            }
            
            # 发送报告开始事件
            yield {
                "event": "report_start",
                "data": json.dumps({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }),
            }
            
            # 使用流式 writer 生成报告
            async for chunk in writer_node_streaming(final_state):
                yield {
                    "event": "report_chunk",
                    "data": json.dumps({
                        "content": chunk,
                    }),
                }
            
            # 发送完成事件
            yield {
                "event": "complete",
                "data": json.dumps({
                    "sources_count": len(final_state.get("sources", [])),
                    "iterations": last_iteration,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }),
            }
            
            yield {
                "event": "node_end",
                "data": json.dumps({
                    "node": "writer",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }),
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": str(e),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }),
            }

    async def sse_format():
        """格式化为 SSE 格式"""
        async for event in event_generator():
            event_type = event.get("event", "message")
            data = event.get("data", "{}")
            yield f"event: {event_type}\ndata: {data}\n\n"

    return StreamingResponse(
        sse_format(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
