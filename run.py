#!/usr/bin/env python3
"""
Deep Research Agent 启动脚本
"""
import uvicorn


if __name__ == "__main__":
    print("=" * 50)
    print("  Deep Research Agent")
    print("  启动中...")
    print("=" * 50)
    print()
    print("后端地址: http://localhost:8000")
    print("前端地址: 用浏览器打开 index.html")
    print()
    print("API 文档: http://localhost:8000/docs")
    print("=" * 50)
    print()

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
