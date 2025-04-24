#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcp_process.py
----------------------------------
独立子进程执行 MCP-Server 调用，避免与主进程事件循环冲突。
用法（由 agent.py 间接调用）:
    python mcp_process.py  <  JSON输入  >  JSON输出
输入格式:
{
  "operation": "ping" | "list_detectors" | "detect",
  "params": { ... }            # detect 时: {method, series, params?}
}
输出统一格式:
{ "status": "success", "result": <string | object> }
或
{ "status": "error",   "message": "<error info>" }
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from typing import Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - mcp_process - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_process")

# ---------- 统一 I/O 工具 ----------
def jprint(obj: Dict[str, Any]):
    """安全输出 JSON 到 stdout"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.flush()


# ---------- 核心协程 ----------
async def run_operation(op: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """连接 MCP-Server，执行指定操作"""
    server_params = StdioServerParameters(
        command="python",
        args=["adtk_server.py"],          # 同目录下的 ADTK-Server
    )

    try:
        logger.info("Launching ADTK-MCP server…")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("MCP session ready → op=%s", op)

                # ----------- ping -----------
                if op == "ping":
                    res = await session.call_tool("ping", {})
                    return {"status": "success", "result": res.content[0].text}

                # ----------- list detectors -----------
                if op == "list_detectors":
                    res = await session.call_tool("获取所有检测方法信息", {})
                    return {"status": "success", "result": res.content[0].text}

                # ----------- detect -----------
                if op == "detect":
                    method  = params.get("method")
                    series  = params.get("series")
                    extra   = params.get("params", {})

                    if not method or series is None:
                        return {"status": "error",
                                "message": "detect 需要提供 method 与 series"}

                    arguments = {"series": series}
                    arguments.update(extra)

                    res = await session.call_tool(method, arguments)
                    return {"status": "success", "result": res.content[0].text}

                # ----------- unknown -----------
                return {"status": "error", "message": f"未知 operation: {op}"}

    except Exception as e:
        logger.error("MCP 操作失败: %s", e)
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ---------- main ----------
async def _amain():
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw or "{}")
        op     = payload.get("operation") or ""
        params = payload.get("params", {})

        if not op:
            jprint({"status": "error", "message": "operation 不能为空"})
            return

        result = await run_operation(op, params)
        jprint(result)

    except Exception as e:
        logger.error("顶层异常: %s", e)
        traceback.print_exc()
        jprint({"status": "error", "message": str(e)})


if __name__ == "__main__":
    asyncio.run(_amain())
