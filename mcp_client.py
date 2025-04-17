#mcp.client.py

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SERVER_SCRIPT = Path(__file__).with_name("adtk_fastmcp_server.py")

class ADTKFastMCPClient:
    """轻量级 MCP 客户端，采用 stdio 与本地服务器沟通。"""

    def __init__(self, port: int = 7777):
        self._params = StdioServerParameters(
            command="python",
            args=[str(SERVER_SCRIPT), "--port", str(port)],
        )
        self._session: ClientSession | None = None

    async def connect(self):
        if self._session is None:
            read, write = await stdio_client(self._params)
            self._session = ClientSession(read, write)
            await self._session.initialize()
            logger.info("Connected to local FastMCP server via stdio")
        return self

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    async def get_all_detectors(self) -> Dict[str, Any]:
        res = await self._session.call_tool("获取所有检测方法信息", {})
        return json.loads(res.content[0].text)

    async def detect(
        self,
        method: str,
        series: List[List[float]],
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = {"series": series}
        if params:
            payload.update(params)
        res = await self._session.call_tool(method, payload)
        return json.loads(res.content[0].text)

# ----------------------------------------------------------------------
# Singleton helper
# ----------------------------------------------------------------------
_client_singleton: ADTKFastMCPClient | None = None

async def get_mcp_client(port: int = 7777) -> ADTKFastMCPClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = ADTKFastMCPClient(port)
        await _client_singleton.connect()
    return _client_singleton

async def call_tool(tool_name: str, payload: dict | None = None) -> dict:
    """
    兼容早期代码里 `from mcp_client import call_tool` 的调用写法。
    等同于直接 session.call_tool。
    """
    payload = payload or {}
    client = await get_mcp_client()        # 单例
    res = await client._session.call_tool(tool_name, payload)
    return json.loads(res.content[0].text)

if __name__ == "__main__":
    async def _demo():
        cli = await get_mcp_client()
        meta = await cli.get_all_detectors()
        print("detectors:", list(meta.keys())[:5], "...")
        demo_series = [[i, float(i % 7)] for i in range(50)]
        res = await cli.detect("IQR异常检测", demo_series, {"c": 3.0})
        print(res)
        await cli.close()
    asyncio.run(_demo())
