# mcp_client.py  (适配 MCP 1.6 STDIO)
from __future__ import annotations
import asyncio, json, logging
from typing import Any, Dict, List

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SERVER_CMD = ["python", "adtk_server.py"]

class MCPClient:
    _inst: "MCPClient|None" = None
    def __init__(self): self._sess: ClientSession|None = None

    async def _ensure(self):
        if self._sess: return
        logger.info("拉起 ADTK Server (STDIO)…")
        params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
        self._ctx_mgr = stdio_client(params)
        read, write = await self._ctx_mgr.__aenter__()
        self._sess = ClientSession(read, write)
        await self._sess.initialize()
        logger.info("STDIO 连接就绪")

    async def list_detectors(self) -> Dict[str,Any]:
        await self._ensure()
        res = await self._sess.call_tool("获取所有检测方法信息", {})
        return json.loads(res.content[0].text)

    async def detect(self, tool:str, series:List[List[float]], params:Dict[str,Any]|None=None):
        await self._ensure()
        payload = {"series": series, **(params or {})}
        res = await self._sess.call_tool(tool, payload)
        return json.loads(res.content[0].text)

    async def close(self):
        if self._sess:
            await self._sess.close()
            await self._ctx_mgr.__aexit__(None,None,None)
            self._sess = None

# 单例 getter
async def get_mcp_client() -> MCPClient:
    if MCPClient._inst is None:
        MCPClient._inst = MCPClient()
    return MCPClient._inst
