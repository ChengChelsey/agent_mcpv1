# mcp_client.py  —— 直接覆盖
from __future__ import annotations
import asyncio, json, logging
from typing import Any, Dict, List
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SERVER_CMD = ["python", "adtk_server.py"]   # 启动姿势

class MCPClient:
    _inst: "MCPClient|None" = None
    def __init__(self):
        self._sess: ClientSession|None = None
        self._ctx = None

    async def connect(self):
        if self._sess:
            return
        logger.info("通过 STDIO 启动 ADTK Server …")
        params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
        self._ctx = stdio_client(params)
        read, write = await self._ctx.__aenter__()
        self._sess = ClientSession(read, write)
        await self._sess.initialize()
        logger.info("STDIO 连接就绪")

    async def close(self):
        if self._sess:
            await self._sess.close()
            self._sess = None
        if self._ctx:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None

    # 高层 API
    async def list_detectors(self):
        res = await self._sess.call_tool("获取所有检测方法信息", {})
        return json.loads(res.content[0].text)

    async def detect(self, tool:str, series:List[List[float]], params:Dict[str,Any]|None=None):
        payload = {"series": series}
        if params: payload.update(params)
        res = await self._sess.call_tool(tool, payload)
        return json.loads(res.content[0].text)

async def get_mcp_client() -> MCPClient:
    if MCPClient._inst is None:
        MCPClient._inst = MCPClient()
        await MCPClient._inst.connect()
    return MCPClient._inst
