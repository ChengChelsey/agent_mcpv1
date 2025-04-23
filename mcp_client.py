# mcp_client.py

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

SERVER_PATH = Path(__file__).with_name("adtk_server.py")

class ADTKFastMCPClient:
    """轻量级 MCP 客户端，采用 stdio 与本地服务器沟通。"""

    def __init__(self, port: int = 7777):
        self._params = StdioServerParameters(
            command="python",
            args=[str(SERVER_PATH), "--port", str(port)],
        )
        self._session: ClientSession | None = None

    async def connect(self):
        if self._session is None:
            try:
                logger.info("正在通过 stdio 连接 MCP 服务器...")
            # 使用 async with 语法处理异步上下文管理器
                async with stdio_client(self._params) as (read, write):
                    self._session = ClientSession(read, write)
                    await self._session.initialize()
                    logger.info("成功通过 stdio 连接到 MCP 服务器")
            except Exception as e:
                logger.error(f"连接 MCP 服务器失败: {e}")
                raise
        return self

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("已关闭MCP连接")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    async def get_all_detectors(self) -> Dict[str, Any]:
        """获取所有检测器信息"""
        if not self._session:
            raise RuntimeError("未连接到MCP服务器")
        
        logger.info("正在获取检测方法信息...")
        res = await self._session.call_tool("获取所有检测方法信息", {})
        logger.info("成功获取检测方法信息")
        return json.loads(res.content[0].text)

    async def detect(
        self,
        method: str,
        series: List[List[float]],
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """执行异常检测"""
        if not self._session:
            raise RuntimeError("未连接到MCP服务器")
            
        payload = {"series": series}
        if params:
            payload.update(params)
            
        logger.info(f"正在执行检测: {method}")
        res = await self._session.call_tool(method, payload)
        logger.info(f"检测完成: {method}")
        return json.loads(res.content[0].text)

# ----------------------------------------------------------------------
# Singleton helper
# ----------------------------------------------------------------------
_client_singleton: ADTKFastMCPClient | None = None

async def get_mcp_client(port: int = 7777) -> ADTKFastMCPClient:
    """获取MCP客户端单例"""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = ADTKFastMCPClient(port)
        await _client_singleton.connect()
    return _client_singleton

async def call_tool_async(tool_name: str, payload: dict | None = None) -> dict:
    """异步调用MCP工具"""
    payload = payload or {}
    client = await get_mcp_client()
    res = await client._session.call_tool(tool_name, payload)
    return json.loads(res.content[0].text)

def call_tool(tool_name: str, payload: dict | None = None) -> dict:
    """同步包装器，用于调用异步函数"""
    return asyncio.run(call_tool_async(tool_name, payload))

if __name__ == "__main__":
    async def _demo():
        try:
            cli = await get_mcp_client()
            print("成功连接到MCP服务器")
            
            # 测试ping工具
            try:
                ping_result = await cli._session.call_tool("ping", {})
                print(f"Ping测试: {ping_result.content[0].text}")
            except Exception as e:
                print(f"Ping测试失败: {e}")
            
            # 获取检测器信息
            meta = await cli.get_all_detectors()
            print(f"检测器数量: {len(meta)}")
            if meta:
                print("可用检测器:", list(meta.keys())[:5])
            
            # 测试检测功能
            demo_series = [[i, float(i % 7)] for i in range(50)]
            res = await cli.detect("IQR异常检测", demo_series, {"c": 3.0})
            print("检测结果:", res)
            
            await cli.close()
        except Exception as e:
            print(f"运行示例出错: {e}")
            import traceback
            traceback.print_exc()
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_demo())