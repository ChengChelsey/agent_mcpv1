# mcp_client.py
"""
更新的MCP客户端 - 修复了参数验证问题
"""
from __future__ import annotations
import asyncio
import json
import logging
import sys
import os
import traceback
from typing import Dict, List, Any, Optional, Tuple

# 导入MCP库
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mcp_client")

# 服务器命令
SERVER_CMD = ["python", "fixed_adtk_server_v3.py"]

class MCPClient:
    """ADTK MCP客户端类"""
    _instance = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        """初始化客户端"""
        self._session = None
        self._read = None
        self._write = None
        self._context_manager = None
        self._initialized = False
        
    @classmethod
    async def get_instance(cls) -> MCPClient:
        """获取单例实例"""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = MCPClient()
            return cls._instance
    
    async def ensure_connected(self):
        """确保已连接到服务器"""
        if self._initialized:
            return
            
        logger.info("连接ADTK服务器...")
        server_params = StdioServerParameters(
            command=SERVER_CMD[0],
            args=SERVER_CMD[1:]
        )
        
        try:
            # 创建连接
            self._context_manager = stdio_client(server_params)
            self._read, self._write = await self._context_manager.__aenter__()
            
            # 创建会话
            self._session = ClientSession(self._read, self._write)
            await self._session.initialize()
            self._initialized = True
            logger.info("ADTK服务器连接和初始化成功")
            
        except Exception as e:
            logger.error(f"连接ADTK服务器失败: {e}")
            if self._context_manager:
                try:
                    await self._context_manager.__aexit__(type(e), e, None)
                except Exception as close_error:
                    logger.error(f"关闭连接失败: {close_error}")
                self._context_manager = None
            raise
    
    async def list_detectors(self) -> Dict[str, Any]:
        """获取所有检测器信息"""
        await self.ensure_connected()
        
        try:
            result = await self._session.call_tool("获取所有检测方法信息", {})
            if not result.content:
                logger.error("获取检测器信息返回空内容")
                return {}
                
            return json.loads(result.content[0].text)
        except Exception as e:
            logger.error(f"获取检测器信息失败: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    async def get_all_detectors(self) -> Dict[str, Any]:
        """获取所有检测方法信息（别名）"""
        return await self.list_detectors()
    
    async def detect(self, method: str, series: List[List[float]], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行异常检测
        
        Args:
            method: 检测方法名称
            series: 时序数据 [[timestamp, value], ...]
            params: 检测参数
            
        Returns:
            检测结果
        """
        await self.ensure_connected()
        
        # 准备参数 - 必须包含series
        arguments = {"series": series}
        if params:
            arguments.update(params)
        
        logger.info(f"调用检测方法: {method}, 参数: {json.dumps(arguments)[:100]}...")
        
        try:
            # 调用工具
            result = await self._session.call_tool(method, arguments)
            
            if not result.content:
                logger.error(f"检测方法 {method} 返回空内容")
                return {"error": "服务器返回空内容"}
                
            try:
                result_text = result.content[0].text
                logger.info(f"原始结果: {result_text[:100]}...")
                return json.loads(result_text)
            except json.JSONDecodeError as e:
                logger.error(f"解析结果JSON失败: {e}")
                logger.error(f"原始文本: {result.content[0].text}")
                return {"error": f"解析结果失败: {str(e)}"}
                
        except Exception as e:
            logger.error(f"执行检测失败: {e}")
            traceback.print_exc()
            return {"error": f"执行检测失败: {str(e)}"}
    
    async def close(self):
        """关闭客户端连接"""
        logger.info("关闭MCP客户端...")
        
        # 关闭会话
        if self._session:
            try:
                await self._session.close()
            except Exception as e:
                logger.error(f"关闭会话失败: {e}")
            self._session = None
        
        # 关闭连接上下文
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"关闭连接上下文失败: {e}")
            self._context_manager = None
        
        self._initialized = False
        logger.info("MCP客户端已关闭")

# 获取客户端实例
async def get_mcp_client() -> MCPClient:
    """获取MCP客户端实例"""
    return await MCPClient.get_instance()