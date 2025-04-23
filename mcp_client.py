# mcp_client.py
"""
基于直接成功连接代码的 MCP 客户端
"""
from __future__ import annotations
import asyncio
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple

# 导入MCP库
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mcp_client")

class MCPClient:
    """ADTK MCP客户端类"""
    _instance = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        """初始化客户端"""
        # 只保存必要的状态
        self._session = None
        self._context_mgr = None
        self._read = None
        self._write = None
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
        
        # 服务器参数 - 与成功测试完全一致
        server_params = StdioServerParameters(
            command="python", 
            args=["adtk_server.py"]
        )
        
        try:
            logger.info("连接ADTK服务器...")
            
            # 使用与直接测试相同的方式创建连接
            self._context_mgr = stdio_client(server_params)
            self._read, self._write = await self._context_mgr.__aenter__()
            logger.info("连接成功，创建会话...")
            
            # 创建会话
            self._session = ClientSession(self._read, self._write)
            await self._session.initialize()
            logger.info("会话初始化成功")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"连接ADTK服务器失败: {e}")
            if self._context_mgr:
                try:
                    await self._context_mgr.__aexit__(type(e), e, None)
                except Exception as close_err:
                    logger.error(f"关闭连接失败: {close_err}")
                self._context_mgr = None
            raise
    
    async def call_tool(self, tool_name: str, args: Dict[str, Any] = None):
        """直接调用工具"""
        if args is None:
            args = {}
            
        await self.ensure_connected()
        logger.info(f"调用工具: {tool_name}")
        return await self._session.call_tool(tool_name, args)
    
    async def list_detectors(self) -> Dict[str, Any]:
        """获取所有检测器信息"""
        await self.ensure_connected()
        
        try:
            logger.info("获取检测器列表...")
            result = await self._session.call_tool("获取所有检测方法信息", {})
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
        
        logger.info(f"执行{method}...")
        
        try:
            result = await self._session.call_tool(method, arguments)
            
            try:
                result_text = result.content[0].text
                logger.info(f"检测原始结果: {result_text[:100]}...")
                
                parsed_result = json.loads(result_text)
                
                if "error" in parsed_result:
                    logger.error(f"检测失败: {parsed_result['error']}")
                else:
                    anomalies = parsed_result.get("anomalies", [])
                    logger.info(f"检测成功，发现 {len(anomalies)} 个异常点")
                    
                return parsed_result
            except json.JSONDecodeError as e:
                logger.error(f"解析JSON失败: {e}")
                logger.error(f"原始文本: '{result.content[0].text}'")
                return {"error": f"解析JSON失败: {str(e)}"}
                
        except Exception as e:
            logger.error(f"执行检测失败: {e}")
            traceback.print_exc()
            return {"error": f"执行检测失败: {str(e)}"}
    
    async def close(self):
        """关闭客户端连接"""
        logger.info("关闭MCP客户端...")
        
        if not self._initialized:
            logger.info("客户端未初始化，无需关闭")
            return
        
        self._session = None
        
        if self._context_mgr:
            try:
                await self._context_mgr.__aexit__(None, None, None)
                logger.info("连接已关闭")
            except Exception as e:
                logger.error(f"关闭连接失败: {e}")
            finally:
                self._context_mgr = None
        
        self._initialized = False

# 获取客户端实例
async def get_mcp_client() -> MCPClient:
    """获取MCP客户端实例并确保连接"""
    client = await MCPClient.get_instance()
    await client.ensure_connected()
    return client