"""
MCP客户端模块

实现连接到ADTK MCP服务器并调用异常检测方法的客户端。
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp_client")

class MCPClient:
    """MCP客户端类，用于连接ADTK MCP服务器并调用异常检测方法"""
    
    def __init__(self, server_path="adtk_server.py", server_port=7777):
        """初始化MCP客户端
        
        Args:
            server_path: ADTK服务器脚本路径
            server_port: 服务器端口
        """
        self.server_path = server_path
        self.server_port = server_port
        self.session = None
        self.available_tools = []
        self.exit_stack = None
    
    async def connect(self):
        """连接到ADTK MCP服务器"""
        from contextlib import AsyncExitStack
        
        self.exit_stack = AsyncExitStack()
        
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_path, str(self.server_port)],
            env=None
        )
        
        try:
            # 启动MCP服务器并建立通信
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            # 初始化会话
            await self.session.initialize()
            
            # 获取可用工具列表
            response = await self.session.list_tools()
            self.available_tools = response.tools
            
            logger.info(f"已连接到ADTK MCP服务器，可用工具: {[tool.name for tool in self.available_tools]}")
            return True
        except Exception as e:
            logger.error(f"连接ADTK MCP服务器失败: {e}")
            return False
    
    async def disconnect(self):
        """断开与ADTK MCP服务器的连接"""
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.exit_stack = None
            self.session = None
            logger.info("已断开与ADTK MCP服务器的连接")
    
    async def get_all_detectors(self) -> Dict:
        """获取所有可用的检测方法信息"""
        if not self.session:
            logger.error("未连接到ADTK MCP服务器")
            return {"error": "未连接到ADTK MCP服务器"}
        
        try:
            result = await self.session.call_tool("获取所有检测方法信息", {})
            return json.loads(result.content[0].text)
        except Exception as e:
            logger.error(f"获取检测方法信息失败: {e}")
            return {"error": f"获取检测方法信息失败: {e}"}
    
    async def detect_anomalies(self, tool_name: str, data: Dict, params: Dict) -> Dict:
        """使用指定方法检测异常
        
        Args:
            tool_name: 检测方法名称，例如"IQR异常检测"
            data: 时间序列数据
            params: 检测参数
            
        Returns:
            检测结果
        """
        if not self.session:
            logger.error("未连接到ADTK MCP服务器")
            return {"error": "未连接到ADTK MCP服务器"}
        
        try:
            # 准备参数
            data_json = json.dumps(data, ensure_ascii=False)
            params_json = json.dumps(params, ensure_ascii=False)
            
            # 调用工具
            result = await self.session.call_tool(tool_name, {
                "data": data_json,
                "params": params_json
            })
            
            return json.loads(result.content[0].text)
        except Exception as e:
            logger.error(f"调用 {tool_name} 失败: {e}")
            return {"error": f"调用 {tool_name} 失败: {e}"}

# 单例模式实现
_mcp_client_instance = None

async def get_mcp_client() -> MCPClient:
    """获取MCP客户端单例"""
    global _mcp_client_instance
    
    if _mcp_client_instance is None:
        _mcp_client_instance = MCPClient()
        success = await _mcp_client_instance.connect()
        if not success:
            _mcp_client_instance = None
            raise Exception("连接ADTK MCP服务器失败")
    
    return _mcp_client_instance

# 使用例子
async def example_usage():
    """MCP客户端使用示例"""
    try:
        # 获取客户端实例
        client = await get_mcp_client()
        
        # 获取所有检测方法信息
        detector_info = await client.get_all_detectors()
        print("检测方法信息:", json.dumps(detector_info, ensure_ascii=False, indent=2)[:200] + "...")
        
        # 模拟时间序列数据
        import numpy as np
        timestamps = list(range(0, 100))
        values = list(np.sin(np.array(timestamps) * 0.1))
        # 添加异常点
        values[30] = 3.0
        values[60] = -3.0
        
        series_data = {"series": [[t, v] for t, v in zip(timestamps, values)]}
        
        # 使用IQR方法检测异常
        iqr_result = await client.detect_anomalies("IQR异常检测", series_data, {"c": 3.0})
        print("IQR检测结果:", json.dumps(iqr_result, ensure_ascii=False, indent=2))
        
        # 断开连接
        await client.disconnect()
        
    except Exception as e:
        print(f"示例执行失败: {e}")

if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())