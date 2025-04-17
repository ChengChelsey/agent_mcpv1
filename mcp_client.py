"""
MCP客户端模块

实现连接到ADTK MCP服务器并调用异常检测方法的客户端。
适配MCP 1.6.0版本。
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
import traceback

# 导入MCP 1.6.0版本的客户端组件
from mcp import MCPClient
from mcp.client import MCPClientParameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp_client")

class ADTKMCPClient:
    """ADTK MCP客户端类，用于连接ADTK MCP服务器并调用异常检测方法"""
    
    def __init__(self, host="localhost", port=7777):
        """初始化ADTK MCP客户端
        
        Args:
            host: 服务器主机名
            port: 服务器端口
        """
        self.host = host
        self.port = port
        self.client = None
        self.available_tools = []
        logger.info(f"初始化ADTK MCP客户端，服务器: {host}:{port}")
    
    async def connect(self):
        """连接到ADTK MCP服务器"""
        try:
            # 设置客户端参数
            client_params = MCPClientParameters(host=self.host, port=self.port)
            
            # 创建客户端
            self.client = MCPClient(client_params)
            
            # 初始化连接
            await self.client.initialize()
            
            # 获取可用工具列表
            tools_response = await self.client.list_tools()
            self.available_tools = tools_response.tools
            
            logger.info(f"已连接到ADTK MCP服务器，可用工具: {[tool.name for tool in self.available_tools]}")
            return True
        except Exception as e:
            logger.error(f"连接ADTK MCP服务器失败: {e}")
            traceback.print_exc()
            return False
    
    async def disconnect(self):
        """断开与ADTK MCP服务器的连接"""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("已断开与ADTK MCP服务器的连接")
    
    async def get_all_detectors(self) -> Dict:
        """获取所有可用的检测方法信息"""
        if not self.client:
            logger.error("未连接到ADTK MCP服务器")
            return {"error": "未连接到ADTK MCP服务器"}
        
        try:
            result = await self.client.call_tool("获取所有检测方法信息", {})
            return json.loads(result.content[0].text)
        except Exception as e:
            logger.error(f"获取检测方法信息失败: {e}")
            traceback.print_exc()
            return {"error": f"获取检测方法信息失败: {e}"}
    
    async def detect_anomalies(self, method_name: str, data: Dict, params: Dict) -> Dict:
        """使用指定方法检测异常
        
        Args:
            method_name: 检测方法名称，例如"IQR异常检测"
            data: 时间序列数据
            params: 检测参数
            
        Returns:
            检测结果
        """
        if not self.client:
            logger.error("未连接到ADTK MCP服务器")
            return {"error": "未连接到ADTK MCP服务器"}
        
        try:
            # 准备参数
            data_json = json.dumps(data, ensure_ascii=False)
            params_json = json.dumps(params, ensure_ascii=False)
            
            # 调用工具
            result = await self.client.call_tool(method_name, {
                "data": data_json,
                "params": params_json
            })
            
            return json.loads(result.content[0].text)
        except Exception as e:
            logger.error(f"调用 {method_name} 失败: {e}")
            traceback.print_exc()
            return {"error": f"调用 {method_name} 失败: {e}"}

# 单例模式实现
_adtk_client_instance = None

async def get_mcp_client(host="localhost", port=7777) -> ADTKMCPClient:
    """获取MCP客户端单例"""
    global _adtk_client_instance
    
    if _adtk_client_instance is None:
        _adtk_client_instance = ADTKMCPClient(host, port)
        success = await _adtk_client_instance.connect()
        if not success:
            _adtk_client_instance = None
            raise Exception("连接ADTK MCP服务器失败")
    
    return _adtk_client_instance

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
        print(f"示例执行错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())