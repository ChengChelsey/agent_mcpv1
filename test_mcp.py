#!/usr/bin/env python3
"""
直接测试 - 没有任何封装，与成功测试完全一致
"""
import asyncio
import json
import logging
import sys

# 导入MCP库
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("direct_test")

async def test():
    """直接测试连接，没有任何封装"""
    try:
        # 服务器参数
        server_params = StdioServerParameters(
            command="python",
            args=["adtk_server.py"]
        )
        
        logger.info("连接ADTK服务器...")
        
        # 建立连接
        async with stdio_client(server_params) as (read, write):
            logger.info("连接成功，创建会话...")
            
            # 创建并初始化会话
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("会话初始化成功")
                
                # 测试ping
                logger.info("测试ping...")
                ping_result = await session.call_tool("ping", {})
                logger.info(f"Ping结果: {ping_result.content[0].text}")
                
                # 获取检测器列表
                logger.info("获取检测器列表...")
                detectors_result = await session.call_tool("获取所有检测方法信息", {})
                detectors = json.loads(detectors_result.content[0].text)
                logger.info(f"可用检测器: {list(detectors.keys())}")
        
        logger.info("测试成功并正常关闭连接")
        return True
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test())
    sys.exit(0 if success else 1)