#!/usr/bin/env python3
"""
测试ADTK-MCP集成v3
"""
import asyncio
import datetime
import random
import json
import logging
import sys
import traceback

# 导入MCP库
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_v3")

async def test_adtk():
    """测试ADTK服务器"""
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
                
                # 创建测试数据
                logger.info("创建测试数据...")
                now = int(datetime.datetime.now().timestamp())
                series = [[now + i, 0.2 + 0.01 * random.random()] for i in range(60)]
                series[30][1] = 1.0  # 添加异常点
                
                # 执行IQR检测
                logger.info("执行IQR异常检测...")
                iqr_result = await session.call_tool("IQR异常检测", {
                    "series": series,
                    "c": 2.5
                })
                
                # 解析并显示结果
                try:
                    result_text = iqr_result.content[0].text
                    logger.info(f"IQR检测原始结果: {result_text[:100]}...")
                    
                    result = json.loads(result_text)
                    
                    if "error" in result:
                        logger.error(f"IQR检测失败: {result['error']}")
                    else:
                        anomalies = result.get("anomalies", [])
                        logger.info(f"检测成功，发现 {len(anomalies)} 个异常点")
                        logger.info(f"结果摘要: {json.dumps(result, ensure_ascii=False)[:100]}...")
                except json.JSONDecodeError as e:
                    logger.error(f"解析JSON失败: {e}")
                    logger.error(f"原始文本: '{result_text}'")
                
                logger.info("测试完成")
                return True
    
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_adtk())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("测试被中断")
        sys.exit(130)