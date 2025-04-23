# test_mcp.py
import asyncio
import logging
from mcp_client import get_mcp_client

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_mcp")

async def test_mcp():
    try:
        logger.info("正在连接MCP客户端...")
        client = await get_mcp_client()
        logger.info("连接成功，正在获取检测器信息...")
        
        try:
            # 测试ping
            ping_response = await client._session.call_tool("ping", {})
            logger.info(f"Ping响应: {ping_response.content[0].text}")
        except Exception as e:
            logger.error(f"Ping测试失败: {e}")
        
        # 获取检测器信息
        detectors = await client.get_all_detectors()
        logger.info(f"获取到 {len(detectors)} 个检测器")
        logger.info(f"检测器列表: {list(detectors.keys())}")
        
        await client.close()
        return True
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # 先手动启动服务器
    print("请确保已经在另一个终端运行了adtk_server.py")
    input("按Enter继续...")
    
    asyncio.run(test_mcp())