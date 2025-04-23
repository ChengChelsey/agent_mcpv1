#!/usr/bin/env python3
"""
时序异常检测系统主入口
只开启交互 CLI，真正的 ADTK MCP Server 由 mcp_client 在首次调用时
通过 STDIO 自动拉起。
"""

import asyncio
import signal
import sys
import logging
from concurrent.futures import ThreadPoolExecutor

import agent  # 你的 ReAct 智能体
from mcp_client import get_mcp_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main")


def interactive_mode():
    """交互式命令行界面"""
    print("\n===== 时序异常检测系统 =====")
    print("输入 exit / quit 退出\n")
    
    while True:
        try:
            q = input("查询 > ").strip()
            if q.lower() in {"exit", "quit", "q"}:
                break
            if q:
                agent.chat(q)
        except KeyboardInterrupt:
            print("\n输入被中断")
            break
        except Exception as e:
            logger.error("处理查询出错: %s", e, exc_info=True)


async def cleanup_resources():
    """清理所有异步资源"""
    try:
        logger.info("清理MCP资源...")
        client = await get_mcp_client()
        await client.close()
        logger.info("MCP资源已清理")
    except Exception as e:
        logger.error(f"清理资源出错: {e}")


def _sig_handler(sig, frame):
    """信号处理函数"""
    print("\n收到退出信号，正在关闭…")
    
    # 确保清理资源
    try:
        asyncio.run(cleanup_resources())
    except Exception as e:
        logger.error(f"信号处理中清理资源失败: {e}")
        
    sys.exit(0)


async def main():
    """主函数"""
    # 注册信号处理
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        # 交互 CLI 放到线程，保持 asyncio 主循环空闲
        with ThreadPoolExecutor() as pool:
            try:
                pool.submit(interactive_mode).result()
            except KeyboardInterrupt:
                logger.info("主程序被中断")
    finally:
        # 确保在程序结束时清理资源
        await cleanup_resources()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被中断，正在清理资源...")
        try:
            asyncio.run(cleanup_resources())
        except Exception as e:
            logger.error(f"清理资源失败: {e}")
    except Exception as e:
        logger.error(f"程序出错: {e}", exc_info=True)
        # 即使出错也尝试清理资源
        try:
            asyncio.run(cleanup_resources())
        except Exception as cleanup_err:
            logger.error(f"清理资源失败: {cleanup_err}")