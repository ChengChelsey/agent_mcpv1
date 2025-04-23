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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main")


def interactive_mode():
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
            break
        except Exception as e:
            logger.error("处理查询出错: %s", e, exc_info=True)


def _sig_handler(sig, frame):
    print("\n收到退出信号，正在关闭…")
    sys.exit(0)


async def main():
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # 交互 CLI 放到线程，保持 asyncio 主循环空闲
    with ThreadPoolExecutor() as pool:
        pool.submit(interactive_mode).result()


if __name__ == "__main__":
    asyncio.run(main())
