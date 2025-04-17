#!/usr/bin/env python3
"""
时序异常检测系统主入口

该脚本启动整个时序异常检测系统，包括ADTK MCP服务器和交互式命令行界面。
"""

import os
import sys
import time
import argparse
import asyncio
import subprocess
import signal
import logging
from concurrent.futures import ThreadPoolExecutor

import agent
from mcp_client import get_mcp_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

def start_adtk_server(port=7777):
    """启动ADTK MCP服务器进程"""
    logger.info(f"启动ADTK MCP服务器，端口: {port}")
    
    # 使用subprocess启动服务器进程
    server_process = subprocess.Popen(
        ["python", "adtk_server.py", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # 等待服务器启动
    time.sleep(2)
    
    # 检查服务器是否成功启动
    if server_process.poll() is not None:
        logger.error("ADTK MCP服务器启动失败")
        output, _ = server_process.communicate()
        logger.error(f"服务器输出: {output}")
        return None
    
    logger.info("ADTK MCP服务器启动成功")
    return server_process

async def check_mcp_connection():
    """检查MCP客户端连接"""
    try:
        client = await get_mcp_client()
        detector_info = await client.get_all_detectors()
        if "error" in detector_info:
            logger.error(f"MCP连接检查失败: {detector_info['error']}")
            return False
        logger.info("MCP连接检查成功")
        return True
    except Exception as e:
        logger.error(f"MCP连接检查失败: {e}")
        return False

def interactive_mode():
    """交互式命令行模式"""
    print("\n===== 时序异常检测系统 =====")
    print("输入 'exit' 或 'quit' 退出系统")
    
    while True:
        try:
            user_query = input("\n请输入查询: ")
            if user_query.lower() in ["exit", "quit", "退出", "q"]:
                break
            
            # 调用agent处理查询
            agent.chat(user_query)
            
        except KeyboardInterrupt:
            print("\n接收到终止信号，准备退出系统...")
            break
        except Exception as e:
            logger.error(f"处理查询出错: {e}")
            print(f"处理查询出错: {e}")

def signal_handler(sig, frame):
    """信号处理函数"""
    print("\n接收到终止信号，准备退出系统...")
    sys.exit(0)

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="时序异常检测系统")
    parser.add_argument("--port", type=int, default=7777, help="ADTK MCP服务器端口")
    args = parser.parse_args()
    
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动ADTK MCP服务器
    server_process = start_adtk_server(args.port)
    if not server_process:
        logger.error("系统启动失败")
        return
    
    # 检查MCP连接
    connection_ok = await check_mcp_connection()
    if not connection_ok:
        logger.error("MCP连接失败，系统无法正常工作")
        server_process.terminate()
        return
    
    try:
        # 启动交互式模式
        with ThreadPoolExecutor() as executor:
            # 在线程中运行交互式模式
            future = executor.submit(interactive_mode)
            future.result()  # 等待交互式模式结束
    finally:
        # 停止ADTK MCP服务器
        logger.info("停止ADTK MCP服务器")
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

if __name__ == "__main__":
    asyncio.run(main())