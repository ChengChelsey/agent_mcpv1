# utils/mcp_detector.py
"""
高层接口：给定序列 + 要调用的 MCP 工具列表
→ 返回 DetectionResult 列表，或直接出图
"""
import asyncio
import json
from typing import List, Tuple, Dict, Any
from mcp_client import get_mcp_client  # 修改导入方式
from utils.adtk_adapter import json_to_detection_result
from output.visualization import generate_summary_echarts_html

Series = List[Tuple[int, float]]  # [(timestamp, value)]

async def call_tool_async(tool_name: str, payload: dict | None = None) -> dict:
    """异步调用工具函数"""
    payload = payload or {}
    client = await get_mcp_client()
    res = await client._session.call_tool(tool_name, payload)
    return json.loads(res.content[0].text)

def call_tool(tool_name: str, payload: dict | None = None) -> dict:
    """同步包装器，用于调用异步函数"""
    return asyncio.run(call_tool_async(tool_name, payload))

async def detect_with_mcp_async(series: Series,
                          tools: List[str],
                          tool_params: Dict[str, Dict[str, Any]] | None = None):
    """异步版本的MCP检测"""
    results = []
    for tl in tools:
        payload = {"series": series}
        if tool_params and tl in tool_params:
            payload.update(tool_params[tl])
        client = await get_mcp_client()
        raw_response = await client._session.call_tool(tl, payload)
        import json
        raw = json.loads(raw_response.content[0].text)
        results.append(json_to_detection_result(raw))
    return results

def detect_with_mcp(series: Series,
                    tools: List[str],
                    tool_params: Dict[str, Dict[str, Any]] | None = None):
    """同步包装器，用于异步检测函数"""
    return asyncio.run(detect_with_mcp_async(series, tools, tool_params))

async def detect_and_plot_async(series: Series,
                         tools: List[str],
                         html_out: str | None = None,
                         title: str = "时序异常检测汇总图",
                         tool_params: Dict[str, Dict[str, Any]] | None = None):
    """异步版本的检测并出图函数"""
    det_results = await detect_with_mcp_async(series, tools, tool_params)
    return generate_summary_echarts_html(
        series1=series,
        detection_results=det_results,
        output_path=html_out,
        title=title
    )

def detect_and_plot(series: Series,
                    tools: List[str],
                    html_out: str | None = None,
                    title: str = "时序异常检测汇总图",
                    tool_params: Dict[str, Dict[str, Any]] | None = None):
    """同步包装器，用于异步检测并出图函数"""
    return asyncio.run(detect_and_plot_async(series, tools, html_out, title, tool_params))