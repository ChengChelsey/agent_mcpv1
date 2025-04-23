# utils/mcp_detector.py
"""
高层接口：给定序列 + 要调用的 MCP 工具列表
→ 返回 DetectionResult 列表，或直接出图
"""
import asyncio
import json
from typing import List, Tuple, Dict, Any
from mcp_client import get_mcp_client
from utils.adtk_adapter import json_to_detection_result
from output.visualization import generate_summary_echarts_html

Series = List[Tuple[int, float]]  # [(timestamp, value)]

async def detect_with_mcp_async(series: Series,
                          tools: List[str],
                          tool_params: Dict[str, Dict[str, Any]] | None = None):
    """异步版本的MCP检测"""
    results = []
    client = await get_mcp_client()
    
    for tool_name in tools:
        try:
            payload = {"series": series}
            if tool_params and tool_name in tool_params:
                payload.update(tool_params[tool_name])
            
            raw_result = await client.detect(tool_name, series, 
                                          tool_params.get(tool_name) if tool_params else None)
            
            # 将结果转换为DetectionResult
            detection_result = json_to_detection_result(raw_result)
            
            # 添加工具参数到结果中
            if tool_params and tool_name in tool_params:
                detection_result.parameters = tool_params[tool_name]
                
            results.append(detection_result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"调用工具 {tool_name} 时出错: {e}")
    
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
                         tool_params: Dict[str, Dict[str, Any]] | None = None,
                         series2: Series | None = None):
    """异步版本的检测并出图函数"""
    det_results = await detect_with_mcp_async(series, tools, tool_params)
    return generate_summary_echarts_html(
        series1=series,
        series2=series2,
        detection_results=det_results,
        output_path=html_out,
        title=title
    )

def detect_and_plot(series: Series,
                    tools: List[str],
                    html_out: str | None = None,
                    title: str = "时序异常检测汇总图",
                    tool_params: Dict[str, Dict[str, Any]] | None = None,
                    series2: Series | None = None):
    """同步包装器，用于异步检测并出图函数"""
    return asyncio.run(detect_and_plot_async(series, tools, html_out, title, tool_params, series2))