# utils/mcp_detector.py
"""
高层接口：给定序列 + 要调用的 MCP 工具列表
→ 返回 DetectionResult 列表，或直接出图
"""
from typing import List, Tuple, Dict, Any
from mcp_client import call_tool           # 你的客户端封装
from utils.adtk_adapter import json_to_detection_result
from output.visualization import generate_summary_echarts_html

Series = List[Tuple[int, float]]  # [(timestamp, value)]

def detect_with_mcp(series: Series,
                    tools: List[str],
                    tool_params: Dict[str, Dict[str, Any]] | None = None):
    results = []
    for tl in tools:
        payload = {"series": series}
        if tool_params and tl in tool_params:
            payload.update(tool_params[tl])
        raw = call_tool(tl, payload)
        results.append(json_to_detection_result(raw))
    return results

def detect_and_plot(series: Series,
                    tools: List[str],
                    html_out: str | None = None,
                    title: str = "时序异常检测汇总图",
                    tool_params: Dict[str, Dict[str, Any]] | None = None):
    det_results = detect_with_mcp(series, tools, tool_params)
    return generate_summary_echarts_html(
        series1=series,
        detection_results=det_results,
        output_path=html_out,
        title=title
    )
