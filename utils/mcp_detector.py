"""
修复版 mcp_detector.py 
提供高层接口：给定序列 + 要调用的 MCP 工具列表
→ 返回 DetectionResult 列表，或直接出图
"""
import asyncio
import json
import os
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from mcp_client import get_mcp_client
from utils.adtk_adapter import json_to_detection_result
from output.visualization import generate_summary_echarts_html

# 配置日志
logger = logging.getLogger(__name__)

# 类型别名定义
Series = List[Tuple[int, float]]  # [(timestamp, value)]

# 导入检测方法描述（若存在）
try:
    from detector_descriptions import DETECTOR_DESCRIPTIONS
except ImportError:
    # 如果不存在，使用空字典
    DETECTOR_DESCRIPTIONS = {}

def get_detector_metadata():
    """获取检测器的元数据描述"""
    return DETECTOR_DESCRIPTIONS

async def get_all_detectors_async():
    """异步获取所有可用的检测方法"""
    try:
        client = await get_mcp_client()
        detector_info = await client.list_detectors()
        
        # 增强检测器信息
        for name, info in detector_info.items():
            # 如果有本地描述信息，则融合
            if name in DETECTOR_DESCRIPTIONS:
                local_desc = DETECTOR_DESCRIPTIONS[name]
                info.update({
                    "principle": local_desc.get("principle", ""),
                    "suitable_features": local_desc.get("suitable_features", []),
                    "limitations": local_desc.get("limitations", []),
                    "output_type": local_desc.get("output_type", "point"),
                    "parameters": local_desc.get("parameters", {})
                })
            
        return detector_info
    except Exception as e:
        logger.error(f"获取检测方法信息错误: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"获取检测方法信息失败: {str(e)}"}

def get_all_detectors():
    """同步版本，获取所有可用的检测方法"""
    # 这个版本简单明了，直接使用asyncio.run
    try:
        # 创建新的事件循环运行异步任务
        result = asyncio.new_event_loop().run_until_complete(get_all_detectors_async())
        return result
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            logger.error("在已有事件循环中调用get_all_detectors()，请直接使用异步版本")
            return {"error": "请在异步环境中使用get_all_detectors_async()"}
        else:
            logger.error(f"获取检测方法错误: {e}")
            return {"error": str(e)}
    except Exception as e:
        logger.error(f"获取检测方法错误: {e}")
        return {"error": str(e)}

async def detect_with_mcp_async(series: Series,
                          tools: List[str],
                          tool_params: Dict[str, Dict[str, Any]] | None = None):
    """异步版本的MCP检测
    
    Args:
        series: 时间序列数据
        tools: 要使用的检测方法名称列表
        tool_params: 各方法的参数，键为方法名，值为参数字典
    
    Returns:
        检测结果列表
    """
    results = []
    client = await get_mcp_client()
    
    for tool_name in tools:
        try:
            # 获取参数
            params = None
            if tool_params and tool_name in tool_params:
                params = tool_params[tool_name]
            
            # 执行检测
            raw_result = await client.detect(tool_name, series, params)
            
            # 检查错误
            if isinstance(raw_result, dict) and "error" in raw_result:
                logger.error(f"检测方法 {tool_name} 返回错误: {raw_result['error']}")
                # 添加错误结果
                raw_result["method"] = tool_name
                results.append(raw_result)
                continue
            
            # 将结果转换为DetectionResult
            detection_result = raw_result
            
            # 添加工具参数到结果中
            if params:
                if "parameters" not in detection_result:
                    detection_result["parameters"] = {}
                detection_result["parameters"].update(params)
                
            results.append(detection_result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"调用工具 {tool_name} 时出错: {e}")
            results.append({"error": str(e), "method": tool_name})
    
    return results

def detect_with_mcp(series: Series,
                    tools: List[str],
                    tool_params: Dict[str, Dict[str, Any]] | None = None):
    """同步包装器，用于异步检测函数
    
    Args:
        series: 时间序列数据
        tools: 要使用的检测方法名称列表
        tool_params: 各方法的参数，键为方法名，值为参数字典
    
    Returns:
        检测结果列表
    """
    try:
        # 创建新的事件循环运行异步任务
        result = asyncio.new_event_loop().run_until_complete(
            detect_with_mcp_async(series, tools, tool_params)
        )
        return result
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            logger.error("在已有事件循环中调用detect_with_mcp()，请直接使用异步版本")
            return [{"error": "请在异步环境中使用detect_with_mcp_async()", "method": "错误"}]
        else:
            logger.error(f"检测错误: {e}")
            return [{"error": str(e), "method": "错误"}]
    except Exception as e:
        logger.error(f"检测错误: {e}")
        return [{"error": str(e), "method": "错误"}]

async def detect_and_plot_async(series: Series,
                         tools: List[str],
                         html_out: str | None = None,
                         title: str = "时序异常检测汇总图",
                         tool_params: Dict[str, Dict[str, Any]] | None = None,
                         series2: Series | None = None):
    """异步版本的检测并出图函数
    
    Args:
        series: 时间序列数据
        tools: 要使用的检测方法名称列表
        html_out: 输出HTML文件路径，为None则不保存
        title: 图表标题
        tool_params: 各方法的参数，键为方法名，值为参数字典
        series2: 第二个时间序列数据，用于比较
    
    Returns:
        (图表路径, tooltip_map)的元组
    """
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
    try:
        # 创建新的事件循环运行异步任务
        result = asyncio.new_event_loop().run_until_complete(
            detect_and_plot_async(series, tools, html_out, title, tool_params, series2)
        )
        return result
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            logger.error("在已有事件循环中调用detect_and_plot()，请直接使用异步版本")
            return None, {}
        else:
            logger.error(f"检测并出图错误: {e}")
            return None, {}
    except Exception as e:
        logger.error(f"检测并出图错误: {e}")
        return None, {}

async def execute_single_detection_async(method: str, series: Series, params: Dict[str, Any] = None):
    """执行单个检测方法
    
    Args:
        method: 检测方法名称
        series: 时间序列数据
        params: 检测参数
        
    Returns:
        检测结果
    """
    try:
        client = await get_mcp_client()
        result = await client.detect(method, series, params or {})
        return result
    except Exception as e:
        logger.error(f"执行检测方法 {method} 失败: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "method": method}

def execute_single_detection(method: str, series: Series, params: Dict[str, Any] = None):
    """同步包装器，执行单个检测方法"""
    try:
        # 创建新的事件循环运行异步任务
        result = asyncio.new_event_loop().run_until_complete(
            execute_single_detection_async(method, series, params)
        )
        return result
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            logger.error(f"在已有事件循环中调用execute_single_detection()，请直接使用异步版本")
            return {"error": "请在异步环境中使用execute_single_detection_async()", "method": method}
        else:
            logger.error(f"执行检测错误: {e}")
            return {"error": str(e), "method": method}
    except Exception as e:
        logger.error(f"执行检测错误: {e}")
        return {"error": str(e), "method": method}