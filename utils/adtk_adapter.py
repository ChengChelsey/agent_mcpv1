"""
优化版 adtk_adapter.py
适配 ADTK 异常检测结果的处理
"""

import json
from typing import Dict, Any, List, Union, Tuple, Optional

# 设置默认检测结果结构
DEFAULT_DETECTION_RESULT = {
    "method": "未知方法",
    "visual_type": "point",
    "anomalies": [],
    "intervals": [],
    "anomaly_scores": [],
    "explanation": [],
    "anomaly_ratio": 0.0,
    "parameters": {}
}

def json_to_detection_result(obj: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """将JSON字符串或字典转换为统一的检测结果格式
    
    Args:
        obj: JSON字符串或字典形式的检测结果
        
    Returns:
        标准化的检测结果字典
    """
    try:
        # 如果是字符串，解析为字典
        data = json.loads(obj) if isinstance(obj, str) else obj
        
        # 处理错误情况
        if isinstance(data, dict) and "error" in data:
            result = DEFAULT_DETECTION_RESULT.copy()
            result["method"] = data.get("method", "未知方法")
            result["error"] = data["error"]
            return result
            
        # 创建标准化的结果
        result = DEFAULT_DETECTION_RESULT.copy()
        
        # 复制已有字段
        for key in DEFAULT_DETECTION_RESULT:
            if key in data:
                result[key] = data[key]
        
        # 确保方法名存在
        if "method" not in result or not result["method"]:
            result["method"] = data.get("method", "未知方法")
            
        # 处理区间数据
        if "intervals" in data:
            # 确保区间格式一致，转为元组列表
            result["intervals"] = [tuple(interval) for interval in data["intervals"]]
        
        # 添加异常比例
        if "anomaly_ratio" not in result:
            # 计算异常比例
            anomaly_count = len(result["anomalies"])
            interval_count = len(result["intervals"]) * 3  # 区间计为3个点
            total_anomalies = anomaly_count + interval_count
            
            # 从结果中提取序列长度，若不存在则设为0避免除零错误
            series_length = data.get("series_length", 0)
            if series_length > 0:
                result["anomaly_ratio"] = total_anomalies / series_length
            else:
                result["anomaly_ratio"] = 0.0
                
        return result
        
    except Exception as e:
        # 发生错误时返回基本信息
        return {
            "method": "解析失败",
            "visual_type": "none",
            "anomalies": [],
            "intervals": [],
            "explanation": [f"结果解析失败: {str(e)}"],
            "error": str(e),
            "anomaly_ratio": 0.0
        }

def format_detection_result_for_report(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将检测结果格式化为报告使用的格式
    
    Args:
        results: 检测结果列表
        
    Returns:
        格式化后的结果列表
    """
    formatted_results = []
    
    for result in results:
        # 跳过错误结果
        if "error" in result:
            continue
            
        # 基本信息
        formatted = {
            "method": result.get("method", "未知方法"),
            "anomaly_count": len(result.get("anomalies", [])),
            "interval_count": len(result.get("intervals", [])),
            "anomaly_ratio": result.get("anomaly_ratio", 0.0),
            "visual_type": result.get("visual_type", "point")
        }
        
        # 添加解释信息
        explanations = result.get("explanation", [])
        if explanations:
            # 最多取前3条解释
            formatted["explanations"] = explanations[:3]
            
        # 添加参数信息
        if "parameters" in result:
            formatted["parameters"] = result["parameters"]
            
        formatted_results.append(formatted)
        
    return formatted_results

def merge_detection_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """合并多个检测结果为综合结果
    
    Args:
        results: 检测结果列表
        
    Returns:
        综合结果
    """
    if not results:
        return DEFAULT_DETECTION_RESULT.copy()
        
    all_anomalies = set()
    all_intervals = []
    explanations = []
    
    for result in results:
        # 跳过错误结果
        if "error" in result:
            continue
            
        # 收集异常点
        all_anomalies.update(result.get("anomalies", []))
        
        # 收集异常区间
        all_intervals.extend(result.get("intervals", []))
        
        # 收集解释
        for explanation in result.get("explanation", []):
            if explanation and explanation not in explanations:
                explanations.append(f"{result.get('method', '未知方法')}: {explanation}")
    
    # 创建综合结果
    merged = {
        "method": "综合检测",
        "visual_type": "point",
        "anomalies": sorted(list(all_anomalies)),
        "intervals": all_intervals,
        "explanation": explanations,
        "anomaly_ratio": 0.0  # 稍后计算
    }
    
    # 计算综合异常比例
    if "series_length" in results[0]:
        series_length = results[0].get("series_length", 0)
        if series_length > 0:
            total_anomalies = len(merged["anomalies"]) + len(merged["intervals"]) * 3
            merged["anomaly_ratio"] = total_anomalies / series_length
    
    return merged