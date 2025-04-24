#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版 ReAct 智能体
支持 MCP 调用 ADTK 检测方法的完整流程
"""
from __future__ import annotations

import asyncio  # NEW
import datetime
import hashlib
import json
import os
import re
import time
import traceback
from typing import Any, Dict, List, Tuple, Optional
import requests

import config
import numpy as np
from django.conf import settings  # noqa: F401 (kept, may be used elsewhere)

from utils.ts_cache import load_series_from_cache
from utils.time_utils import parse_time_expressions, format_timestamp, group_anomaly_times
from utils.ts_features import extract_time_series_features
from utils.mcp_detector import detect_with_mcp
from mcp_client import get_mcp_client
from output.report_generator import generate_llm_report, generate_report_html
from output.visualization import generate_summary_echarts_html
import subprocess

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AIOPS_BACKEND_DOMAIN = 'https://aiopsbackend.cstcloud.cn'
LLM_URL = 'http://10.16.1.16:58000/v1/chat/completions'
AUTH = ('chelseyyycheng@outlook.com', 'UofV1uwHwhVp9tcTue')

CACHE_DIR = "cached_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_data_from_backend(ip:str, start_ts:int, end_ts:int, field:str):
    """从后端API获取时序数据"""
    url = f"{AIOPS_BACKEND_DOMAIN}/api/v1/monitor/mail/metric/format-value/?start={start_ts}&end={end_ts}&instance={ip}&field={field}"
    resp = requests.get(url, auth=AUTH)
    if resp.status_code!=200:
        return f"后端请求失败: {resp.status_code} => {resp.text}"
    j = resp.json()
    results = j.get("results", [])
    if not results:
        return []
    vals = results[0].get("values", [])
    arr = []
    from datetime import datetime
    def parse_ts(s):
        try:
            dt = datetime.strptime(s,"%Y-%m-%d %H:%M:%S")
            return int(dt.timestamp())
        except:
            return 0
    for row in vals:
        if len(row)>=2:
            tstr,vstr = row[0], row[1]
            t = parse_ts(tstr)
            try:
                v = float(vstr)
            except:
                v = 0.0
            arr.append([t,v])
    return arr

# 优化后的工具列表，改进了描述和逻辑
tools = [
    {
        "name": "解析用户自然语言时间",
        "description": "返回一个list，每个元素是{start, end, start_str, end_str, error}. 如果不确定，可向用户澄清。",
        "parameters": {
            "type": "object",
            "properties": {
                "raw_text": {"type": "string"}
            },
            "required": ["raw_text"]
        }
    },
    {
        "name": "请求智能运管后端Api，获取指标项的时序数据",
        "description": "从后端或本地缓存获取IP在指定时间范围(field)的时序数据(list of [int_ts, val])。注意start/end必须是形如'YYYY-MM-DD HH:MM:SS'的确定时间。",
        "parameters": {
            "type": "object",
            "properties": {
                "ip": {
                    "type": "string",
                    "description": "要查询的 IP，如 '192.168.0.110'"
                },
                "start": {
                    "type": "string",
                    "description": "开始时间，格式 '2025-03-24 00:00:00'"
                },
                "end": {
                    "type": "string",
                    "description": "结束时间，格式 '2025-03-24 23:59:59'"
                },
                "field": {
                    "type": "string",
                    "description": "监控项名称，如 'cpu_rate'"
                }
            },
            "required": ["ip", "start", "end", "field"]
        }
    },
    {
        "name": "请求智能运管后端Api，查询监控实例有哪些监控项",
        "description": "返回指定IP下可用的监控项列表（可选项）",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "系统服务名称 (一般填 '主机监控')"
                },
                "instance": {
                    "type": "string",
                    "description": "监控实例 IP"
                }
            },
            "required": ["service", "instance"]
        }
    },
    {
        "name": "请求智能运管后端Api，查询监控服务的资产情况和监控实例",
        "description": "查询一个监控服务的所有资产/IP等信息",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "要查询的系统服务名称"
                }
            },
            "required": ["service"]
        }
    },
    {
        "name": "请求智能运管后端Api，查询监控实例之间的拓扑关联关系",
        "description": "查询指定IP的上联、下联监控实例等信息",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "系统服务名称"
                },
                "instance_ip": {
                    "type": "string",
                    "description": "监控实例IP"
                }
            },
            "required": ["service", "instance_ip"]
        }
    },
    {
        "name": "分析时序数据特征",
        "description": "分析时序数据的统计特性、趋势、季节性、波动性等特征，返回特征字典用于方法选择",
        "parameters": {
            "type": "object",
            "properties": {
                "series": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": [
                            {"type": "number"},
                            {"type": "number"}
                        ]
                    },
                    "description": "时序数据，格式为 [[timestamp, value], ...]"
                }
            },
            "required": ["series"]
        }
    },
    {
        "name": "获取异常检测方法信息",
        "description": "从MCP服务器获取所有可用的异常检测方法信息",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "执行异常检测",
        "description": "使用指定的方法对时序数据进行异常检测",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "检测方法名称，如'IQR异常检测'"
                },
                "series": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": [
                            {"type": "number"},
                            {"type": "number"}
                        ]
                    },
                    "description": "时序数据，格式为 [[timestamp, value], ...]"
                },
                "params": {
                    "type": "object",
                    "description": "检测参数，不提供则使用默认值"
                }
            },
            "required": ["method", "series"]
        }
    },
    {
        "name": "计算综合异常评分",
        "description": "根据多个检测方法的结果计算综合异常评分",
        "parameters": {
            "type": "object",
            "properties": {
                "detection_results": {
                    "type": "array",
                    "items": {
                        "type": "object"
                    },
                    "description": "多个检测方法的结果列表"
                }
            },
            "required": ["detection_results"]
        }
    },
    {
        "name": "生成异常检测报告",
        "description": "根据异常检测结果生成综合报告和可视化图表",
        "parameters": {
            "type": "object",
            "properties": {
                "series": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": [
                            {"type": "number"},
                            {"type": "number"}
                        ]
                    },
                    "description": "时序数据，格式为 [[timestamp, value], ...]"
                },
                "detection_results": {
                    "type": "array",
                    "items": {
                        "type": "object"
                    },
                    "description": "多个检测方法的结果列表"
                },
                "composite_score": {
                    "type": "number",
                    "description": "综合异常评分"
                },
                "classification": {
                    "type": "string",
                    "description": "异常分类，如'正常'、'轻度异常'、'高置信度异常'"
                },
                "metadata": {
                    "type": "object",
                    "description": "元数据，如IP、指标名称、时间范围等"
                },
                "is_multi_series": {
                    "type": "boolean",
                    "description": "是否是多序列分析"
                },
                "series2": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": [
                            {"type": "number"},
                            {"type": "number"}
                        ]
                    },
                    "description": "第二个时序数据，多序列分析时使用"
                }
            },
            "required": ["series", "detection_results", "composite_score", "classification", "metadata"]
        }
    }
]

###############################################################################
# 新增工具函数
###############################################################################

def build_method_catalog(detector_info):
    """
    将检测器信息格式化为大模型易于理解的格式
    """
    catalog = []
    for name, info in detector_info.items():
        method_info = {
            "name": name,
            "description": info.get("description", ""),
            "principle": info.get("principle", ""),
            "suitable_features": info.get("suitable_features", []),
            "limitations": info.get("limitations", []),
            "default_params": info.get("default_params", {})
        }
        catalog.append(method_info)
    
    return json.dumps(catalog, ensure_ascii=False, indent=2)[:4000]  # 避免超出token长度限制

def select_default_methods(detector_info, max_methods=5):
    """当LLM选择失败时提供默认方法"""
    default_methods = []
    
    # 优先选择的方法
    priority_methods = [
        "IQR异常检测",
        "分位数异常检测",
        "水平位移检测",
        "季节性异常检测",
        "自回归异常检测"
    ]
    
    # 从优先方法中选择可用的
    available_methods = list(detector_info.keys())
    for method in priority_methods:
        if method in available_methods and len(default_methods) < max_methods:
            default_methods.append({
                "method": method,
                "params": {},
                "weight": 1.0 / min(max_methods, len(priority_methods)),
                "reason": "默认推荐方法"
            })
    
    # 如果优先方法不足，从其他可用方法中补充
    if len(default_methods) < max_methods:
        remaining = [m for m in available_methods if m not in priority_methods]
        for method in remaining[:max_methods-len(default_methods)]:
            default_methods.append({
                "method": method,
                "params": {},
                "weight": 1.0 / min(max_methods, len(priority_methods) + len(remaining)),
                "reason": "补充推荐方法"
            })
    
    return default_methods

def llm_select_methods(features, detector_info, max_methods=5):
    """
    让大模型根据时序特征选择合适的检测方法
    """
    system_prompt = (
        "你是时序异常检测专家。"
        "我将提供：1) 序列特征 JSON；2) 可用检测方法列表(JSON)。\n"
        "请你基于特征选择 **最多 "
        f"{max_methods} 个** 最合适的方法，"
        "并返回 JSON: {\"selected\":[{\"method\":\"...\",\"params\":{...},\"weight\":数值,\"reason\":\"选择理由\"}]}。\n"
        "理由需要解释为什么该方法适合这个时序数据的特征。\n"
        "权重是0-1之间的值，表示该方法对最终结果的影响程度，所有权重之和应为1。\n"
        "如果默认参数即可，请留空 params。\n"
        "注意：你只能从提供的可用方法列表中选择方法，不要创造新的方法名。"
    )

    # 提取方法名列表，明确告诉模型可用的方法
    available_methods = []
    for name, info in detector_info.items():
        method_info = {
            "name": name,
            "description": info.get("description", ""),
            "suitable_features": info.get("suitable_features", []),
            "default_params": info.get("default_params", {})
        }
        available_methods.append(method_info)
    
    # 告诉模型具体可用的方法名
    method_names = [m["name"] for m in available_methods]
    
    user_msg = (
        "序列特征:\n" + json.dumps(features, ensure_ascii=False, indent=2) +
        "\n\n可用方法列表:\n" + json.dumps(available_methods, ensure_ascii=False, indent=2) +
        "\n\n仅可用的方法名称:\n" + json.dumps(method_names, ensure_ascii=False)
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]

    response = llm_call(messages)
    if not response:
        logger.warning("LLM未返回结果，使用默认方法")
        return select_default_methods(detector_info, max_methods)

    try:
        content = response.get("content", "")
        # 尝试找到JSON部分
        json_start = content.find("{")
        json_end = content.rfind("}")
        
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end+1]
            parsed = json.loads(json_str)
            selected = parsed.get("selected", [])
            
            # 验证选择的方法是否在可用列表中
            valid_selected = []
            for method_info in selected:
                method_name = method_info.get("method")
                if method_name in method_names:
                    valid_selected.append(method_info)
                else:
                    logger.warning(f"LLM选择了不可用的方法: {method_name}")
            
            if valid_selected:
                # 确保权重合计为1
                total_weight = sum(m.get("weight", 0) for m in valid_selected)
                if total_weight > 0:
                    for m in valid_selected:
                        m["weight"] = round(m.get("weight", 0) / total_weight, 2)
                return valid_selected
                
        logger.warning("LLM返回格式无效或没有选择有效方法，使用默认方法")
        return select_default_methods(detector_info, max_methods)
    except Exception as e:
        logger.error(f"LLM返回的方法选择结果解析失败: {e}")
        return select_default_methods(detector_info, max_methods)
    
async def get_detection_methods():
    """从MCP服务器获取异常检测方法信息并添加适用场景描述"""
    try:
        client = await get_mcp_client()
        detector_info = await client.list_detectors()
        
        # 增强检测器信息
        for name, info in detector_info.items():
            # 如果存在描述信息，将其添加到每个方法
            from detector_descriptions import DETECTOR_DESCRIPTIONS
            if name in DETECTOR_DESCRIPTIONS:
                local_desc = DETECTOR_DESCRIPTIONS[name]
                info.update({
                    "principle": local_desc.get("principle", ""),
                    "suitable_features": local_desc.get("suitable_features", []),
                    "limitations": local_desc.get("limitations", []),
                    "parameters": local_desc.get("parameters", {})
                })
        
        return detector_info
    except Exception as e:
        logger.error(f"获取检测方法信息错误: {e}")
        traceback.print_exc()
        return {"error": f"获取检测方法信息失败: {str(e)}"}

def get_detector_suitable_features(name):
    """为检测器生成适用特征描述"""
    suitable_features = {
        "IQR异常检测": [
            "适用于大多数数据分布，尤其是近似正态分布数据",
            "对于有明显离群值的数据效果好",
            "不受极端值影响",
            "适合处理数据范围变化的情况"
        ],
        "广义ESD检测": [
            "适用于近似正态分布的数据",
            "对有少量离群值的数据效果好",
            "适合需要统计显著性的场景",
            "有明确的异常点数量上限"
        ],
        "分位数异常检测": [
            "适用于任意分布数据",
            "对非正态分布数据也有效",
            "处理噪声数据时稳健",
            "能适应不同数据范围"
        ],
        "阈值异常检测": [
            "适用于有明确阈值边界的数据",
            "适合监控明确上下限的指标",
            "简单直观，计算开销小",
            "对定义明确的异常有高检出率"
        ],
        "持续性异常检测": [
            "适用于应该保持相对稳定的数据",
            "能检测出持续偏离正常水平的异常",
            "适合检测系统状态变化",
            "对渐变异常敏感"
        ],
        "水平位移检测": [
            "适用于存在明显水平跳变的数据",
            "能检测出均值突变",
            "适合系统配置变更、环境变化等场景",
            "对阶跃变化敏感"
        ],
        "波动性变化检测": [
            "适用于应保持稳定波动幅度的数据",
            "能检测出波动性突变",
            "适合检测系统不稳定性增加",
            "对方差变化敏感"
        ],
        "季节性异常检测": [
            "适用于有明显周期性模式的数据",
            "能检测出违背季节性模式的异常",
            "适合日常、周度或月度循环的数据",
            "需要数据至少包含2-3个完整周期"
        ],
        "自回归异常检测": [
            "适用于有时间相关性的数据",
            "能检测出违背历史模式的异常",
            "适合有短期依赖关系的数据",
            "对短期预测偏差敏感"
        ]
    }
    
    return suitable_features.get(name, ["适用于一般时序数据"])

def get_detector_limitations(name):
    """为检测器生成局限性描述"""
    limitations = {
        "IQR异常检测": [
            "不考虑时间序列的时间依赖性",
            "对于多峰分布效果较差",
            "无法检测趋势性异常",
            "对季节性变化敏感度低"
        ],
        "广义ESD检测": [
            "要求数据近似正态分布",
            "不适合有强烈季节性的数据",
            "计算开销较大",
            "需要指定最大异常数量"
        ],
        "分位数异常检测": [
            "静态阈值可能不适合动态变化的数据",
            "无法检测趋势变化",
            "对极端值过于敏感",
            "不考虑时间序列的时序特性"
        ],
        "阈值异常检测": [
            "需要预先知道合理阈值",
            "不适应数据分布变化",
            "过于简单可能导致误报",
            "无法检测复杂模式异常"
        ],
        "持续性异常检测": [
            "对短期波动敏感度低",
            "可能错过短暂的异常尖峰",
            "窗口大小选择影响检测效果",
            "不适用于有明显趋势的数据"
        ],
        "水平位移检测": [
            "无法检测渐变异常",
            "对噪声敏感",
            "可能将正常的季节性变化误判为异常",
            "窗口大小选择影响检测效果"
        ],
        "波动性变化检测": [
            "需要足够的历史数据建立基线",
            "对初始波动性假设敏感",
            "可能忽略均值变化",
            "窗口大小选择影响检测效果"
        ],
        "季节性异常检测": [
            "需要准确指定季节性周期",
            "数据不足时效果差",
            "对非季节性异常检测能力弱",
            "计算开销大"
        ],
        "自回归异常检测": [
            "需要足够的训练数据",
            "对参数选择敏感",
            "计算开销大",
            "不适合有长期依赖性的数据"
        ]
    }
    
    return limitations.get(name, ["需要适当调整参数以适应具体数据"])
async def execute_detection(method, series_data, params=None):
    """执行异常检测"""
    try:
        # 指定参数为空字典而非None
        params = params or {}
        
        # 简化日志
        logger.info(f"执行检测方法: {method}, 参数: {json.dumps(params)[:100]}")
        
        client = await get_mcp_client()
        result = await client.detect(method, series_data, params)
        
        # 检查结果
        if isinstance(result, dict) and "error" in result:
            logger.error(f"检测错误: {result['error']}")
            return {"error": result["error"], "method": method}
        
        return result
    except Exception as e:
        logger.error(f"执行异常检测错误: {e}")
        traceback.print_exc()
        return {"error": f"执行异常检测失败: {str(e)}", "method": method}
    
async def cleanup_mcp_resources():
    """清理MCP资源"""
    try:
        client = await get_mcp_client()
        await client.close()
        logger.info("MCP资源已清理")
    except Exception as e:
        logger.error(f"清理MCP资源出错: {e}")

def calculate_composite_score(detection_results):
    """计算综合异常评分"""
    try:
        if not detection_results or len(detection_results) == 0:
            return {"score": 0.0, "classification": "正常", "anomalies": [], "anomaly_count": 0}
        
        total_weight = 0.0
        weighted_score = 0.0
        all_anomalies = set()
        
        for result in detection_results:
            # 跳过有错误的结果
            if "error" in result:
                continue
            
            # 获取权重
            weight = result.get("parameters", {}).get("weight", 0.5)
            total_weight += weight
            
            # 获取异常比例作为分数
            anomaly_ratio = result.get("anomaly_ratio", 0)
            # 转换为0-1的评分，使用对数变换防止大量异常点导致评分过高
            method_score = min(0.8, 0.2 + 0.3 * np.log10(1 + anomaly_ratio * 100)) if anomaly_ratio > 0 else 0
            
            # 累加权重评分
            weighted_score += weight * method_score
            
            # 收集所有异常点
            all_anomalies.update(result.get("anomalies", []))
        
        # 计算最终得分
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # 确定分类
        if final_score >= config.HIGH_ANOMALY_THRESHOLD:
            classification = "高置信度异常"
        elif final_score >= config.MILD_ANOMALY_THRESHOLD:
            classification = "轻度异常"
        else:
            classification = "正常"
        
        return {
            "score": final_score,
            "classification": classification,
            "anomaly_count": len(all_anomalies),
            "anomalies": sorted(list(all_anomalies))
        }
    except Exception as e:
        logger.error(f"计算综合评分错误: {e}")
        traceback.print_exc()
        return {"score": 0.0, "classification": "正常", "anomalies": [], "anomaly_count": 0, "error": str(e)}

async def generate_detection_report(series_data, detection_results, composite_score, classification, metadata, is_multi_series=False, series2=None):
    """生成异常检测报告"""
    try:
        # 生成报告目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ip = metadata.get("ip", "unknown")
        field = metadata.get("field", "unknown")
        
        base_dir = f"output/plots/{ip}_{field}_{timestamp}"
        os.makedirs(base_dir, exist_ok=True)
        
        # 生成图表
        chart_title = f"{ip} {field} 异常检测汇总图"
        if is_multi_series and metadata.get("ip2"):
            chart_title = f"{ip} vs {metadata.get('ip2')} {field} 对比异常检测汇总图"
            
        summary_path = os.path.join(base_dir, "summary.html")
        chart_path, tooltip_map = generate_summary_echarts_html(
            series_data, 
            series2, 
            detection_results, 
            summary_path, 
            title=chart_title
        )
        
        # 生成LLM分析
        user_query = metadata.get("user_query", "")
        
        # 处理composite_score为字典或浮点数的情况
        anomalies = []
        if isinstance(composite_score, dict):
            anomalies = composite_score.get("anomalies", [])
            score_value = composite_score.get("score", 0.0)
        else:
            # 如果是浮点数，创建一个新的字典
            score_value = composite_score
            
        result_dict = {
            "classification": classification,
            "composite_score": score_value,
            "anomaly_times": anomalies,
            "method_results": detection_results,
            "user_query": user_query
        }
        
        # 多序列时添加异常区间
        if is_multi_series:
            anomaly_intervals = group_anomaly_times(anomalies)
            result_dict["anomaly_intervals"] = anomaly_intervals
            
        llm_analysis = generate_llm_report(result_dict, "multi" if is_multi_series else "single")
        
        # 生成最终报告
        final_report_path = os.path.join(base_dir, "final_report.html")
        generate_report_html(
            user_query, chart_path, detection_results, tooltip_map, final_report_path, 
            composite_score=score_value,
            classification=classification,
            llm_analysis=llm_analysis,
            is_multi_series=is_multi_series
        )
        
        return {
            "report_path": final_report_path,
            "chart_path": chart_path,
            "llm_analysis": llm_analysis
        }
    except Exception as e:
        logger.error(f"生成报告错误: {e}")
        traceback.print_exc()
        return {"error": f"生成报告失败: {str(e)}"}

def run_mcp_operation(operation: str, params: dict | None = None):
    """通过 mcp_process.py 子进程执行 MCP 操作，完全隔离事件循环"""
    params = params or {}
    input_json = json.dumps({"operation": operation, "params": params})

    proc = subprocess.Popen(
        ["python", "mcp_process.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc.communicate(input_json)

    if proc.returncode != 0:
        logger.error(f"MCP 子进程异常退出：{stderr.strip()}")
        return {"error": stderr.strip() or "子进程执行失败"}

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        logger.error(f"无法解析子进程输出：{stdout[:200]}...")
        return {"error": f"无法解析子进程输出：{stdout[:200]}..."}

    if result.get("status") != "success":
        return {"error": result.get("message", "未知错误")}

    # 确保返回的结果是有效的JSON
    inner = result["result"]
    try:
        if isinstance(inner, str) and inner.strip():
            return json.loads(inner)
        return inner
    except json.JSONDecodeError:
        logger.error(f"无法解析内部结果JSON：{inner[:200]}...")
        return {"error": f"无法解析内部结果：{inner[:200]}..."}

def get_all_detectors() -> dict:
    """列出所有检测方法（同步、隔离）"""
    return run_mcp_operation("list_detectors")

def execute_detection(method: str, series: list, params: dict | None = None) -> dict:
    """执行单个检测（同步、隔离）"""
    return run_mcp_operation("detect", {
        "method": method,
        "series": series,
        "params": params or {}
    })
###############################################################################
# 工具函数
###############################################################################

def monitor_item_list(ip):
    """查询监控实例有哪些监控项"""
    url = f'{AIOPS_BACKEND_DOMAIN}/api/v1/monitor/mail/machine/field/?instance={ip}'
    resp = requests.get(url=url, auth=AUTH)
    if resp.status_code == 200:
        items = json.loads(resp.text)
        result = {}
        for x in items:
            result[x.get('field')] = x.get('purpose')
        return result
    else:
        return f"查询监控项失败: {resp.status_code} => {resp.text}"

def get_service_asset(service):
    """查询监控服务的资产情况和监控实例"""
    url = f'{AIOPS_BACKEND_DOMAIN}/api/v1/property/mail/?ordering=num_id&page=1&page_size=2000'
    resp = requests.get(url=url, auth=AUTH)
    if resp.status_code == 200:
        text = json.loads(resp.text)
        results = text.get('results',[])
        item_list = []
        for r in results:
            r["category"] = r.get("category",{}).get("name")
            r["ip_set"] = [_.get("ip") for _ in r.get('ip_set',[])]
            for k in ["num_id","creation","modification","remark","sort_weight","monitor_status"]:
                r.pop(k, None)
            for k,v in list(r.items()):
                if not v or v == "无":
                    r.pop(k)
            item_list.append(r)
        return item_list
    else:
        return f"查询失败: {resp.status_code} => {resp.text}"

def get_service_asset_edges(service, instance_ip):
    """查询监控实例之间的拓扑关联关系"""
    url = f'{AIOPS_BACKEND_DOMAIN}/api/v1/property/mail/topology/search?instance={instance_ip}'
    resp = requests.get(url=url, auth=AUTH)
    if resp.status_code == 200:
        return json.loads(resp.text)
    else:
        return f"查询拓扑失败: {resp.status_code} => {resp.text}"
    
def get_series_data(ip: str, start: str, end: str, field: str):
    """从缓存或后端获取时序数据"""
    try:
        return load_series_from_cache(ip, field, start, end)
    except Exception as e:
        return str(e) 

###############################################################################
# LLM 调用
###############################################################################

def llm_call(messages):
    """调用LLM"""
    data={
      "model":"Qwen2.5-14B-Instruct",
      "temperature":0.1,
      "messages":messages
    }
    r= requests.post(LLM_URL, json=data)
    if r.status_code==200:
        jj= r.json()
        if "choices" in jj and len(jj["choices"])>0:
            return jj["choices"][0]["message"]
        else:
            return None
    else:
        logger.error(f"LLM调用错误: {r.status_code}, {r.text}")
        return None

def parse_llm_response(txt):
    """解析LLM回复中的XML格式内容"""
    pat_thought = r"<思考过程>(.*?)</思考过程>"
    pat_action  = r"<工具调用>(.*?)</工具调用>"
    pat_inparam = r"<调用参数>(.*?)</调用参数>"
    pat_final   = r"<最终答案>(.*?)</最终答案>"
    pat_supplement = r"<补充请求>(.*?)</补充请求>"
    def ext(pattern):
        m = re.search(pattern, txt, flags=re.S)
        return m.group(1) if m else ""

    return {
        "thought": ext(pat_thought),
        "action":  ext(pat_action),
        "action_input": ext(pat_inparam),
        "final_answer": ext(pat_final),
        "supplement": ext(pat_supplement)
    }

def get_detection_methods_sync() -> Dict[str, Any]:
    """同步获取检测器信息（内部使用 asyncio.run）"""
    async def _inner():
        try:
            client = await get_mcp_client()
            detector_info = await client.list_detectors()
            return detector_info
        except Exception as e:
            logger.error(f"获取检测方法信息错误: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    return asyncio.run(_inner())

###############################################################################
# 主要ReAct流程
###############################################################################
def react(llm_text: str, session_state: dict) -> Tuple[Any, bool]:
    """
    解析 LLM XML-style 回复并执行相应工具。
    返回 (result, done)；done=True 表示 <最终答案> 已给出。
    """
    parsed = parse_llm_response(llm_text)
    action      = parsed["action"].strip()
    inp_str     = parsed["action_input"].strip()
    final_ans   = parsed["final_answer"].strip()
    supplement  = parsed["supplement"].strip()

    # ---------- 补充信息 ----------
    if supplement:
        return {"type": "supplement", "content": supplement}, False

    # ---------- 最终答案 ----------
    if final_ans and not action:      # 有答案且无工具调用
        return final_ans, True

    # ---------- 工具调用 ----------
    if not action:                    # 没指定工具
        return "缺少 <工具调用> 标签或工具名为空", False

    try:
        params: Dict[str, Any] = json.loads(inp_str or "{}")
    except Exception:
        return f"无法解析 <调用参数> JSON：{inp_str}", False

    # 1. 解析自然语言时间 -----------------------------------------
    if action == "解析用户自然语言时间":
        return parse_time_expressions(params["raw_text"]), False

    # 2. 获取时序数据 ---------------------------------------------
    if action == "请求智能运管后端Api，获取指标项的时序数据":
        data = get_series_data(**params)
        session_state["series"] = data
        return data, False

    # 3. 时序特征提取 + 选择方法 ------------------------------------
    if action == "分析时序数据特征":
        series = params["series"]
        features = extract_time_series_features(series)

        detector_info = get_all_detectors()       # 同步拿到
        selected = llm_select_methods(features, detector_info)

        # 缓存
        session_state.update({
            "series": series,
            "features": features,
            "selected_methods": selected
        })
        return {"features": features, "selected_methods": selected}, False

    # 4. 执行单个 MCP 检测 ----------------------------------------
    if action == "执行异常检测":
        method  = params["method"]
        series  = params["series"]
        p       = params.get("params", {})
        result  = execute_detection(method, series, p)
        return result, False

    # 5. **批量** 执行选定方法 -------------------------------------
    if action == "批量执行选定检测方法":
            if not session_state.get("series") or not session_state.get("selected_methods"):
                return "缺少序列数据或方法列表，无法执行批量检测", False

            try:
                det_results = []
                available_methods = get_all_detectors()
                available_method_names = list(available_methods.keys()) if isinstance(available_methods, dict) and "error" not in available_methods else []
        
                success_count = 0
                for item in session_state["selected_methods"]:
                    method = item["method"]
            
            # 检查方法是否可用
                    if method not in available_method_names:
                        logger.warning(f"跳过不可用的方法: {method}")
                        continue
            
                    params = item.get("params", {})
                    weight = item.get("weight", 0.5)
            
            # 执行检测
                    result = execute_detection(method, session_state["series"], params)
            
            # 检查结果是否有错误
                    if "error" in result:
                        logger.warning(f"方法 {method} 执行出错: {result['error']}")
                        continue
                
            # 添加权重到结果
                    if "parameters" not in result:
                        result["parameters"] = {}
                    result["parameters"]["weight"] = weight
            
                    det_results.append(result)
                    success_count += 1
        
                if not det_results:
                    return {"error": "所有选定的检测方法都执行失败"}, False
            
                logger.info(f"成功执行 {success_count}/{len(session_state['selected_methods'])} 个检测方法")
        
        # 计算综合得分
                comp = calculate_composite_score(det_results)
                session_state["current_detection_results"] = det_results
                session_state["composite_score"] = comp
                return {"detection_results": det_results, "composite_score": comp}, False
            except Exception as e:
                logger.error(f"批量执行检测方法出错: {e}")
                return {"error": f"批量执行检测方法失败: {str(e)}"}, False
    # 6. 计算综合分数 ---------------------------------------------
    if action == "计算综合异常评分":
        comp = calculate_composite_score(params["detection_results"])
        session_state["composite_score"] = comp
        return comp, False

    # 7. 生成报告 --------------------------------------------------
    if action == "生成异常检测报告":
        comp_score = params["composite_score"]
        if not isinstance(comp_score, dict):
            comp_score = {
                "score": comp_score,
                "classification": params["classification"],
                "anomalies": [],
                "anomaly_count": 0
            }
    
        result = asyncio.run(generate_detection_report(
            params["series"],
            params["detection_results"],
            comp_score,  # 使用处理后的comp_score
            params["classification"],
            params["metadata"],
            params.get("is_multi_series", False),
            params.get("series2")
        ))
        asyncio.run(cleanup_mcp_resources())
        return result, False

    # 8. 其它简单查询 ---------------------------------------------
    if action == "请求智能运管后端Api，查询监控实例有哪些监控项":
        return monitor_item_list(params["instance"]), False

    if action == "请求智能运管后端Api，查询监控服务的资产情况和监控实例":
        return get_service_asset(params["service"]), False

    if action == "请求智能运管后端Api，查询监控实例之间的拓扑关联关系":
        return get_service_asset_edges(params["service"], params["instance_ip"]), False

    # 未识别工具 ---------------------------------------------------
    return f"未知工具调用: {action}", False

def _to_builtin(x):
    """将各种类型转换为Python内置类型，特别是numpy类型"""
    if isinstance(x, dict):
        return {k: _to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, (np.generic,)):          # numpy 标量 → builtin
        return x.item()
    return x

def shorten_tool_result(res):
    """
    将工具返回值做简要摘要，避免往 LLM 里写入过大的内容。
    现在支持自动把 numpy.* 标量转换为 json 可序列化的 builtin。
    """
    res = _to_builtin(res)  # ★ 先做通用转换

    if isinstance(res, dict) and "error" in res:
        return json.dumps({"error": res["error"]}, ensure_ascii=False)
    
    # 处理分位数异常检测结果，确保异常点被正确输出
    if isinstance(res, dict) and "method" in res and "anomalies" in res:
        method = res.get("method", "未知方法")
        anomalies = res.get("anomalies", [])
        if anomalies:
            anomaly_summary = f"检测到 {len(anomalies)} 个异常点"
            if len(anomalies) <= 5:
                anomaly_summary += f"：{anomalies}"
            else:
                anomaly_summary += f"，前5个：{anomalies[:5]}"
            return f"{method} {anomaly_summary}"
        else:
            return f"{method} 未检测到异常"

    if isinstance(res, list):
        # 特殊：解析时间表达式的结果
        if res and isinstance(res[0], dict) and "start" in res[0] and "end" in res[0]:
            for item in res:
                if "error" not in item or not item["error"]:
                    if "start_str" not in item:
                        item["start_str"] = datetime.datetime.fromtimestamp(item["start"]).strftime("%Y-%m-%d %H:%M:%S")
                    if "end_str" not in item:
                        item["end_str"] = datetime.datetime.fromtimestamp(item["end"]).strftime("%Y-%m-%d %H:%M:%S")
            time_results = [
                {"start_time": it.get("start_str",""), "end_time": it.get("end_str",""),
                 "start": it.get("start",0), "end": it.get("end",0)} if not it.get("error")
                else {"error": it["error"]}
                for it in res
            ]
            return json.dumps(time_results, ensure_ascii=False)

        # 长列表给出前后各 5 条
        if len(res) > 10:
            summary = {"总条目数": len(res), "前5条": res[:5], "后5条": res[-5:]}
            return json.dumps(summary, ensure_ascii=False, indent=2)
        return json.dumps(res, ensure_ascii=False)

    elif isinstance(res, dict):
        # 特殊处理：选择的检测方法
        if "selected_methods" in res:
            methods = res["selected_methods"]
            method_summary = [{"method": m.get("method"), "weight": m.get("weight", 0.5), "reason": m.get("reason", "")} for m in methods]
            return json.dumps({"selected_methods": method_summary, "features_summary": "提取的时序特征数据"}, ensure_ascii=False, indent=2)
        
        summary = {}
        for k, v in res.items():
            if isinstance(v, list):
                summary[k] = f"[List len={len(v)}]"
            elif isinstance(v, str) and len(v) > 300:
                summary[k] = v[:300] + f"...(omitted, length={len(v)})"
            else:
                summary[k] = v
        return json.dumps(summary, ensure_ascii=False)

    elif isinstance(res, str) and len(res) > 300:
        return res[:300] + f"...(omitted, length={len(res)})"
    else:
        return str(res)
    
def chat(user_query):
    """主对话函数"""
    session_state = {
        "series": None,
        "features": None,
        "selected_methods": None,
        "current_detection_results": None,
        "composite_score": None
    }
    system_prompt = f'''你是一个严格遵守格式规范的用于运维功能，运维数据可视化，运行于生产环境的ReAct智能体，你叫小助手，必须按以下格式处理请求：

    你的工具列表如下:
    {json.dumps(tools, ensure_ascii=False, indent=2)}
    当前时间为: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    处理规则：
    1.请根据当前时间来推断用户输入的具体日期。
    2.使用下流程进行序列异常检测:
      a. 获取时序数据
      b. 分析时序数据特征
      c. 获取所有检测方法信息
      d. 根据特征和方法信息，选择适合的方法(约3-5个方法)
      e. 对每个选定的方法执行异常检测
      f. 计算综合异常评分
      g. 生成异常检测报告
    3.如果用户输入多个时间区间，但是没有明显的比较词汇，则要在<补充请求>里提问，示例:
    <思考过程>我不知道用户是要对这些时间的数据分别进行单序列分析还是一起多序列分析，我需要确认</思考过程> <工具调用></工具调用> <调用参数></调用参数> <最终答案></最终答案> <补充请求>请问您是想对每段数据进行单序列分析，还是需要多序列的对比分析</补充请求> 
    4.如过用户输入2个时间区间，并且用户输入包含"对比"、"相比"、"比较"、"环比"、"VS"、"vs"、"变化"、"相较于"等明显比较词汇，则按多序列方式处理。
    5.根据用户的输入来判断是否要调用工具以及调用哪个工具,判断不确定的时候可以使用<补充请求>来询问用户
    6.你每次只能调用一个工具，不能在同一次响应中调用多个工具，如果有多个任务，请分轮执行。
    7.不能伪造数据
    8.严格按照以下xml格式生成响应文本：
    ```
    <思考过程>你的思考过程</思考过程>
    <工具调用>工具名称，不调用则为空</工具调用>
    <调用参数>工具输入参数{{json}}</调用参数>
    <最终答案>用户问题的最终结果（知道问题的最终答案时返回）</最终答案>
    <补充请求>系统请求用户补充信息</补充请求>
    ```
    【重要的原则】：
    1."解析用户自然语言时间"工具无法解析时间时，请你先自己根据当前时间和用户语义来计算正确的时间。
    2.当你的工具调用遇到错误时（例如"无效的field"），你必须主动思考如何解决这个问题，而不是立即询问用户。
    例如，如果出现"无效的field"错误，你应该自己主动调用"请求智能运管后端Api，查询监控实例有哪些监控项"工具来查询可用的监控项。
    3.模糊的信息通过<补充请求>来询问用户，明确的信息直接调用工具。
    4.在选择异常检测方法时，要充分考虑时序数据的特征和各检测方法的适用场景，选择最合适的方法，不要全部使用也不要随意选择。
    5.检测方法的数量控制在3-5个，太少会缺少验证，太多会增加不必要的计算。
    6.异常检测完成后，必须使用"计算综合异常评分"工具计算总分，然后使用"生成异常检测报告"生成可视化报告。
    7.最终的异常检测报告应该包含：综合评分、异常点列表、多种检测方法结果、可视化图表。
    8.整个分析流程中，只要遇到一个步骤出错，就应该尝试修复而不是放弃整个分析。
    
    '''
    
    history=[]
    history.append({"role":"system","content":system_prompt})
    history.append({"role":"user","content": user_query})

    round_no=1
    max_round=15
    pending_context = None 
    had_supplement = False
    # 记录原始用户查询
    original_user_query = user_query

    while round_no <= max_round:
        logger.info(f"=== 第 {round_no} 轮 ===")

        # 调 LLM
        ans = llm_call(history)
        if not ans:
            logger.error("LLM 返回空，结束对话")
            break
        logger.info(f"LLM 回复: {ans['content'][:200]}…")
        history.append(ans)

        # 处理 LLM 输出
        result, done = react(ans["content"], session_state)

        # ---------- 补充请求 ----------
        if isinstance(result, dict) and result.get("type") == "supplement":
            print(f"\n小助手: {result['content']}")
            user_input = input("你: ").strip() or "好的"
            history.append({"role": "user",
                            "content": f"补充回答: {user_input}"})
            round_no += 1
            continue

        # 把工具返回值写回历史（摘要）
        short_res = shorten_tool_result(result)
        history.append({"role": "user",
                        "content": f"<工具调用结果>: {short_res}"})

        # ---------- 终局 ----------
        if done:
            logger.info("=== 最终答案 ===")
            print(result)
            asyncio.run(cleanup_mcp_resources())
            return result

        round_no += 1

    # 超出轮次
    logger.warning("超过最大轮次，终止对话")
    asyncio.run(cleanup_mcp_resources())
    return "会话结束，未能在限定轮次内完成分析。"
def init_detection_system():
    """初始化检测系统，获取可用方法"""
    logger.info("初始化检测系统，获取可用方法...")
    try:
        methods = get_all_detectors()
        if isinstance(methods, dict) and "error" not in methods:
            available_methods = list(methods.keys())
            logger.info(f"可用检测方法: {available_methods}")
            return True
        else:
            error = methods.get("error", "未知错误")
            logger.error(f"获取检测方法失败: {error}")
            return False
    except Exception as e:
        logger.error(f"初始化检测系统失败: {e}")
        return False
    
if __name__ == '__main__':
    init_detection_system()
    chat('请分析192.168.0.110这台主机上周星期一的cpu利用率，检测是否有异常')