#! /usr/bin/env python3
import re
import json
import datetime
import requests
import time
import json, datetime, traceback
import os  
import traceback
import hashlib
import config 
import dateparser

from typing import Any, Dict
from django.conf import settings

from output.report_generator import get_anomaly_detection_report
from analysis.single_series import analyze_single_series
from analysis.multi_series import analyze_multi_series
from output.report_generator import generate_report_single, generate_report_multi
from output.visualization import generate_summary_echarts_html
from utils.ts_cache import ensure_cache_file, load_series_from_cache
from utils.time_utils import parse_time_expressions
from utils.ts_features import analyze_time_series_features, select_detection_methods

# MCP客户端
from mcp_client import get_mcp_client
AIOPS_BACKEND_DOMAIN = 'https://aiopsbackend.cstcloud.cn'
LLM_URL = 'http://10.16.1.16:58000/v1/chat/completions'
AUTH = ('chelseyyycheng@outlook.com', 'UofV1uwHwhVp9tcTue')

CACHE_DIR = "cached_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_data_from_backend(ip:str, start_ts:int, end_ts:int, field:str):
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

tools = [
     {
        "name":"解析用户自然语言时间",
        "description":"返回一个list，每个元素是{start, end, error}. 如果不确定，可向用户澄清。",
        "parameters":{
            "type":"object",
            "properties":{
                "raw_text":{"type":"string"}
            },
            "required":["raw_text"]
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
            "required": ["ip","start","end","field"]
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
            "required": ["service","instance"]
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
            "required": ["service","instance_ip"]
        }
    },
    {
        "name": "单序列异常检测(文件)",
        "description": "对单序列 [int_ts,val] 进行多方法分析, 生成报告和ECharts HTML",
        "parameters": {
            "type": "object",
            "properties": {
                "ip":    {"type": "string"},
                "field": {"type": "string"},
                "start": {"type": "string"},
                "end":   {"type": "string"}
            },
            "required": ["ip","field","start","end"]
        }
    },
    {
        "name": "多序列对比异常检测(文件)",
        "description": "对两组 [int_ts,val] 进行对比分析, 生成报告和ECharts HTML",
        "parameters": {
            "type": "object",
            "properties": {
                "ip1":    {"type": "string"},
                "field1": {"type": "string"},
                "start1": {"type": "string"},
                "end1":   {"type": "string"},
                "ip2":    {"type": "string"},
                "field2": {"type": "string"},
                "start2": {"type": "string"},
                "end2":   {"type": "string"}
            },
            "required": ["ip1","field1","start1","end1","ip2","field2","start2","end2"]
        }
    },
    {
        "name": "分析时序数据特征",
        "description": "分析时序数据的统计特性、趋势、季节性、波动性等特征",
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
        "name": "选择适合的异常检测方法",
        "description": "根据时序数据特征和检测方法信息，选择最适合的检测方法",
        "parameters": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": "时序数据特征"
                },
                "detector_info": {
                    "type": "object",
                    "description": "检测方法信息"
                }
            },
            "required": ["features", "detector_info"]
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
                    "description": "检测参数"
                }
            },
            "required": ["method", "series", "params"]
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
        "description": "根据异常检测结果生成综合报告",
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
                }
            },
            "required": ["series", "detection_results", "composite_score", "classification", "metadata"]
        }
    }
]

###############################################################################
# 新增工具函数
###############################################################################

async def analyze_ts_features(series_data):
    """分析时序数据特征"""
    try:
        features = analyze_time_series_features(series_data)
        return features
    except Exception as e:
        print(f"分析时序特征错误: {e}")
        traceback.print_exc()
        return {"error": f"分析时序特征失败: {str(e)}"}

async def get_detection_methods():
    """从MCP服务器获取异常检测方法信息"""
    try:
        client = await get_mcp_client()
        detector_info = await client.get_all_detectors()
        return detector_info
    except Exception as e:
        print(f"获取检测方法信息错误: {e}")
        traceback.print_exc()
        return {"error": f"获取检测方法信息失败: {str(e)}"}

async def select_best_detection_methods(features, detector_info):
    """根据时序特征和检测方法信息，选择最适合的检测方法"""
    try:
        selected_methods = select_detection_methods(features, detector_info)
        return selected_methods
    except Exception as e:
        print(f"选择检测方法错误: {e}")
        traceback.print_exc()
        return {"error": f"选择检测方法失败: {str(e)}"}
async def execute_detection(method, series_data, params):
    """执行异常检测
    
    Args:
        method: 检测方法名称，如 "IQR异常检测"
        series_data: 时间序列数据
        params: 检测参数
    
    Returns:
        检测结果
    """
    try:
        # 准备数据
        data = {"series": series_data}
        
        # 获取MCP客户端
        client = await get_mcp_client()
        
        # 执行检测
        result = await client.detect_anomalies(method, data, params)
        
        # 检查是否有错误
        if isinstance(result, dict) and "error" in result:
            logger.error(f"检测错误: {result['error']}")
            return {"error": result["error"], "method": method}
        
        return result
    except Exception as e:
        logger.error(f"执行异常检测错误: {e}")
        traceback.print_exc()
        return {"error": f"执行异常检测失败: {str(e)}", "method": method}

def calculate_composite_score(detection_results):
    """计算综合异常评分"""
    try:
        if not detection_results or len(detection_results) == 0:
            return {"score": 0.0, "classification": "正常"}
        
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
        print(f"计算综合评分错误: {e}")
        traceback.print_exc()
        return {"error": f"计算综合评分失败: {str(e)}"}

async def generate_detection_report(series_data, detection_results, composite_score, classification, metadata):
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
        summary_path = os.path.join(base_dir, "summary.html")
        chart_path, tooltip_map = generate_summary_echarts_html(
            series_data, 
            None,  # 单序列检测，无第二序列
            detection_results, 
            summary_path, 
            title=chart_title
        )
        
        # 生成LLM分析
        user_query = metadata.get("user_query", "")
        result_dict = {
            "classification": classification,
            "composite_score": composite_score,
            "anomaly_times": composite_score.get("anomalies", []),
            "method_results": detection_results,
            "user_query": user_query
        }
        
        llm_analysis = generate_llm_report(result_dict, "single")
        
        # 生成最终报告
        final_report_path = os.path.join(base_dir, "final_report.html")
        generate_report_html(
            user_query, chart_path, detection_results, tooltip_map, final_report_path, 
            composite_score=composite_score["score"],
            classification=classification,
            llm_analysis=llm_analysis,
            is_multi_series=False
        )
        
        return {
            "report_path": final_report_path,
            "chart_path": chart_path,
            "llm_analysis": llm_analysis
        }
    except Exception as e:
        print(f"生成报告错误: {e}")
        traceback.print_exc()
        return {"error": f"生成报告失败: {str(e)}"}

###############################################################################

def monitor_item_list(ip):
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
    url = f'{AIOPS_BACKEND_DOMAIN}/api/v1/property/mail/topology/search?instance={instance_ip}'
    resp = requests.get(url=url, auth=AUTH)
    if resp.status_code == 200:
        return json.loads(resp.text)
    else:
        return f"查询拓扑失败: {resp.status_code} => {resp.text}"
    
def get_series_data(ip: str, start: str, end: str, field: str):
    try:
        return load_series_from_cache(ip, field, start, end)
    except Exception as e:
        return str(e) 

def validate_multi_series_params(action_input):
    #验证多序列对比异常检测的参数，目前只能处理两个序列的对比

    sequence_nums = set()
    for key in action_input.keys():
        if key.startswith(('ip', 'field', 'start', 'end')) and len(key) > 2 and key[2:].isdigit():
            sequence_nums.add(int(key[2:]))
    
    if len(sequence_nums) > 2 or max(sequence_nums, default=0) > 2:
        return False
    required_pairs = [
        ('ip1', 'field1', 'start1', 'end1'),
        ('ip2', 'field2', 'start2', 'end2')
    ]
    
    for pair in required_pairs:
        if not all(param in action_input for param in pair):
            return False
    return True

import numpy as np
from output.report_generator import generate_llm_report, generate_report_html

###############################################################################
def single_series_detect(ip, field, start, end, user_query=""):

    try:
        series = load_series_from_cache(ip, field, start, end)
        query_text = user_query or f"分析 {ip} 在 {start} ~ {end} 的 {field} 数据"
        
        result = generate_report_single(series, ip, field, query_text)
        result["user_query"] = query_text
        return result
    except Exception as e:
        print(f"单序列分析生成报告失败: {e}")
        traceback.print_exc()
        
        #基本分析作为备选
        try:
            series = load_series_from_cache(ip, field, start, end)
            analysis_result = analyze_single_series(series)
            return {
                "classification": analysis_result["classification"],
                "composite_score": analysis_result["composite_score"],
                "anomaly_times": analysis_result["anomaly_times"],
                "report_path": "N/A - 报告生成失败",
                "user_query": user_query or f"分析 {ip} 在 {start} ~ {end} 的 {field} 数据"
            }
        except:
            return {"error": f"无法加载或处理数据: {str(e)}"}

def multi_series_detect(ip1, field1, start1, end1, ip2, field2, start2, end2, user_query=""):
    try:
        series1 = load_series_from_cache(ip1, field1, start1, end1)
        series2 = load_series_from_cache(ip2, field2, start2, end2)
        
        query_text = user_query or f"对比分析 {ip1} 在 {start1} 和 {ip2} 在 {start2} 的 {field1}/{field2} 指标"

        result = generate_report_multi(series1, series2, ip1, ip2, field1, query_text)
        result["user_query"] = query_text
        return result
    except Exception as e:
        print(f"多序列分析生成报告失败: {e}")
        traceback.print_exc()
        
        try:
            series1 = load_series_from_cache(ip1, field1, start1, end1)
            series2 = load_series_from_cache(ip2, field2, start2, end2)
            analysis_result = analyze_multi_series(series1, series2)
            return {
                "classification": analysis_result["classification"],
                "composite_score": analysis_result["composite_score"],
                "anomaly_times": analysis_result["anomaly_times"],
                "anomaly_intervals": analysis_result.get("anomaly_intervals", []),
                "report_path": "N/A - 报告生成失败",
                "user_query": query_text
            }
        except:
            return {"error": f"无法加载或处理数据: {str(e)}"}

###############################################################################

def llm_call(messages):
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
        print("Error:", r.status_code, r.text)
        return None

def parse_llm_response(txt):
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

def react(llm_text):
    parsed = parse_llm_response(llm_text)
    action = parsed["action"]
    inp_str = parsed["action_input"]
    final_ans = parsed["final_answer"]
    supplement = parsed["supplement"]
    is_final = False

    if action and inp_str:
        try:
            action_input = json.loads(inp_str)
        except:
            return f"无法解析调用参数JSON: {inp_str}", False

        if action == "解析用户自然语言时间":
            return parse_time_expressions(action_input["raw_text"]), False
        if action == "请求智能运管后端Api，获取指标项的时序数据":
            data = get_series_data(**action_input)
            return data, False
        if action == "请求智能运管后端Api，查询监控实例有哪些监控项":
            return monitor_item_list(action_input["instance"]), False
        elif action == "请求智能运管后端Api，查询监控服务的资产情况和监控实例":
            return get_service_asset(action_input["service"]), False
        elif action == "请求智能运管后端Api，查询监控实例之间的拓扑关联关系":
            return get_service_asset_edges(action_input["service"], action_input["instance_ip"]), False
        elif action == "分析时序数据特征":
            # 异步函数需要特殊处理
            result = asyncio.run(analyze_ts_features(action_input["series"]))
            return result, False
            
        elif action == "获取异常检测方法信息":
            result = asyncio.run(get_detection_methods())
            return result, False
            
        elif action == "选择适合的异常检测方法":
            result = asyncio.run(select_best_detection_methods(action_input["features"], action_input["detector_info"]))
            return result, False
            
        elif action == "执行异常检测":
            result = asyncio.run(execute_detection(action_input["method"], action_input["series"], action_input["params"]))
            return result, False
            
        elif action == "计算综合异常评分":
            result = calculate_composite_score(action_input["detection_results"])
            return result, False
            
        elif action == "生成异常检测报告":
            result = asyncio.run(generate_detection_report(
                action_input["series"],
                action_input["detection_results"],
                action_input["composite_score"],
                action_input["classification"],
                action_input["metadata"]
            ))
            return result, False
        
        else:
            return f"未知工具调用: {action}", False
        
    if supplement.strip():
        return {"type": "supplement", "content": supplement}

    if final_ans.strip():
        is_final = True
        return final_ans, is_final
    return ("格式不符合要求，必须使用：<思考过程></思考过程> <工具调用></工具调用> <调用参数></调用参数> <最终答案></最终答案>", is_final)

def shorten_tool_result(res):
    if isinstance(res, list):
        if len(res) > 0 and isinstance(res[0], dict) and "start" in res[0] and "end" in res[0]:
            for item in res:
                if "error" not in item or not item["error"]:
                    if "start_str" not in item:
                        start_dt = datetime.datetime.fromtimestamp(item["start"])
                        item["start_str"] = start_dt.strftime("%Y-%m-%d %H:%M:%S") 
                    if "end_str" not in item:
                        end_dt = datetime.datetime.fromtimestamp(item["end"])
                        item["end_str"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            
            #一个简单的的摘要
            time_results = []
            for item in res:
                if "error" in item and item["error"]:
                    time_results.append({"error": item["error"]})
                else:
                    time_results.append({
                        "start_time": item.get("start_str", ""),
                        "end_time": item.get("end_str", ""),
                        "start": item.get("start", 0),
                        "end": item.get("end", 0)
                    })
            return json.dumps(time_results, ensure_ascii=False)
        if len(res) > 10:
            summary = {"总条目数": len(res), "前5条": res[:5], "后5条": res[-5:]}
            return json.dumps(summary, ensure_ascii=False, indent=2)
            
        return json.dumps(res, ensure_ascii=False)
    elif isinstance(res, dict):
        summary = {}
        for k,v in res.items():
            if isinstance(v, list):
                summary[k] = f"[List len={len(v)}]"
            elif isinstance(v, str) and len(v)>300:
                summary[k] = v[:300] + f"...(omitted, length={len(v)})"
            else:
                summary[k] = v
        return json.dumps(summary, ensure_ascii=False)
    elif isinstance(res, str) and len(res)>300:
        return res[:300] + f"...(omitted, length={len(res)})"
    else:
        return str(res)

def chat(user_query):
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
    4.如过用户输入2个时间区间，并且用户输入包含"对比"、"相比"、"比较"、"环比"、"VS"、"vs"、"变化"、"相较于"等明显比较词汇，则调用'多序列对比异常检测(文件)'。
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
    4.在选择异常检测方法时，请根据时序数据的特征和每种检测方法的适用场景，选择最合适的方法，不要全部使用。
    5.检测方法的数量控制在3-5个，太少会缺少验证，太多会增加不必要的计算。
    
    '''
    history=[]
    history.append({"role":"system","content":system_prompt})
    history.append({"role":"user","content": user_query})

    round_num=1
    max_round=15
    pending_context = None 
    had_supplement = False
    # 记录原始用户查询
    original_user_query = user_query

    while True:
        print(f"=== 第{round_num}轮对话 ===")

        if pending_context:
            ans = llm_call(pending_context["history"])
            pending_context = None  
        else:
            ans = llm_call(history)
            
        if not ans:
            print("大模型返回None,结束")
            return

        print(ans["content"])

        history.append(ans)
        txt = ans.get("content","")
        res = react(txt)

        if isinstance(res, dict) and res.get("type") == "supplement":
            print(f"\n小助手: {res['content']}")
            try:
                user_input = input("你: ")
                if not user_input.strip():
                    user_input = "默认继续单序列分析"
            except Exception as e:
                print(f"无法获取用户输入: {e}")
                user_input = "默认继续单序列分析"
            
            supplement_response = f"对于您的问题 '{res['content']}'，我的回答是: {user_input}"
            history.append({"role": "user", "content": supplement_response})
            
            had_supplement = True
            round_num += 1
            continue

        result, done = res
        
        if action == "执行异常检测" or action == "生成异常检测报告":
            try:
                action_input_match = re.search(r'<调用参数>(.*?)</调用参数>', txt, re.DOTALL)
                if action_input_match:
                    action_input = json.loads(action_input_match.group(1))
                    action_input["user_query"] = original_user_query
                    updated_txt = re.sub(r'<调用参数>.*?</调用参数>', 
                                         f'<调用参数>{json.dumps(action_input, ensure_ascii=False)}</调用参数>', 
                                         txt, flags=re.DOTALL)
                    history[-1]["content"] = updated_txt
            except Exception as e:
                print(f"处理调用参数出错: {str(e)}")

        short_result = shorten_tool_result(result)
        history.append({
            "role":"user",
            "content": f"<工具调用结果>: {short_result}"
        })

        if done:
            print("===最终输出===")
            print(result)
            return result

        round_num+=1
        if round_num>max_round:
            print("超出上限")
            return

if __name__ == '__main__':
    # chat('你好')
    chat(
        '请分析192.168.0.110这台主机上周星期一和上上周星期一还有前天的cpu利用率，并作图给出分析报告')
