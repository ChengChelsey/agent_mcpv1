# output/analysis_summary.py
import datetime
from typing import List, Dict, Any

def generate_text_analysis(detection_results: List, composite_score: float, classification: str, 
                          series_info: Dict, is_multi_series: bool = False) -> str:
    """
    生成易于理解的文字分析报告
    
    参数:
        detection_results: 检测结果列表
        composite_score: 综合得分
        classification: 分类结果 ('正常', '轻度异常', '高置信度异常')
        series_info: 关于数据的信息 {"ip": str, "field": str, "start": str, "end": str}
        is_multi_series: 是否是多序列分析
        
    返回:
        str: 结构化的文字分析报告
    """
    # 提取基本信息
    ip = series_info.get("ip", "未知主机")
    field = series_info.get("field", "未知指标")
    start_time = series_info.get("start", "未知开始时间")
    end_time = series_info.get("end", "未知结束时间")
    
    # 检测统计信息
    total_anomalies = sum(len(result.anomalies) for result in detection_results)
    total_intervals = sum(len(result.intervals) for result in detection_results)
    methods_with_anomalies = [r.method for r in detection_results if len(r.anomalies) > 0 or len(r.intervals) > 0]
    
    # 格式化时间
    try:
        start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        period_desc = f"{start_dt.strftime('%Y年%m月%d日 %H:%M')} 至 {end_dt.strftime('%Y年%m月%d日 %H:%M')}"
    except:
        period_desc = f"{start_time} 至 {end_time}"
    
    # 根据评分生成总体分析
    if classification == "正常":
        summary_line = f"系统运行正常，未发现明显异常。"
        if total_anomalies > 0:
            summary_line += f" 虽然检测到 {total_anomalies} 个可能的异常点，但整体影响较小，综合评分为 {composite_score:.2f}，低于异常阈值。"
    elif classification == "轻度异常":
        summary_line = f"系统运行存在轻微异常，综合评分为 {composite_score:.2f}。"
        if total_anomalies > 0:
            summary_line += f" 共检测到 {total_anomalies} 个异常点，可能需要关注。"
    else:  # 高置信度异常
        summary_line = f"系统运行存在明显异常，综合评分为 {composite_score:.2f}，超过高置信阈值。"
        if total_anomalies > 0:
            summary_line += f" 共检测到 {total_anomalies} 个异常点，强烈建议进一步分析。"
    
    # 构建详细分析报告
    if is_multi_series:
        ip2 = series_info.get("ip2", "第二组数据")
        analysis_text = f"""## {ip} vs {ip2} {field} 对比分析报告

**分析时段**: {period_desc}

**总体分析**: {summary_line}

**检测方法详情**:
"""
    else:
        analysis_text = f"""## {ip} {field} 异常检测报告

**分析时段**: {period_desc}

**总体分析**: {summary_line}

**检测方法详情**:
"""

    # 添加各个检测方法的详细分析
    for result in detection_results:
        # 跳过无异常的方法
        if not result.anomalies and not result.intervals:
            continue
            
        method_desc = result.description
        
        # 处理不同类型的异常
        if result.visual_type == "point" and result.anomalies:
            analysis_text += f"\n- **{result.method}**: 检测到 {len(result.anomalies)} 个异常点。"
            
            # 添加代表性异常的解释
            if result.explanation and len(result.explanation) > 0:
                max_examples = min(3, len(result.explanation))
                analysis_text += " 典型异常表现为: "
                for i in range(max_examples):
                    # 尝试将时间戳转换为可读时间
                    try:
                        ts = result.anomalies[i]
                        ts_str = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                        analysis_text += f"{ts_str} - {result.explanation[i]}; "
                    except:
                        analysis_text += f"{result.explanation[i]}; "
        
        elif result.visual_type in ("range", "curve") and result.intervals:
            analysis_text += f"\n- **{result.method}**: 检测到 {len(result.intervals)} 个异常区间。"
            
            # 添加代表性区间的解释
            if result.explanation and len(result.explanation) > 0:
                max_examples = min(2, len(result.explanation))
                analysis_text += " 典型异常区间表现为: "
                for i in range(max_examples):
                    try:
                        start, end = result.intervals[i]
                        start_str = datetime.datetime.fromtimestamp(start).strftime("%H:%M:%S")
                        end_str = datetime.datetime.fromtimestamp(end).strftime("%H:%M:%S")
                        analysis_text += f"{start_str}至{end_str} - {result.explanation[i]}; "
                    except:
                        analysis_text += f"{result.explanation[i]}; "
        
        elif result.visual_type == "none" and result.explanation:
            analysis_text += f"\n- **{result.method}**: {' '.join(result.explanation[:2])}"
    
    # 添加建议和结论
    analysis_text += "\n\n**建议**:\n"
    
    if classification == "正常":
        analysis_text += "- 系统运行状态良好，可继续当前运维策略。"
    elif classification == "轻度异常":
        analysis_text += f"- 关注 {field} 指标的变化趋势，尤其是 {', '.join(methods_with_anomalies[:2])} 检测到的异常点。\n"
        analysis_text += "- 考虑设置该指标的监控告警，避免问题恶化。"
    else:  # 高置信度异常
        analysis_text += f"- 建议立即排查 {field} 指标异常的根本原因，特别是 {', '.join(methods_with_anomalies[:2])} 检测到的问题。\n"
        analysis_text += "- 检查系统配置和相关依赖组件状态。\n"
        analysis_text += "- 考虑与其他指标交叉分析，确定影响范围。"
    
    return analysis_text