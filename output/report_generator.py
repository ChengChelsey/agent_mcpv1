# output/report_generator.py
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import config
from output.visualization import generate_summary_echarts_html

def generate_text_analysis(detection_results, composite_score, classification, series_info, is_multi_series=False):
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
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
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
        ip1 = series_info.get("ip", "主机")
        ip2 = series_info.get("ip2", "主机")
        field = series_info.get("field", "指标")
        
        # 更具体的多序列分析摘要
        analysis_text = f"""## {ip1} {field} 对比分析报告

**分析时段**: {period_desc}

**总体分析**: {summary_line}

**对比结果摘要**: 对比分析了两个时间段的{field}数据，{"发现明显的差异模式" if classification != "正常" else "两个序列整体模式相似"}。
"""
        if total_anomalies > 0:
            analysis_text += f"共检测到 {total_anomalies} 个异常点和 {total_intervals} 个异常区间。\n"
        
        analysis_text += "\n**检测方法详情**:\n"
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
                        ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
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
                        start_str = datetime.fromtimestamp(start).strftime("%H:%M:%S")
                        end_str = datetime.fromtimestamp(end).strftime("%H:%M:%S")
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

def generate_report_single(series, ip, field, user_query):
    """
    为单序列异常检测生成汇总报告
    
    参数:
        series: 时序数据 [(ts, value), ...]
        ip: 主机IP
        field: 指标字段
        user_query: 用户查询
    
    返回:
        dict: 报告结果包含图表路径和分析结果
    """
    # 执行异常检测
    from analysis.single_series import analyze_single_series
    result = analyze_single_series(series)
    method_results = result["method_results"]
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"output/plots/{ip}_{field}_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)
    
    # 提取时间信息构建 series_info
    start_time = datetime.fromtimestamp(series[0][0]).strftime("%Y-%m-%d %H:%M:%S") if series else "未知"
    end_time = datetime.fromtimestamp(series[-1][0]).strftime("%Y-%m-%d %H:%M:%S") if series else "未知"
    
    series_info = {
        "ip": ip,
        "field": field,
        "start": start_time,
        "end": end_time
    }
    
    # 生成分析报告文字
    analysis_text = generate_text_analysis(
        method_results, 
        result["composite_score"], 
        result["classification"], 
        series_info,
        is_multi_series=False
    )
    
    # 生成汇总图表
    summary_path = os.path.join(base_dir, "summary.html")
    chart_path, tooltip_map = generate_summary_echarts_html(
        series, None, method_results, summary_path, 
        title=f"{ip} {field} 异常检测汇总图"
    )
    
    # 生成最终报告
    final_report_path = os.path.join(base_dir, "final_report.html")
    generate_report_single_series(
        user_query, chart_path, method_results, tooltip_map, final_report_path, 
        analysis_text=analysis_text, composite_score=result["composite_score"],
        classification=result["classification"]  # 确保传递分类和得分
    )
    
    # 返回结果
    return {
        "classification": result["classification"],
        "composite_score": result["composite_score"],
        "anomaly_times": result["anomaly_times"],
        "method_results": [mr.to_dict() for mr in method_results],
        "report_path": final_report_path,
        "analysis_text": analysis_text
    }

def generate_report_multi(series1, series2, ip1, ip2, field, user_query):
    """
    为多序列对比异常检测生成汇总报告
    
    参数:
        series1: 第一个时序数据 [(ts, value), ...]
        series2: 第二个时序数据 [(ts, value), ...]
        ip1: 第一个主机IP
        ip2: 第二个主机IP
        field: 指标字段
        user_query: 用户查询
    
    返回:
        dict: 报告结果包含图表路径和分析结果
    """
    # 执行多序列对比异常检测
    from analysis.multi_series import analyze_multi_series
    result = analyze_multi_series(series1, series2)
    method_results = result["method_results"]
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"output/plots/{ip1}_{ip2}_{field}_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)
    
    # 提取时间信息构建 series_info
    start_time = datetime.fromtimestamp(series1[0][0]).strftime("%Y-%m-%d %H:%M:%S") if series1 else "未知"
    end_time = datetime.fromtimestamp(series1[-1][0]).strftime("%Y-%m-%d %H:%M:%S") if series1 else "未知"
    
    series_info = {
        "ip": ip1,
        "ip2": ip2,
        "field": field,
        "start": start_time,
        "end": end_time
    }
    
    # 生成分析报告文字
    analysis_text = generate_text_analysis(
        method_results, 
        result["composite_score"], 
        result["classification"], 
        series_info,
        is_multi_series=True
    )
    
    # 生成汇总图表 - 传递两个序列
    summary_path = os.path.join(base_dir, "summary.html")
    chart_path, tooltip_map = generate_summary_echarts_html(
        series1, series2, method_results, summary_path,
        title=f"{ip1} vs {ip2} {field} 对比异常检测汇总图"
    )
    
    # 生成最终报告
    final_report_path = os.path.join(base_dir, "final_report.html")
    generate_report_single_series(
        user_query, chart_path, method_results, tooltip_map, final_report_path,
        analysis_text=analysis_text, composite_score=result["composite_score"], 
        classification=result["classification"]  # 确保传递分类和得分
    )
    
    # 返回结果
    return {
        "classification": result["classification"],
        "composite_score": result["composite_score"],
        "anomaly_times": result["anomaly_times"],
        "anomaly_intervals": result["anomaly_intervals"],
        "method_results": [mr.to_dict() for mr in method_results],
        "report_path": final_report_path,
        "analysis_text": analysis_text
    }
def generate_report_single_series(user_query, chart_path, detection_results, tooltip_map, output_path, analysis_text=None, composite_score=0.0, classification="正常"):
    """
    生成单序列异常检测HTML报告
    
    参数:
        user_query: 用户原始查询
        chart_path: 图表HTML路径
        detection_results: 检测结果列表
        tooltip_map: 异常点/区间映射
        output_path: 输出HTML报告路径
        analysis_text: 可选，分析文本报告
        composite_score: 综合得分
        classification: 分类结果
    
    返回:
        str: 生成的报告路径
    """
    # 使用传入的综合评分和分类，而不是重新计算
    # 设置报告样式类
    if classification == "高置信度异常":
        alert_class = "alert-danger"
    elif classification == "轻度异常":
        alert_class = "alert-warning"
    else:
        alert_class = "alert-success"
    
    # 构建异常点说明HTML
    anomaly_explanations_html = ""
    for point_id, info in tooltip_map.items():
        # 格式化时间戳
        if "ts" in info:
            time_str = datetime.fromtimestamp(info["ts"]).strftime("%Y-%m-%d %H:%M:%S")
            anomaly_explanations_html += f"""
            <div class="card mb-2">
                <div class="card-header">
                    <strong>异常点 #{point_id}</strong> - {time_str}
                </div>
                <div class="card-body">
                    <p><strong>方法:</strong> {info["method"]}</p>
                    <p><strong>值:</strong> {info.get("value", "N/A")}</p>
                    <p><strong>说明:</strong> {info["explanation"]}</p>
                </div>
            </div>
            """
        # 区间异常
        elif "ts_start" in info and "ts_end" in info:
            start_str = datetime.fromtimestamp(info["ts_start"]).strftime("%Y-%m-%d %H:%M:%S")
            end_str = datetime.fromtimestamp(info["ts_end"]).strftime("%Y-%m-%d %H:%M:%S")
            anomaly_explanations_html += f"""
            <div class="card mb-2">
                <div class="card-header">
                    <strong>异常区间 #{point_id}</strong> - {start_str} 至 {end_str}
                </div>
                <div class="card-body">
                    <p><strong>方法:</strong> {info["method"]}</p>
                    <p><strong>说明:</strong> {info["explanation"]}</p>
                </div>
            </div>
            """
    
    # 如果没有异常，显示正常信息
    if not tooltip_map:
        anomaly_explanations_html = """
        <div class="alert alert-success">
            <p>未检测到异常点。</p>
        </div>
        """
    
    # 构建方法摘要HTML
    methods_summary_html = ""
    for result in detection_results:
        methods_summary_html += f"""
        <div class="card mb-2">
            <div class="card-header">
                <strong>{result.method}</strong>
            </div>
            <div class="card-body">
                <p>{result.description}</p>
            </div>
        </div>
        """
    
    # 如果提供了分析文本，则转换成HTML格式
    formatted_text = ""
    if analysis_text:
        # 替换Markdown格式为HTML格式，避免使用复杂的f-string转义
        formatted_text = analysis_text.replace('\n', '<br>')
        formatted_text = re.sub(r'## (.*)', r'<h3>\1</h3>', formatted_text)
        formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_text)
        formatted_text = re.sub(r'- (.*)', r'• \1<br>', formatted_text)
    
    # 构建完整HTML报告
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>异常检测报告</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; }}
        .iframe-container {{ width: 100%; height: 650px; border: none; }}
        .markdown-content {{ line-height: 1.6; }}
        .markdown-content h3 {{ margin-top: 20px; margin-bottom: 15px; color: #333; }}
        .markdown-content strong {{ font-weight: 600; color: #444; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mt-4 mb-4">时序数据异常检测报告</h1>
        
        <div class="card mb-4">
            <div class="card-header">
                <strong>用户问题</strong>
            </div>
            <div class="card-body">
                <p>{user_query}</p>
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header">
                <strong>分析结论</strong>
            </div>
            <div class="card-body">
                <div class="alert {alert_class}">
                    <h4>综合判定: {classification}</h4>
                    <p>综合得分: {composite_score:.2f}</p>
                </div>
                
                <!-- 将分析报告摘要内容放到分析结论中 -->
                <div class="markdown-content mt-3">
                    {formatted_text if analysis_text else ""}
                </div>
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header">
                <strong>异常检测图表</strong>
            </div>
            <div class="card-body p-0">
                <iframe class="iframe-container" src="{os.path.basename(chart_path)}"></iframe>
            </div>
        </div>
        
        <h2 class="mt-4 mb-3">异常点详细说明</h2>
        {anomaly_explanations_html}
        
        <h2 class="mt-4 mb-3">检测方法摘要</h2>
        {methods_summary_html}
        
        <footer class="mt-5 text-center text-muted">
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path