# output/report_generator.py
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import config
from output.visualization import generate_summary_echarts_html

def convert_markdown_to_html(markdown_text):
    """
    将Markdown文本转换为HTML
    
    参数:
        markdown_text: Markdown格式的文本
    
    返回:
        str: 转换后的HTML文本
    """
    # 转换标题
    html = markdown_text
    
    # 转换## 标题
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    
    # 转换### 标题
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # 转换#### 标题
    html = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # 转换# 标题
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # 转换加粗
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    
    # 转换斜体
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    
    # 转换无序列表
    # 先把所有无序列表项转换为临时标记
    html = re.sub(r'^- (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # 将连续的<li>项组合成<ul>列表
    li_pattern = r'(<li>.*?</li>)+'
    
    def replace_list(match):
        list_items = match.group(0)
        return f'<ul>{list_items}</ul>'
    
    html = re.sub(li_pattern, replace_list, html, flags=re.DOTALL)
    
    # 转换有序列表 (数字后跟点的格式)
    html = re.sub(r'^(\d+)\. (.*?)$', r'<li>\2</li>', html, flags=re.MULTILINE)
    
    # 转换段落 (连续的非标签文本)
    # 先按换行符分割
    paragraphs = []
    buffer = ""
    
    for line in html.split('\n'):
        line = line.strip()
        if line and not line.startswith('<') and not line.endswith('>'):
            if buffer:
                buffer += " " + line
            else:
                buffer = line
        else:
            if buffer:
                paragraphs.append(f'<p>{buffer}</p>')
                buffer = ""
            if line:
                paragraphs.append(line)
    
    if buffer:
        paragraphs.append(f'<p>{buffer}</p>')
    
    html = '\n'.join(paragraphs)
    
    return html

def generate_llm_report(result, report_type="single", user_query=""):
    """
    使用大模型生成异常检测分析报告
    
    参数:
        result: 异常检测结果
        report_type: 报告类型，"single"单序列或"multi"多序列
        user_query: 用户的原始查询
        
    返回:
        str: 大模型生成的分析报告
    """
    # 构建完整的上下文信息
    classification = result.get('classification', '未知')
    score = result.get('composite_score', 0)
    
    # 获取异常点数量
    anomaly_count = len(result.get('anomaly_times', []))
    interval_count = len(result.get('anomaly_intervals', []))
    
    # 提取方法结果信息
    method_results = result.get('method_results', [])
    method_details = []
    
    for method in method_results:
        if isinstance(method, dict):
            # 如果已经是字典形式
            method_info = method
        else:
            # 如果是对象，转为字典
            method_info = method.to_dict() if hasattr(method, 'to_dict') else vars(method)
        
        # 提取关键信息
        method_name = method_info.get('method', '未知方法')
        anomalies = method_info.get('anomalies', [])
        intervals = method_info.get('intervals', [])
        explanations = method_info.get('explanation', [])
        
        if anomalies or intervals or explanations:
            detail = {
                "method": method_name,
                "anomaly_count": len(anomalies),
                "interval_count": len(intervals),
                "explanations": explanations[:3]  # 限制长度
            }
            method_details.append(detail)
    
    # 构建提示文本
    if report_type == "single":
        prompt = f"""
请为以下单序列异常检测结果生成一份详细的分析报告，报告应包含以下方面：
1. 总体异常状况摘要
2. 各检测方法发现的问题分析
3. 可能的原因分析
4. 建议的后续操作

检测结果信息：
- 用户查询: {user_query}
- 总体分类: {classification}
- 综合得分: {score:.2f}
- 发现异常点数: {anomaly_count}

检测方法详情:
{json.dumps(method_details, ensure_ascii=False, indent=2)}

请注意以下要求：
- 生成的报告应该使用Markdown格式，包含标题(##)、子标题(###)、列表项等
- 每个部分应该使用二级标题(##)，内容要精炼但信息量丰富
- 不要简单重复上述数据，而是提供有价值的见解和分析
- 总长度控制在600字以内
"""
    else:  # multi
        prompt = f"""
请为以下多序列对比异常检测结果生成一份详细的分析报告，报告应包含以下方面：
1. 两个序列的对比摘要
2. 差异模式与趋势分析
3. 各检测方法发现的问题详解
4. 异常的可能原因
5. 建议的后续操作

检测结果信息：
- 用户查询: {user_query}
- 总体分类: {classification}
- 综合得分: {score:.2f}
- 发现异常点数: {anomaly_count}
- 发现异常区间数: {interval_count}

检测方法详情:
{json.dumps(method_details, ensure_ascii=False, indent=2)}

请注意以下要求：
- 生成的报告应该使用Markdown格式，包含标题(##)、子标题(###)、列表项等
- 每个部分应该使用二级标题(##)，内容要精炼但信息量丰富
- 不要简单重复上述数据，而是提供有价值的见解和分析
- 总长度控制在600字以内
"""

    # 调用大模型生成报告
    messages = [
        {"role": "system", "content": "你是一位专业的系统运维分析师，擅长解读时序数据异常检测结果并提供深入分析。你的回复应使用规范的Markdown格式。"},
        {"role": "user", "content": prompt}
    ]
    
    try:
        # 导入位于上层的llm_call函数
        import sys, os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from agent import llm_call
        
        response = llm_call(messages)
        if response and 'content' in response:
            return response['content']
        else:
            print("LLM返回格式不正确或为空")
            return f"## 异常检测报告\n\n**分类**: {classification}\n**得分**: {score:.2f}\n**异常点数**: {anomaly_count}"
    except Exception as e:
        print(f"LLM调用失败: {e}")
        return f"## 异常检测报告\n\n**分类**: {classification}\n**得分**: {score:.2f}\n**异常点数**: {anomaly_count}"


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
    
    # 生成汇总图表
    summary_path = os.path.join(base_dir, "summary.html")
    chart_path, tooltip_map = generate_summary_echarts_html(
        series, None, method_results, summary_path, 
        title=f"{ip} {field} 异常检测汇总图"
    )
    
    # 调用大模型生成详细的分析报告
    print(f"调用大模型生成报告，用户查询: {user_query}")
    llm_analysis = generate_llm_report(result, "single", user_query)
    print(f"大模型生成的报告长度: {len(llm_analysis)}")
    
    # 生成最终报告
    final_report_path = os.path.join(base_dir, "final_report.html")
    generate_report_html(
        user_query, chart_path, method_results, tooltip_map, final_report_path, 
        composite_score=result["composite_score"],
        classification=result["classification"],
        llm_analysis=llm_analysis
    )
    
    # 返回结果
    return {
        "classification": result["classification"],
        "composite_score": result["composite_score"],
        "anomaly_times": result["anomaly_times"],
        "method_results": [mr.to_dict() for mr in method_results],
        "report_path": final_report_path,
        "llm_analysis": llm_analysis
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
    
    # 生成汇总图表 - 传递两个序列
    summary_path = os.path.join(base_dir, "summary.html")
    chart_path, tooltip_map = generate_summary_echarts_html(
        series1, series2, method_results, summary_path,
        title=f"{ip1} vs {ip2} {field} 对比异常检测汇总图"
    )
    
    # 调用大模型生成详细的分析报告
    print(f"调用大模型生成报告，用户查询: {user_query}")
    llm_analysis = generate_llm_report(result, "multi", user_query)
    print(f"大模型生成的报告长度: {len(llm_analysis)}")
    
    # 生成最终报告
    final_report_path = os.path.join(base_dir, "final_report.html")
    generate_report_html(
        user_query, chart_path, method_results, tooltip_map, final_report_path,
        composite_score=result["composite_score"], 
        classification=result["classification"],
        llm_analysis=llm_analysis,
        is_multi_series=True
    )
    
    # 返回结果
    return {
        "classification": result["classification"],
        "composite_score": result["composite_score"],
        "anomaly_times": result["anomaly_times"],
        "anomaly_intervals": result["anomaly_intervals"],
        "method_results": [mr.to_dict() for mr in method_results],
        "report_path": final_report_path,
        "llm_analysis": llm_analysis
    }


def generate_report_html(user_query, chart_path, detection_results, tooltip_map, output_path, 
                          composite_score=0.0, classification="正常", llm_analysis=None, is_multi_series=False):
    """
    生成异常检测HTML报告
    
    参数:
        user_query: 用户原始查询
        chart_path: 图表HTML路径
        detection_results: 检测结果列表
        tooltip_map: 异常点/区间映射
        output_path: 输出HTML报告路径
        composite_score: 综合得分
        classification: 分类结果
        llm_analysis: 大模型生成的分析报告
        is_multi_series: 是否为多序列分析
    
    返回:
        str: 生成的报告路径
    """
    # 设置报告样式类
    if classification == "高置信度异常":
        alert_class = "alert-danger"
    elif classification == "轻度异常":
        alert_class = "alert-warning"
    else:
        alert_class = "alert-success"
    
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
    
    # 处理LLM分析文本 - 使用更完善的Markdown解析
    formatted_llm_analysis = ""
    if llm_analysis:
        # 使用更完善的Markdown到HTML转换
        formatted_llm_analysis = convert_markdown_to_html(llm_analysis)
    
    report_title = "多序列对比异常检测报告" if is_multi_series else "时序数据异常检测报告"
    
    # 构建完整HTML报告
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{report_title}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; }}
        .iframe-container {{ width: 100%; height: 650px; border: none; }}
        .markdown-content {{ line-height: 1.6; }}
        .markdown-content h3 {{ margin-top: 20px; margin-bottom: 15px; color: #333; font-size: 1.4rem; }}
        .markdown-content h2 {{ margin-top: 25px; margin-bottom: 15px; color: #222; font-size: 1.6rem; }}
        .markdown-content h1 {{ margin-top: 30px; margin-bottom: 20px; color: #111; font-size: 1.8rem; }}
        .markdown-content p {{ margin-bottom: 15px; font-size: 1rem; }}
        .markdown-content strong {{ font-weight: 600; color: #444; }}
        .markdown-content ul, .markdown-content ol {{ margin-bottom: 15px; padding-left: 25px; }}
        .markdown-content li {{ margin-bottom: 5px; }}
        .info-box {{ 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            padding: 10px; 
            margin-bottom: 15px; 
            background-color: #f9f9f9; 
        }}
        .hint-text {{
            font-style: italic;
            color: #666;
            margin-top: 5px;
            font-size: 0.9em;
        }}
        .ai-analysis {{
            background-color: #f8f9fa;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 20px 0;
        }}
        .ai-analysis-title {{
            color: #17a2b8;
            font-weight: 600;
            margin-bottom: 10px;
            font-size: 1.2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mt-4 mb-4">{report_title}</h1>
        
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
                
                <!-- AI分析报告 -->
                <div class="ai-analysis">
                    <div class="ai-analysis-title">AI辅助分析报告</div>
                    <div class="markdown-content">
                        {formatted_llm_analysis if llm_analysis else "未生成AI分析报告"}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header">
                <strong>异常检测图表</strong>
            </div>
            <div class="card-body p-0">
                <div class="info-box">
                    <p class="hint-text">提示: 将鼠标悬停在异常点或异常区间上可查看详细信息。异常点/区间以红色标记并带有编号。</p>
                </div>
                <iframe class="iframe-container" src="{os.path.basename(chart_path)}"></iframe>
            </div>
        </div>
        
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

def get_anomaly_detection_report(result, detection_type):
    """
    根据异常检测结果生成报告
    
    参数:
        result: 异常检测结果
        detection_type: 检测类型 ("single" 或 "multi")
        
    返回:
        str: 格式化的报告文本
    """
    # 使用大模型生成报告
    llm_analysis = generate_llm_report(result, detection_type, result.get("user_query", ""))
    
    # 准备报告标题和基本信息
    if detection_type == "single":
        title = "单序列异常检测报告"
        anomaly_info = f"**异常点数**：{len(result['anomaly_times'])}"
    else:  # multi
        title = "多序列对比异常检测报告"
        anomaly_info = f"**异常段数**：{len(result.get('anomaly_intervals', []))}"
    
    # 生成最终报告
    final_report = f"""
## {title}

**结论**：{result['classification']}  
**综合得分**：{result['composite_score']:.2f}  
{anomaly_info}

{llm_analysis}

**图表路径**：{result['report_path']}
"""
    return final_report