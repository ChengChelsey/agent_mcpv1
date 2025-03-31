# output/visualization.py
import json
import uuid
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

def generate_summary_echarts_html(series1, series2=None, detection_results=None, output_path=None, title="时序异常检测汇总图"):
    """
    生成融合所有方法的汇总异常检测可视化图表
    
    参数:
        series1: 第一个时序数据 [(timestamp, value), ...]
        series2: 可选，第二个时序数据，用于多序列对比
        detection_results: 多个检测器结果列表 [DetectionResult, ...]
        output_path: 输出HTML文件路径
        title: 图表标题
    
    返回:
        output_path: 输出的HTML文件路径
        tooltip_map: 异常点映射，用于报告中解释关联
    """
    if detection_results is None:
        detection_results = []
        
    chart_id = f"chart_summary_{uuid.uuid4().hex[:8]}"

    # 主序列数据 - 修改名称和数据格式
    series_list = [{
        "name": series2 is not None and "上周CPU利用率" or "原始序列",
        "type": "line",
        "data": [[t * 1000, v] for t, v in series1],
        "symbolSize": 0,  # 减小正常点的大小
        "lineStyle": {"width": 2},
        "itemStyle": {"color": "#5470C6"}
    }]
    
    # 添加第二个序列（如果存在）
    if series2 is not None:
        # 将两个序列的时间范围对齐
        min_time = min(series1[0][0], series2[0][0])
        offset1 = series1[0][0] - min_time  # 第一个序列相对于较早开始时间的偏移
        offset2 = series2[0][0] - min_time  # 第二个序列相对于较早开始时间的偏移
        
        # 将时间戳调整为相对时间，这样两个序列可以在图表上对齐
        adjusted_series2 = [(t - offset2 + offset1, v) for t, v in series2]
        
        series_list.append({
            "name": "这周CPU利用率",
            "type": "line",
            "data": [[t * 1000, v] for t, v in adjusted_series2],
            "symbolSize": 0,  # 减小正常点的大小
            "lineStyle": {"width": 2},
            "itemStyle": {"color": "#91CC75"}
        })
        
        # 添加差值曲线
        if len(series1) == len(adjusted_series2):
            diff_data = []
            for i in range(len(series1)):
                diff_value = round(series1[i][1] - adjusted_series2[i][1], 3)
                diff_data.append([series1[i][0] * 1000, diff_value])
                
            series_list.append({
                "name": "差值曲线",
                "type": "line",
                "data": diff_data,
                "symbolSize": 0,  # 减小正常点的大小
                "lineStyle": {"width": 1, "type": "dashed"},
                "itemStyle": {"color": "#EE6666"}
            })

    mark_points = []
    mark_areas = []
    extra_series = []
    tooltip_map = {}

    point_counter = 1  # 异常点序号
    explanation_counter = 1  # 解释编号

    for result in detection_results:
        # 处理点异常
        if result.visual_type == "point":
            for i, ts in enumerate(result.anomalies):
                value = next((v for t, v in series1 if t == ts), None)
                if value is None:
                    continue
                
                # 构建标签和提示信息
                explanation = ""
                if i < len(result.explanation):
                    explanation = result.explanation[i]
                
                # 存储对应关系，用于报告生成
                tooltip_map[point_counter] = {
                    "method": result.method,
                    "ts": ts,
                    "value": value,
                    "explanation": explanation
                }
                explanation_counter += 1
                
                # 添加标记点，将异常信息存储在数据中供自定义tooltip使用
                mark_points.append({
                    "coord": [ts * 1000, value],
                    "symbol": "circle",
                    "symbolSize": 8,
                    "itemStyle": {"color": "red"},
                    "label": {"formatter": f"#{point_counter}", "show": True, "position": "top"},
                    "anomalyInfo": {
                        "id": point_counter,
                        "method": result.method,
                        "timestamp": format_timestamp(ts),
                        "value": value,
                        "explanation": explanation
                    }
                })
                point_counter += 1

        # 处理区间异常
        if result.visual_type in ("range", "curve"):
            for i, (start, end) in enumerate(result.intervals):
                # 获取区间解释
                area_explanation = ""
                if i < len(result.explanation):
                    area_explanation = result.explanation[i]
                
                # 存储对应关系
                tooltip_map[point_counter] = {
                    "method": result.method,
                    "ts_start": start,
                    "ts_end": end,
                    "explanation": area_explanation
                }
                
                # 添加标记区域，将异常信息存储在数据中供自定义tooltip使用
                mark_areas.append({
                    "itemStyle": {"color": "rgba(255, 100, 100, 0.2)"},
                    "label": {"show": True, "position": "top", "formatter": f"#{point_counter}"},
                    "xAxis": start * 1000,
                    "xAxis2": end * 1000,
                    "anomalyInfo": {
                        "id": point_counter,
                        "method": result.method,
                        "startTime": format_timestamp(start),
                        "endTime": format_timestamp(end),
                        "explanation": area_explanation
                    }
                })
                
                point_counter += 1
                explanation_counter += 1

        # 处理辅助曲线 - 使用第二个y轴
        if result.visual_type == "curve" and result.auxiliary_curve:
            # 检查方法名来确定是否使用第二个Y轴
            use_second_yaxis = result.method in ["CUSUM", "TrendDriftCUSUM"]
            yAxisIndex = 1 if use_second_yaxis else 0
            
            curve_data = [[t * 1000, v] for t, v in result.auxiliary_curve]
            extra_series.append({
                "name": f"{result.method} 辅助曲线",
                "type": "line",
                "yAxisIndex": yAxisIndex,  # 使用右侧Y轴
                "data": curve_data,
                "lineStyle": {"type": "dashed", "width": 1.5},
                "itemStyle": {"color": "#EE6666"},
                "showSymbol": False
            })

    # 合并所有系列
    series_list.extend(extra_series)
    
    # 检查是否需要第二个Y轴
    need_second_yaxis = any(s.get("yAxisIndex", 0) == 1 for s in series_list)

    # 构建 ECharts 选项
    option = {
        "title": {"text": title, "left": "center"},
        "tooltip": {
            "trigger": "axis", 
            "axisPointer": {"type": "cross"}, 
            "confine": True,
            "backgroundColor": "rgba(255, 255, 255, 0.9)",
            "borderColor": "#ccc",
            "borderWidth": 1,
            "textStyle": {
                "color": "#333"
            },
            "extraCssText": "box-shadow: 0 0 8px rgba(0, 0, 0, 0.1);"
        },
        "legend": {"top": 30, "data": [s["name"] for s in series_list]},
        "grid": {"left": "3%", "right": need_second_yaxis and "8%" or "4%", "bottom": "8%", "containLabel": True},
        "toolbox": {
            "feature": {
                "saveAsImage": {},
                "dataZoom": {},
                "restore": {}
            }
        },
        "xAxis": {
            "type": "time",
            "name": "时间",
            "axisLabel": {
                "formatter": "{yyyy}-{MM}-{dd} {HH}:{mm}",
                "rotate": 30,  # 旋转标签以避免重叠
                "margin": 15   # 增加边距
            }
        },
        "yAxis": [
            {"type": "value", "name": "数值", "position": "left"}
        ],
        "series": series_list,
        "dataZoom": [
            {"type": "slider", "show": True, "xAxisIndex": [0], "start": 0, "end": 100, "bottom": 10},
            {"type": "inside", "xAxisIndex": [0], "start": 0, "end": 100}
        ]
    }
    
    # 添加第二个Y轴（如果需要）
    if need_second_yaxis:
        option["yAxis"].append({
            "type": "value",
            "name": "辅助曲线值",
            "position": "right",
            "splitLine": {"show": False}
        })

    # 整合标记点和标记区域
    if mark_points:
        series_list[0]["markPoint"] = {
            "data": mark_points, 
            "symbolSize": 8,
            "emphasis": {"scale": True},
            "label": {"show": True}
        }
    
    if mark_areas:
        series_list[0]["markArea"] = {
            "data": [[{"xAxis": area["xAxis"], "itemStyle": area["itemStyle"], "anomalyInfo": area["anomalyInfo"]}, 
                     {"xAxis": area["xAxis2"]}] for area in mark_areas],
            "emphasis": {"focus": "self"}
        }

    # 生成HTML，添加自定义JavaScript处理异常点的tooltip
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #container {{ width: 100%; height: 100vh; }}
        #{chart_id} {{ width: 100%; height: 650px; }}
        
        /* 自定义异常点提示样式 */
        .anomaly-tooltip {{
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 10px;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.1);
            font-family: Arial, sans-serif;
            font-size: 14px;
            color: #333;
            max-width: 300px;
        }}
        .anomaly-tooltip-title {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #d23;
        }}
        .anomaly-tooltip-item {{
            display: flex;
            margin: 2px 0;
        }}
        .anomaly-tooltip-label {{
            font-weight: bold;
            margin-right: 8px;
            color: #666;
            min-width: 60px;
        }}
        .anomaly-tooltip-value {{
            color: #333;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="{chart_id}"></div>
    </div>
    <script>
        var chart = echarts.init(document.getElementById('{chart_id}'));
        var option = {json.dumps(option, ensure_ascii=False)};
        
        // 自定义tooltip处理函数
        chart.on('mouseover', function(params) {{
            if (params.componentType === 'markPoint' && params.data && params.data.anomalyInfo) {{
                var info = params.data.anomalyInfo;
                var content = 
                    '<div class="anomaly-tooltip">' +
                    '<div class="anomaly-tooltip-title">异常点 #' + info.id + '</div>' +
                    '<div class="anomaly-tooltip-item"><span class="anomaly-tooltip-label">方法:</span><span class="anomaly-tooltip-value">' + info.method + '</span></div>' + 
                    '<div class="anomaly-tooltip-item"><span class="anomaly-tooltip-label">时间:</span><span class="anomaly-tooltip-value">' + info.timestamp + '</span></div>' + 
                    '<div class="anomaly-tooltip-item"><span class="anomaly-tooltip-label">数值:</span><span class="anomaly-tooltip-value">' + info.value.toFixed(2) + '</span></div>' + 
                    '<div class="anomaly-tooltip-item"><span class="anomaly-tooltip-label">说明:</span><span class="anomaly-tooltip-value">' + info.explanation + '</span></div>' +
                    '</div>';
                
                chart.setOption({{
                    tooltip: {{
                        formatter: content,
                        extraCssText: 'padding: 0; border: none; background: transparent;'
                    }}
                }});
            }}
            else if (params.componentType === 'markArea' && params.data && params.data[0] && params.data[0].anomalyInfo) {{
                var info = params.data[0].anomalyInfo;
                var content = 
                    '<div class="anomaly-tooltip">' +
                    '<div class="anomaly-tooltip-title">异常区间 #' + info.id + '</div>' +
                    '<div class="anomaly-tooltip-item"><span class="anomaly-tooltip-label">方法:</span><span class="anomaly-tooltip-value">' + info.method + '</span></div>' + 
                    '<div class="anomaly-tooltip-item"><span class="anomaly-tooltip-label">开始:</span><span class="anomaly-tooltip-value">' + info.startTime + '</span></div>' + 
                    '<div class="anomaly-tooltip-item"><span class="anomaly-tooltip-label">结束:</span><span class="anomaly-tooltip-value">' + info.endTime + '</span></div>' + 
                    '<div class="anomaly-tooltip-item"><span class="anomaly-tooltip-label">说明:</span><span class="anomaly-tooltip-value">' + info.explanation + '</span></div>' +
                    '</div>';
                
                chart.setOption({{
                    tooltip: {{
                        formatter: content,
                        extraCssText: 'padding: 0; border: none; background: transparent;'
                    }}
                }});
            }}
        }});
        
        chart.on('mouseout', function(params) {{
            if (params.componentType === 'markPoint' || params.componentType === 'markArea') {{
                // 恢复默认tooltip
                chart.setOption({{
                    tooltip: {{
                        formatter: null,
                        extraCssText: option.tooltip.extraCssText
                    }}
                }});
            }}
        }});
        
        chart.setOption(option);
        window.addEventListener('resize', function() {{
            chart.resize();
        }});
    </script>
</body>
</html>"""

    # 确保输出目录存在
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path, tooltip_map
    else:
        return html, tooltip_map

def format_timestamp(ts):
    """将时间戳格式化为可读时间"""
    from datetime import datetime
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(ts)

# 向后兼容函数
def generate_echarts_html_single(series, anomalies, title="单序列异常检测"):
    """
    向后兼容函数 - 生成单序列异常检测图表
    """
    from detectors.base import DetectionResult
    import time
    
    # 将旧格式转换为检测结果对象
    result = DetectionResult(
        method="Legacy",
        anomalies=anomalies,
        description="旧版接口生成的异常检测图表",
        visual_type="point"
    )
    
    path = f"output/plots/legacy_single_{int(time.time())}.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    chart_path, _ = generate_summary_echarts_html(
        series, None, [result], path, title
    )
    
    return chart_path

def generate_echarts_html_multi(series1, series2, anomalies, title="多序列对比异常检测"):
    """
    向后兼容函数 - 生成多序列对比异常检测图表
    """
    from detectors.base import DetectionResult
    import time
    
    # 将旧格式转换为检测结果对象
    result = DetectionResult(
        method="Legacy",
        anomalies=anomalies,
        description="旧版接口生成的异常检测图表",
        visual_type="point"
    )
    
    path = f"output/plots/legacy_multi_{int(time.time())}.html"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    chart_path, _ = generate_summary_echarts_html(
        series1, series2, [result], path, title
    )
    
    return chart_path