# output/visualization.py
import json
import uuid
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union
from utils.time_utils import format_timestamp

def process_anomaly_points(detection_results, series_data, timestamps):
    
    mark_points = []
    tooltip_map = {}
    point_counter = 1
    
    for result in detection_results:
        if not hasattr(result, 'visual_type') or result.visual_type != "point":
            continue
            
        anomalies = result.anomalies if hasattr(result, 'anomalies') else []
        explanations = result.explanation if hasattr(result, 'explanation') else []
        
        for i, ts in enumerate(anomalies):
            value = next((v for t, v in series_data if t == ts), None)
            if value is None:
                continue

            explanation = ""
            if i < len(explanations):
                explanation = explanations[i]
            
            tooltip_map[point_counter] = {
                "method": result.method if hasattr(result, 'method') else "未知方法",
                "ts": ts,
                "value": value,
                "explanation": explanation
            }
            
            mark_points.append({
                "coord": [ts * 1000, value],
                "symbol": "circle",
                "symbolSize": 8,
                "itemStyle": {"color": "red"},
                "label": {"formatter": f"#{point_counter}", "show": True, "position": "top"},
                "anomalyInfo": {
                    "id": point_counter,
                    "method": result.method if hasattr(result, 'method') else "未知方法",
                    "timestamp": format_timestamp(ts),
                    "value": value,
                    "explanation": explanation
                }
            })
            
            point_counter += 1
            
    return mark_points, tooltip_map

def process_anomaly_ranges(detection_results, timestamps):
    
    mark_areas = []
    tooltip_map = {}
    point_counter = 1
    
    for result in detection_results:
        if not hasattr(result, 'visual_type') or result.visual_type not in ("range", "curve"):
            continue
            
        intervals = result.intervals if hasattr(result, 'intervals') else []
        explanations = result.explanation if hasattr(result, 'explanation') else []
        
        for i, (start, end) in enumerate(intervals):
            area_explanation = ""
            if i < len(explanations):
                area_explanation = explanations[i]
                
            tooltip_map[point_counter] = {
                "method": result.method if hasattr(result, 'method') else "未知方法",
                "ts_start": start,
                "ts_end": end,
                "explanation": area_explanation
            }
            
            mark_areas.append({
                "itemStyle": {"color": "rgba(255, 100, 100, 0.2)"},
                "label": {"show": True, "position": "top", "formatter": f"#{point_counter}"},
                "xAxis": start * 1000,
                "xAxis2": end * 1000,
                "anomalyInfo": {
                    "id": point_counter,
                    "method": result.method if hasattr(result, 'method') else "未知方法",
                    "startTime": format_timestamp(start),
                    "endTime": format_timestamp(end),
                    "explanation": area_explanation
                }
            })
            
            point_counter += 1
            
    return mark_areas, tooltip_map

def process_auxiliary_curves(detection_results, timestamps):
    #处理辅助曲线数据
    
    extra_series = []
    for result in detection_results:
        # 跳过非曲线类型或没有辅助曲线的结果
        if not hasattr(result, 'visual_type') or result.visual_type != "curve":
            continue
        if not hasattr(result, 'auxiliary_curve') or not result.auxiliary_curve:
            continue
    
        #是否使用第二个Y轴
        use_second_yaxis = False
        if hasattr(result, 'method'):
            use_second_yaxis = result.method in ["CUSUM", "TrendDriftCUSUM"]
            
        yAxisIndex = 1 if use_second_yaxis else 0
        curve_data = [[t * 1000, v] for t, v in result.auxiliary_curve]
        
        extra_series.append({
            "name": f"{result.method if hasattr(result, 'method') else '辅助'} 辅助曲线",
            "type": "line",
            "yAxisIndex": yAxisIndex,
            "data": curve_data,
            "lineStyle": {"type": "dashed", "width": 1.5},
            "itemStyle": {"color": "#EE6666"},
            "showSymbol": False
        })
            
    return extra_series

def prepare_series_data(series1, series2=None):
   
    series_list = [{
        "name": series2 is not None and "上周CPU利用率" or "原始序列",
        "type": "line",
        "data": [[t * 1000, v] for t, v in series1],
        "symbolSize": 0,  # 减小正常点的大小
        "lineStyle": {"width": 2},
        "itemStyle": {"color": "#5470C6"}
    }]

    if series2 is not None:
        #时间范围对齐
        min_time = min(series1[0][0], series2[0][0])
        offset1 = series1[0][0] - min_time 
        offset2 = series2[0][0] - min_time
        
        adjusted_series2 = [(t - offset2 + offset1, v) for t, v in series2]
        
        series_list.append({
            "name": "这周CPU利用率",
            "type": "line",
            "data": [[t * 1000, v] for t, v in adjusted_series2],
            "symbolSize": 0,
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
                "symbolSize": 0, 
                "lineStyle": {"width": 1, "type": "dashed"},
                "itemStyle": {"color": "#EE6666"}
            })
    
    return series_list

def generate_summary_echarts_html(series1, series2=None, detection_results=None, output_path=None, title="时序异常检测汇总图"):
    
    if detection_results is None:
        detection_results = []
        
    chart_id = f"chart_summary_{uuid.uuid4().hex[:8]}"
    timestamps = [t for t, _ in series1]
    
    series_list = prepare_series_data(series1, series2)
    
    mark_points, tooltip_map_points = process_anomaly_points(detection_results, series1, timestamps)
    mark_areas, tooltip_map_areas = process_anomaly_ranges(detection_results, timestamps)
    extra_series = process_auxiliary_curves(detection_results, timestamps)
    
    series_list.extend(extra_series)
    
    tooltip_map = {**tooltip_map_points, **tooltip_map_areas}
    
    need_second_yaxis = any(s.get("yAxisIndex", 0) == 1 for s in series_list)

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
                "rotate": 30,
                "margin": 15
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
    
    if need_second_yaxis:
        option["yAxis"].append({
            "type": "value",
            "name": "辅助曲线值",
            "position": "right",
            "splitLine": {"show": False}
        })

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

    #异常点的tooltip
    html = generate_html_template(chart_id, option, title)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path, tooltip_map
    else:
        return html, tooltip_map

def generate_html_template(chart_id, option, title):
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #container {{ width: 100%; height: 100vh; }}
        #{chart_id} {{ width: 100%; height: 650px; }}
        
        /* 异常点提示样式 */
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
                //恢复默认tooltip
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