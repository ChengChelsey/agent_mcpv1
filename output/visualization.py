import json

def generate_echarts_html_single(series, anomalies=None):
    if anomalies is None:
        anomalies = []

    data_points = []
    anomaly_points = []
    for (t, v) in series:
        data_points.append([t * 1000, v])
        if t in anomalies:
            anomaly_points.append({
                "coord": [t * 1000, v],
                "symbol": "triangle",
                "symbolSize": 6,
                "itemStyle": {"color": "red"},
                "name": "异常"
            })

    option = {
        "title": {"text": "CPU Comparison", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "time",
            "name": "Time (Hours)"
        },
        "yAxis": {
            "type": "value",
            "name": "Value"
        },
        "legend": {
            "data": ["序列"],
            "top": 30,
            "left": "center"
        },
        "series": [
            {
                "name": "序列",
                "type": "line",
                "data": data_points,
                "symbolSize": 4,
                "lineStyle": {"width": 2},
                "itemStyle": {"color": "#5470C6"},
                "markPoint": {
                    "data": anomaly_points
                }
            }
        ]
    }

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\">
    <title>Single Series ECharts</title>
    <script src=\"https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js\"></script>
</head>
<body>
<div id=\"chart\" style=\"width:1000px; height:500px;\"></div>
<script>
var chart = echarts.init(document.getElementById('chart'));
var option = {json.dumps(option, ensure_ascii=False)};
chart.setOption(option);
</script>
</body>
</html>
"""
    return html

def generate_echarts_html_multi(series1, series2, anomalies=None):
    if anomalies is None:
        anomalies = []

    data1 = []
    data2 = []
    diff = []
    anomaly_marks = []

    for i in range(len(series1)):
        t, v1 = series1[i]
        _, v2 = series2[i]
        ts = t * 1000
        data1.append([ts, v1])
        data2.append([ts, v2])
        diff.append([ts, v1 - v2])
        if t in anomalies:
            anomaly_marks.append({
                "coord": [ts, v1],
                "symbol": "circle",
                "symbolSize": 3,
                "itemStyle": {"color": "red"},
                "name": "异常"
            })

    option = {
        "title": {"text": "CPU Comparison", "left": "center"},
        "tooltip": {"trigger": "axis"},
        "legend": {
            "data": ["序列1", "序列2", "差值(序列1 - 序列2)"],
            "top": 30,
            "left": "center"
        },
        "xAxis": {
            "type": "time",
            "name": "Time (Hours)"
        },
        "yAxis": {
            "type": "value",
            "name": "Value / Difference"
        },
        "series": [
            {
                "name": "序列1",
                "type": "line",
                "data": data1,
                "symbolSize": 4,
                "lineStyle": {"width": 2, "color": "#5470C6"},
                "itemStyle": {"color": "#5470C6"},
                "markPoint": {"data": anomaly_marks}
            },
            {
                "name": "序列2",
                "type": "line",
                "data": data2,
                "symbolSize": 4,
                "lineStyle": {"width": 2, "type": "dashed", "color": "#91CC75"},
                "itemStyle": {"color": "#91CC75"},
                "markPoint": {"data": anomaly_marks}
            },
            {
                "name": "差值(序列1 - 序列2)",
                "type": "line",
                "data": diff,
                "symbol": "none",
                "lineStyle": {"width": 2, "color": "#73C0DE"},
                "itemStyle": {"color": "#73C0DE"}
            }
        ]
    }

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\">
    <title>Multi Series ECharts</title>
    <script src=\"https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js\"></script>
</head>
<body>
<div id=\"chart\" style=\"width:1000px; height:500px;\"></div>
<script>
var chart = echarts.init(document.getElementById('chart'));
var option = {json.dumps(option, ensure_ascii=False)};
chart.setOption(option);
</script>
</body>
</html>
"""
    return html
