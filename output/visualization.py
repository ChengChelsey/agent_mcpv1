# output/visualization.py
import json

def generate_echarts_html_single(series, anomalies=None):
    """
    series: [ [int_ts, val], ... ]
    anomalies: [ t_int, ... ]
    """
    if anomalies is None:
        anomalies=[]
    data_points = []
    anomaly_points=[]
    for (t,v) in series:
        data_points.append([t*1000, v])
        if t in anomalies:
            anomaly_points.append({"coord":[t*1000,v], "name":"异常"})

    option = {
        "title":{"text":"单序列折线图"},
        "tooltip":{"trigger":"axis"},
        "xAxis":{"type":"time"},
        "yAxis":{"type":"value"},
        "series":[
            {
              "name":"值",
              "type":"line",
              "data": data_points,
              "markPoint":{"data": anomaly_points}
            }
        ]
    }
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Single Series ECharts</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js"></script>
</head>
<body>
<div id="chart" style="width:800px; height:400px;"></div>
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
    """
    两条线
    series1, series2 => [ [t,val], ...]
    anomalies => union of anomaly times
    """
    if anomalies is None:
        anomalies=[]
    # prepare
    data1=[]
    data2=[]
    mark_points=[]
    for (t,v) in series1:
        data1.append([t*1000, v])
        if t in anomalies:
            mark_points.append({"coord":[t*1000, v], "name":"异常"})
    for (t,v) in series2:
        data2.append([t*1000, v])
        if t in anomalies:
            mark_points.append({"coord":[t*1000, v], "name":"异常"})

    option = {
        "title":{"text":"多序列对比折线图"},
        "tooltip":{"trigger":"axis"},
        "legend":{"data":["序列1","序列2"]},
        "xAxis":{"type":"time"},
        "yAxis":{"type":"value"},
        "series":[
            {
              "name":"序列1",
              "type":"line",
              "data":data1,
              "markPoint":{"data": mark_points}
            },
            {
              "name":"序列2",
              "type":"line",
              "data":data2,
              "markPoint":{"data": mark_points}
            }
        ]
    }
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Multi Series ECharts</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js"></script>
</head>
<body>
<div id="chart" style="width:800px; height:400px;"></div>
<script>
var chart = echarts.init(document.getElementById('chart'));
var option = {json.dumps(option, ensure_ascii=False)};
chart.setOption(option);
</script>
</body>
</html>
"""
    return html
