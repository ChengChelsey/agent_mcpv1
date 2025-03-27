# 文件: detectors/stl_detector.py

import math
try:
    from statsmodels.tsa.seasonal import STL
except ImportError:
    STL = None

def detect_stl_residual(series, period=None):
    """
    使用STL分解（若可用），分析残差异常。
    series: list of [ts, val]
    period: 季节期(若无则可设None)
    """
    if not series:
        return {
            "method":"STL",
            "anomalies":[],
            "scores":[],
            "description":"无数据"
        }
    values = [row[1] for row in series]
    n = len(values)
    if n < 3:
        return {
            "method":"STL",
            "anomalies":[],
            "scores":[],
            "description":"数据点太少"
        }
    # 如 period 未指定,可默认=7等. 这里仅演示:
    if not period:
        period = 24  # 如果是一天周期之类

    anomalies = []
    scores = []

    # 若statsmodels没装或period不合理 => 做个简化
    if STL is None or period>=n:
        # 简单移动平均
        window = max(1, min(n//5,10))
        trend = []
        for i in range(n):
            left_i = max(0,i-window)
            right_i= min(n,i+window+1)
            sub= values[left_i:right_i]
            trend_val = sum(sub)/len(sub)
            trend.append(trend_val)
        resid = [v - t for v,t in zip(values,trend)]
    else:
        stl = STL(values, period=period, robust=True)
        res = stl.fit()
        resid = res.resid

    # 分析 resid => zscore
    from statistics import mean, pstdev
    m = mean(resid)
    s = pstdev(resid)
    if s==0:
        return {
            "method":"STL",
            "anomalies":[],
            "scores":[],
            "description":"STL残差无波动"
        }
    threshold = 3
    for i, r_val in enumerate(resid):
        if abs(r_val-m)>threshold*s:
            anomalies.append(series[i][0])
            scores.append(abs(r_val-m)/s - threshold)
    desc = f"STL分解残差发现{len(anomalies)}个>3σ异常点" if anomalies else "未见STL异常"
    return {
        "method":"STL",
        "anomalies": anomalies,
        "scores": scores,
        "description": desc
    }
