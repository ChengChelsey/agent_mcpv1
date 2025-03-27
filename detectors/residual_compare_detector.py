# 文件: detectors/residual_compare_detector.py

import statistics

def detect_residual_compare(series1, series2, threshold=3.0):
    """
    两序列做差 => 分析(离群).
    (可认为是"残差对比",非常简化)
    threshold: 超过mean±threshold*stdev则异常
    """
    if not series1 or not series2:
        return {
            "method":"ResidualComparison",
            "anomalies":[],
            "scores":[],
            "description":"无数据或序列为空"
        }
    # 要对齐
    # (若multi_series里已对齐,这里只简单按ts查找)
    dict2 = {row[0]: row[1] for row in series2}
    diffs = []
    common_ts = []
    for (ts, v1) in series1:
        if ts in dict2:
            diffs.append(v1 - dict2[ts])
            common_ts.append(ts)
    if not diffs:
        return {
            "method":"ResidualComparison",
            "anomalies":[],
            "scores":[],
            "description":"两序列无重叠时间戳"
        }
    m = statistics.mean(diffs)
    s = statistics.pstdev(diffs) if len(diffs)>1 else 0
    if s==0:
        return {
            "method":"ResidualComparison",
            "anomalies":[],
            "scores":[],
            "description":"残差无波动"
        }
    anomalies = []
    scores = []
    for i, d in enumerate(diffs):
        if abs(d-m) > threshold*s:
            anomalies.append(common_ts[i])
            scores.append(abs(d-m)/s)
    desc = f"残差比较发现{len(anomalies)}个离群点" if anomalies else "未见残差异常"
    return {
        "method":"ResidualComparison",
        "anomalies":anomalies,
        "scores":scores,
        "description":desc
    }
