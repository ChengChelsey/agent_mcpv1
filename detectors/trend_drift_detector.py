# 文件: detectors/trend_drift_detector.py

def detect_trend_drift(series1, series2, drift_threshold=5.0):
    """
    将两序列做差 => CUSUM => 看是否有趋势漂移
    """
    if not series1 or not series2:
        return {
            "method":"TrendDriftCUSUM",
            "anomalies":[],
            "scores":[],
            "description":"无数据"
        }
    # 同样先对齐
    dict2 = {row[0]: row[1] for row in series2}
    diffs=[]
    ts_list=[]
    for (ts,v1) in series1:
        if ts in dict2:
            diffs.append(v1 - dict2[ts])
            ts_list.append(ts)
    if not diffs:
        return {
            "method":"TrendDriftCUSUM",
            "anomalies":[],
            "scores":[],
            "description":"序列无对齐"
        }
    pos_sum=0.0
    neg_sum=0.0
    anomalies=[]
    scores=[]
    for i, d in enumerate(diffs):
        pos_sum = max(0, pos_sum+d)
        neg_sum = min(0, neg_sum+d)
        if pos_sum>drift_threshold or neg_sum<-drift_threshold:
            anomalies.append(ts_list[i])
            scores.append(abs(pos_sum) if abs(pos_sum)>abs(neg_sum) else abs(neg_sum))
            pos_sum=0
            neg_sum=0
    desc = f"趋势漂移检测到{len(anomalies)}处超阈值" if anomalies else "未见明显趋势漂移"
    return {
        "method":"TrendDriftCUSUM",
        "anomalies":anomalies,
        "scores":scores,
        "description":desc
    }
