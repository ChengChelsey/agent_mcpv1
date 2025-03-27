# 文件: detectors/cusum_detector.py

def detect_cusum(series, drift_threshold=5.0):
    """
    使用CUSUM算法检测趋势漂移.
    drift_threshold: 超过该累计偏差则视为异常

    series: list of [ts, val]
    return: { "method":"CUSUM", "anomalies":[...], "scores":[...], "description":... }
    """
    if not series:
        return {
            "method":"CUSUM",
            "anomalies":[],
            "scores":[],
            "description":"无数据"
        }
    vals = [row[1] for row in series]
    mean_val = sum(vals)/len(vals)
    pos_sum = 0.0
    neg_sum = 0.0
    anomalies = []
    scores = []
    for i, (ts, val) in enumerate(series):
        diff = val - mean_val
        pos_sum = max(0, pos_sum+diff)
        neg_sum = min(0, neg_sum+diff)
        # 超过阈值 => 异常
        if pos_sum > drift_threshold or neg_sum < -drift_threshold:
            anomalies.append(ts)
            score = abs(pos_sum) if abs(pos_sum)>abs(neg_sum) else abs(neg_sum)
            scores.append(score)
            pos_sum = 0
            neg_sum = 0
    desc = f"CUSUM检测到{len(anomalies)}个疑似趋势漂移异常." if anomalies else "未见CUSUM异常"
    return {
        "method":"CUSUM",
        "anomalies": anomalies,
        "scores": scores,
        "description": desc
    }
