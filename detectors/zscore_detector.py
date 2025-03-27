# 文件: detectors/zscore_detector.py
import statistics

def detect_zscore(series, threshold=3.0):
    """
    使用Z-Score方法全局检测离群点。
    series: list of [ts, val], ts为int秒, val为float
    threshold: 超过多少σ才视为异常,默认为3
    """
    if not series:
        return {
            "method":"Z-Score",
            "anomalies":[],
            "scores":[],
            "description":"无数据"
        }

    vals = [row[1] for row in series]
    mean_val = statistics.mean(vals)
    stdev_val = statistics.pstdev(vals)
    if stdev_val == 0:
        return {
            "method":"Z-Score",
            "anomalies":[],
            "scores":[],
            "description":"方差=0,无显著波动"
        }

    anomalies = []
    scores = []
    for (ts, val) in series:
        z = (val - mean_val)/stdev_val
        if abs(z) > threshold:
            anomalies.append(ts)
            scores.append(abs(z))

    desc = f"Z-Score检测到{len(anomalies)}个> {threshold}σ的异常点." if anomalies else "未发现Z-Score异常"
    return {
        "method":"Z-Score",
        "anomalies": anomalies,
        "scores": scores,
        "description": desc
    }
