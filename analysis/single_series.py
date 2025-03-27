# 文件: analysis/single_series.py

import config
from detectors import zscore_detector, cusum_detector, stl_detector

def analyze_single_series(series):
    """
    对单序列时序数据进行异常检测，调用Z-Score、CUSUM、STL等方法并融合结果。
    series: list of [int_ts, float_val], 已按时间排序或未排序皆可(内部方法会自行处理)
    
    返回 dict:
    {
      "method_results": [
         { "method":"Z-Score", "anomalies":[...], "scores":[...], "description":"..." },
         { "method":"CUSUM",   ... },
         { "method":"STL",     ... },
      ],
      "composite_score": 0.55,
      "classification": "轻度异常",
      "anomaly_times": [...]
    }
    """
    # 分别调用三个检测器
    res_z     = zscore_detector.detect_zscore(series)
    res_cusum = cusum_detector.detect_cusum(series)
    res_stl   = stl_detector.detect_stl_residual(series)

    method_results = [res_z, res_cusum, res_stl]

    # 计算加权综合评分
    total_weight = 0.0
    composite_score = 0.0
    length = len(series) if series else 1

    for res in method_results:
        m_name = res["method"]
        weight = config.WEIGHTS_SINGLE.get(m_name, 0.0)
        total_weight += weight

        anomalies_count = len(res["anomalies"])
        method_score = anomalies_count / length
        # 让方法分数最小值0.1 => 避免特别小分数被忽视
        if 0 < method_score < 0.1:
            method_score = 0.1

        composite_score += weight * method_score

    if total_weight > 0:
        composite_score /= total_weight

    # 根据score判断分类
    if composite_score >= config.HIGH_ANOMALY_THRESHOLD:
        classification = "高置信度异常"
    elif composite_score >= config.MILD_ANOMALY_THRESHOLD:
        classification = "轻度异常"
    else:
        classification = "正常"

    # 合并 anomalies
    all_anomalies = set()
    for r in method_results:
        for ts in r["anomalies"]:
            all_anomalies.add(ts)

    return {
        "method_results": method_results,
        "composite_score": composite_score,
        "classification": classification,
        "anomaly_times": sorted(all_anomalies)
    }
