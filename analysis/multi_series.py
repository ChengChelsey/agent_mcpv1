# 文件: analysis/multi_series.py

import config
from detectors import residual_compare_detector, trend_drift_detector, change_rate_detector, trend_slope_detector
from analysis.data_alignment import align_series

def analyze_multi_series(series1, series2, align=True):
    """
    对两组时序进行多方法对比异常检测；可在内部先插值对齐(align_series)再检测。
    series1, series2: list of [int_ts, float_val]
    align: bool, 是否先做插值对齐
    
    返回 dict:
    {
      "method_results": [...],
      "composite_score": 0.65,
      "classification": "轻度异常",
      "anomaly_times": [...]
    }
    """
    # 1) 先可选插值对齐
    if align:
        series1, series2 = align_series(series1, series2, method="linear", fill_value="extrapolate")

    # 2) 调用多种方法
    res_residual = residual_compare_detector.detect_residual_compare(series1, series2)
    res_drift    = trend_drift_detector.detect_trend_drift(series1, series2)
    res_change   = change_rate_detector.detect_change_rate(series1, series2)
    res_slope    = trend_slope_detector.detect_trend_slope(series1, series2)

    method_results = [res_residual, res_drift, res_change, res_slope]

    # 3) 计算加权综合评分
    total_weight = 0.0
    composite_score = 0.0
    length = max(len(series1), len(series2)) or 1

    for res in method_results:
        m_name = res["method"]
        weight = config.WEIGHTS_MULTI.get(m_name, 0.0)
        total_weight += weight

        anomalies_count = len(res["anomalies"])
        method_score = anomalies_count / length
        if 0 < method_score < 0.1:
            method_score = 0.1

        composite_score += weight * method_score

    if total_weight > 0:
        composite_score /= total_weight

    if composite_score >= config.HIGH_ANOMALY_THRESHOLD:
        classification = "高置信度异常"
    elif composite_score >= config.MILD_ANOMALY_THRESHOLD:
        classification = "轻度异常"
    else:
        classification = "正常"

    # 合并 anomalies
    all_anoms = set()
    for r in method_results:
        for ts in r["anomalies"]:
            all_anoms.add(ts)

    return {
        "method_results": method_results,
        "composite_score": composite_score,
        "classification": classification,
        "anomaly_times": sorted(all_anoms)
    }
