# 文件: analysis/multi_series.py

import config
from detectors import residual_compare_detector, trend_drift_detector, change_rate_detector, trend_slope_detector
from analysis.data_alignment import align_series

def group_anomaly_times(anomalies, max_gap=1800):
    if not anomalies:
        return []
    intervals = []
    cur_start = anomalies[0]
    cur_end   = anomalies[0]
    for t in anomalies[1:]:
        if t - cur_end <= max_gap:
            cur_end = t
        else:
            intervals.append((cur_start, cur_end))
            cur_start = t
            cur_end   = t

    intervals.append((cur_start, cur_end))
    return intervals


def analyze_multi_series(series1, series2, align=True):
    #先插值对齐(align_series)再检测。

    if align:
        series1, series2 = align_series(series1, series2, method="linear", fill_value="extrapolate")

    res_residual = residual_compare_detector.detect_residual_compare(series1, series2)
    res_drift    = trend_drift_detector.detect_trend_drift(series1, series2)
    res_change   = change_rate_detector.detect_change_rate(series1, series2)
    res_slope    = trend_slope_detector.detect_trend_slope(series1, series2)

    method_results = [res_residual, res_drift, res_change, res_slope]

    #加权
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

    all_anoms = set()
    for r in method_results:
        for ts in r["anomalies"]:
            all_anoms.add(ts)
            
    anomaly_list = sorted(all_anoms)
    intervals = group_anomaly_times(anomaly_list, max_gap=1800)

    return {
        "method_results": method_results,
        "composite_score": composite_score,
        "classification": classification,
        "anomaly_times": anomaly_list,       
        "anomaly_intervals": intervals       
    }
