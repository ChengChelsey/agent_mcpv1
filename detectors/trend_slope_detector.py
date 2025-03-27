# 文件: detectors/trend_slope_detector.py

def detect_trend_slope(series1, series2, window=5, slope_threshold=0.5):
    """
    滑动窗口对比两个序列的局部斜率差异
    window: 窗口大小(数据点数)
    slope_threshold: 当斜率比差距大于此阈值则异常
    """
    if not series1 or not series2:
        return {
            "method":"TrendSlope",
            "anomalies":[],
            "scores":[],
            "description":"数据不足"
        }
    dict2 = {row[0]:row[1] for row in series2}
    s1_sorted = sorted(series1,key=lambda x:x[0])

    anomalies=[]
    scores=[]
    # 用window做局部斜率
    for i in range(len(s1_sorted)-window+1):
        segment1 = s1_sorted[i:i+window]
        # 同步 segment2
        seg2 = []
        for (ts,v1) in segment1:
            if ts in dict2:
                seg2.append((ts, dict2[ts]))
        if len(seg2)<2:
            continue
        # slope1
        slope1 = (segment1[-1][1]-segment1[0][1])/(window-1)
        slope2 = (seg2[-1][1]-seg2[0][1])/(len(seg2)-1)
        ratio = 0.0
        if abs(slope1)>1e-9 and abs(slope2)>1e-9:
            ratio = abs(slope1-slope2)/max(abs(slope1),abs(slope2))
        if ratio> slope_threshold:
            anomalies.append(segment1[-1][0])
            scores.append(ratio)
    desc = f"趋势斜率对比发现{len(anomalies)}处异常" if anomalies else "趋势斜率基本一致"
    return {
        "method":"TrendSlope",
        "anomalies":anomalies,
        "scores":scores,
        "description":desc
    }
