# config.py

WEIGHTS_SINGLE = {
    "Z-Score": 0.3,
    "CUSUM":   0.3,
    "STL":     0.4
}

WEIGHTS_MULTI = {
    "ResidualComparison": 0.3,
    "TrendDriftCUSUM":    0.3,
    "ChangeRate":         0.2,
    "TrendSlope":         0.2
}

# 阈值：决定综合分数达到多少算高置信度异常
HIGH_ANOMALY_THRESHOLD = 0.7
MILD_ANOMALY_THRESHOLD = 0.4
