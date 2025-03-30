import numpy as np
from detectors.base import DetectionResult

class ChangeRateDetector:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def detect(self, series1: list[tuple[int, float]], series2: list[tuple[int, float]]) -> DetectionResult:
        timestamps = [t for t, _ in series1]
        rate_diffs = []

        for i in range(1, len(series1)):
            delta1 = series1[i][1] - series1[i - 1][1]
            delta2 = series2[i][1] - series2[i - 1][1]
            rate_diff = abs(delta1 - delta2)
            rate_diffs.append((timestamps[i], rate_diff))

        # 仅文字解释，找出偏差较大的前几段
        sorted_diff = sorted(rate_diffs, key=lambda x: -x[1])
        top = sorted_diff[:3]
        explanations = [
            f"{ts}: 变化速率差值为 {round(diff, 3)}，差异较大"
            for ts, diff in top
        ]

        return DetectionResult(
            method="ChangeRate",
            description="ChangeRate 用于比较两个序列的局部变化速度，检测出速率差异较大的时间点",
            visual_type="none",
            explanation=explanations
        )
