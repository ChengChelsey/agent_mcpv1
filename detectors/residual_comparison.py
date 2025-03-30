import numpy as np
from detectors.base import DetectionResult

class ResidualComparisonDetector:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def detect(self, series1: list[tuple[int, float]], series2: list[tuple[int, float]]) -> DetectionResult:
        residuals = []
        timestamps = [t for t, _ in series1]

        for (_, v1), (_, v2) in zip(series1, series2):
            residuals.append(v1 - v2)

        residuals = np.array(residuals)
        mean = np.mean(residuals)
        std = np.std(residuals)
        z_scores = (residuals - mean) / std

        anomalies = []
        scores = []
        explanations = []

        for i, z in enumerate(z_scores):
            if abs(z) > self.threshold:
                anomalies.append(timestamps[i])
                scores.append(round(abs(z), 3))
                explanations.append(f"残差Z值={z:.2f}，两序列差异大")

        return DetectionResult(
            method="ResidualComparison",
            anomalies=anomalies,
            anomaly_scores=scores,
            description=f"基于残差Z分数检测出 {len(anomalies)} 个异常点（阈值={self.threshold}）",
            visual_type="point",
            explanation=explanations
        )
