# detectors/trend_slope.py
import numpy as np
from detectors.base import DetectionResult

class TrendSlopeDetector:
    def __init__(self, window: int = 5, threshold: float = 0.2, slope_threshold: float = None):
        """
        初始化趋势斜率检测器
        
        参数:
            window: 滑动窗口大小
            threshold: 斜率差异阈值 (与slope_threshold相同，为向后兼容)
            slope_threshold: 斜率差异阈值（如果提供，优先使用）
        """
        self.window = window
        # 支持两种参数名
        self.threshold = slope_threshold if slope_threshold is not None else threshold
    
    def detect(self, series1: list, series2: list) -> DetectionResult:
        """
        检测两个时间序列之间的趋势斜率差异
        
        参数:
            series1: 第一个时间序列 [(timestamp, value), ...]
            series2: 第二个时间序列 [(timestamp, value), ...]
            
        返回:
            DetectionResult: 检测结果对象
        """
        if not series1 or not series2 or len(series1) < self.window or len(series2) < self.window:
            return DetectionResult(
                method="TrendSlope",
                description="数据点不足进行趋势斜率分析",
                visual_type="none"
            )
            
        def calc_slope(values):
            x = np.arange(len(values))
            A = np.vstack([x, np.ones(len(values))]).T
            m, _ = np.linalg.lstsq(A, values, rcond=None)[0]
            return m

        slopes1, slopes2, timestamps = [], [], []

        for i in range(len(series1) - self.window + 1):
            window1 = [v for _, v in series1[i:i + self.window]]
            window2 = [v for _, v in series2[i:i + self.window]]
            
            try:
                slope1 = calc_slope(window1)
                slope2 = calc_slope(window2)
                slopes1.append(slope1)
                slopes2.append(slope2)
                timestamps.append(series1[i + self.window // 2][0])
            except Exception as e:
                print(f"计算斜率时出错: {e}")
                continue

        if not timestamps:
            return DetectionResult(
                method="TrendSlope",
                description="无法计算有效的趋势斜率",
                visual_type="none"
            )

        slope_diff = np.abs(np.array(slopes1) - np.array(slopes2))
        sorted_indices = np.argsort(-slope_diff)
        
        # 找出差异最大的几个点
        anomalies = []
        scores = []
        explanations = []
        
        for i in sorted_indices[:min(3, len(sorted_indices))]:
            ts = timestamps[i]
            diff = slope_diff[i]
            if diff > self.threshold:
                anomalies.append(ts)
                scores.append(float(diff))
                explanations.append(f"趋势斜率差值为 {diff:.3f}，高于阈值 {self.threshold}")
        
        return DetectionResult(
            method="TrendSlope",
            anomalies=anomalies,
            anomaly_scores=scores,
            description=f"TrendSlope 检测两个序列在滑动窗口下的局部趋势方向差异，发现 {len(anomalies)} 个异常点",
            visual_type="point",
            explanation=explanations
        )