# detectors/cusum.py
import numpy as np
from typing import List, Tuple, Optional
from detectors.base import DetectionResult

class CUSUMDetector:
    """
    CUSUM (累积和) 检测器，用于检测时间序列的累积偏移
    """
    def __init__(self, drift_threshold: float = 5.0, k: float = 0.5):
        """
        初始化CUSUM检测器
        
        参数:
            drift_threshold: CUSUM阈值，超过此值视为异常
            k: 灵敏度参数，较小的值对小偏移更敏感
        """
        self.drift_threshold = drift_threshold
        self.k = k
    
    def detect(self, series: List[Tuple[int, float]]) -> DetectionResult:
        """
        对时间序列执行CUSUM检测
        
        参数:
            series: 时间序列数据 [(timestamp, value), ...]
            
        返回:
            DetectionResult: 检测结果对象
        """
        if not series:
            return DetectionResult(
                method="CUSUM",
                description="无数据进行CUSUM分析",
                visual_type="none"
            )
        
        # 提取时间戳和值
        timestamps = [t for t, _ in series]
        values = np.array([v for _, v in series])
        
        # 计算数据平均值
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 1.0
        
        # 初始化CUSUM值
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))
        
        # 计算正负CUSUM
        for i in range(1, len(values)):
            # 正向累积
            cusum_pos[i] = max(0, cusum_pos[i-1] + (values[i] - mean)/std - self.k)
            # 负向累积
            cusum_neg[i] = max(0, cusum_neg[i-1] - (values[i] - mean)/std - self.k)
        
        # 合并正负CUSUM
        cusum_combined = np.maximum(cusum_pos, cusum_neg)
        
        # 找出超过阈值的异常点
        anomalies = []
        scores = []
        for i, c in enumerate(cusum_combined):
            if c > self.drift_threshold:
                anomalies.append(timestamps[i])
                scores.append(float(c))
        
        # 生成解释文本
        explanations = [
            f"CUSUM值={scores[i]:.2f}，累计偏移超过阈值({self.drift_threshold})"
            for i in range(len(anomalies))
        ]
        
        # 构建区间
        from analysis.multi_series import group_anomaly_times
        intervals = group_anomaly_times(anomalies)
        
        # 构建CUSUM曲线数据
        cum_curve = [(timestamps[i], float(cusum_combined[i])) for i in range(len(timestamps))]
        
        return DetectionResult(
            method="CUSUM",
            anomalies=anomalies,
            anomaly_scores=scores,
            intervals=intervals,
            auxiliary_curve=cum_curve,
            description=f"CUSUM累积偏移检测到 {len(intervals)} 个异常区段，共 {len(anomalies)} 个高偏移点",
            visual_type="curve",
            explanation=explanations
        )