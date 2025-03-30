# detectors/zscore.py
import numpy as np
from typing import List, Tuple, Optional
from detectors.base import DetectionResult

class ZScoreDetector:
    """
    Z-Score 异常检测器，基于标准差倍数检测异常点
    """
    def __init__(self, threshold: float = 3.0):
        """
        初始化Z-Score检测器
        
        参数:
            threshold: Z-Score阈值，超过此值视为异常
        """
        self.threshold = threshold
    
    def detect(self, series: List[Tuple[int, float]]) -> DetectionResult:
        """
        对时间序列执行Z-Score检测
        
        参数:
            series: 时间序列数据 [(timestamp, value), ...]
            
        返回:
            DetectionResult: 检测结果对象
        """
        if not series:
            return DetectionResult(
                method="Z-Score",
                description="无数据进行Z-Score分析",
                visual_type="none"
            )
        
        # 提取时间戳和值
        timestamps = [t for t, _ in series]
        values = np.array([v for _, v in series])
        
        # 计算均值和标准差
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 1.0
        
        # 计算Z-Score
        z_scores = (values - mean) / std if std > 0 else np.zeros_like(values)
        
        # 找出超过阈值的异常点
        anomalies = []
        scores = []
        explanations = []
        
        for i, z in enumerate(z_scores):
            if abs(z) > self.threshold:
                anomalies.append(timestamps[i])
                scores.append(float(abs(z)))
                # 解释是高于均值还是低于均值
                direction = "高于" if z > 0 else "低于"
                explanations.append(f"Z-Score={z:.2f}，{direction}均值{abs(z):.2f}个标准差")
        
        return DetectionResult(
            method="Z-Score",
            anomalies=anomalies,
            anomaly_scores=scores,
            description=f"使用Z-Score方法(阈值={self.threshold})检测到{len(anomalies)}个异常点",
            visual_type="point",
            explanation=explanations
        )