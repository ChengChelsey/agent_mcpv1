# detectors/base.py
from typing import List, Tuple, Optional, Dict, Any, Union

class DetectionResult:
    """
    统一的异常检测结果类，支持多种可视化展示方式
    
    属性:
        method: 检测方法名称
        anomalies: 异常时间点列表
        anomaly_scores: 对应每个异常点的分数
        intervals: 异常区间列表 [(start, end), ...]
        auxiliary_curve: 辅助曲线数据 [(ts, value), ...]
        description: 检测方法的描述
        visual_type: 可视化类型 "point"|"range"|"curve"|"none"
        explanation: 对应每个异常点/区间的解释文本
    """
    def __init__(
        self,
        method: str,
        anomalies: Optional[List[int]] = None,
        anomaly_scores: Optional[List[float]] = None,
        intervals: Optional[List[Tuple[int, int]]] = None,
        auxiliary_curve: Optional[List[Tuple[int, float]]] = None,
        description: str = "",
        visual_type: str = "point",  # point | range | curve | none
        explanation: Optional[List[str]] = None,
    ):
        self.method = method
        self.anomalies = anomalies or []
        self.anomaly_scores = anomaly_scores or []
        self.intervals = intervals or []
        self.auxiliary_curve = auxiliary_curve or []
        self.description = description
        self.visual_type = visual_type
        self.explanation = explanation or []
    
    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式，便于序列化和传输"""
        return {
            "method": self.method,
            "anomalies": self.anomalies,
            "anomaly_scores": self.anomaly_scores,
            "intervals": self.intervals,
            "auxiliary_curve": self.auxiliary_curve,
            "description": self.description,
            "visual_type": self.visual_type,
            "explanation": self.explanation
        }