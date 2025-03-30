# detectors/__init__.py
"""
异常检测器模块，包含各种用于时序数据异常检测的算法
"""

# 明确导出检测器类以简化导入
from detectors.zscore import ZScoreDetector
from detectors.cusum import CUSUMDetector
from detectors.stl_detector import STLDetector
from detectors.residual_comparison import ResidualComparisonDetector
from detectors.trend_drift_cusum import TrendDriftCUSUMDetector
from detectors.change_rate import ChangeRateDetector
from detectors.trend_slope import TrendSlopeDetector
from detectors.base import DetectionResult

# 为向后兼容性导出旧的函数名
from detectors.zscore import ZScoreDetector as detect_zscore
from detectors.cusum import CUSUMDetector as detect_cusum
from detectors.stl_detector import STLDetector as detect_stl_residual
from detectors.residual_comparison import ResidualComparisonDetector as detect_residual_compare
from detectors.trend_drift_cusum import TrendDriftCUSUMDetector as detect_trend_drift
from detectors.change_rate import ChangeRateDetector as detect_change_rate
from detectors.trend_slope import TrendSlopeDetector as detect_trend_slope