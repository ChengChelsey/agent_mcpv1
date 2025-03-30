# detectors/trend_drift_cusum.py
import numpy as np
from detectors.base import DetectionResult
from utils.time_utils import group_anomaly_times

class TrendDriftCUSUMDetector:
    def __init__(self, threshold: float = 5.0):
        """
        初始化趋势漂移检测器
        
        参数:
            threshold: CUSUM阈值，超过此值视为异常
        """
        self.threshold = threshold
    
    def detect(self, series1: list, series2: list) -> DetectionResult:
        """
        检测两个时间序列之间的趋势漂移
        """
        if not series1 or not series2 or len(series1) < 10 or len(series2) < 10:
            return DetectionResult(
                method="TrendDriftCUSUM",
                description="数据点不足进行趋势漂移分析(至少需要10个点)",
                visual_type="none"
            )
        
        # 计算两个序列之间的差异
        residuals = []
        timestamps = [t for t, _ in series1]
        values1 = [v for _, v in series1]
        values2 = [v for _, v in series2]
        
        for (_, v1), (_, v2) in zip(series1, series2):
            residuals.append(v1 - v2)
        
        # 计算基本统计量，用于初步评估差异
        mean_abs_residual = np.mean(np.abs(residuals))
        max_abs_residual = np.max(np.abs(residuals))
        relative_diff = mean_abs_residual / (np.mean(np.abs(values1)) + 1e-10)
        
        # 检查整体差异是否足够大
        if max_abs_residual < 0.05 and mean_abs_residual < 0.01:
            return DetectionResult(
                method="TrendDriftCUSUM",
                description=f"两序列几乎相同，最大差异仅{max_abs_residual:.3f}，平均差异{mean_abs_residual:.3f}",
                visual_type="none"
            )
        
        # 相对差异小于10%
        if relative_diff < 0.1:
            return DetectionResult(
                method="TrendDriftCUSUM",
                description=f"两序列差异不显著(相对差异{relative_diff:.1%})，无需进行漂移分析",
                visual_type="none"
            )
        
        # 计算CUSUM前先对数据进行平滑处理
        def smooth_data(data, window=3):
            """简单移动平均平滑"""
            if len(data) < window:
                return data
            smoothed = np.convolve(data, np.ones(window)/window, mode='same')
            # 边缘处理
            smoothed[:window//2] = data[:window//2]
            smoothed[-window//2:] = data[-window//2:]
            return smoothed
        
        # 平滑残差
        smoothed_residuals = smooth_data(residuals, window=5)
        
        # 计算残差的标准差和均值
        mean = np.mean(smoothed_residuals)
        std = np.std(smoothed_residuals) 
        
        # 防止除零
        if std < 1e-10:
            std = 1.0
            
        # 标准化残差
        norm_residuals = [(r - mean) / std for r in smoothed_residuals]
        
        # 计算CUSUM，增大控制因子
        control_factor = 1.0  # 增大控制因子，减少灵敏度
        cum_sum_pos = [0]
        cum_sum_neg = [0]
        
        # 分别跟踪上升和下降趋势
        for r in norm_residuals:
            cum_sum_pos.append(max(0, cum_sum_pos[-1] + r - control_factor))
            cum_sum_neg.append(max(0, cum_sum_neg[-1] - r - control_factor))
        
        cum_sum_pos = cum_sum_pos[1:]  # 移除初始0
        cum_sum_neg = cum_sum_neg[1:]  # 移除初始0
        
        # 取两个方向CUSUM的最大值
        cum_sum = [max(p, n) for p, n in zip(cum_sum_pos, cum_sum_neg)]
        
        # 找出超过阈值的点，但增加持续性要求
        anomalies = []
        scores = []
        consecutive_count = 0
        required_consecutive = 3  # 至少需要连续3个点超过阈值
        
        for i, c in enumerate(cum_sum):
            if c > self.threshold:
                consecutive_count += 1
                if consecutive_count >= required_consecutive:
                    # 只将第一个连续异常点加入列表
                    if consecutive_count == required_consecutive:
                        anomalies.append(timestamps[i - required_consecutive + 1])
                        scores.append(float(c))
                    # 将当前点也加入
                    anomalies.append(timestamps[i])
                    scores.append(float(c))
            else:
                consecutive_count = 0
        
        # 分组为区间，要求至少5分钟
        intervals = group_anomaly_times(anomalies, max_gap=300)  # 5分钟间隔
        
        # 过滤短区间和弱区间
        filtered_intervals = []
        explanations = []
        
        for interval in intervals:
            start, end = interval
            duration = end - start
            
            # 获取区间内的CUSUM值
            interval_indices = [i for i, ts in enumerate(timestamps) if start <= ts <= end]
            if not interval_indices:
                continue
                
            interval_scores = [cum_sum[i] for i in interval_indices]
            avg_score = np.mean(interval_scores) if interval_scores else 0
            max_score = np.max(interval_scores) if interval_scores else 0
                
            # 过滤条件：区间至少5分钟且平均CUSUM值显著高于阈值
            # 同时最大值也要显著高于阈值
            if duration >= 300 and avg_score > self.threshold * 1.3 and max_score > self.threshold * 1.5:
                filtered_intervals.append(interval)
                explanations.append(
                    f"区间{formatted_timestamp(start)}至{formatted_timestamp(end)}的CUSUM值平均为{avg_score:.1f}，最大值{max_score:.1f}，超过阈值{self.threshold}，表明两序列存在持续趋势差异"
                )
        
        # 如果没有异常区间，返回无异常结果
        if not filtered_intervals:
            return DetectionResult(
                method="TrendDriftCUSUM",
                description=f"趋势漂移检测未发现明显的持续性异常区段",
                visual_type="none"
            )
        
        # 构建辅助曲线数据
        aux_curve = [(timestamps[i], float(cum_sum[i])) for i in range(len(timestamps))]
        
        return DetectionResult(
            method="TrendDriftCUSUM",
            anomalies=[],  # 不使用点异常，只用区间
            anomaly_scores=[],
            intervals=filtered_intervals,
            auxiliary_curve=aux_curve,
            description=f"趋势漂移检测发现 {len(filtered_intervals)} 个明显异常区段，相对差异{relative_diff:.1%}",
            visual_type="range",
            explanation=explanations
        )

def formatted_timestamp(ts):
    """将时间戳格式化为可读时间"""
    from datetime import datetime
    try:
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    except:
        return str(ts)