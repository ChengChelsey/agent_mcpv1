# detectors/stl_detector.py
import numpy as np
import traceback
from typing import List, Tuple, Optional
from statsmodels.tsa.seasonal import STL
from detectors.base import DetectionResult

class STLDetector:
    """
    使用STL (Season-Trend-Loess) 分解的异常检测器
    """
    def __init__(self, seasonal: int = 24, z_threshold: float = 3.0):
        """
        初始化STL检测器
        
        参数:
            seasonal: 季节性周期长度，默认为24（假设每小时的数据有天周期）
            z_threshold: 残差Z-Score阈值
        """
        self.seasonal = seasonal
        self.z_threshold = z_threshold
    
    def detect(self, series: List[Tuple[int, float]]) -> DetectionResult:
        """对时间序列执行STL分解检测"""
        if not series:
            return DetectionResult(method="STL", description="无数据进行STL分解", visual_type="none")
            
        # 提取时间戳和值
        timestamps = [t for t, _ in series]
        values = np.array([v for _, v in series])
        
        # 基础验证：检查数据长度
        if len(values) < 2:
            print(f"STL 数据点数不足: {len(values)}")
            return DetectionResult(
                method="STL", 
                description="数据点数不足以进行分析",
                visual_type="none"
            )
            
        # 验证：检查是否存在非法值
        if np.isnan(values).any() or np.isinf(values).any():
            # 清理非法值
            is_valid = ~(np.isnan(values) | np.isinf(values))
            if np.sum(is_valid) < len(values) * 0.9:  # 如果有效数据不足90%
                print(f"STL 数据含有过多无效值: {len(values) - np.sum(is_valid)} / {len(values)}")
                return DetectionResult(
                    method="STL", 
                    description="数据含有过多无效值，无法分析",
                    visual_type="none"
                )
            # 替换无效值
            valid_mean = np.mean(values[is_valid])
            values_clean = np.copy(values)
            values_clean[~is_valid] = valid_mean
            values = values_clean
        
        # 验证：几乎不变的数据
        value_range = np.max(values) - np.min(values)
        if value_range < 1e-6 or np.std(values) < 1e-6:
            print("STL 数据几乎不变")
            return DetectionResult(
                method="STL", 
                description="数据几乎不变，无需进行异常检测",
                visual_type="none"
            )
            
        # 尝试自动确定合适的季节性参数
        def estimate_seasonality(ts_values, ts_timestamps):
            """估计时间序列的季节性周期"""
            try:
                # 检查采样间隔
                if len(ts_timestamps) < 2:
                    return max(2, min(len(ts_values) // 4, 10))
                    
                intervals = [ts_timestamps[i+1] - ts_timestamps[i] for i in range(len(ts_timestamps)-1)]
                if not intervals:
                    return 2  # 默认最小季节性
                    
                median_interval = sorted(intervals)[len(intervals)//2]
                print(f"数据点的中位数时间间隔: {median_interval} 秒")
                
                # 数据量太少，使用默认值
                if len(ts_values) < 48:
                    return max(3, min(len(ts_values) // 4, 11))  # 确保是奇数
                
                # 常见周期性估计
                if 3500 <= median_interval <= 3700:  # ~1小时
                    # 对于小时级数据，根据数据长度估计
                    if len(ts_values) >= 48:  # 超过2天的数据
                        return 24  # 一天的周期
                    else:
                        return max(3, len(ts_values) // 4)
                elif 80000 <= median_interval <= 90000:  # ~1天
                    return 7  # 一周的周期
                elif 290000 <= median_interval <= 310000:  # ~3-4天
                    return 4  # 约2周一个周期
                else:
                    # 如果有固定模式的峰值（例如每小时定时任务）
                    # 尝试检测这种模式
                    try:
                        # 获取峰值
                        peaks = []
                        for i in range(1, len(ts_values)-1):
                            if ts_values[i] > ts_values[i-1] and ts_values[i] > ts_values[i+1]:
                                if ts_values[i] > np.mean(ts_values) + 1.5 * np.std(ts_values):
                                    peaks.append(i)
                        
                        if len(peaks) >= 3:
                            # 计算峰值之间的间隔
                            peak_intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
                            if peak_intervals:
                                avg_interval = sum(peak_intervals) / len(peak_intervals)
                                print(f"检测到明显的峰值模式，平均间隔: {avg_interval} 个点")
                                if 10 <= avg_interval <= 20:
                                    return int(avg_interval)
                    except Exception as e:
                        print(f"峰值检测失败: {e}")
                    
                    # 使用默认值（奇数）
                    default_seasonal = max(3, min(len(ts_values) // 8, 23))
                    # 确保是奇数
                    if default_seasonal % 2 == 0:
                        default_seasonal += 1
                    return default_seasonal
            except Exception as e:
                print(f"季节性估计错误: {e}")
                # 返回奇数值
                default_val = max(3, min(len(ts_values) // 6, 11))
                return default_val if default_val % 2 == 1 else default_val + 1
        
        try:
            # 一、尝试使用更适合的季节性参数
            if self.seasonal < 2 or self.seasonal > len(values) // 2:
                estimated_seasonal = estimate_seasonality(values, timestamps)
                print(f"自动估计的季节性参数: {estimated_seasonal}")
                # 确保季节性参数在合理范围
                self.seasonal = max(2, min(estimated_seasonal, len(values) // 2))
                # 确保季节性参数是奇数，以便于简化分析
                if self.seasonal % 2 == 0:
                    self.seasonal += 1
                print(f"最终使用的季节性参数: {self.seasonal}")
            
            # 如果数据不足以执行标准STL (至少需要2*seasonal的数据)
            if len(values) < 2 * self.seasonal:
                print(f"数据长度({len(values)})不足以进行STL分解(需要至少{2*self.seasonal}点)")
                # 使用简化季节性分解
                return self._perform_simplified_analysis(timestamps, values)
                
            # 二、尝试不同配置进行STL分解
            stl_result = None
            stl_error = None
            
            # 尝试的参数组合
            param_combinations = [
                # 周期性参数, 鲁棒性, 周期性平滑度
                (self.seasonal, True, None),
                (self.seasonal, False, None),
                (max(2, self.seasonal // 2), True, None),
                (max(2, self.seasonal // 2), False, None),
                (max(2, self.seasonal * 2), True, None)
            ]
            
            for seasonal_param, robust_param, seasonal_deg in param_combinations:
                try:
                    stl_config = {
                        'seasonal': seasonal_param,
                        'robust': robust_param
                    }
                    if seasonal_deg is not None:
                        stl_config['seasonal_deg'] = seasonal_deg
                        
                    print(f"尝试STL配置: {stl_config}")
                    stl = STL(values, **stl_config)
                    stl_result = stl.fit()
                    print(f"STL分解成功，使用配置: {stl_config}")
                    break  # 成功找到可行配置
                except Exception as e:
                    stl_error = str(e)
                    print(f"STL配置 {stl_config} 失败: {e}")
                    continue
                    
            # 如果所有STL配置都失败
            if stl_result is None:
                print(f"所有STL配置都失败，最后错误: {stl_error}")
                return self._perform_simplified_analysis(timestamps, values)
                
            # 三、成功执行STL分解，计算残差异常
            resid = stl_result.resid
            
            # 计算残差的Z-Score
            resid_mean = np.mean(resid)
            resid_std = np.std(resid) if np.std(resid) > 0 else 1.0
            resid_z = (resid - resid_mean) / resid_std
            
            # 找出残差超过阈值的异常点
            anomalies = []
            scores = []
            explanations = []
            
            for i, z in enumerate(resid_z):
                if abs(z) > self.z_threshold:
                    anomalies.append(timestamps[i])
                    scores.append(float(abs(z)))
                    direction = "高于" if z > 0 else "低于"
                    explanations.append(f"STL残差Z-Score={z:.2f}，该点{direction}预期值{abs(z):.2f}个标准差")
            
            # 构建残差曲线
            resid_curve = [(timestamps[i], float(resid[i])) for i in range(len(timestamps))]
            
            return DetectionResult(
                method="STL",
                anomalies=anomalies,
                anomaly_scores=scores,
                auxiliary_curve=resid_curve,
                description=f"STL分解(季节性周期={self.seasonal}) + 残差Z-Score(阈值={self.z_threshold})检测到{len(anomalies)}个异常点",
                visual_type="point",
                explanation=explanations
            )
            
        except Exception as e:
            tb = traceback.format_exc()
            print(f"STL分解失败: {e}\n{tb}")
            return self._perform_simplified_analysis(timestamps, values)
    
    def _perform_simplified_analysis(self, timestamps, values):
        """
        当STL分解失败时使用简化的分析方法
        """
        try:
            print("使用简化的滑动窗口分析作为替代")
            # 使用简单的移动平均作为替代方法
            window_size = max(3, min(self.seasonal, len(values) // 3))
            # 确保窗口大小是奇数
            if window_size % 2 == 0:
                window_size += 1
            
            print(f"使用窗口大小: {window_size}")
            
            # 使用简单的移动平均作为趋势
            try:
                # 首先尝试scipy的medfilt
                from scipy.signal import medfilt
                padded_values = np.pad(values, (window_size//2, window_size//2), mode='edge')
                trend = medfilt(padded_values, kernel_size=window_size)[window_size//2:window_size//2+len(values)]
            except Exception as e:
                print(f"中值滤波失败: {e}，使用滑动平均")
                # 如果失败，使用手动实现的滑动平均
                trend = np.zeros_like(values)
                half_window = window_size // 2
                
                for i in range(len(values)):
                    start = max(0, i - half_window)
                    end = min(len(values), i + half_window + 1)
                    trend[i] = np.mean(values[start:end])
            
            # 计算残差
            residuals = values - trend
            
            # 计算残差Z-Score
            resid_mean = np.mean(residuals)
            resid_std = np.std(residuals) if np.std(residuals) > 0 else 1.0
            resid_z = (residuals - resid_mean) / resid_std
            
            # 找出异常点
            anomalies = []
            scores = []
            explanations = []
            
            for i, z in enumerate(resid_z):
                if abs(z) > self.z_threshold:
                    anomalies.append(timestamps[i])
                    scores.append(float(abs(z)))
                    direction = "高于" if z > 0 else "低于"
                    explanations.append(f"简化残差Z-Score={z:.2f}，该点{direction}趋势{abs(z):.2f}个标准差")
            
            # 构建残差曲线
            resid_curve = [(timestamps[i], float(residuals[i])) for i in range(len(timestamps))]
            
            return DetectionResult(
                method="STL-Simplified",
                anomalies=anomalies,
                anomaly_scores=scores,
                auxiliary_curve=resid_curve,
                description=f"简化滑动窗口分析(窗口={window_size})检测到{len(anomalies)}个异常点",
                visual_type="point",
                explanation=explanations
            )
            
        except Exception as e:
            print(f"简化分析也失败了: {e}")
            # 最后的后备方案：使用Z-Score
            from detectors.zscore import ZScoreDetector
            zscore_result = ZScoreDetector(threshold=self.z_threshold).detect(list(zip(timestamps, values)))
            
            # 确保不重复Z-Score结果
            zscore_result.method = "STL-Fallback"
            zscore_result.description = f"STL分解失败，使用Z-Score作为替代方法"
            
            return zscore_result