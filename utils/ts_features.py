"""
时序数据特征分析工具

实现时间序列数据特征的提取和分析功能。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ts_features")

def analyze_time_series_features(series_data: List[List]) -> Dict:
    """分析时间序列数据的特征
    
    Args:
        series_data: 时间序列数据，格式为 [[timestamp, value], ...]
        
    Returns:
        包含时间序列特征的字典
    """
    try:
        # 提取时间戳和值
        timestamps = [item[0] for item in series_data]
        values = [item[1] for item in series_data]
        
        # 创建Series用于分析
        ts = pd.Series(values, index=timestamps)
        
        # 计算基本统计特征
        features = {
            "长度": len(ts),
            "最小值": float(ts.min()),
            "最大值": float(ts.max()),
            "平均值": float(ts.mean()),
            "中位数": float(ts.median()),
            "标准差": float(ts.std()),
            "四分位距": float(ts.quantile(0.75) - ts.quantile(0.25)),
            "非空值比例": float((~ts.isna()).mean()),
            "变异系数": float(ts.std() / ts.mean()) if ts.mean() != 0 else float('nan'),
        }
        
        # 检测平稳性
        is_stationary = check_stationarity(ts)
        features["平稳性"] = is_stationary
        
        # 检测季节性
        has_seasonality, seasonality_period = check_seasonality(ts)
        features["季节性"] = has_seasonality
        if has_seasonality and seasonality_period is not None:
            features["季节周期"] = seasonality_period
        
        # 检测趋势
        has_trend, trend_strength = check_trend(ts)
        features["趋势"] = has_trend
        features["趋势强度"] = trend_strength
        
        # 检测波动性
        volatility, volatility_description = calculate_volatility(ts)
        features["波动性指数"] = volatility
        features["波动性描述"] = volatility_description
        
        # 异常值初步检测
        outlier_ratio = estimate_outlier_ratio(ts)
        features["异常值比例估计"] = outlier_ratio
        
        # 计算一阶差分序列的统计特性（变化率特性）
        if len(ts) > 1:
            diff = ts.diff().dropna()
            features["变化率统计"] = {
                "平均变化率": float(diff.mean()),
                "最大变化率": float(diff.max()),
                "最小变化率": float(diff.min()),
                "变化率标准差": float(diff.std())
            }
        
        # 检测自相关性
        autocorr_lag1 = calculate_autocorrelation(ts, 1)
        features["自相关系数(lag=1)"] = autocorr_lag1
        
        # 数据分布特征
        skewness = float(stats.skew(ts.dropna()))
        kurtosis = float(stats.kurtosis(ts.dropna()))
        features["偏度"] = skewness  # 正值表示右偏，负值表示左偏
        features["峰度"] = kurtosis  # 正值表示尖峰，负值表示平坦
        
        return features
        
    except Exception as e:
        logger.error(f"特征分析错误: {str(e)}")
        return {"error": f"特征分析失败: {str(e)}"}

def check_stationarity(ts: pd.Series) -> bool:
    """检查时间序列是否平稳
    
    简单实现：通过比较前半部分和后半部分的均值和方差来判断
    """
    if len(ts) < 10:  # 数据点太少
        return False
        
    # 将序列分为前半部分和后半部分
    mid_point = len(ts) // 2
    first_half = ts[:mid_point]
    second_half = ts[mid_point:]
    
    # 比较均值
    mean_diff = abs(first_half.mean() - second_half.mean())
    mean_threshold = ts.std() * 0.5  # 允许的均值差异阈值
    
    # 比较方差
    var_ratio = first_half.var() / second_half.var() if second_half.var() > 0 else float('inf')
    var_threshold = 2.0  # 允许的方差比率阈值
    
    return mean_diff < mean_threshold and 1/var_threshold < var_ratio < var_threshold

def check_seasonality(ts: pd.Series) -> Tuple[bool, Optional[int]]:
    """检查时间序列是否具有季节性，并返回可能的季节周期"""
    if len(ts) < 10:  # 数据点太少
        return False, None
        
    # 计算自相关
    n = min(len(ts) // 2, 50)  # 最多检查50个lag
    acf = compute_acf(ts, n)
    
    # 查找自相关的峰值
    peaks = []
    for i in range(2, len(acf) - 1):
        if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.3:  # 只考虑自相关大于0.3的峰值
            peaks.append(i)
    
    if len(peaks) > 0:
        # 如果找到多个峰值，返回第一个作为可能的季节周期
        return True, peaks[0]
    else:
        return False, None

def compute_acf(ts: pd.Series, n: int) -> np.ndarray:
    """计算自相关函数"""
    result = np.zeros(n+1)
    y = ts - ts.mean()
    variance = np.sum(y ** 2) / len(y)
    
    for lag in range(n+1):
        covariance = np.sum(y[:-lag] * y[lag:]) / (len(y) - lag) if lag > 0 else variance
        result[lag] = covariance / variance
        
    return result

def check_trend(ts: pd.Series) -> Tuple[bool, float]:
    """检查时间序列是否具有趋势，并返回趋势强度"""
    if len(ts) < 10:  # 数据点太少
        return False, 0.0
        
    # 简单线性回归检测趋势
    x = np.arange(len(ts))
    y = ts.values
    
    # 计算斜率
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return False, 0.0
        
    slope = numerator / denominator
    
    # 计算R²
    y_pred = x_mean + slope * (x - x_mean)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # 如果R²较高且斜率不接近0，则认为有趋势
    has_trend = r_squared > 0.3 and abs(slope) > 0.01 * abs(y.mean())
    
    return has_trend, float(r_squared)

def calculate_volatility(ts: pd.Series) -> Tuple[float, str]:
    """计算时间序列的波动性，并返回描述性结果"""
    if len(ts) < 5:
        return 0.0, "数据点不足"
        
    # 计算变异系数
    cv = ts.std() / abs(ts.mean()) if ts.mean() != 0 else float('inf')
    
    # 计算平均绝对百分比变化
    changes = abs(ts.pct_change().dropna())
    mean_change = changes.mean() if len(changes) > 0 else 0
    
    # 计算波动性指数（综合考虑变异系数和平均变化率）
    volatility_index = 0.7 * min(cv, 1.0) + 0.3 * min(mean_change * 10, 1.0)
    
    # 波动性描述
    if volatility_index < 0.2:
        volatility_description = "低"
    elif volatility_index < 0.5:
        volatility_description = "中"
    else:
        volatility_description = "高"
    
    return float(volatility_index), volatility_description

def estimate_outlier_ratio(ts: pd.Series) -> float:
    """估计时间序列中的异常值比例"""
    if len(ts) < 5:
        return 0.0
        
    # 使用Z-score简单估计
    z_scores = abs((ts - ts.mean()) / ts.std())
    outlier_ratio = (z_scores > 3).mean()
    
    return float(outlier_ratio)

def calculate_autocorrelation(ts: pd.Series, lag: int) -> float:
    """计算指定滞后期的自相关系数"""
    if len(ts) <= lag:
        return 0.0
    
    # 去除均值
    y = ts - ts.mean()
    
    # 计算自相关
    numerator = np.sum(y[:-lag] * y[lag:])
    denominator = np.sqrt(np.sum(y[:-lag]**2) * np.sum(y[lag:]**2))
    
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)

def recommend_detection_methods(features: Dict) -> List[Dict]:
    """根据时序特征推荐适合的检测方法
    
    这个函数根据数据特征，推荐最适合的异常检测方法
    
    Returns:
        包含推荐方法名称和推荐理由的列表
    """
    recommended = []
    
    # 基本方法，适用于大多数情况
    recommended.append({
        "方法": "IQR异常检测",
        "理由": "适用于大多数场景，对数据分布要求较低，能检测点异常"
    })
    
    # 根据特征推荐方法
    if features.get("平稳性", False):
        recommended.append({
            "方法": "广义ESD检测",
            "理由": "数据呈现平稳性，适合使用统计检验方法"
        })
    
    if features.get("波动性描述") == "高":
        recommended.append({
            "方法": "波动性变化检测",
            "理由": "数据波动性较高，适合检测波动性的突变"
        })
    
    if features.get("趋势", False):
        recommended.append({
            "方法": "水平位移检测",
            "理由": "数据存在明显趋势，适合检测水平位移异常"
        })
    
    if features.get("季节性", False):
        recommended.append({
            "方法": "季节性异常检测",
            "理由": "数据具有季节性模式，适合检测偏离季节模式的异常"
        })
    
    # 如果数据点较多，添加一些更复杂的方法
    if features.get("长度", 0) > 30:
        recommended.append({
            "方法": "持续性异常检测",
            "理由": "数据点数量充足，适合检测与前序值的异常偏差"
        })
        
        if features.get("自相关系数(lag=1)", 0) > 0.5:
            recommended.append({
                "方法": "自回归异常检测",
                "理由": "数据具有较强的自相关性，适合使用自回归模型"
            })
    
    # 如果检测到明显的异常比例
    if features.get("异常值比例估计", 0) > 0.01:
        recommended.append({
            "方法": "分位数异常检测",
            "理由": "数据中可能存在异常值，适合使用分位数方法"
        })
    
    # 返回不超过5个的推荐方法
    return recommended[:5]

# 直接使用的异常检测方法选择函数
def select_detection_methods(features: Dict, detector_info: Dict) -> List[Dict]:
    """根据时序特征和检测器信息，选择最适合的异常检测方法
    
    Args:
        features: 时序数据特征
        detector_info: 检测器信息
        
    Returns:
        选定的异常检测方法列表，包含方法名称、参数和权重
    """
    selected_methods = []
    
    # 获取单变量检测器信息
    single_var_detectors = detector_info.get("单变量检测器", [])
    
    # 根据特征提取重要信息
    data_length = features.get("长度", 0)
    is_stationary = features.get("平稳性", False)
    has_seasonality = features.get("季节性", False)
    seasonality_period = features.get("季节周期")
    has_trend = features.get("趋势", False)
    volatility = features.get("波动性描述", "中")
    auto_corr = features.get("自相关系数(lag=1)", 0)
    outlier_ratio = features.get("异常值比例估计", 0)
    
    # 定义权重和选择条件
    detector_scores = {}
    
    # 对每个检测器评分
    for detector in single_var_detectors:
        score = 0.0
        method_name = detector.get("类名", "")
        
        # 根据特征对每个方法进行评分
        if method_name == "InterQuartileRangeAD":
            # IQR方法适用于大多数情况
            score = 0.7
            
        elif method_name == "GeneralizedESDTestAD":
            # 广义ESD方法适用于平稳数据
            score = 0.5
            if is_stationary:
                score += 0.3
            
        elif method_name == "QuantileAD":
            # 分位数方法适用于有明显异常的数据
            score = 0.4
            if outlier_ratio > 0.01:
                score += 0.3
            
        elif method_name == "ThresholdAD":
            # 阈值方法适用于有明确阈值的数据
            score = 0.3  # 默认较低分数，因为需要人工设置阈值
            
        elif method_name == "PersistAD":
            # 持续性方法适用于有自相关性的数据
            score = 0.4
            if auto_corr > 0.3:
                score += 0.3
                
        elif method_name == "LevelShiftAD":
            # 水平位移方法适用于有趋势的数据
            score = 0.4
            if has_trend:
                score += 0.3
                
        elif method_name == "VolatilityShiftAD":
            # 波动性变化方法适用于高波动性数据
            score = 0.3
            if volatility == "高":
                score += 0.4
                
        elif method_name == "SeasonalAD":
            # 季节性方法适用于有季节性的数据
            score = 0.3
            if has_seasonality:
                score += 0.5
                
        elif method_name == "AutoregressionAD":
            # 自回归方法适用于强自相关数据
            score = 0.3
            if auto_corr > 0.5:
                score += 0.4
        
        # 数据长度不足时减分
        if data_length < 30 and method_name in ["SeasonalAD", "AutoregressionAD", "VolatilityShiftAD"]:
            score -= 0.3
        
        # 记录评分
        detector_scores[method_name] = score
    
    # 选择评分最高的3-5个方法
    sorted_detectors = sorted(detector_scores.items(), key=lambda x: x[1], reverse=True)
    top_detectors = sorted_detectors[:min(5, len(sorted_detectors))]
    
    # 确保至少有3个方法（如果可能）
    if len(top_detectors) < 3 and len(sorted_detectors) >= 3:
        top_detectors = sorted_detectors[:3]
    
    # 构建方法参数
    for method_name, score in top_detectors:
        # 跳过评分太低的方法
        if score < 0.3:
            continue
            
        # 设置方法的默认参数
        params = {}
        
        # 根据特征设置适当的参数
        if method_name == "InterQuartileRangeAD":
            c = 3.0
            if outlier_ratio > 0.05:
                c = 4.0  # 异常值比例高时，提高阈值减少误报
            params = {"c": c}
            
        elif method_name == "GeneralizedESDTestAD":
            params = {"alpha": 0.05}
            
        elif method_name == "QuantileAD":
            params = {"low": 0.05, "high": 0.95}
            
        elif method_name == "ThresholdAD":
            # 使用中位数和IQR估计合适的阈值
            median = features.get("中位数", 0)
            iqr = features.get("四分位距", 1)
            params = {
                "low": median - 3 * iqr if iqr > 0 else None,
                "high": median + 3 * iqr if iqr > 0 else None
            }
            
        elif method_name == "PersistAD":
            window = 5 if data_length > 50 else 3
            params = {"window": window, "c": 3.0, "side": "both"}
            
        elif method_name == "LevelShiftAD":
            window = 10 if data_length > 100 else 5
            params = {"window": window, "c": 6.0, "side": "both"}
            
        elif method_name == "VolatilityShiftAD":
            window = 15 if data_length > 150 else 10
            params = {"window": window, "c": 6.0, "side": "both", "agg": "std"}
            
        elif method_name == "SeasonalAD":
            freq = seasonality_period if seasonality_period else None
            params = {"freq": freq, "c": 3.0, "trend": has_trend}
            
        elif method_name == "AutoregressionAD":
            n_steps = 3 if auto_corr > 0.7 else 1
            params = {"n_steps": n_steps, "c": 3.0, "side": "both"}
        
        # 添加权重
        params["weight"] = score
        
        # 添加到选定方法列表
        for detector in single_var_detectors:
            if detector.get("类名") == method_name:
                method_display_name = next((name for name, m in {
                    "IQR异常检测": "InterQuartileRangeAD",
                    "广义ESD检测": "GeneralizedESDTestAD",
                    "分位数异常检测": "QuantileAD",
                    "阈值异常检测": "ThresholdAD",
                    "持续性异常检测": "PersistAD",
                    "水平位移检测": "LevelShiftAD",
                    "波动性变化检测": "VolatilityShiftAD",
                    "季节性异常检测": "SeasonalAD",
                    "自回归异常检测": "AutoregressionAD"
                }.items() if m == method_name), method_name)
                
                selected_methods.append({
                    "方法名称": method_display_name,
                    "类名": method_name,
                    "参数": params,
                    "理由": detector.get("适用场景", ""),
                    "评分": score
                })
                break
    
    return selected_methods