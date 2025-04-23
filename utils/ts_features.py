# utils/ts_features.py
"""
时序特征分析和检测方法选择
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf

def analyze_time_series_features(series_data: List[List[float]]) -> Dict[str, Any]:
    """
    分析时序数据特征，包括：
    - 基本统计量
    - 趋势性
    - 季节性
    - 平稳性
    - 异常值特征
    
    Args:
        series_data: 时序数据，格式为 [[timestamp, value], ...]
        
    Returns:
        特征字典
    """
    # 转换为pandas Series
    timestamps = [item[0] for item in series_data]
    values = [item[1] for item in series_data]
    
    # 检查数据是否为空
    if not values:
        return {"error": "输入数据为空"}
    
    # 创建pandas Series
    ts = pd.Series(values, index=pd.to_datetime(timestamps, unit='s'))
    
    # 计算基本统计量
    basic_stats = {
        "length": len(ts),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "missing_values": int(np.isnan(values).sum()),
    }
    
    # 计算偏度和峰度
    basic_stats["skewness"] = float(stats.skew(values))
    basic_stats["kurtosis"] = float(stats.kurtosis(values))
    
    # 计算平稳性
    stationarity = {}
    try:
        adf_result = adfuller(values)
        stationarity = {
            "adf_statistic": float(adf_result[0]),
            "adf_pvalue": float(adf_result[1]),
            "is_stationary": adf_result[1] < 0.05,
        }
    except:
        stationarity = {
            "adf_statistic": None,
            "adf_pvalue": None,
            "is_stationary": False,
        }
    
    # 自相关性分析
    autocorrelation = {}
    try:
        acf_values = acf(values, nlags=min(40, len(values)//2))
        pacf_values = pacf(values, nlags=min(40, len(values)//2))
        
        # 判断是否有显著自相关
        significant_acf = np.any(np.abs(acf_values[1:]) > 1.96/np.sqrt(len(values)))
        
        autocorrelation = {
            "has_autocorrelation": significant_acf,
            "acf_peak": float(np.max(np.abs(acf_values[1:]))),
        }
    except:
        autocorrelation = {
            "has_autocorrelation": False,
            "acf_peak": 0.0,
        }
    
    # 季节性检测
    seasonality = {}
    try:
        # 重新采样为规则间隔数据
        ts_regular = ts.resample('1min').mean().interpolate()
        
        # 尝试不同周期检测季节性
        potential_periods = [
            24,            # 每天24小时
            24*7,          # 每周
            24*30,         # 每月
        ]
        
        max_strength = 0
        best_period = None
        
        for period in potential_periods:
            if len(ts_regular) >= period * 2:  # 至少需要两个完整周期
                try:
                    result = seasonal_decompose(ts_regular, model='additive', period=period)
                    # 计算季节性强度
                    seasonal_strength = np.std(result.seasonal) / np.std(ts_regular)
                    if seasonal_strength > max_strength:
                        max_strength = seasonal_strength
                        best_period = period
                except:
                    continue
        
        seasonality = {
            "has_seasonality": max_strength > 0.1,  # 季节性强度阈值
            "seasonal_strength": float(max_strength),
            "detected_period": best_period,
        }
    except:
        seasonality = {
            "has_seasonality": False,
            "seasonal_strength": 0.0,
            "detected_period": None,
        }
    
    # 趋势分析
    trend = {}
    try:
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        trend = {
            "slope": float(slope),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "has_trend": p_value < 0.05 and abs(r_value) > 0.3,
        }
    except:
        trend = {
            "slope": 0.0,
            "r_squared": 0.0,
            "p_value": 1.0,
            "has_trend": False,
        }
    
    # 异常值检测（简单的Z分数法）
    outliers = {}
    try:
        z_scores = np.abs(stats.zscore(values))
        outlier_indices = np.where(z_scores > 3)[0]
        outlier_count = len(outlier_indices)
        outlier_ratio = outlier_count / len(values) if values else 0
        
        outliers = {
            "outlier_count": int(outlier_count),
            "outlier_ratio": float(outlier_ratio),
            "has_outliers": outlier_count > 0,
        }
    except:
        outliers = {
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "has_outliers": False,
        }
    
    # 波动性分析
    volatility = {}
    try:
        returns = np.diff(values) / values[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        volatility = {
            "volatility": float(np.std(returns)),
            "max_change": float(np.max(np.abs(returns))),
            "is_volatile": np.std(returns) > 0.05,  # 波动性阈值
        }
    except:
        volatility = {
            "volatility": 0.0,
            "max_change": 0.0,
            "is_volatile": False,
        }
    
    # 汇总特征
    return {
        "basic_stats": basic_stats,
        "stationarity": stationarity,
        "autocorrelation": autocorrelation,
        "seasonality": seasonality,
        "trend": trend,
        "outliers": outliers,
        "volatility": volatility,
    }

def select_detection_methods(features: Dict[str, Any], detector_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    根据时序特征和检测方法信息，选择最适合的检测方法
    
    Args:
        features: 时序特征
        detector_info: 检测方法信息
        
    Returns:
        选定的检测方法列表，每个方法包含名称、参数和权重
    """
    selected_methods = []
    
    # 获取各特征指标
    try:
        has_outliers = features.get("outliers", {}).get("has_outliers", False)
        has_trend = features.get("trend", {}).get("has_trend", False)
        has_seasonality = features.get("seasonality", {}).get("has_seasonality", False)
        is_stationary = features.get("stationarity", {}).get("is_stationary", False)
        has_autocorrelation = features.get("autocorrelation", {}).get("has_autocorrelation", False)
        is_volatile = features.get("volatility", {}).get("is_volatile", False)
        
        # 为每种可能的异常模式选择合适的方法
        
        # 1. 处理离群值检测
        if has_outliers:
            # IQR检测非常适合离群值检测
            if "IQR异常检测" in detector_info:
                selected_methods.append({
                    "method": "IQR异常检测",
                    "parameters": {"c": 2.5},  # 稍微降低阈值使其更敏感
                    "weight": 0.8,
                    "reason": "数据含有明显离群点，IQR方法适合检测孤立的异常值"
                })
            
            # 分位数异常检测也适合
            if "分位数异常检测" in detector_info:
                selected_methods.append({
                    "method": "分位数异常检测",
                    "parameters": {"low": 0.05, "high": 0.95},
                    "weight": 0.7,
                    "reason": "数据分布有长尾特征，分位数方法可以有效处理"
                })
        
        # 2. 处理趋势变化
        if has_trend:
            # 自回归适合趋势变化
            if "自回归异常检测" in detector_info:
                selected_methods.append({
                    "method": "自回归异常检测",
                    "parameters": {"c": 3.0, "n_steps": 1},
                    "weight": 0.7,
                    "reason": "数据存在趋势，自回归方法可以检测趋势中的异常"
                })
            
            # 水平位移检测适合趋势突变
            if "水平位移检测" in detector_info:
                selected_methods.append({
                    "method": "水平位移检测",
                    "parameters": {"window": 5, "c": 6.0},
                    "weight": 0.6,
                    "reason": "检测数据中可能存在的水平位移异常"
                })
        
        # 3. 处理季节性异常
        if has_seasonality:
            if "季节性异常检测" in detector_info:
                # 设置合适的周期
                period = features.get("seasonality", {}).get("detected_period")
                if period:
                    selected_methods.append({
                        "method": "季节性异常检测",
                        "parameters": {"freq": int(period), "c": 3.0},
                        "weight": 0.8,
                        "reason": f"数据有明显周期({period})，季节性检测可识别周期中的异常模式"
                    })
        
        # 4. 处理波动性变化
        if is_volatile:
            if "波动性变化检测" in detector_info:
                selected_methods.append({
                    "method": "波动性变化检测",
                    "parameters": {"window": 10, "c": 5.0},
                    "weight": 0.7,
                    "reason": "数据波动性较大，适合检测波动性突变"
                })
        
        # 5. 通用方法（无论特征如何，这些方法通常都有效）
        
        # 持续性异常检测
        if "持续性异常检测" in detector_info:
            window_size = 3  # 默认窗口大小
            if has_autocorrelation:
                window_size = 5  # 自相关数据使用更大窗口
            
            selected_methods.append({
                "method": "持续性异常检测",
                "parameters": {"window": window_size, "c": 3.0},
                "weight": 0.6,
                "reason": "检测持续的异常情况"
            })
        
        # 如果方法数量不足，添加一些通用方法
        if len(selected_methods) < 3:
            if "广义ESD检测" in detector_info and "广义ESD检测" not in [m["method"] for m in selected_methods]:
                selected_methods.append({
                    "method": "广义ESD检测",
                    "parameters": {"alpha": 0.05},
                    "weight": 0.5,
                    "reason": "通用的异常检测方法，适用于多种场景"
                })
            
            if "阈值异常检测" in detector_info and "阈值异常检测" not in [m["method"] for m in selected_methods]:
                # 根据数据特征设置合理的阈值
                basic_stats = features.get("basic_stats", {})
                mean = basic_stats.get("mean", 0)
                std = basic_stats.get("std", 1)
                
                selected_methods.append({
                    "method": "阈值异常检测",
                    "parameters": {"low": mean - 2.5*std, "high": mean + 2.5*std},
                    "weight": 0.4,
                    "reason": "基于统计特征设置的自适应阈值检测"
                })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"选择检测方法时发生错误: {e}")
        
        # 出错时返回一些基本方法
        if "IQR异常检测" in detector_info:
            selected_methods.append({
                "method": "IQR异常检测",
                "parameters": {"c": 3.0},
                "weight": 0.7,
                "reason": "基础异常检测方法"
            })
        
        if "持续性异常检测" in detector_info:
            selected_methods.append({
                "method": "持续性异常检测",
                "parameters": {"window": 3, "c": 3.0},
                "weight": 0.6,
                "reason": "检测持续的异常情况"
            })
    
    # 最多返回5个方法
    return selected_methods[:5]