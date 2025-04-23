"""utils/ts_features.py
精简版 – 仅负责提取可被大模型直接使用的关键时序特征。
完全删除了检测方法选择逻辑。
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import STL

Series = List[Tuple[int, float]]  # [(timestamp, value)]


def extract_time_series_features(series: Series) -> Dict[str, Any]:
    """提取关键时序特征，供大模型判断采用何种检测器。

    返回字段均为 **原子特征**，不含任何算法选择逻辑。
    """

    if not series:
        return {"error": "输入序列为空"}

    ts_idx = pd.to_datetime([t for t, _ in series], unit="s")
    values = np.array([v for _, v in series], dtype=float)
    ts = pd.Series(values, index=ts_idx)

    # -- 基本统计量
    basic = {
        "length": int(len(ts)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "missing_ratio": float(np.isnan(values).sum() / len(values)),
    }

    # -- 偏度 / 峰度
    try:
        basic["skewness"] = float(stats.skew(values, nan_policy="omit"))
        basic["kurtosis"] = float(stats.kurtosis(values, nan_policy="omit"))
    except Exception:
        basic["skewness"], basic["kurtosis"] = None, None

    # -- 平稳性（ADF）
    stationarity = {}
    try:
        adf_stat, adf_p, *_ = adfuller(values, autolag="AIC")
        stationarity = {
            "adf_stat": float(adf_stat),
            "adf_pvalue": float(adf_p),
            "is_stationary": bool(adf_p < 0.05),
        }
    except Exception:
        stationarity = {"adf_stat": None, "adf_pvalue": None, "is_stationary": False}

    # -- 自相关
    autocorr = {}
    try:
        acf_vals = acf(values, nlags=min(40, len(values)//2))
        autocorr = {
            "acf_peak_abs": float(np.max(np.abs(acf_vals[1:]))),
            "has_autocorr": bool(np.any(np.abs(acf_vals[1:]) > 1.96/np.sqrt(len(values))))
        }
    except Exception:
        autocorr = {"acf_peak_abs": None, "has_autocorr": False}

    # -- 趋势（线性拟合斜率）
    trend = {}
    try:
        x = np.arange(len(values))
        slope, intercept, r_val, p_val, _ = stats.linregress(x, values)
        trend = {
            "slope": float(slope),
            "r_squared": float(r_val**2),
            "p_value": float(p_val),
            "has_trend": bool(p_val < 0.05 and abs(r_val) > 0.3),
        }
    except Exception:
        trend = {"slope": None, "r_squared": None, "p_value": None, "has_trend": False}

    # -- 季节性（STL）
    seasonality = {}
    try:
        if len(ts) >= 48:  # 至少两天(假设小时级别)的数据
            res = STL(ts, period=24, robust=True).fit()
            strength = np.std(res.seasonal) / (np.std(res.resid) + 1e-9)
            seasonality = {
                "has_seasonality": bool(strength > 0.1),
                "strength": float(strength),
            }
        else:
            seasonality = {"has_seasonality": False, "strength": None}
    except Exception:
        seasonality = {"has_seasonality": False, "strength": None}

    # -- 波动性
    volatility = {}
    try:
        diffs = np.diff(values)
        volatility = {
            "volatility_std": float(np.std(diffs)),
            "max_abs_change": float(np.max(np.abs(diffs))) if diffs.size else 0.0,
        }
    except Exception:
        volatility = {"volatility_std": None, "max_abs_change": None}

    # -- 离群值概况（简单 z-score）
    outliers = {}
    try:
        if basic["std"]:
            z_scores = np.abs((values - basic["mean"]) / basic["std"])
            outlier_idx = np.where(z_scores > 3)[0]
            outliers = {
                "outlier_count": int(outlier_idx.size),
                "outlier_ratio": float(outlier_idx.size / len(values)),
            }
        else:
            outliers = {"outlier_count": 0, "outlier_ratio": 0.0}
    except Exception:
        outliers = {"outlier_count": None, "outlier_ratio": None}

    return {
        "basic": basic,
        "stationarity": stationarity,
        "autocorr": autocorr,
        "trend": trend,
        "seasonality": seasonality,
        "volatility": volatility,
        "outliers": outliers,
    }
