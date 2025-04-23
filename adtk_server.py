#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构版 ADTK-MCP Server：统一封装异常检测逻辑，减少冗余代码
最终修复版
"""
import sys
import json
import logging
import traceback
from typing import List, Dict, Any, Optional

import pandas as pd
from mcp.server.fastmcp import FastMCP
from adtk.detector import *
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# 配置日志
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("adtk_server")

# 创建 MCP Server
mcp = FastMCP("ADTK-Server")

# 检测器元数据
DETECTOR_META = {}

# ======================== 通用转换函数 ========================
def convert_to_series(ts_data):
    if not ts_data:
        raise ValueError("时间序列数据为空")
    idx = pd.to_datetime([t for t, _ in ts_data], unit='s')
    values = [v for _, v in ts_data]
    return pd.Series(values, index=idx)

def convert_to_dataframe(multi_ts_data):
    if not multi_ts_data:
        raise ValueError("多变量时间序列数据为空")
    idx = pd.to_datetime([item["timestamp"] for item in multi_ts_data], unit='s')
    df = pd.DataFrame([item["features"] for item in multi_ts_data], index=idx)
    return df

# ======================== 通用执行函数 ========================
def format_detection_result(name, events, total):
    points, ranges = [], []
    for e in events:
        if isinstance(e, tuple):
            ranges.append([int(e[0].timestamp()), int(e[1].timestamp())])
        else:
            points.append(int(e.timestamp()))
    ratio = (len(points) + len(ranges) * 3) / max(total, 1)
    return {
        "method": name,
        "visual_type": "range" if ranges else "point" if points else "none",
        "anomalies": points,
        "intervals": ranges,
        "anomaly_ratio": round(ratio, 6),
        "explanation": [f"{name} 检出异常"] * (len(points) + len(ranges))
    }

def run_univariate_detection(series, detector_cls, name, **kwargs):
    try:
        ts = convert_to_series(series)
        detector = detector_cls(**kwargs)
        detector.fit(ts)
        events = detector.detect(ts, return_list=True)
        return format_detection_result(name, events, len(ts))
    except Exception as e:
        logger.error(f"{name} 检测失败: {e}")
        return {"error": str(e)}

def run_multivariate_detection(data, detector_cls, name, base_model=None, **kwargs):
    try:
        df = convert_to_dataframe(data)
        detector = detector_cls(base_model, **kwargs) if base_model else detector_cls(**kwargs)
        detector.fit(df)
        events = detector.detect(df, return_list=True)
        return format_detection_result(name, events, len(df))
    except Exception as e:
        logger.error(f"{name} 检测失败: {e}")
        return {"error": str(e)}

# ======================== 单变量检测器 ========================

# IQR异常检测
@mcp.tool(name="IQR异常检测", description="基于四分位数范围的异常检测")
def iqr_detector(series: List[List[float]], c: float = 3.0) -> Dict[str, Any]:
    logger.info(f"执行IQR异常检测, 参数: c={c}")
    return run_univariate_detection(series, InterQuartileRangeAD, "InterQuartileRangeAD", c=c)

# 广义ESD检测
@mcp.tool(name="广义ESD检测", description="Generalized ESD统计检验异常检测")
def esd_detector(series: List[List[float]], alpha: float = 0.05) -> Dict[str, Any]:
    logger.info(f"执行广义ESD检测, 参数: alpha={alpha}")
    return run_univariate_detection(series, GeneralizedESDTestAD, "GeneralizedESDTestAD", alpha=alpha)

# 分位数异常检测
@mcp.tool(name="分位数异常检测", description="基于分位数的异常检测")
def quantile_detector(series: List[List[float]], low: float = 0.05, high: float = 0.95) -> Dict[str, Any]:
    logger.info(f"执行分位数异常检测, 参数: low={low}, high={high}")
    return run_univariate_detection(series, QuantileAD, "QuantileAD", low=low, high=high)

# 阈值异常检测
@mcp.tool(name="阈值异常检测", description="基于固定阈值的异常检测")
def threshold_detector(series: List[List[float]], low: Optional[float] = None, high: Optional[float] = None) -> Dict[str, Any]:
    logger.info(f"执行阈值异常检测, 参数: low={low}, high={high}")
    return run_univariate_detection(series, ThresholdAD, "ThresholdAD", low=low, high=high)

# 持续性异常检测
@mcp.tool(name="持续性异常检测", description="检测持续异常状态")
def persist_detector(series: List[List[float]], window: int = 1, c: float = 3.0) -> Dict[str, Any]:
    logger.info(f"执行持续性异常检测, 参数: window={window}, c={c}")
    return run_univariate_detection(series, PersistAD, "PersistAD", window=window, c=c)

# 水平位移检测
@mcp.tool(name="水平位移检测", description="检测数据水平的突变")
def level_shift_detector(series: List[List[float]], window: int = 5, c: float = 6.0) -> Dict[str, Any]:
    logger.info(f"执行水平位移检测, 参数: window={window}, c={c}")
    return run_univariate_detection(series, LevelShiftAD, "LevelShiftAD", window=window, c=c)

# 波动性变化检测
@mcp.tool(name="波动性变化检测", description="检测数据波动性的变化")
def volatility_shift_detector(series: List[List[float]], window: int = 10, c: float = 6.0) -> Dict[str, Any]:
    logger.info(f"执行波动性变化检测, 参数: window={window}, c={c}")
    return run_univariate_detection(series, VolatilityShiftAD, "VolatilityShiftAD", window=window, c=c)

# 季节性异常检测
@mcp.tool(name="季节性异常检测", description="检测季节性数据的异常")
def seasonal_detector(series: List[List[float]], freq: Optional[int] = None, c: float = 3.0) -> Dict[str, Any]:
    logger.info(f"执行季节性异常检测, 参数: freq={freq}, c={c}")
    return run_univariate_detection(series, SeasonalAD, "SeasonalAD", freq=freq, c=c)

# 自回归异常检测
@mcp.tool(name="自回归异常检测", description="基于自回归模型的异常检测")
def autoregression_detector(series: List[List[float]], n_steps: int = 1, step_size: int = 1, c: float = 3.0) -> Dict[str, Any]:
    logger.info(f"执行自回归异常检测, 参数: n_steps={n_steps}, step_size={step_size}, c={c}")
    return run_univariate_detection(series, AutoregressionAD, "AutoregressionAD", n_steps=n_steps, step_size=step_size, c=c)

# ======================== 多变量检测器 ========================
@mcp.tool(name="最小聚类异常检测", description="使用KMeans聚类检测多变量异常")
def min_cluster_detector(data: List[Dict[str, Any]], n_clusters: int = 2) -> Dict[str, Any]:
    logger.info(f"执行最小聚类异常检测, 参数: n_clusters={n_clusters}")
    return run_multivariate_detection(
        data, MinClusterDetector, "MinClusterDetector", 
        base_model=KMeans(n_clusters=n_clusters)
    )

@mcp.tool(name="离群值异常检测", description="使用IsolationForest检测多变量异常")
def outlier_detector(data: List[Dict[str, Any]], contamination: float = 0.05) -> Dict[str, Any]:
    logger.info(f"执行离群值异常检测, 参数: contamination={contamination}")
    return run_multivariate_detection(
        data, OutlierDetector, "OutlierDetector", 
        base_model=IsolationForest(contamination=contamination)
    )

@mcp.tool(name="PCA异常检测", description="使用PCA检测多变量异常")
def pca_detector(data: List[Dict[str, Any]], k: int = 1, c: float = 3.0) -> Dict[str, Any]:
    logger.info(f"执行PCA异常检测, 参数: k={k}, c={c}")
    return run_multivariate_detection(data, PcaAD, "PcaAD", k=k, c=c)

@mcp.tool(name="回归异常检测", description="使用线性回归检测多变量异常")
def regression_detector(data: List[Dict[str, Any]], target: str, c: float = 3.0, side: str = "both") -> Dict[str, Any]:
    logger.info(f"执行回归异常检测, 参数: target={target}, c={c}, side={side}")
    try:
        df = convert_to_dataframe(data)
        if target not in df.columns:
            return {"error": f"目标列 {target} 不存在，可用列：{list(df.columns)}"}
        detector = RegressionAD(target=target, regressor=LinearRegression(), c=c, side=side)
        detector.fit(df)
        events = detector.detect(df, return_list=True)
        return format_detection_result("RegressionAD", events, len(df))
    except Exception as e:
        logger.error(f"RegressionAD 检测失败: {e}")
        return {"error": str(e)}

# 注册检测器元数据
DETECTOR_META = {
    # 单变量检测器
    "IQR异常检测": {
        "category": "univariate", 
        "method": "InterQuartileRangeAD", 
        "default_params": {"c": 3.0}, 
        "description": "基于四分位数范围的异常检测"
    },
    "广义ESD检测": {
        "category": "univariate", 
        "method": "GeneralizedESDTestAD", 
        "default_params": {"alpha": 0.05}, 
        "description": "Generalized ESD统计检验异常检测"
    },
    "分位数异常检测": {
        "category": "univariate", 
        "method": "QuantileAD", 
        "default_params": {"low": 0.05, "high": 0.95}, 
        "description": "基于分位数的异常检测"
    },
    "阈值异常检测": {
        "category": "univariate", 
        "method": "ThresholdAD", 
        "default_params": {}, 
        "description": "基于固定阈值的异常检测"
    },
    "持续性异常检测": {
        "category": "univariate", 
        "method": "PersistAD", 
        "default_params": {"window": 1, "c": 3.0}, 
        "description": "检测持续异常状态"
    },
    "水平位移检测": {
        "category": "univariate", 
        "method": "LevelShiftAD", 
        "default_params": {"window": 5, "c": 6.0}, 
        "description": "检测数据水平的突变"
    },
    "波动性变化检测": {
        "category": "univariate", 
        "method": "VolatilityShiftAD", 
        "default_params": {"window": 10, "c": 6.0}, 
        "description": "检测数据波动性的变化"
    },
    "季节性异常检测": {
        "category": "univariate", 
        "method": "SeasonalAD", 
        "default_params": {"freq": None, "c": 3.0}, 
        "description": "检测季节性数据的异常"
    },
    "自回归异常检测": {
        "category": "univariate", 
        "method": "AutoregressionAD", 
        "default_params": {"n_steps": 1, "step_size": 1, "c": 3.0}, 
        "description": "基于自回归模型的异常检测"
    },
    
    # 多变量检测器
    "最小聚类异常检测": {
        "category": "multivariate",
        "method": "MinClusterDetector",
        "default_params": {"n_clusters": 2},
        "description": "使用KMeans聚类检测多变量异常"
    },
    "离群值异常检测": {
        "category": "multivariate",
        "method": "OutlierDetector",
        "default_params": {"contamination": 0.05},
        "description": "使用IsolationForest检测多变量异常"
    },
    "PCA异常检测": {
        "category": "multivariate",
        "method": "PcaAD",
        "default_params": {"k": 1, "c": 3.0},
        "description": "使用PCA检测多变量异常"
    },
    "回归异常检测": {
        "category": "multivariate",
        "method": "RegressionAD",
        "default_params": {"c": 3.0, "side": "both", "target": "需要指定"},
        "description": "使用线性回归检测多变量异常"
    }
}

# ======================== 工具 ========================
@mcp.tool(name="获取所有检测方法信息", description="获取所有可用检测方法的信息")
def get_all_detectors() -> Dict[str, Any]:
    """返回所有注册的检测器信息"""
    logger.info("调用: 获取所有检测方法信息")
    return DETECTOR_META

@mcp.tool(name="ping", description="测试服务器连接")
def ping() -> str:
    logger.info("调用: ping")
    return "pong"

# ======================== 主入口 ========================
if __name__ == "__main__":
    try:
        logger.info("启动ADTK-MCP服务器...")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("服务器中断")
    except Exception as e:
        logger.error(f"运行失败: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)