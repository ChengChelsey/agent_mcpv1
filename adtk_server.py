#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""ADTK FastMCP Server

全部 ADTK 检测器自动注册为 MCP tool，
并新增统一入口 `获取所有检测方法信息` 供 LLM 一次性获取元数据。
需依赖 mcp>=1.6。
"""
import json
from typing import Any, Callable, Dict, List, Tuple
import argparse
import pandas as pd
import logging
from adtk.detector import (
    AutoregressionAD,
    GeneralizedESDTestAD,
    InterQuartileRangeAD,
    LevelShiftAD,
    MinClusterDetector,
    OutlierDetector,
    PcaAD,
    PersistAD,
    QuantileAD,
    RegressionAD,
    SeasonalAD,
    ThresholdAD,
    VolatilityShiftAD,
)
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from mcp.server.fastmcp import FastMCP

import sys, logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,          # ★ 关键：写 stderr，别污染 stdout
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("adtk_server")

# 创建MCP服务器实例
mcp = FastMCP("adtk-server")

# ---------------------------------------------------------------------------
# 全局元数据字典
# ---------------------------------------------------------------------------
DETECTOR_META: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _list_to_series(series: List[Tuple[int, float]]) -> pd.Series:
    if not series:
        raise ValueError("series 不能为空")
    idx = pd.to_datetime([t for t, _ in series], unit="s")
    return pd.Series([v for _, v in series], index=idx)


def _fit_or_detect(detector, data):
    return detector.fit_detect(data) if hasattr(detector, "fit_detect") else detector.detect(data)

def _wrap(det_name: str, events, total_points: int):
    """
    events: ADTK 返回的 event list
    total_points: 该序列的总点数
    """
    points, ranges = [], []
    for ev in events:
        if isinstance(ev, tuple):          # 区间
            ranges.append([int(ev[0].timestamp()),
                           int(ev[1].timestamp())])
        else:                              # 单点
            points.append(int(ev.timestamp()))

    # === 1) 计算 anomaly_ratio 作为该方法的「原始分」 ===
    #    · 一个单点算 1
    #    · 一个区间按 3 倍权重折算成点数（跟你原先的一致）
    anomaly_units = len(points) + len(ranges) * 3
    anomaly_ratio = anomaly_units / max(total_points, 1)

    return json.dumps({
        "method": det_name,
        "visual_type": "range" if ranges else "point" if points else "none",
        "anomalies": points,
        "intervals": ranges,
        "anomaly_ratio": round(anomaly_ratio, 6),      # 交给 agent 去 log→score
        # ↓ 其余字段只是保持兼容（想用可以用，不想用忽略）
        "explanation": [f"{det_name} 检出异常"] * anomaly_units,
    }, ensure_ascii=False)


def _register_univariate(det_cls, tool_name, description, default_kwargs=None):
    default_kwargs = default_kwargs or {}

    @mcp.tool(name=tool_name, description=description)
    def _tool(series: List[Tuple[int, float]], **kwargs):
        logger.info(f"正在执行 {tool_name} 检测")
        params = {**default_kwargs, **kwargs}
        ts = _list_to_series(series)
        det = det_cls(**params)
        events = det.detect(ts, return_list=True)
        result = _wrap(det_cls.__name__, events, len(ts))
        logger.info(f"{tool_name} 检测完成")
        return result

    DETECTOR_META[tool_name] = {
        "category": "univariate",
        "method": det_cls.__name__,
        "default_params": default_kwargs,
        "description": description,
    }

def _register_multivariate(factory: Callable, tool_name, description, default_kwargs=None):
    default_kwargs = default_kwargs or {}

    @mcp.tool(name=tool_name, description=description)
    def _tool(series: List[Dict[str, Any]], **kwargs):
        logger.info(f"正在执行 {tool_name} 检测")
        params = {**default_kwargs, **kwargs}
        idx = pd.to_datetime([d["timestamp"] for d in series], unit="s")
        df = pd.DataFrame([d["features"] for d in series], index=idx)
        det = factory(params)
        events = det.detect(df, return_list=True)
        result = _wrap(det.__class__.__name__, events, len(df))
        logger.info(f"{tool_name} 检测完成")
        return result

    DETECTOR_META[tool_name] = {
        "category": "multivariate",
        "method": tool_name,
        "default_params": default_kwargs,
        "description": description,
    }
# ---------------------------------------------------------------------------
# 注册所有检测器
# ---------------------------------------------------------------------------
_register_univariate(InterQuartileRangeAD, "IQR异常检测", "使用 IQR 方法检测离群值", {"c": 3.0})
_register_univariate(GeneralizedESDTestAD, "广义ESD检测", "Generalized ESD Test", {"alpha": 0.05})
_register_univariate(QuantileAD, "分位数异常检测", "基于分位数阈值检测", {"low": 0.05, "high": 0.95})
_register_univariate(ThresholdAD, "阈值异常检测", "固定上下阈值检测")
_register_univariate(PersistAD, "持续性异常检测", "窗口内持续异常检测", {"window": 1, "c": 3.0})
_register_univariate(LevelShiftAD, "水平位移检测", "检测时间序列水平漂移", {"window": 5, "c": 6.0})
_register_univariate(VolatilityShiftAD, "波动性变化检测", "检测波动性突变", {"window": 10, "c": 6.0})
_register_univariate(SeasonalAD, "季节性异常检测", "检测周期模式异常", {"freq": None, "c": 3.0})
_register_univariate(AutoregressionAD, "自回归异常检测", "基于自回归残差检测", {"n_steps": 1, "step_size": 1, "c": 3.0})

_register_multivariate(lambda p: MinClusterDetector(KMeans(n_clusters=p.get("n_clusters", 2))),
    "最小聚类异常检测", "KMeans + MinClusterDetector", {"n_clusters": 2})
_register_multivariate(lambda p: OutlierDetector(IsolationForest(contamination=p.get("contamination", 0.05))),
    "离群值异常检测", "IsolationForest + OutlierDetector", {"contamination": 0.05})
_register_multivariate(lambda p: PcaAD(k=p.get("k", 1), c=p.get("c", 5.0)),
    "PCA异常检测", "主成分分析异常检测", {"k": 1, "c": 5.0})
_register_multivariate(lambda p: RegressionAD(target=p["target"], regressor=LinearRegression(), c=p.get("c", 3.0), side=p.get("side", "both")),
    "回归异常检测", "多变量线性回归残差异常检测", {"c": 3.0, "side": "both"})

# ---------------------------------------------------------------------------
# 统一入口：获取所有检测方法信息
# ---------------------------------------------------------------------------
@mcp.tool(name="获取所有检测方法信息", description="列出所有 ADTK 检测器元数据")
def get_all_detectors() -> Dict[str, Any]:
    logger.info("调用获取所有检测方法信息")
    return DETECTOR_META

# 添加简单的测试工具
@mcp.tool(name="ping", description="测试MCP连接")
def ping() -> str:
    logger.info("收到ping请求")
    return "pong from ADTK server"

# ---------------------------------------------------------------------------
# 运行入口
# ---------------------------------------------------------------------------
def main():
    import argparse, logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7777,
                        help="HTTP 端口（客户端也要用同一端口）")
    args = parser.parse_args()

    logging.getLogger("adtk_server").info("启动 FastMCP (HTTP-SSE) 端口 %d", args.port)
    from mcp.server.fastmcp import FastMCP
    FastMCP._default_port = args.port          # 强制端口
    mcp.run()                                  # 保持原写法

if __name__ == "__main__":
    mcp.run()    