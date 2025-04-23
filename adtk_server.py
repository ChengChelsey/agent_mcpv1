#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADTK-FastMCP 服务器（STDIO 模式，适配 mcp==1.6.*）

▪ 自动注册常用 ADTK 单变量 / 多变量检测器为 MCP Tool
▪ 提供统一工具：『获取所有检测方法信息』
▪ 保留一个『ping』测试工具
"""

from __future__ import annotations
import json
import os, sys, logging
sys.stdout = open(os.devnull, "w")    
from typing import Any, Dict, List, Tuple, Callable

# ────────────────── 日志：全部输出到 stderr ──────────────────
root = logging.getLogger()
for h in list(root.handlers):
    root.removeHandler(h)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("adtk_server")

# ────────────────── ADTK & 依赖 ──────────────────
import pandas as pd
from adtk.detector import (
    InterQuartileRangeAD, GeneralizedESDTestAD, QuantileAD, ThresholdAD,
    PersistAD, LevelShiftAD, VolatilityShiftAD, SeasonalAD, AutoregressionAD,
    MinClusterDetector, OutlierDetector, PcaAD, RegressionAD
)
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# ────────────────── MCP ──────────────────
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("adtk-stdio")
DETECTOR_META: Dict[str, Dict[str, Any]] = {}         # 元数据汇总

# ────────────────── 工具注册辅助 ──────────────────
def _to_series(ts_list: List[Tuple[int, float]]) -> pd.Series:
    if not ts_list:
        raise ValueError("series 不能为空")
    idx = pd.to_datetime([t for t, _ in ts_list], unit="s")
    return pd.Series([v for _, v in ts_list], index=idx)

def _wrap(det_name: str, events, total_points: int) -> str:
    pts, rng = [], []
    for ev in events:
        if isinstance(ev, tuple):
            rng.append([int(ev[0].timestamp()), int(ev[1].timestamp())])
        else:
            pts.append(int(ev.timestamp()))
    ratio = (len(pts) + len(rng) * 3) / max(total_points, 1)
    return json.dumps(
        {
            "method": det_name,
            "visual_type": "range" if rng else "point" if pts else "none",
            "anomalies": pts,
            "intervals": rng,
            "anomaly_ratio": round(ratio, 6),
            "explanation": [f"{det_name} 检出异常"] * (len(pts) + len(rng)),
        },
        ensure_ascii=False,
    )

def _register_univariate(
    det_cls, tool_name: str, description: str, default_kwargs: dict | None = None
):
    default_kwargs = default_kwargs or {}

    @mcp.tool(name=tool_name, description=description)
    def _(series: List[Tuple[int, float]], **kw):
        logger.info("执行单变量检测：%s", tool_name)
        ts = _to_series(series)
        det = det_cls(**{**default_kwargs, **kw})
        events = det.detect(ts, return_list=True)
        return _wrap(det_cls.__name__, events, len(ts))

    DETECTOR_META[tool_name] = {
        "category": "univariate",
        "method": det_cls.__name__,
        "default_params": default_kwargs,
        "description": description,
    }

def _register_multivariate(
    factory: Callable[[dict], Any],
    tool_name: str,
    description: str,
    default_kwargs: dict | None = None,
):
    default_kwargs = default_kwargs or {}

    @mcp.tool(name=tool_name, description=description)
    def _(series: List[Dict[str, Any]], **kw):
        logger.info("执行多变量检测：%s", tool_name)
        idx = pd.to_datetime([d["timestamp"] for d in series], unit="s")
        df = pd.DataFrame([d["features"] for d in series], index=idx)
        det = factory({**default_kwargs, **kw})
        events = det.detect(df, return_list=True)
        return _wrap(det.__class__.__name__, events, len(df))

    DETECTOR_META[tool_name] = {
        "category": "multivariate",
        "method": tool_name,
        "default_params": default_kwargs,
        "description": description,
    }

# ────────────────── 注册所有检测器 ──────────────────
# —— 单变量 ——（需要可自行增删）
_register_univariate(InterQuartileRangeAD, "IQR异常检测",
                     "Inter-Quartile Range 方法", {"c": 3.0})
_register_univariate(GeneralizedESDTestAD, "广义ESD检测",
                     "Generalized ESD", {"alpha": 0.05})
_register_univariate(QuantileAD, "分位数异常检测",
                     "分位数阈值检测", {"low": 0.05, "high": 0.95})
_register_univariate(ThresholdAD, "阈值异常检测",
                     "固定上下阈值")
_register_univariate(PersistAD, "持续性异常检测",
                     "窗口持续异常", {"window": 1, "c": 3.0})
_register_univariate(LevelShiftAD, "水平位移检测",
                     "水平漂移", {"window": 5, "c": 6.0})
_register_univariate(VolatilityShiftAD, "波动性变化检测",
                     "波动性突变", {"window": 10, "c": 6.0})
_register_univariate(SeasonalAD, "季节性异常检测",
                     "周期模式异常", {"freq": None, "c": 3.0})
_register_univariate(AutoregressionAD, "自回归异常检测",
                     "AR 残差检测", {"n_steps": 1, "step_size": 1, "c": 3.0})

# —— 多变量 ——
_register_multivariate(
    lambda p: MinClusterDetector(KMeans(n_clusters=p.get("n_clusters", 2))),
    "最小聚类异常检测", "KMeans + MinClusterDetector", {"n_clusters": 2},
)
_register_multivariate(
    lambda p: OutlierDetector(IsolationForest(contamination=p.get("contamination", 0.05))),
    "离群值异常检测", "IsolationForest + OutlierDetector", {"contamination": 0.05},
)
_register_multivariate(
    lambda p: PcaAD(k=p.get("k", 1), c=p.get("c", 5.0)),
    "PCA异常检测", "主成分分析异常检测", {"k": 1, "c": 5.0},
)
_register_multivariate(
    lambda p: RegressionAD(
        target=p["target"],
        regressor=LinearRegression(),
        c=p.get("c", 3.0),
        side=p.get("side", "both"),
    ),
    "回归异常检测", "多变量线性回归残差异常检测",
    {"c": 3.0, "side": "both", "target": "..."},
)

# ────────────────── 通用工具 ──────────────────
@mcp.tool(name="获取所有检测方法信息", description="列出所有 ADTK 检测器元数据")
def get_all_detectors() -> Dict[str, Any]:
    return DETECTOR_META

@mcp.tool(name="ping", description="测试 MCP 连通性")
def ping() -> str:
    return "pong"

# ────────────────── 运行（STDIO） ──────────────────
if __name__ == "__main__":
    mcp.run()     # FastMCP.run() 会启动 STDIO 服务器
