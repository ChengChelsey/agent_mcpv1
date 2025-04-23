#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版的ADTK-MCP服务器 (v4)
增加了模型训练步骤
"""
import sys
import json
import logging
import traceback
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("adtk_server")

# 导入MCP
try:
    from mcp.server.fastmcp import FastMCP, Context
except ImportError as e:
    logger.error(f"导入MCP依赖失败: {e}")
    sys.exit(1)

# 导入ADTK
try:
    import pandas as pd
    import numpy as np
    from adtk.detector import (
        InterQuartileRangeAD, 
        GeneralizedESDTestAD, 
        QuantileAD,
        ThresholdAD,
        PersistAD, 
        LevelShiftAD, 
        VolatilityShiftAD, 
        SeasonalAD, 
        AutoregressionAD
    )
except ImportError as e:
    logger.error(f"导入ADTK依赖失败: {e}")
    sys.exit(1)

# 创建MCP服务器
mcp = FastMCP("ADTK-Server")

# 检测器元数据
DETECTOR_META = {}

# 辅助函数：时间序列转换
def convert_to_series(ts_data):
    """将时间戳-值对列表转换为pandas.Series"""
    if not ts_data:
        raise ValueError("时间序列数据为空")
        
    try:
        idx = pd.to_datetime([t for t, _ in ts_data], unit='s')
        values = [v for _, v in ts_data]
        return pd.Series(values, index=idx)
    except Exception as e:
        logger.error(f"转换时间序列失败: {e}")
        raise ValueError(f"转换时间序列失败: {str(e)}")

# 辅助函数：格式化检测结果
def format_detection_result(detector_name, events, total_points):
    """格式化检测结果"""
    points = []
    ranges = []
    
    try:
        for event in events:
            if isinstance(event, tuple):  # 区间
                ranges.append([int(event[0].timestamp()), int(event[1].timestamp())])
            else:  # 点
                points.append(int(event.timestamp()))
                
        # 计算异常比例
        anomaly_ratio = (len(points) + len(ranges) * 3) / max(total_points, 1)
        
        # 构建结果
        result = {
            "method": detector_name,
            "visual_type": "range" if ranges else "point" if points else "none",
            "anomalies": points,
            "intervals": ranges,
            "anomaly_ratio": round(anomaly_ratio, 6),
            "explanation": [f"{detector_name} 检出异常"] * (len(points) + len(ranges))
        }
        
        return result
    except Exception as e:
        logger.error(f"格式化检测结果失败: {e}")
        return {"error": f"格式化检测结果失败: {str(e)}"}

# IQR异常检测
@mcp.tool(name="IQR异常检测", description="基于四分位数范围的异常检测")
def iqr_detector(series: List[List[float]], c: float = 3.0) -> Dict[str, Any]:
    """IQR异常检测实现"""
    logger.info(f"执行IQR异常检测, 参数c = {c}")
    
    try:
        # 转换时间序列
        ts = convert_to_series(series)
        logger.info(f"时间序列长度: {len(ts)}")
        
        # 创建检测器
        detector = InterQuartileRangeAD(c=c)
        
        # 先训练模型
        logger.info("训练IQR检测器...")
        detector.fit(ts)
        
        # 执行检测
        logger.info("执行IQR检测...")
        events = detector.detect(ts, return_list=True)
        logger.info(f"检测到 {len(events)} 个异常事件")
        
        # 返回结果
        result = format_detection_result("InterQuartileRangeAD", events, len(ts))
        return result
    except Exception as e:
        logger.error(f"IQR异常检测失败: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# 广义ESD检测
@mcp.tool(name="广义ESD检测", description="Generalized ESD统计检验异常检测")
def esd_detector(series: List[List[float]], alpha: float = 0.05) -> Dict[str, Any]:
    """广义ESD检测实现"""
    logger.info(f"执行广义ESD检测, 参数alpha = {alpha}")
    
    try:
        # 转换时间序列
        ts = convert_to_series(series)
        logger.info(f"时间序列长度: {len(ts)}")
        
        # 创建检测器
        detector = GeneralizedESDTestAD(alpha=alpha)
        
        # 先训练模型
        logger.info("训练ESD检测器...")
        detector.fit(ts)
        
        # 执行检测
        logger.info("执行ESD检测...")
        events = detector.detect(ts, return_list=True)
        logger.info(f"检测到 {len(events)} 个异常事件")
        
        # 返回结果
        result = format_detection_result("GeneralizedESDTestAD", events, len(ts))
        return result
    except Exception as e:
        logger.error(f"广义ESD检测失败: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# 分位数异常检测
@mcp.tool(name="分位数异常检测", description="基于分位数的异常检测")
def quantile_detector(series: List[List[float]], low: float = 0.05, high: float = 0.95) -> Dict[str, Any]:
    """分位数异常检测实现"""
    logger.info(f"执行分位数异常检测, 参数low = {low}, high = {high}")
    
    try:
        # 转换时间序列
        ts = convert_to_series(series)
        logger.info(f"时间序列长度: {len(ts)}")
        
        # 创建检测器
        detector = QuantileAD(low=low, high=high)
        
        # 先训练模型
        logger.info("训练分位数检测器...")
        detector.fit(ts)
        
        # 执行检测
        logger.info("执行分位数检测...")
        events = detector.detect(ts, return_list=True)
        logger.info(f"检测到 {len(events)} 个异常事件")
        
        # 返回结果
        result = format_detection_result("QuantileAD", events, len(ts))
        return result
    except Exception as e:
        logger.error(f"分位数异常检测失败: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# 阈值异常检测
@mcp.tool(name="阈值异常检测", description="基于固定阈值的异常检测")
def threshold_detector(series: List[List[float]], low: Optional[float] = None, high: Optional[float] = None) -> Dict[str, Any]:
    """阈值异常检测实现"""
    logger.info(f"执行阈值异常检测, 参数low = {low}, high = {high}")
    
    try:
        # 转换时间序列
        ts = convert_to_series(series)
        logger.info(f"时间序列长度: {len(ts)}")
        
        # 创建检测器
        detector = ThresholdAD(low=low, high=high)
        
        # 无需训练，但为了代码一致性，仍然调用fit
        logger.info("准备阈值检测...")
        detector.fit(ts)
        
        # 执行检测
        logger.info("执行阈值检测...")
        events = detector.detect(ts, return_list=True)
        logger.info(f"检测到 {len(events)} 个异常事件")
        
        # 返回结果
        result = format_detection_result("ThresholdAD", events, len(ts))
        return result
    except Exception as e:
        logger.error(f"阈值异常检测失败: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# 持续性异常检测
@mcp.tool(name="持续性异常检测", description="检测持续异常状态")
def persist_detector(series: List[List[float]], window: int = 1, c: float = 3.0) -> Dict[str, Any]:
    """持续性异常检测实现"""
    logger.info(f"执行持续性异常检测, 参数window = {window}, c = {c}")
    
    try:
        # 转换时间序列
        ts = convert_to_series(series)
        logger.info(f"时间序列长度: {len(ts)}")
        
        # 创建检测器
        detector = PersistAD(window=window, c=c)
        
        # 先训练模型
        logger.info("训练持续性检测器...")
        detector.fit(ts)
        
        # 执行检测
        logger.info("执行持续性检测...")
        events = detector.detect(ts, return_list=True)
        logger.info(f"检测到 {len(events)} 个异常事件")
        
        # 返回结果
        result = format_detection_result("PersistAD", events, len(ts))
        return result
    except Exception as e:
        logger.error(f"持续性异常检测失败: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# 水平位移检测
@mcp.tool(name="水平位移检测", description="检测数据水平的突变")
def level_shift_detector(series: List[List[float]], window: int = 5, c: float = 6.0) -> Dict[str, Any]:
    """水平位移检测实现"""
    logger.info(f"执行水平位移检测, 参数window = {window}, c = {c}")
    
    try:
        # 转换时间序列
        ts = convert_to_series(series)
        logger.info(f"时间序列长度: {len(ts)}")
        
        # 创建检测器
        detector = LevelShiftAD(window=window, c=c)
        
        # 先训练模型
        logger.info("训练水平位移检测器...")
        detector.fit(ts)
        
        # 执行检测
        logger.info("执行水平位移检测...")
        events = detector.detect(ts, return_list=True)
        logger.info(f"检测到 {len(events)} 个异常事件")
        
        # 返回结果
        result = format_detection_result("LevelShiftAD", events, len(ts))
        return result
    except Exception as e:
        logger.error(f"水平位移检测失败: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# 波动性变化检测
@mcp.tool(name="波动性变化检测", description="检测数据波动性的变化")
def volatility_shift_detector(series: List[List[float]], window: int = 10, c: float = 6.0) -> Dict[str, Any]:
    """波动性变化检测实现"""
    logger.info(f"执行波动性变化检测, 参数window = {window}, c = {c}")
    
    try:
        # 转换时间序列
        ts = convert_to_series(series)
        logger.info(f"时间序列长度: {len(ts)}")
        
        # 创建检测器
        detector = VolatilityShiftAD(window=window, c=c)
        
        # 先训练模型
        logger.info("训练波动性变化检测器...")
        detector.fit(ts)
        
        # 执行检测
        logger.info("执行波动性变化检测...")
        events = detector.detect(ts, return_list=True)
        logger.info(f"检测到 {len(events)} 个异常事件")
        
        # 返回结果
        result = format_detection_result("VolatilityShiftAD", events, len(ts))
        return result
    except Exception as e:
        logger.error(f"波动性变化检测失败: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# 季节性异常检测
@mcp.tool(name="季节性异常检测", description="检测季节性数据的异常")
def seasonal_detector(series: List[List[float]], freq: Optional[int] = None, c: float = 3.0) -> Dict[str, Any]:
    """季节性异常检测实现"""
    logger.info(f"执行季节性异常检测, 参数freq = {freq}, c = {c}")
    
    try:
        # 转换时间序列
        ts = convert_to_series(series)
        logger.info(f"时间序列长度: {len(ts)}")
        
        # 创建检测器
        detector = SeasonalAD(freq=freq, c=c)
        
        # 先训练模型
        logger.info("训练季节性检测器...")
        detector.fit(ts)
        
        # 执行检测
        logger.info("执行季节性检测...")
        events = detector.detect(ts, return_list=True)
        logger.info(f"检测到 {len(events)} 个异常事件")
        
        # 返回结果
        result = format_detection_result("SeasonalAD", events, len(ts))
        return result
    except Exception as e:
        logger.error(f"季节性异常检测失败: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# 自回归异常检测
@mcp.tool(name="自回归异常检测", description="基于自回归模型的异常检测")
def autoregression_detector(series: List[List[float]], n_steps: int = 1, step_size: int = 1, c: float = 3.0) -> Dict[str, Any]:
    """自回归异常检测实现"""
    logger.info(f"执行自回归异常检测, 参数n_steps = {n_steps}, step_size = {step_size}, c = {c}")
    
    try:
        # 转换时间序列
        ts = convert_to_series(series)
        logger.info(f"时间序列长度: {len(ts)}")
        
        # 创建检测器
        detector = AutoregressionAD(n_steps=n_steps, step_size=step_size, c=c)
        
        # 先训练模型
        logger.info("训练自回归检测器...")
        detector.fit(ts)
        
        # 执行检测
        logger.info("执行自回归检测...")
        events = detector.detect(ts, return_list=True)
        logger.info(f"检测到 {len(events)} 个异常事件")
        
        # 返回结果
        result = format_detection_result("AutoregressionAD", events, len(ts))
        return result
    except Exception as e:
        logger.error(f"自回归异常检测失败: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# 注册检测器信息
DETECTOR_META = {
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
    }
}

# 获取检测器列表工具
@mcp.tool(name="获取所有检测方法信息", description="获取所有可用检测方法的信息")
def get_all_detectors() -> Dict[str, Any]:
    """返回所有注册的检测器信息"""
    logger.info("调用: 获取所有检测方法信息")
    return DETECTOR_META

# 测试工具
@mcp.tool(name="ping", description="测试服务器连接")
def ping() -> str:
    """简单的连接测试"""
    logger.info("调用: ping")
    return "pong"

# 主入口
if __name__ == "__main__":
    try:
        logger.info("启动ADTK服务器...")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("服务器被中断")
    except Exception as e:
        logger.error(f"服务器运行出错: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)