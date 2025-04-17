"""
ADTK MCP服务器

这个模块实现了一个MCP服务器，将ADTK库中的异常检测算法作为工具暴露出来。
适配MCP 1.6.0版本，使用@mcptool装饰器。
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from io import StringIO
import contextlib
import inspect
import traceback

# ADTK 库
import adtk
from adtk.detector import *

# 多变量检测所需的库
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# 为MCP 1.6.0版本导入正确的模块
from mcp import MCPServer
from mcp.server import MCPServerParameters
from mcp.core.tool import Tool, ToolCall
from mcp.server.tools import mcptool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("adtk_server")

# 默认检测方法权重配置
DEFAULT_WEIGHTS = {
    "InterQuartileRangeAD": 0.5,
    "GeneralizedESDTestAD": 0.6,
    "QuantileAD": 0.4,
    "ThresholdAD": 0.3,
    "PersistAD": 0.5,
    "LevelShiftAD": 0.6,
    "VolatilityShiftAD": 0.5,
    "SeasonalAD": 0.7,
    "AutoregressionAD": 0.5,
    "MinClusterDetector": 0.4,
    "OutlierDetector": 0.5,
    "PcaAD": 0.6,
    "RegressionAD": 0.5
}

class ADTKServer(MCPServer):
    """ADTK MCP服务器类，注册ADTK库中的异常检测方法作为工具"""
    
    def __init__(self, server_params: MCPServerParameters):
        super().__init__(server_params)
        self.detector_cache = {}  # 缓存创建的检测器实例
    
    @mcptool("获取所有检测方法信息", "获取所有可用的ADTK检测方法信息")
    async def get_all_detectors(self) -> str:
        """获取所有可用的ADTK检测方法信息"""
        try:
            # 捕获print_all_models()的输出
            output = StringIO()
            with contextlib.redirect_stdout(output):
                adtk.detector.print_all_models()
            
            models_text = output.getvalue()
            
            # 获取检测器类和描述信息
            detector_info = self._extract_detector_info()
            
            # 添加原始文档以便大模型理解
            detector_info["原始方法文档"] = models_text
            
            return json.dumps(detector_info, ensure_ascii=False)
        except Exception as e:
            logger.error(f"获取检测方法信息错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"获取检测方法信息失败: {str(e)}"})
    
    def _extract_detector_info(self) -> Dict:
        """提取检测器信息，包括类名、描述、参数等"""
        detector_info = {
            "单变量检测器": [],
            "多变量检测器": []
        }
        
        # 获取所有检测器类
        detector_classes = [obj for name, obj in inspect.getmembers(sys.modules["adtk.detector"]) 
                           if inspect.isclass(obj) and (name.endswith('AD') or 'Detector' in name)]
        
        for cls in detector_classes:
            # 提取类的文档字符串
            doc_string = inspect.getdoc(cls) or ""
            
            # 提取参数信息
            parameters = []
            if hasattr(cls, "__init__"):
                sig = inspect.signature(cls.__init__)
                for param_name, param in sig.parameters.items():
                    if param_name not in ['self', 'args', 'kwargs']:
                        param_info = {
                            "名称": param_name,
                            "默认值": None if param.default is param.empty else str(param.default),
                            "类型": str(param.annotation) if param.annotation is not param.empty else "未指定"
                        }
                        parameters.append(param_info)
            
            # 判断是单变量还是多变量检测器
            category = "多变量检测器" if "HD" in cls.__name__ or cls.__name__ in ["MinClusterDetector", "OutlierDetector", "PcaAD", "RegressionAD"] else "单变量检测器"
            
            # 构建检测器信息
            detector_data = {
                "类名": cls.__name__,
                "描述": doc_string.split("\n\n")[0] if doc_string else "",
                "参数": parameters,
                "默认权重": DEFAULT_WEIGHTS.get(cls.__name__, 0.5),
                "适用场景": self._get_detector_scenario(cls.__name__)
            }
            
            detector_info[category].append(detector_data)
        
        return detector_info
    
    def _get_detector_scenario(self, class_name: str) -> str:
        """返回检测器的适用场景描述"""
        scenarios = {
            "InterQuartileRangeAD": "适用于检测偏离数据中位数过远的异常点，对数据分布没有特定要求",
            "GeneralizedESDTestAD": "适用于近似正态分布的数据的异常检测，特别适合需要统计显著性的场景",
            "QuantileAD": "适用于有明显上下阈值的数据，对于长尾分布或偏斜分布表现良好",
            "ThresholdAD": "适用于已知明确阈值的监控指标，如CPU使用率、内存使用率等",
            "PersistAD": "适用于应该具有连续性或稳定性的指标，能够检测数据的突变点",
            "LevelShiftAD": "适用于检测时序数据中的水平位移或层级变化，如系统重启后的性能变化",
            "VolatilityShiftAD": "适用于需要检测波动性突变的指标，如网络延迟、响应时间的不稳定",
            "SeasonalAD": "适用于具有明显周期性的数据，如每天、每周或每月有规律波动的指标",
            "AutoregressionAD": "适用于具有自相关性的时序数据，当前值可以由过去值预测的情况",
            "MinClusterDetector": "适用于多变量时序数据的异常检测，能够识别具有相似模式的异常组",
            "OutlierDetector": "适用于多变量时序数据的离群点检测，特别是数据维度高时",
            "PcaAD": "适用于高维数据的异常检测，能够捕捉变量之间的相关性异常",
            "RegressionAD": "适用于变量间存在线性关系的多变量数据，如某些指标可以预测其他指标"
        }
        return scenarios.get(class_name, "通用异常检测方法")
    
    @mcptool("IQR异常检测", "使用IQR方法进行异常检测")
    async def detect_with_iqr(self, data: str, params: str) -> str:
        """使用IQR方法进行异常检测
        
        Args:
            data: JSON字符串，包含时间序列数据，格式为 {"series": [[timestamp, value], ...]}
            params: JSON字符串，包含检测参数，如 {"c": 3.0}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            series_data = data_dict.get("series", [])
            c = params_dict.get("c", 3.0)
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("InterQuartileRangeAD", 0.5))
            
            # 转换为pandas Series
            timestamps = [point[0] for point in series_data]
            values = [point[1] for point in series_data]
            ts = pd.Series(values, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建检测器
            detector = InterQuartileRangeAD(c=c)
            
            # 进行检测
            anomalies = detector.fit_detect(ts)
            
            # 提取异常点
            anomaly_timestamps = []
            anomaly_values = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    value = float(ts[idx])
                    anomaly_timestamps.append(timestamp)
                    anomaly_values.append(value)
            
            # 准备结果
            result = {
                "method": "InterQuartileRangeAD",
                "anomalies": anomaly_timestamps,
                "anomaly_values": anomaly_values,
                "parameters": {"c": c, "weight": weight},
                "total_points": len(series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(series_data) if len(series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"IQR检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"IQR检测失败: {str(e)}"})
    
    @mcptool("广义ESD检测", "使用广义ESD检验进行异常检测")
    async def detect_with_esd(self, data: str, params: str) -> str:
        """使用广义ESD检验进行异常检测
        
        Args:
            data: JSON字符串，包含时间序列数据，格式为 {"series": [[timestamp, value], ...]}
            params: JSON字符串，包含检测参数，如 {"alpha": 0.05}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            series_data = data_dict.get("series", [])
            alpha = params_dict.get("alpha", 0.05)
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("GeneralizedESDTestAD", 0.6))
            
            # 转换为pandas Series
            timestamps = [point[0] for point in series_data]
            values = [point[1] for point in series_data]
            ts = pd.Series(values, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建检测器
            detector = GeneralizedESDTestAD(alpha=alpha)
            
            # 进行检测
            anomalies = detector.fit_detect(ts)
            
            # 提取异常点
            anomaly_timestamps = []
            anomaly_values = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    value = float(ts[idx])
                    anomaly_timestamps.append(timestamp)
                    anomaly_values.append(value)
            
            # 准备结果
            result = {
                "method": "GeneralizedESDTestAD",
                "anomalies": anomaly_timestamps,
                "anomaly_values": anomaly_values,
                "parameters": {"alpha": alpha, "weight": weight},
                "total_points": len(series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(series_data) if len(series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"ESD检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"ESD检测失败: {str(e)}"})
    
    @mcptool("分位数异常检测", "使用分位数方法进行异常检测")
    async def detect_with_quantile(self, data: str, params: str) -> str:
        """使用分位数方法进行异常检测
        
        Args:
            data: JSON字符串，包含时间序列数据，格式为 {"series": [[timestamp, value], ...]}
            params: JSON字符串，包含检测参数，如 {"low": 0.05, "high": 0.95}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            series_data = data_dict.get("series", [])
            low = params_dict.get("low", 0.05)
            high = params_dict.get("high", 0.95)
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("QuantileAD", 0.4))
            
            # 转换为pandas Series
            timestamps = [point[0] for point in series_data]
            values = [point[1] for point in series_data]
            ts = pd.Series(values, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建检测器
            detector = QuantileAD(low=low, high=high)
            
            # 进行检测
            anomalies = detector.fit_detect(ts)
            
            # 提取异常点
            anomaly_timestamps = []
            anomaly_values = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    value = float(ts[idx])
                    anomaly_timestamps.append(timestamp)
                    anomaly_values.append(value)
            
            # 准备结果
            result = {
                "method": "QuantileAD",
                "anomalies": anomaly_timestamps,
                "anomaly_values": anomaly_values,
                "parameters": {"low": low, "high": high, "weight": weight},
                "total_points": len(series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(series_data) if len(series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"分位数检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"分位数检测失败: {str(e)}"})
    
    @mcptool("阈值异常检测", "使用固定阈值进行异常检测")
    async def detect_with_threshold(self, data: str, params: str) -> str:
        """使用固定阈值进行异常检测
        
        Args:
            data: JSON字符串，包含时间序列数据，格式为 {"series": [[timestamp, value], ...]}
            params: JSON字符串，包含检测参数，如 {"low": 10, "high": 90}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            series_data = data_dict.get("series", [])
            low = params_dict.get("low")
            high = params_dict.get("high")
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("ThresholdAD", 0.3))
            
            # 转换为pandas Series
            timestamps = [point[0] for point in series_data]
            values = [point[1] for point in series_data]
            ts = pd.Series(values, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建检测器
            detector = ThresholdAD(low=low, high=high)
            
            # 进行检测
            anomalies = detector.detect(ts)  # 注意ThresholdAD不需要fit
            
            # 提取异常点
            anomaly_timestamps = []
            anomaly_values = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    value = float(ts[idx])
                    anomaly_timestamps.append(timestamp)
                    anomaly_values.append(value)
            
            # 准备结果
            result = {
                "method": "ThresholdAD",
                "anomalies": anomaly_timestamps,
                "anomaly_values": anomaly_values,
                "parameters": {"low": low, "high": high, "weight": weight},
                "total_points": len(series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(series_data) if len(series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"阈值检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"阈值检测失败: {str(e)}"})
    
    @mcptool("持续性异常检测", "使用持续性检测方法进行异常检测")
    async def detect_with_persist(self, data: str, params: str) -> str:
        """使用持续性检测方法进行异常检测
        
        Args:
            data: JSON字符串，包含时间序列数据，格式为 {"series": [[timestamp, value], ...]}
            params: JSON字符串，包含检测参数，如 {"window": 5, "c": 3.0, "side": "both"}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            series_data = data_dict.get("series", [])
            window = params_dict.get("window", 1)
            c = params_dict.get("c", 3.0)
            side = params_dict.get("side", "both")
            min_periods = params_dict.get("min_periods", None)
            agg = params_dict.get("agg", "median")
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("PersistAD", 0.5))
            
            # 转换为pandas Series
            timestamps = [point[0] for point in series_data]
            values = [point[1] for point in series_data]
            ts = pd.Series(values, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建检测器
            detector = PersistAD(window=window, c=c, side=side, min_periods=min_periods, agg=agg)
            
            # 进行检测
            anomalies = detector.fit_detect(ts)
            
            # 提取异常点
            anomaly_timestamps = []
            anomaly_values = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    value = float(ts[idx])
                    anomaly_timestamps.append(timestamp)
                    anomaly_values.append(value)
            
            # 准备结果
            result = {
                "method": "PersistAD",
                "anomalies": anomaly_timestamps,
                "anomaly_values": anomaly_values,
                "parameters": {
                    "window": window,
                    "c": c,
                    "side": side,
                    "min_periods": min_periods,
                    "agg": agg,
                    "weight": weight
                },
                "total_points": len(series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(series_data) if len(series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"持续性检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"持续性检测失败: {str(e)}"})
    
    @mcptool("水平位移检测", "使用水平位移检测方法进行异常检测")
    async def detect_with_level_shift(self, data: str, params: str) -> str:
        """使用水平位移检测方法进行异常检测
        
        Args:
            data: JSON字符串，包含时间序列数据，格式为 {"series": [[timestamp, value], ...]}
            params: JSON字符串，包含检测参数，如 {"window": 5, "c": 6.0, "side": "both"}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            series_data = data_dict.get("series", [])
            window = params_dict.get("window", 5)
            c = params_dict.get("c", 6.0)
            side = params_dict.get("side", "both")
            min_periods = params_dict.get("min_periods", None)
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("LevelShiftAD", 0.6))
            
            # 转换为pandas Series
            timestamps = [point[0] for point in series_data]
            values = [point[1] for point in series_data]
            ts = pd.Series(values, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建检测器
            detector = LevelShiftAD(window=window, c=c, side=side, min_periods=min_periods)
            
            # 进行检测
            anomalies = detector.fit_detect(ts)
            
            # 提取异常点
            anomaly_timestamps = []
            anomaly_values = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    value = float(ts[idx])
                    anomaly_timestamps.append(timestamp)
                    anomaly_values.append(value)
            
            # 准备结果
            result = {
                "method": "LevelShiftAD",
                "anomalies": anomaly_timestamps,
                "anomaly_values": anomaly_values,
                "parameters": {
                    "window": window,
                    "c": c,
                    "side": side,
                    "min_periods": min_periods,
                    "weight": weight
                },
                "total_points": len(series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(series_data) if len(series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"水平位移检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"水平位移检测失败: {str(e)}"})
    
    @mcptool("波动性变化检测", "使用波动性变化检测方法进行异常检测")
    async def detect_with_volatility_shift(self, data: str, params: str) -> str:
        """使用波动性变化检测方法进行异常检测
        
        Args:
            data: JSON字符串，包含时间序列数据，格式为 {"series": [[timestamp, value], ...]}
            params: JSON字符串，包含检测参数，如 {"window": 10, "c": 6.0, "side": "both", "agg": "std"}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            series_data = data_dict.get("series", [])
            window = params_dict.get("window", 10)
            c = params_dict.get("c", 6.0)
            side = params_dict.get("side", "both")
            min_periods = params_dict.get("min_periods", None)
            agg = params_dict.get("agg", "std")
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("VolatilityShiftAD", 0.5))
            
            # 转换为pandas Series
            timestamps = [point[0] for point in series_data]
            values = [point[1] for point in series_data]
            ts = pd.Series(values, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建检测器
            detector = VolatilityShiftAD(window=window, c=c, side=side, min_periods=min_periods, agg=agg)
            
            # 进行检测
            anomalies = detector.fit_detect(ts)
            
            # 提取异常点
            anomaly_timestamps = []
            anomaly_values = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    value = float(ts[idx])
                    anomaly_timestamps.append(timestamp)
                    anomaly_values.append(value)
            
            # 准备结果
            result = {
                "method": "VolatilityShiftAD",
                "anomalies": anomaly_timestamps,
                "anomaly_values": anomaly_values,
                "parameters": {
                    "window": window,
                    "c": c,
                    "side": side,
                    "min_periods": min_periods,
                    "agg": agg,
                    "weight": weight
                },
                "total_points": len(series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(series_data) if len(series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"波动性变化检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"波动性变化检测失败: {str(e)}"})
    
    @mcptool("季节性异常检测", "使用季节性异常检测方法进行异常检测")
    async def detect_with_seasonal(self, data: str, params: str) -> str:
        """使用季节性异常检测方法进行异常检测
        
        Args:
            data: JSON字符串，包含时间序列数据，格式为 {"series": [[timestamp, value], ...]}
            params: JSON字符串，包含检测参数，如 {"freq": 24, "c": 3.0, "side": "both", "trend": false}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            series_data = data_dict.get("series", [])
            freq = params_dict.get("freq")
            c = params_dict.get("c", 3.0)
            side = params_dict.get("side", "both")
            trend = params_dict.get("trend", False)
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("SeasonalAD", 0.7))
            
            # 转换为pandas Series
            timestamps = [point[0] for point in series_data]
            values = [point[1] for point in series_data]
            ts = pd.Series(values, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建检测器
            detector = SeasonalAD(freq=freq, c=c, side=side, trend=trend)
            
            # 进行检测
            anomalies = detector.fit_detect(ts)
            
            # 提取异常点
            anomaly_timestamps = []
            anomaly_values = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    value = float(ts[idx])
                    anomaly_timestamps.append(timestamp)
                    anomaly_values.append(value)
            
            # 准备结果
            result = {
                "method": "SeasonalAD",
                "anomalies": anomaly_timestamps,
                "anomaly_values": anomaly_values,
                "parameters": {
                    "freq": freq,
                    "c": c,
                    "side": side,
                    "trend": trend,
                    "weight": weight
                },
                "total_points": len(series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(series_data) if len(series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"季节性检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"季节性检测失败: {str(e)}"})
    
    @mcptool("自回归异常检测", "使用自回归异常检测方法进行异常检测")
    async def detect_with_autoregression(self, data: str, params: str) -> str:
        """使用自回归异常检测方法进行异常检测
        
        Args:
            data: JSON字符串，包含时间序列数据，格式为 {"series": [[timestamp, value], ...]}
            params: JSON字符串，包含检测参数，如 {"n_steps": 1, "c": 3.0, "side": "both"}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            series_data = data_dict.get("series", [])
            n_steps = params_dict.get("n_steps", 1)
            step_size = params_dict.get("step_size", 1)
            c = params_dict.get("c", 3.0)
            side = params_dict.get("side", "both")
            regressor = params_dict.get("regressor")  # 默认为None，使用线性回归
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("AutoregressionAD", 0.5))
            
            # 转换为pandas Series
            timestamps = [point[0] for point in series_data]
            values = [point[1] for point in series_data]
            ts = pd.Series(values, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建检测器
            detector = AutoregressionAD(n_steps=n_steps, step_size=step_size, c=c, side=side)
            
            # 进行检测
            anomalies = detector.fit_detect(ts)
            
            # 提取异常点
            anomaly_timestamps = []
            anomaly_values = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    value = float(ts[idx])
                    anomaly_timestamps.append(timestamp)
                    anomaly_values.append(value)
            
            # 准备结果
            result = {
                "method": "AutoregressionAD",
                "anomalies": anomaly_timestamps,
                "anomaly_values": anomaly_values,
                "parameters": {
                    "n_steps": n_steps,
                    "step_size": step_size,
                    "c": c,
                    "side": side,
                    "weight": weight
                },
                "total_points": len(series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(series_data) if len(series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"自回归检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"自回归检测失败: {str(e)}"})
    
    @mcptool("最小聚类异常检测", "使用最小聚类异常检测方法进行异常检测")
    async def detect_with_min_cluster(self, data: str, params: str) -> str:
        """使用最小聚类异常检测方法进行异常检测（多变量检测）
        
        Args:
            data: JSON字符串，包含多变量时间序列数据，格式为 {"series": [{"timestamp": t, "features": [v1, v2, ...]}, ...]}
            params: JSON字符串，包含检测参数，如 {"n_clusters": 2}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            multi_series_data = data_dict.get("series", [])
            n_clusters = params_dict.get("n_clusters", 2)
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("MinClusterDetector", 0.4))
            
            # 转换为pandas DataFrame
            timestamps = [item["timestamp"] for item in multi_series_data]
            features_data = [item["features"] for item in multi_series_data]
            
            # 创建DataFrame
            df = pd.DataFrame(features_data, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建聚类模型
            model = KMeans(n_clusters=n_clusters)
            
            # 创建检测器
            detector = MinClusterDetector(model)
            
            # 进行检测
            anomalies = detector.fit_detect(df)
            
            # 提取异常点
            anomaly_timestamps = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    anomaly_timestamps.append(timestamp)
            
            # 准备结果
            result = {
                "method": "MinClusterDetector",
                "anomalies": anomaly_timestamps,
                "parameters": {"n_clusters": n_clusters, "weight": weight},
                "total_points": len(multi_series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(multi_series_data) if len(multi_series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"最小聚类检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"最小聚类检测失败: {str(e)}"})
    
    @mcptool("离群值异常检测", "使用离群值检测方法进行异常检测")
    async def detect_with_outlier(self, data: str, params: str) -> str:
        """使用离群值检测方法进行异常检测（多变量检测）
        
        Args:
            data: JSON字符串，包含多变量时间序列数据，格式为 {"series": [{"timestamp": t, "features": [v1, v2, ...]}, ...]}
            params: JSON字符串，包含检测参数，如 {"contamination": 0.05}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            multi_series_data = data_dict.get("series", [])
            contamination = params_dict.get("contamination", 0.05)
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("OutlierDetector", 0.5))
            
            # 转换为pandas DataFrame
            timestamps = [item["timestamp"] for item in multi_series_data]
            features_data = [item["features"] for item in multi_series_data]
            
            # 创建DataFrame
            df = pd.DataFrame(features_data, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建离群值检测模型
            model = IsolationForest(contamination=contamination)
            
            # 创建检测器
            detector = OutlierDetector(model)
            
            # 进行检测
            anomalies = detector.fit_detect(df)
            
            # 提取异常点
            anomaly_timestamps = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    anomaly_timestamps.append(timestamp)
            
            # 准备结果
            result = {
                "method": "OutlierDetector",
                "anomalies": anomaly_timestamps,
                "parameters": {"contamination": contamination, "weight": weight},
                "total_points": len(multi_series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(multi_series_data) if len(multi_series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"离群值检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"离群值检测失败: {str(e)}"})
    
    @mcptool("PCA异常检测", "使用PCA进行异常检测")
    async def detect_with_pca(self, data: str, params: str) -> str:
        """使用PCA进行异常检测（多变量检测）
        
        Args:
            data: JSON字符串，包含多变量时间序列数据，格式为 {"series": [{"timestamp": t, "features": [v1, v2, ...]}, ...]}
            params: JSON字符串，包含检测参数，如 {"k": 1, "c": 5.0}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            multi_series_data = data_dict.get("series", [])
            k = params_dict.get("k", 1)
            c = params_dict.get("c", 5.0)
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("PcaAD", 0.6))
            
            # 转换为pandas DataFrame
            timestamps = [item["timestamp"] for item in multi_series_data]
            features_data = [item["features"] for item in multi_series_data]
            
            # 创建DataFrame
            df = pd.DataFrame(features_data, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建检测器
            detector = PcaAD(k=k, c=c)
            
            # 进行检测
            anomalies = detector.fit_detect(df)
            
            # 提取异常点
            anomaly_timestamps = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    anomaly_timestamps.append(timestamp)
            
            # 准备结果
            result = {
                "method": "PcaAD",
                "anomalies": anomaly_timestamps,
                "parameters": {"k": k, "c": c, "weight": weight},
                "total_points": len(multi_series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(multi_series_data) if len(multi_series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"PCA检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"PCA检测失败: {str(e)}"})
    
    @mcptool("回归异常检测", "使用回归进行异常检测")
    async def detect_with_regression(self, data: str, params: str) -> str:
        """使用回归进行异常检测（多变量检测）
        
        Args:
            data: JSON字符串，包含多变量时间序列数据，格式为 {"series": [{"timestamp": t, "features": {"feature1": v1, "feature2": v2, ...}}, ...]}
            params: JSON字符串，包含检测参数，如 {"target": "feature1", "c": 3.0, "side": "both"}
            
        Returns:
            检测结果的JSON字符串
        """
        try:
            # 解析输入数据和参数
            data_dict = json.loads(data)
            params_dict = json.loads(params)
            
            multi_series_data = data_dict.get("series", [])
            target = params_dict.get("target")
            c = params_dict.get("c", 3.0)
            side = params_dict.get("side", "both")
            weight = params_dict.get("weight", DEFAULT_WEIGHTS.get("RegressionAD", 0.5))
            
            if not target:
                return json.dumps({"error": "必须指定目标特征 (target)"})
            
            # 转换为pandas DataFrame
            timestamps = [item["timestamp"] for item in multi_series_data]
            features_data = [item["features"] for item in multi_series_data]
            
            # 创建DataFrame
            df = pd.DataFrame(features_data, index=pd.to_datetime(timestamps, unit='s'))
            
            # 创建线性回归模型
            regressor = LinearRegression()
            
            # 创建检测器
            detector = RegressionAD(target=target, regressor=regressor, c=c, side=side)
            
            # 进行检测
            anomalies = detector.fit_detect(df)
            
            # 提取异常点
            anomaly_timestamps = []
            for idx, is_anomaly in anomalies.items():
                if is_anomaly:
                    # 转换为整数时间戳
                    timestamp = int(idx.timestamp())
                    anomaly_timestamps.append(timestamp)
            
            # 准备结果
            result = {
                "method": "RegressionAD",
                "anomalies": anomaly_timestamps,
                "parameters": {"target": target, "c": c, "side": side, "weight": weight},
                "total_points": len(multi_series_data),
                "anomaly_count": len(anomaly_timestamps),
                "anomaly_ratio": len(anomaly_timestamps) / len(multi_series_data) if len(multi_series_data) > 0 else 0
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"回归检测错误: {str(e)}")
            traceback.print_exc()
            return json.dumps({"error": f"回归检测失败: {str(e)}"})

# 启动服务器的函数
async def start_server(port=7777):
    """启动ADTK MCP服务器"""
    try:
        server_params = MCPServerParameters(port=port)
        server = ADTKServer(server_params)
        
        logger.info(f"启动ADTK MCP服务器，端口: {port}")
        await server.start()
        return server
    except Exception as e:
        logger.error(f"启动ADTK MCP服务器失败: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import asyncio
    
    port = 7777
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    # 创建事件循环
    loop = asyncio.get_event_loop()
    
    # 启动服务器
    server = None
    try:
        server = loop.run_until_complete(start_server(port))
        if server:
            # 保持服务器运行
            loop.run_forever()
    except KeyboardInterrupt:
        logger.info("接收到终止信号，准备关闭服务器...")
    finally:
        # 关闭服务器
        if server:
            loop.run_until_complete(server.stop())
        
        # 关闭事件循环
        loop.close()