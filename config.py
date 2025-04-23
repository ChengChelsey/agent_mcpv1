"""
时序异常检测系统配置文件
包含阈值设置、权重配置等
"""

# 异常严重程度阈值
MILD_ANOMALY_THRESHOLD = 0.25  # 轻度异常阈值
HIGH_ANOMALY_THRESHOLD = 0.6   # 高置信度异常阈值

# 单序列检测方法权重配置
WEIGHTS_SINGLE = {
    "IQR异常检测": 0.4,
    "广义ESD检测": 0.3,
    "分位数异常检测": 0.3,
    "阈值异常检测": 0.5,
    "持续性异常检测": 0.4,
    "水平位移检测": 0.5,
    "波动性变化检测": 0.4,
    "季节性异常检测": 0.5,
    "自回归异常检测": 0.4
}

# 多序列对比检测方法权重配置
WEIGHTS_MULTI = {
    "IQR异常检测": 0.3,
    "广义ESD检测": 0.3,
    "分位数异常检测": 0.3,
    "阈值异常检测": 0.4,
    "持续性异常检测": 0.3,
    "水平位移检测": 0.4,
    "波动性变化检测": 0.4,
    "季节性异常检测": 0.4,
    "自回归异常检测": 0.3
}

# 检测方法阈值配置
THRESHOLD_CONFIG = {
    "Z-Score": {
        "threshold": 3.5
    },
    "CUSUM": {
        "drift_threshold": 6.0,
        "k": 0.7
    },
    "IQR异常检测": {
        "c": 3.0
    },
    "广义ESD检测": {
        "alpha": 0.05
    },
    "分位数异常检测": {
        "low": 0.05,
        "high": 0.95
    },
    "持续性异常检测": {
        "window": 1,
        "c": 3.0
    },
    "水平位移检测": {
        "window": 5,
        "c": 6.0
    },
    "波动性变化检测": {
        "window": 10,
        "c": 6.0
    },
    "季节性异常检测": {
        "freq": None,  # 自动推断
        "c": 3.0
    },
    "自回归异常检测": {
        "n_steps": 1,
        "step_size": 1,
        "c": 3.0
    },
    "ResidualComparison": {
        "threshold": 3.5
    },
    "TrendDriftCUSUM": {
        "drift_threshold": 8.0
    },
    "ChangeRate": {
        "threshold": 0.7
    },
    "TrendSlope": {
        "slope_threshold": 0.4,
        "window": 5
    }
}

# 异常区间分组的最大间隔（秒）
MAX_ANOMALY_GAP = 1800  # 30分钟

# 系统路径设置
OUTPUT_DIR = "output/plots"
CACHE_DIR = "cached_data"

# API设置
AIOPS_BACKEND_DOMAIN = 'https://aiopsbackend.cstcloud.cn'
LLM_URL = 'http://10.16.1.16:58000/v1/chat/completions'
AUTH = ('chelseyyycheng@outlook.com', 'UofV1uwHwhVp9tcTue')