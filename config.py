# config.py
import json
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anomaly_detection")

# 各个检测方法的权重配置 - 调整权重使得结果更合理
WEIGHTS_SINGLE = {
    "Z-Score": 0.5,
    "CUSUM": 0.5
}

WEIGHTS_MULTI = {
    "ResidualComparison": 0.4,  # 提高残差对比的权重
    "TrendDriftCUSUM": 0.25,    # 降低CUSUM的权重
    "ChangeRate": 0.15,
    "TrendSlope": 0.2
}

# 综合分数阈值 - 提高阈值，减少误报
HIGH_ANOMALY_THRESHOLD = 0.7
MILD_ANOMALY_THRESHOLD = 0.4

# 默认的各检测方法阈值配置 - 更加严格的阈值
DEFAULT_THRESHOLD_CONFIG = {
    "Z-Score": {
        "threshold": 3.5  # 增大Z-Score阈值，减少误检
    },
    "CUSUM": {
        "drift_threshold": 6.0,  # 增大偏移阈值
        "k": 0.7  # 增大k值，减少对小波动的敏感度
    },
    "ResidualComparison": {
        "threshold": 3.5
    },
    "TrendDriftCUSUM": {
        "drift_threshold": 8.0  # 显著增大阈值减少误报
    },
    "ChangeRate": {
        "threshold": 0.7  # 增大变化率阈值
    },
    "TrendSlope": {
        "slope_threshold": 0.4,  # 增大斜率阈值
        "window": 5
    }
}

# 创建配置目录
CONFIG_DIR = "config"
os.makedirs(CONFIG_DIR, exist_ok=True)
THRESHOLD_CONFIG_PATH = os.path.join(CONFIG_DIR, "threshold_config.json")

# 如果不存在配置文件，创建一个默认配置
if not os.path.exists(THRESHOLD_CONFIG_PATH):
    try:
        with open(THRESHOLD_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_THRESHOLD_CONFIG, f, ensure_ascii=False, indent=2)
        logger.info(f"已创建默认阈值配置文件: {THRESHOLD_CONFIG_PATH}")
    except Exception as e:
        logger.warning(f"无法创建默认配置文件: {e}")

# 加载用户定义的阈值配置
try:
    with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
        USER_THRESHOLD_CONFIG = json.load(f)
        # 合并用户配置与默认配置
        THRESHOLD_CONFIG = DEFAULT_THRESHOLD_CONFIG.copy()
        for method, config in USER_THRESHOLD_CONFIG.items():
            if method in THRESHOLD_CONFIG:
                THRESHOLD_CONFIG[method].update(config)
            else:
                THRESHOLD_CONFIG[method] = config
        logger.info(f"已加载用户阈值配置: {THRESHOLD_CONFIG_PATH}")
except Exception as e:
    logger.warning(f"无法读取阈值配置文件，使用默认值: {e}")
    THRESHOLD_CONFIG = DEFAULT_THRESHOLD_CONFIG

# 数据库配置
DB_CONFIG = {
    'HOST': 'localhost',
    'PORT': 3306,
    'USER': 'aiops',
    'PASSWORD': 'aiops123',
    'NAME': 'aiops_monitoring'
}