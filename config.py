# config.py
import json
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anomaly_detection")

WEIGHTS_SINGLE = {
    "Z-Score": 0.5,
    "CUSUM": 0.5
}

WEIGHTS_MULTI = {
    "ResidualComparison": 0.4, 
    "TrendDriftCUSUM": 0.25,   
    "ChangeRate": 0.15,
    "TrendSlope": 0.2
}

HIGH_ANOMALY_THRESHOLD = 0.7
MILD_ANOMALY_THRESHOLD = 0.4

DEFAULT_THRESHOLD_CONFIG = {
    "Z-Score": {
        "threshold": 3.5 
    },
    "CUSUM": {
        "drift_threshold": 6.0, 
        "k": 0.7  
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

CONFIG_DIR = "config"
os.makedirs(CONFIG_DIR, exist_ok=True)
THRESHOLD_CONFIG_PATH = os.path.join(CONFIG_DIR, "threshold_config.json")

if not os.path.exists(THRESHOLD_CONFIG_PATH):
    try:
        with open(THRESHOLD_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_THRESHOLD_CONFIG, f, ensure_ascii=False, indent=2)
        logger.info(f"已创建默认阈值配置文件: {THRESHOLD_CONFIG_PATH}")
    except Exception as e:
        logger.warning(f"无法创建默认配置文件: {e}")

try:
    with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
        USER_THRESHOLD_CONFIG = json.load(f)
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

