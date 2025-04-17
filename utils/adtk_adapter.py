# utils/adtk_adapter.py
"""把 adtk_server 返回的 JSON 转为 DetectionResult"""

import json
from detectors.base import DetectionResult

def json_to_detection_result(obj: str | dict) -> DetectionResult:
    data = json.loads(obj) if isinstance(obj, str) else obj
    return DetectionResult(
        method         = data["method"],
        visual_type    = data.get("visual_type", "point"),
        anomalies      = data.get("anomalies", []),
        intervals      = [tuple(x) for x in data.get("intervals", [])],
        anomaly_scores = data.get("anomaly_scores", []),
        explanation    = data.get("explanation", []),
    )
