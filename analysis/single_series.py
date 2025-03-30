# analysis/single_series.py
import config
import numpy as np
import logging
from detectors.base import DetectionResult

logger = logging.getLogger("anomaly_detection.single_series")

def analyze_single_series(series):
    """
    对单个时间序列进行异常检测分析
    
    参数:
        series: 时间序列数据 [(timestamp, value), ...]
        
    返回:
        dict: 包含分析结果的字典
    """
    if not series:
        logger.warning("输入时间序列为空")
        return {
            "method_results": [],
            "composite_score": 0,
            "classification": "正常",
            "anomaly_times": []
        }
        
    # 检查数据有效性
    values = [v for _, v in series]
    if len(set(values)) <= 1:
        logger.info("输入时间序列几乎不变，无需检测")
        return {
            "method_results": [],
            "composite_score": 0,
            "classification": "正常",
            "anomaly_times": []
        }
        
    # 导入检测器
    from detectors.zscore import ZScoreDetector
    from detectors.cusum import CUSUMDetector
    from detectors.stl_detector import STLDetector
    
    # 加载阈值配置
    thres = config.THRESHOLD_CONFIG
    
    # 执行异常检测 - Z-Score
    try:
        res_z = ZScoreDetector(
            threshold=thres.get("Z-Score", {}).get("threshold", 3.5)
        ).detect(series)
        logger.info(f"Z-Score 检测到 {len(res_z.anomalies)} 个异常点")
    except Exception as e:
        logger.error(f"Z-Score 检测失败: {e}")
        res_z = DetectionResult(method="Z-Score", description=f"检测失败: {e}")
    
    # 执行异常检测 - CUSUM
    try:
        res_cusum = CUSUMDetector(
            drift_threshold=thres.get("CUSUM", {}).get("drift_threshold", 6.0),
            k=thres.get("CUSUM", {}).get("k", 0.7)
        ).detect(series)
        logger.info(f"CUSUM 检测到 {len(res_cusum.anomalies)} 个异常点, {len(res_cusum.intervals)} 个异常区间")
    except Exception as e:
        logger.error(f"CUSUM 检测失败: {e}")
        res_cusum = DetectionResult(method="CUSUM", description=f"检测失败: {e}")
    
    # 执行异常检测 - STL
    try:
        res_stl = STLDetector(
            seasonal=thres.get("STL", {}).get("seasonal", 24),
            z_threshold=thres.get("STL", {}).get("z_threshold", 3.5)
        ).detect(series)
        logger.info(f"STL 检测到 {len(res_stl.anomalies)} 个异常点")
    except Exception as e:
        logger.error(f"STL 检测失败: {e}")
        res_stl = DetectionResult(method="STL", description=f"检测失败: {e}")
    
    # 检查是否所有方法都失败了
    valid_results = []
    for result in [res_z, res_cusum, res_stl]:
        if len(result.anomalies) > 0 or len(result.intervals) > 0 or result.visual_type != "none":
            valid_results.append(result)
    
    if not valid_results:
        logger.warning("所有检测方法都未找到异常或失败")
        return {
            "method_results": [res_z, res_cusum, res_stl],
            "composite_score": 0,
            "classification": "正常",
            "anomaly_times": []
        }
    
    # 避免STL和Z-Score重复计算
    method_results = []
    method_names = set()
    
    for result in [res_z, res_cusum, res_stl]:
        # 跳过重复方法
        if result.method == "STL-Fallback" and "Z-Score" in method_names:
            logger.info("跳过与Z-Score重复的STL回退结果")
            continue
        method_results.append(result)
        method_names.add(result.method)
    
    # 计算综合得分
    total_weight = 0.0
    composite_score = 0.0
    length = len(series)
    
    for res in method_results:
        # 获取方法权重，确保每个方法都有权重值
        m_name = res.method
        weight = config.WEIGHTS_SINGLE.get(m_name, 0.3)  # 默认0.3
        total_weight += weight
        
        # 计算异常严重程度
        anomalies_count = len(res.anomalies)
        intervals_count = len(res.intervals) * 3  # 区间异常权重更高
        total_count = anomalies_count + intervals_count
        
        if res.visual_type == "none":
            # 纯文本解释类检测器
            method_score = 0
        elif total_count > 0:
            # 使用对数缩放，避免大数据集中的稀释效应
            # 调整公式，使得即使异常比例很小，也能得到一定的分数
            if total_count / length < 0.01:  # 低于1%的异常点
                method_score = 0.2 + 0.3 * (total_count / length) * 100  # 线性调整
            else:
                ratio = total_count / length
                method_score = min(0.9, 0.2 + 0.3 * np.log10(1 + ratio * 100))
        else:
            method_score = 0
        
        logger.info(f"方法 {m_name} 得分: {method_score:.2f}, 权重: {weight}")
        composite_score += weight * method_score
    
    if total_weight > 0:
        composite_score /= total_weight
    
    # 添加得分的置信度调整 - 减少误报
    # 如果只有一个方法检测到异常，降低得分
    methods_with_anomalies = sum(1 for res in method_results 
                              if len(res.anomalies) > 0 or len(res.intervals) > 0)
    if methods_with_anomalies == 1 and len(method_results) > 1:
        logger.info("仅一个方法检测到异常，降低得分")
        composite_score *= 0.8  # 降低20%
    
    # 确定分类
    classification = (
        "高置信度异常" if composite_score >= config.HIGH_ANOMALY_THRESHOLD
        else "轻度异常" if composite_score >= config.MILD_ANOMALY_THRESHOLD
        else "正常"
    )
    
    logger.info(f"综合得分: {composite_score:.2f}, 分类: {classification}")
    
    # 合并所有异常点
    all_anomalies = set()
    for r in method_results:
        all_anomalies.update(r.anomalies)
    
    # 计算异常点占总数据的比例
    anomaly_ratio = len(all_anomalies) / length if length > 0 else 0
    # 如果异常点超过25%，可能是误报，重新评估
    if anomaly_ratio > 0.25:
        logger.warning(f"异常点比例高达 {anomaly_ratio:.1%}，重新评估分类")
        # 对于大量异常点，需要多个方法一致确认才算高置信度异常
        if methods_with_anomalies < len(method_results) * 0.7:  # 少于70%的方法检测到异常
            if classification == "高置信度异常":
                classification = "轻度异常"
                logger.info("降级为轻度异常")
            elif classification == "轻度异常" and anomaly_ratio > 0.4:
                classification = "正常"
                logger.info("降级为正常")
    
    return {
        "method_results": method_results,
        "composite_score": composite_score,
        "classification": classification,
        "anomaly_times": sorted(all_anomalies)
    }