# analysis/multi_series.py
import config
import math
import numpy as np
import logging
from utils.time_utils import group_anomaly_times
from detectors.base import DetectionResult

logger = logging.getLogger("anomaly_detection.multi_series")

def analyze_multi_series(series1, series2, align=True):
    """
    对两个时间序列进行对比分析
    
    参数:
        series1: 第一个时间序列
        series2: 第二个时间序列
        align: 是否对齐时间戳
        
    返回:
        dict: 包含分析结果的字典
    """
    # 输入参数验证
    if not series1 or not series2:
        logger.warning("输入时间序列为空")
        return {
            "method_results": [],
            "composite_score": 0,
            "classification": "正常",
            "anomaly_times": [],
            "anomaly_intervals": []
        }
    
    # 导入放在函数内部避免循环引用
    from detectors.residual_comparison import ResidualComparisonDetector
    from detectors.trend_drift_cusum import TrendDriftCUSUMDetector
    from detectors.change_rate import ChangeRateDetector
    from detectors.trend_slope import TrendSlopeDetector
    from analysis.data_alignment import align_series
    
    # 检查两个序列是否有足够的差异
    values1 = [v for _, v in series1]
    values2 = [v for _, v in series2]
    
    if align:
        try:
            series1, series2 = align_series(series1, series2, method="linear", fill_value="extrapolate")
            logger.info("成功对齐两个时间序列")
        except Exception as e:
            logger.error(f"时间序列对齐失败: {e}")
            # 继续使用原始序列
    
    # 计算两个序列的相似度
    try:
        mean_abs_diff = np.mean(np.abs(np.array(values1) - np.array(values2)))
        relative_diff = mean_abs_diff / (np.mean(np.abs(values1)) + 1e-10)
        
        logger.info(f"两序列的相对差异: {relative_diff:.1%}")
        # 如果差异极小，可能不需要详细分析
        if relative_diff < 0.05:  # 小于5%的差异
            logger.info("序列几乎相同，无需详细分析")
            return {
                "method_results": [],
                "composite_score": 0,
                "classification": "正常",
                "anomaly_times": [],
                "anomaly_intervals": []
            }
    except Exception as e:
        logger.warning(f"计算序列差异失败: {e}")
        # 继续分析

    # 加载阈值配置
    thres = config.THRESHOLD_CONFIG
    
    # 执行各个检测方法
    detection_results = []
    
    # 1. 残差对比方法
    try:
        res_residual = ResidualComparisonDetector(
            threshold=thres.get("ResidualComparison", {}).get("threshold", 3.5)
        ).detect(series1, series2)
        detection_results.append(res_residual)
        logger.info(f"残差对比检测到 {len(res_residual.anomalies)} 个异常点")
    except Exception as e:
        logger.error(f"残差对比检测失败: {e}")
    
    # 2. 趋势漂移CUSUM方法
    try:
        res_drift = TrendDriftCUSUMDetector(
            threshold=thres.get("TrendDriftCUSUM", {}).get("drift_threshold", 8.0)
        ).detect(series1, series2)
        detection_results.append(res_drift)
        logger.info(f"趋势漂移检测到 {len(res_drift.intervals)} 个异常区间")
    except Exception as e:
        logger.error(f"趋势漂移检测失败: {e}")
    
    # 3. 变化率方法
    try:
        res_change = ChangeRateDetector(
            threshold=thres.get("ChangeRate", {}).get("threshold", 0.7)
        ).detect(series1, series2)
        detection_results.append(res_change)
        logger.info(f"变化率检测到 {len(res_change.explanation)} 个文本解释")
    except Exception as e:
        logger.error(f"变化率检测失败: {e}")
    
    # 4. 趋势斜率方法
    try:
        res_slope = TrendSlopeDetector(
            threshold=thres.get("TrendSlope", {}).get("slope_threshold", 0.4),
            window=thres.get("TrendSlope", {}).get("window", 5)
        ).detect(series1, series2)
        detection_results.append(res_slope)
        logger.info(f"趋势斜率检测到 {len(res_slope.anomalies)} 个异常点")
    except Exception as e:
        logger.error(f"趋势斜率检测失败: {e}")

    # 过滤无效结果
    method_results = [
        r for r in detection_results if r is not None
    ]
    
    if not method_results:
        logger.warning("所有检测方法都失败")
        return {
            "method_results": [],
            "composite_score": 0,
            "classification": "正常",
            "anomaly_times": [],
            "anomaly_intervals": []
        }

    # 计算综合得分
    total_weight = 0.0
    composite_score = 0.0
    length = max(len(series1), len(series2)) or 1
    
    # 方法有意义的分数
    method_scores = {}

    for res in method_results:
        m_name = res.method
        weight = config.WEIGHTS_MULTI.get(m_name, 0.25)  # 默认权重0.25
        total_weight += weight
        
        # 计算方法得分
        if res.visual_type == "none" and res.explanation:
            # 纯文本解释型检测器
            # 根据解释的数量和内容评估得分
            has_significant_diff = any(("差异较大" in expl or "明显" in expl) 
                                      for expl in res.explanation)
            method_score = 0.4 if has_significant_diff else 0.2 if res.explanation else 0
        else:
            # 可视化型检测器
            anomaly_count = len(res.anomalies)
            interval_count = len(res.intervals) * 3  # 区间异常权重更高
            total_count = anomaly_count + interval_count
            
            if total_count > 0:
                # 计算异常比例
                if m_name == "TrendDriftCUSUM":
                    # TrendDriftCUSUM方法特殊处理
                    # 根据区间覆盖的时间比例来评估
                    total_duration = 0
                    for start, end in res.intervals:
                        total_duration += (end - start)
                    
                    coverage_ratio = total_duration / (series1[-1][0] - series1[0][0] + 1)
                    # 调整分数，减少过度敏感
                    if coverage_ratio > 0.5:  # 覆盖超过50%
                        method_score = min(0.7, 0.4 + 0.3 * coverage_ratio)
                    else:
                        method_score = 0.3 * coverage_ratio
                else:
                    # 其他方法使用对数缩放的点/区间数量
                    anomaly_ratio = total_count / length
                    if anomaly_ratio < 0.01:  # 低于1%的异常率
                        method_score = 0.2 + 0.3 * (anomaly_ratio * 100)  # 线性调整
                    else:
                        method_score = min(0.8, 0.2 + 0.3 * np.log10(1 + anomaly_ratio * 100))
            else:
                method_score = 0
        
        # 记录各方法得分
        method_scores[m_name] = method_score
        composite_score += weight * method_score
        logger.info(f"方法 {m_name} 得分: {method_score:.2f}, 权重: {weight}")

    if total_weight > 0:
        composite_score /= total_weight

    # 添加置信度调整
    methods_with_anomalies = sum(1 for res in method_results 
                              if (len(res.anomalies) > 0 or 
                                  len(res.intervals) > 0 or 
                                  (res.visual_type == "none" and len(res.explanation) > 0)))
    
    if methods_with_anomalies == 1 and len(method_results) > 1:
        logger.info("仅一个方法检测到异常，降低得分")
        composite_score *= 0.8  # 降低20%
    
    # 检查趋势漂移CUSUM的得分是否过高
    if "TrendDriftCUSUM" in method_scores and method_scores["TrendDriftCUSUM"] > 0.5:
        # 如果其他方法都没有明显异常，可能是误报
        other_methods_score = sum(score for name, score in method_scores.items() 
                                if name != "TrendDriftCUSUM") / max(1, len(method_scores) - 1)
        
        if other_methods_score < 0.2:  # 其他方法平均得分很低
            logger.warning("TrendDriftCUSUM可能误报，降低总得分")
            # 降低综合得分
            composite_score = (composite_score + other_methods_score) / 2

    # 确定分类
    classification = (
        "高置信度异常" if composite_score >= config.HIGH_ANOMALY_THRESHOLD
        else "轻度异常" if composite_score >= config.MILD_ANOMALY_THRESHOLD
        else "正常"
    )
    
    logger.info(f"综合得分: {composite_score:.2f}, 分类: {classification}")

    # 合并所有异常点
    all_anoms = set()
    for r in method_results:
        all_anoms.update(r.anomalies)
    anomaly_list = sorted(all_anoms)
    
    # 检查异常点是否太多（可能是误报）
    anomaly_ratio = len(anomaly_list) / length if length > 0 else 0
    if anomaly_ratio > 0.25:  # 超过25%是异常点
        logger.warning(f"异常点比例高达 {anomaly_ratio:.1%}，重新评估分类")
        # 对于大量异常点，需要多个方法一致确认才算高置信度异常
        if methods_with_anomalies < len(method_results) * 0.7:  # 少于70%的方法都检测到异常
            if classification == "高置信度异常":
                classification = "轻度异常"
                logger.info("降级为轻度异常")
            elif classification == "轻度异常" and anomaly_ratio > 0.4:
                classification = "正常"
                logger.info("降级为正常")
    
    # 分组时间区间
    intervals = group_anomaly_times(anomaly_list)

    return {
        "method_results": method_results,
        "composite_score": composite_score,
        "classification": classification,
        "anomaly_times": anomaly_list,
        "anomaly_intervals": intervals
    }