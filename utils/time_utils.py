# utils/time_utils.py
def group_anomaly_times(anomalies, max_gap=1800):
    """
    将时间戳列表分组为连续的时间区间
    
    参数:
        anomalies: 时间戳列表
        max_gap: 允许的最大间隔秒数
        
    返回:
        list: [(start1, end1), (start2, end2), ...] 区间列表
    """
    if not anomalies:
        return []
    
    # 排序输入
    sorted_anomalies = sorted(anomalies)
    
    intervals = []
    cur_start = sorted_anomalies[0]
    cur_end = sorted_anomalies[0]
    
    for t in sorted_anomalies[1:]:
        if t - cur_end <= max_gap:
            cur_end = t
        else:
            intervals.append((cur_start, cur_end))
            cur_start = t
            cur_end = t
    
    # 添加最后一个区间
    intervals.append((cur_start, cur_end))
    
    return intervals