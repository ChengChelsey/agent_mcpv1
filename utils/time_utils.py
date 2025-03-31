# utils/time_utils.py
import re
import datetime
import dateparser

def parse_time_expressions(raw_text: str):

    #解析自然语言时间表达式
    
    segments = re.split(r'[,\uFF0C\u3001\u0026\u002C\u002F\u0020\u0026\u2014\u2013\u2014\u006E\u005E]|和|与|及|还有|、', raw_text)
    results = []
    
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        dt = dateparser.parse(seg, languages=['zh', 'en'], settings={"PREFER_DATES_FROM": "past"})
        if dt is None:
            results.append({"start": 0, "end": 0, "error": f"无法解析: {seg}"})
        else:
            day_s = datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)
            day_e = datetime.datetime(dt.year, dt.month, dt.day, 23, 59, 59)
            
            start_str = day_s.strftime("%Y-%m-%d %H:%M:%S")
            end_str = day_e.strftime("%Y-%m-%d %H:%M:%S")
            
            results.append({
                "start": int(day_s.timestamp()),
                "end": int(day_e.timestamp()),
                "error": "",
                "start_str": start_str,
                "end_str": end_str
            })
    
    return results

def group_anomaly_times(anomalies, max_gap=1800):

    #将时间戳列表分组为连续的时间区间
    if not anomalies:
        return []
    
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
    
    intervals.append((cur_start, cur_end))
    
    return intervals

def format_timestamp(ts):
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(ts)

def to_timestamp(time_str):
    try:
        dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    except Exception as e:
        raise ValueError(f"时间字符串格式错误: {e}")