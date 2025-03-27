# 文件: detectors/change_rate_detector.py

def detect_change_rate(series1, series2, threshold=0.5):
    """
    对比两序列在相同时间戳上的变化率。
    threshold: 当两者的变化率之差大于该值则视为异常
    """
    if not series1 or not series2:
        return {
            "method":"ChangeRate",
            "anomalies":[],
            "scores":[],
            "description":"数据不足"
        }
    dict2 = {row[0]:row[1] for row in series2}
    s1_sorted = sorted(series1,key=lambda x:x[0])
    anomalies=[]
    scores=[]
    prev_v1=None
    prev_v2=None
    prev_ts=None
    for (ts,v1) in s1_sorted:
        if ts in dict2:
            v2 = dict2[ts]
            if prev_v1 is not None and prev_v2 is not None and abs(prev_v1)>1e-9 and abs(prev_v2)>1e-9:
                rate1 = (v1 - prev_v1)/abs(prev_v1)
                rate2 = (v2 - prev_v2)/abs(prev_v2)
                diff = abs(rate1-rate2)
                if diff>threshold:
                    anomalies.append(ts)
                    scores.append(diff)
            prev_v1=v1
            prev_v2=v2
            prev_ts=ts
    desc = f"变化率检测到{len(anomalies)}个异常" if anomalies else "未发现变化率异常"
    return {
        "method":"ChangeRate",
        "anomalies":anomalies,
        "scores":scores,
        "description":desc
    }
