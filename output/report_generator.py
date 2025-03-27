# output/report_generator.py
import datetime

def _fmt_ts(t_int):
    try:
        dt = datetime.datetime.fromtimestamp(t_int)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(t_int)

def generate_report_single(results_dict, ip, field, start, end):
    lines=[]
    lines.append(f"【单序列异常检测报告】")
    lines.append(f"监控对象: IP={ip}, field={field}, 时间范围={start}~{end}")
    lines.append("检测方法及其结果:")
    for r in results_dict["method_results"]:
        lines.append(f"- {r['method']}: {r.get('description','')} (异常数={len(r['anomalies'])})")

    lines.append(f"综合评分: {results_dict['composite_score']:.2f}")
    lines.append(f"判定: {results_dict['classification']}")
    if results_dict["anomaly_times"]:
        lines.append("异常时间点:")
        for t in results_dict["anomaly_times"]:
            lines.append(f"  - {_fmt_ts(t)}")

    # 更多stats
    es = results_dict.get("extra_stats", {})
    mean_val = es.get("mean", None)
    std_val  = es.get("std", None)
    max_t    = es.get("max_time", None)
    max_v    = es.get("max_value", None)
    if mean_val is not None:
        lines.append(f"平均值: {mean_val:.3f}, 标准差: {std_val:.3f}")
    if max_t:
        lines.append(f"最大值出现在 {_fmt_ts(max_t)}, value={max_v:.3f}")

    return "\n".join(lines)

def generate_report_multi(results_dict, ip1, field1, ip2, field2,
                          start1, end1, start2, end2):
    lines=[]
    lines.append("【多序列对比异常检测报告】")
    lines.append(f"第一组: IP={ip1}, field={field1}, 时间范围={start1}~{end1}")
    lines.append(f"第二组: IP={ip2}, field={field2}, 时间范围={start2}~{end2}")

    lines.append("检测方法及结果:")
    for r in results_dict["method_results"]:
        lines.append(f"- {r['method']}: {r.get('description','')} (异常数={len(r['anomalies'])})")

    lines.append(f"综合评分: {results_dict['composite_score']:.2f}")
    lines.append(f"分类: {results_dict['classification']}")

    anoms = results_dict["anomaly_times"]
    if anoms:
        lines.append("异常时间点(仅显示前20个):")
        for t in anoms[:20]:
            lines.append(f"  - {_fmt_ts(t)}")
        if len(anoms) > 20:
            lines.append(f"... 共 {len(anoms)} 个异常点，后续省略 ...")
    else:
        lines.append("异常时间点: 无")

    # 额外stats
    es = results_dict.get("extra_stats",{})
    mean1= es.get("mean1",0)
    mean2= es.get("mean2",0)
    max_diff_t= es.get("max_diff_time", None)
    max_diff_v= es.get("max_diff_value", None)
    lines.append(f"第一组均值: {mean1:.3f}, 第二组均值: {mean2:.3f}")
    if max_diff_t is not None:
        lines.append(f"最大差值出现在 {_fmt_ts(max_diff_t)}, 差值= {max_diff_v:.3f}")

    # 这里可以再做一个 prompt 给大模型进行“润色”
    # 省略

    return "\n".join(lines)
