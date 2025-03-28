# 文件: output/report_generator.py

import datetime
import json
from openai import OpenAI###
def deepseek_api(prompt):
    client = OpenAI(api_key="3a92ef09aeb6d7e1b5f6e2fe1676a0c45fcf6a62885af08cf3d50ac553eafd4a", base_url="https://uni-api.cstcloud.cn/v1")
    response = client.chat.completions.create(
        model="deepseek-v3:671b",
        messages=[
            {"role": "system", "content": "你是一个乐于助人的助手。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7,
        stream=False
    )
    return response.choices[0].message.content

def _fmt_ts(t_int):
    try:
        dt = datetime.datetime.fromtimestamp(t_int)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(t_int)


def refine_report_with_deepseek(raw_text:str) -> str:
    prompt = f"""
请将以下分析报告语言优化，使其更流畅自然，但不改变任何数值、日期、时间点或结论：
报告内容如下：
{raw_text}

要求：
1. 保持所有数值、日期、时间点不变。
2. 语言专业且易于理解。
3. 不要添加新数据或修改结论。
"""
    try:
        improved = deepseek_api(prompt)
        return improved
    except Exception as e:
        print(f"[refine_report_with_deepseek] 调用失败: {e}")
        return raw_text

def generate_report_single(results_dict, ip, field, start, end, use_deepseek_refine=False):
    """
    生成单序列异常检测报告。
    当 use_deepseek_refine=True 时，会在最后调用 deepseek_api 进行语言优化。
    """
    lines=[]
    lines.append("【单序列异常检测报告】")
    lines.append(f"监控对象: IP={ip}, field={field}, 时间范围={start}~{end}")
    lines.append("检测方法及结果:")
    for r in results_dict["method_results"]:
        lines.append(f"- {r['method']}: {r.get('description','')} (异常数={len(r['anomalies'])})")

    lines.append(f"综合评分: {results_dict['composite_score']:.2f}")
    lines.append(f"判定: {results_dict['classification']}")

    anomaly_list = results_dict.get("anomaly_times", [])
    if anomaly_list:
        lines.append("异常时间点(仅显示前20个):")
        for t in anomaly_list[:20]:
            lines.append(f"  - {_fmt_ts(t)}")
        if len(anomaly_list) > 20:
            lines.append(f"... 共 {len(anomaly_list)} 个异常点，后续省略 ...")
    else:
        lines.append("异常时间点: 无")

    es = results_dict.get("extra_stats", {})
    mean_val = es.get("mean", None)
    std_val  = es.get("std", None)
    max_t    = es.get("max_time", None)
    max_v    = es.get("max_value", None)
    if mean_val is not None:
        lines.append(f"平均值: {mean_val:.3f}, 标准差: {std_val:.3f}")
    if max_t:
        lines.append(f"最大值出现在 {_fmt_ts(max_t)}, value={max_v:.3f}")

    intervals = results_dict.get("anomaly_intervals", [])
    if intervals:
        lines.append("异常区间(合并后):")
        for (st, ed) in intervals[:5]:
            if st == ed:
                lines.append(f"  - {_fmt_ts(st)} (单点异常)")
            else:
                lines.append(f"  - {_fmt_ts(st)} ~ {_fmt_ts(ed)}")
        if len(intervals) > 5:
            lines.append(f"... 共 {len(intervals)} 个区间，后续省略 ...")

    raw_report = "\n".join(lines)

    if use_deepseek_refine:
        return refine_report_with_deepseek(raw_report)
    else:
        return raw_report


def generate_report_multi(results_dict, ip1, field1, ip2, field2,
                          start1, end1, start2, end2,
                          use_deepseek_refine=False):
    """
    生成多序列对比异常检测报告。
    当 use_deepseek_refine=True 时，会调用 deepseek_api 进行语言优化。
    """
    lines=[]
    lines.append("【多序列对比异常检测报告】")
    lines.append(f"第一组: IP={ip1}, field={field1}, 时间范围={start1}~{end1}")
    lines.append(f"第二组: IP={ip2}, field={field2}, 时间范围={start2}~{end2}")

    lines.append("检测方法及结果:")
    for r in results_dict["method_results"]:
        lines.append(f"- {r['method']}: {r.get('description','')} (异常数={len(r['anomalies'])})")

    lines.append(f"综合评分: {results_dict['composite_score']:.2f}")
    lines.append(f"分类: {results_dict['classification']}")

    anomaly_list = results_dict.get("anomaly_times", [])
    if anomaly_list:
        lines.append("异常时间点(仅显示前20个):")
        for t in anomaly_list[:20]:
            lines.append(f"  - {_fmt_ts(t)}")
        if len(anomaly_list) > 20:
            lines.append(f"... 共 {len(anomaly_list)} 个异常点，后续省略 ...")
    else:
        lines.append("异常时间点: 无")

    #如果有区间
    intervals = results_dict.get("anomaly_intervals", [])
    if intervals:
        lines.append("异常区间(合并后):")
        for (st, ed) in intervals[:5]:
            if st == ed:
                lines.append(f"  - {_fmt_ts(st)} (单点)")
            else:
                lines.append(f"  - {_fmt_ts(st)} ~ {_fmt_ts(ed)}")
        if len(intervals) > 5:
            lines.append(f"... 共 {len(intervals)} 个区间，后续省略 ...")

    es = results_dict.get("extra_stats",{})
    mean1= es.get("mean1",0)
    mean2= es.get("mean2",0)
    max_diff_t= es.get("max_diff_time", None)
    max_diff_v= es.get("max_diff_value", None)
    lines.append(f"第一组均值: {mean1:.3f}, 第二组均值: {mean2:.3f}")
    if max_diff_t is not None:
        lines.append(f"最大差值出现在 {_fmt_ts(max_diff_t)}, 差值= {max_diff_v:.3f}")

    raw_report = "\n".join(lines)

    if use_deepseek_refine:
        return refine_report_with_deepseek(raw_report)
    else:
        return raw_report
