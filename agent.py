#! /usr/bin/env python3
import re
import json
import datetime
import requests
import os  
import hashlib
import config 
import dateparser
from django.conf import settings

from analysis.single_series import analyze_single_series
from analysis.multi_series import analyze_multi_series
from output.report_generator import generate_report_single, generate_report_multi
from output.visualization import generate_echarts_html_single, generate_echarts_html_multi

AIOPS_BACKEND_DOMAIN = 'https://aiopsbackend.cstcloud.cn'
LLM_URL = 'http://10.16.1.16:58000/v1/chat/completions'

AUTH = ('chelseyyycheng@outlook.com', 'UofV1uwHwhVp9tcTue')
CACHE_DIR = "cached_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def parse_time_expressions(raw_text:str):
    """
    从用户输入中提取多个时间表达式，逐一用 dateparser.parse 解析。
    返回一个列表，每个元素是 {start_ts, end_ts, error}，单位为 int秒 (整天或近似区间)
    
    - 不使用硬编码的“这周一 => 某年月日”等逻辑，而是让 dateparser 自动识别中文/英文自然语言
    - 为了简单，此处默认:
       如果解析出的datetime成功 => start=该日0点, end=该日23:59:59
       若用户写了带“到/至/到/至/~/--”之类，也可扩展解析区间 (此处演示略)
    - 若解析失败，则 error 填写信息
    """
    # 用一个简单正则来拆分可能的时间表达，如 “这周一和上周一” -> ["这周一","上周一"]
    # 如果用户只写一句，也能只得到1个
    # （可根据需要再提升复杂度，如处理 "2023-03-01到2023-03-05"）
    segments = re.split(r'[,\uFF0C\u3001\u0026\u002C\u002F\u0020\u0026\u2014\u2013\u2014\u006E\u005E]|和|与|及|还有|、', raw_text)
    # 以上只是示例，用各种分隔符号拆分。可根据需求继续完善
    
    results = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # 调用 dateparser
        dt = dateparser.parse(seg, languages=['zh','en'], settings={"PREFER_DATES_FROM":"past"})
        if dt is None:
            # 解析失败
            results.append({"start":0, "end":0, "error":f"无法解析: {seg}"})
        else:
            # 如果只要整天 => 将dt对齐到0点~23:59:59
            day_s = datetime.datetime(dt.year, dt.month, dt.day, 0,0,0)
            day_e = datetime.datetime(dt.year, dt.month, dt.day, 23,59,59)
            results.append({
                "start": int(day_s.timestamp()),
                "end":   int(day_e.timestamp()),
                "error": ""
            })
    return results

def shorten_tool_result(res):
    """
    将工具调用结果res(可能是list、dict、或长字符串)进行简要化, 避免累积大段数据进入下轮对话.
    """
    if isinstance(res, list):
        # 说明可能是时序数据 => 只显示长度
        return f"[List len={len(res)}]"
    elif isinstance(res, dict):
        # 可能是 {analysis:..., report:..., html:...}
        # 只取一部分
        summary = {}
        for k,v in res.items():
            if isinstance(v, list):
                summary[k] = f"[List len={len(v)}]"
            elif isinstance(v, str) and len(v)>300:
                summary[k] = v[:300] + f"...(omitted, length={len(v)})"
            else:
                summary[k] = v
        return json.dumps(summary, ensure_ascii=False)
    elif isinstance(res, str) and len(res)>300:
        return res[:300] + f"...(omitted, length={len(res)})"
    else:
        return str(res)



###############################################################################
# 一、定义“工具”列表 (更精确)
###############################################################################
tools = [
     {
        "name":"解析用户自然语言时间",
        "description":"返回一个list，每个元素是{start, end, error}. 如果不确定，可向用户澄清。",
        "parameters":{
            "type":"object",
            "properties":{
                "raw_text":{"type":"string"}
            },
            "required":["raw_text"]
        }
    },
    {  
        "name": "请求智能运管后端Api，获取指标项的时序数据",
        "description": "从后端或本地缓存获取IP在指定时间范围(field)的时序数据(list of [int_ts, val])。注意start/end必须是形如'YYYY-MM-DD HH:MM:SS'的确定时间。",
        "parameters": {
            "type": "object",
            "properties": {
                "ip": {
                    "type": "string",
                    "description": "要查询的 IP，如 '192.168.0.110'"
                },
                "start": {
                    "type": "string",
                    "description": "开始时间，格式 '2025-03-24 00:00:00'"
                },
                "end": {
                    "type": "string",
                    "description": "结束时间，格式 '2025-03-24 23:59:59'"
                },
                "field": {
                    "type": "string",
                    "description": "监控项名称，如 'cpu_rate'"
                }
            },
            "required": ["ip","start","end","field"]
        }
    },
    {
        "name": "请求智能运管后端Api，查询监控实例有哪些监控项",
        "description": "返回指定IP下可用的监控项列表（可选项）",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "系统服务名称 (一般填 '主机监控')"
                },
                "instance": {
                    "type": "string",
                    "description": "监控实例 IP"
                }
            },
            "required": ["service","instance"]
        }
    },
    {
        "name": "请求智能运管后端Api，查询监控服务的资产情况和监控实例",
        "description": "查询一个监控服务的所有资产/IP等信息",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "要查询的系统服务名称"
                }
            },
            "required": ["service"]
        }
    },
    {
        "name": "请求智能运管后端Api，查询监控实例之间的拓扑关联关系",
        "description": "查询指定IP的上联、下联监控实例等信息",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "系统服务名称"
                },
                "instance_ip": {
                    "type": "string",
                    "description": "监控实例IP"
                }
            },
            "required": ["service","instance_ip"]
        }
    },
    {
        "name": "单序列异常检测(文件)",
        "description": "对单序列 [int_ts,val] 进行多方法分析, 生成报告和ECharts HTML",
        "parameters": {
            "type": "object",
            "properties": {
                "ip":    {"type": "string"},
                "field": {"type": "string"},
                "start": {"type": "string"},
                "end":   {"type": "string"}
            },
            "required": ["ip","field","start","end"]
        }
    },
    {
        "name": "多序列对比异常检测(文件)",
        "description": "对两组 [int_ts,val] 进行对比分析, 生成报告和ECharts HTML",
        "parameters": {
            "type": "object",
            "properties": {
                "ip1":    {"type": "string"},
                "field1": {"type": "string"},
                "start1": {"type": "string"},
                "end1":   {"type": "string"},
                "ip2":    {"type": "string"},
                "field2": {"type": "string"},
                "start2": {"type": "string"},
                "end2":   {"type": "string"}
            },
            "required": ["ip1","field1","start1","end1","ip2","field2","start2","end2"]
        }
    }
]


###############################################################################
# 二、后端数据 API + 本地缓存
###############################################################################

def monitor_item_list(ip):
    """示例: 拿到此IP可用的监控项。"""
    url = f'{AIOPS_BACKEND_DOMAIN}/api/v1/monitor/mail/machine/field/?instance={ip}'
    resp = requests.get(url=url, auth=AUTH)
    if resp.status_code == 200:
        items = json.loads(resp.text)
        result = {}
        for x in items:
            result[x.get('field')] = x.get('purpose')
        return result
    else:
        return f"查询监控项失败: {resp.status_code} => {resp.text}"

def get_service_asset(service):
    """示例: 查询某service下的资产信息"""
    url = f'{AIOPS_BACKEND_DOMAIN}/api/v1/property/mail/?ordering=num_id&page=1&page_size=2000'
    resp = requests.get(url=url, auth=AUTH)
    if resp.status_code == 200:
        text = json.loads(resp.text)
        results = text.get('results',[])
        item_list = []
        for r in results:
            # 做一些清洗
            r["category"] = r.get("category",{}).get("name")
            r["ip_set"] = [_.get("ip") for _ in r.get('ip_set',[])]
            for k in ["num_id","creation","modification","remark","sort_weight","monitor_status"]:
                r.pop(k, None)
            # 移除空值
            for k,v in list(r.items()):
                if not v or v == "无":
                    r.pop(k)
            item_list.append(r)
        return item_list
    else:
        return f"查询失败: {resp.status_code} => {resp.text}"

def get_service_asset_edges(service, instance_ip):
    """示例: 拓扑查询"""
    url = f'{AIOPS_BACKEND_DOMAIN}/api/v1/property/mail/topology/search?instance={instance_ip}'
    resp = requests.get(url=url, auth=AUTH)
    if resp.status_code == 200:
        return json.loads(resp.text)
    else:
        return f"查询拓扑失败: {resp.status_code} => {resp.text}"
    

def _cache_filename(ip, start_ts, end_ts, field):
    key = f"{ip}_{start_ts}_{end_ts}_{field}"
    h = hashlib.md5(key.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def fetch_data_from_backend(ip:str, start_ts:int, end_ts:int, field:str):
    """
    访问后端, 返回 list of [int_ts, float_val]
    """
    url = f"{AIOPS_BACKEND_DOMAIN}/api/v1/monitor/mail/metric/format-value/?start={start_ts}&end={end_ts}&instance={ip}&field={field}"
    resp = requests.get(url, auth=AUTH)
    if resp.status_code!=200:
        return f"后端请求失败: {resp.status_code} => {resp.text}"
    j = resp.json()
    results = j.get("results", [])
    if not results:
        return []
    vals = results[0].get("values", [])
    arr = []
    from datetime import datetime
    def parse_ts(s):
        try:
            dt = datetime.strptime(s,"%Y-%m-%d %H:%M:%S")
            return int(dt.timestamp())
        except:
            return 0
    for row in vals:
        if len(row)>=2:
            tstr,vstr = row[0], row[1]
            t = parse_ts(tstr)
            try:
                v = float(vstr)
            except:
                v = 0.0
            arr.append([t,v])
    return arr

def get_monitor_metric_value(ip, start, end, field):
    """
    start/end是 'YYYY-MM-DD HH:MM:SS' => 转成int => 读缓存 => 无则后端请求 => 写缓存 => 返回
    """
    import datetime
    def to_int(s):
        dt = datetime.datetime.strptime(s,"%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    st_i = to_int(start)
    et_i = to_int(end)
    fpath= _cache_filename(ip, st_i, et_i, field)
    if os.path.exists(fpath):
        print("(已从本地缓存读取)")
        return json.load(open(fpath,"r",encoding="utf-8"))
    else:
        data= fetch_data_from_backend(ip, st_i, et_i, field)
        if isinstance(data,str):
            return data
        with open(fpath,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        print("(已调用后端并写入本地缓存)")
        return data

###############################################################################
# 三、用于(文件)单序列/多序列检测，并生成报告 & 可视化
###############################################################################

def single_series_detect(ip, field, start, end):
    series = get_monitor_metric_value(ip, start, end, field)
    if isinstance(series, str):
        return {"error": series}

    res = analyze_single_series(series)
    rep = generate_report_single(res, ip, field, start, end)
    html = generate_echarts_html_single(series, res["anomaly_times"])
    return {"analysis": res, "report": rep, "html": html}


def multi_series_detect(ip1, field1, start1, end1,
                        ip2, field2, start2, end2):
    s1 = get_monitor_metric_value(ip1, start1, end1, field1)
    if isinstance(s1, str):
        return {"error": s1}
    s2 = get_monitor_metric_value(ip2, start2, end2, field2)
    if isinstance(s2, str):
        return {"error": s2}

    res = analyze_multi_series(s1, s2)
    rep = generate_report_multi(res, ip1, field1, ip2, field2, start1, end1, start2, end2)
    html = generate_echarts_html_multi(s1, s2, res["anomaly_times"])
    return {"analysis": res, "report": rep, "html": html}


###############################################################################
# 四、LLM ReAct Agent
###############################################################################

def llm_call(messages):
    """
    向大模型发请求
    """
    data={
      "model":"Qwen2.5-14B-Instruct",
      "temperature":0.1,
      "messages":messages
    }
    r= requests.post(LLM_URL, json=data)
    if r.status_code==200:
        jj= r.json()
        if "choices" in jj and len(jj["choices"])>0:
            return jj["choices"][0]["message"]
        else:
            return None
    else:
        print("Error:", r.status_code, r.text)
        return None
    
def init_msg(role, content):
    return {"role": role, "content": content}



def parse_llm_response(txt):
    pat_thought = r"<思考过程>(.*?)</思考过程>"
    pat_action  = r"<工具调用>(.*?)</工具调用>"
    pat_inparam = r"<调用参数>(.*?)</调用参数>"
    pat_final   = r"<最终答案>(.*?)</最终答案>"
    def ext(pattern):
        m = re.search(pattern, txt, flags=re.S)
        return m.group(1) if m else ""

    return {
        "thought": ext(pat_thought),
        "action":  ext(pat_action),
        "action_input": ext(pat_inparam),
        "final_answer": ext(pat_final)
    }

def react(llm_text):
    parsed= parse_llm_response(llm_text)
    action= parsed["action"]
    inp_str= parsed["action_input"]
    final_ans= parsed["final_answer"]
    is_final= False

    if action and inp_str:
        # 解析调用参数
        try:
            action_input = json.loads(inp_str)
        except:
            return f"无法解析调用参数JSON: {inp_str}", False

        # 匹配action
        if action == "解析用户自然语言时间":
            return parse_time_expressions(action_input["raw_text"]), False
        elif action == "请求智能运管后端Api，获取指标项的时序数据":
            return get_monitor_metric_value(**action_input), False
        elif action == "请求智能运管后端Api，查询监控实例有哪些监控项":
            return monitor_item_list(action_input["instance"]), False
        elif action == "请求智能运管后端Api，查询监控服务的资产情况和监控实例":
            return get_service_asset(action_input["service"]), False
        elif action == "请求智能运管后端Api，查询监控实例之间的拓扑关联关系":
            return get_service_asset_edges(action_input["service"], action_input["instance_ip"]), False

        elif action == "单序列异常检测(文件)":
            return single_series_detect(**action_input), False

        elif action == "多序列对比异常检测(文件)":
            return multi_series_detect(**action_input), False

        else:
            return f"未知工具调用: {action}", False

    # 如果final_answer不为空，就返回并结束
    if final_ans.strip():
        is_final = True
        return (final_ans,is_final)

    return ("格式不符合要求，必须使用：<思考过程></思考过程> <工具调用></工具调用> <调用参数></调用参数> <最终答案></最终答案>", is_final)

def shorten_tool_result(res, max_len=300):
    """
    对工具返回结果做截断或简要摘要，避免在对话中携带过长文本。
    """
    if isinstance(res, list):
        return f"[List len={len(res)}] (showing 0)"
    elif isinstance(res, dict):
        summary = {}
        for k, v in res.items():
            if isinstance(v, list):
                summary[k] = f"[List len={len(v)}] (omitted)"
            elif isinstance(v, str) and len(v) > max_len:
                summary[k] = v[:max_len] + f"...(omitted, length={len(v)})"
            else:
                summary[k] = v
        return json.dumps(summary, ensure_ascii=False)
    elif isinstance(res, str) and len(res) > max_len:
        return res[:max_len] + f"...(omitted length={len(res)})"
    else:
        return str(res)


def chat(user_query):
    system_prompt = f'''你是一个严格遵守格式规范的用于运维功能，运维数据可视化，运行于生产环境的ReAct智能体，你叫小助手，必须按以下格式处理请求：

    你的工具列表如下:
    {json.dumps(tools, ensure_ascii=False, indent=2)}
    当前时间为: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    处理规则：
    1.请根据当前时间来推断用户输入的时间区间的具体值
    2.如 parse_time_expressions 只返回1个时间区间，则调用'单序列异常检测(文件)'。
    3.如 parse_time_expressions 返回2个时间区间，并且用户输入包含"对比"、"相比"、"比较"、"环比"等比较词汇 ，则调用'多序列对比异常检测(文件)'。
    4.若有超过1个时间区间，但是没有明显的比较词汇，可先在<最终答案>里提问，示例:
    <思考过程>我不知道用户是要对这些时间的数据分别进行单序列分析还是一起多序列分析，我需要确认</思考过程> <工具调用></工具调用> <调用参数></调用参数> <最终答案>请问您是想对每段数据进行单序列分析，还是需要多序列的对比分析？</最终答案>
    5.根据用户的输入来自行判断是否要调用工具以及调用哪个工具
    4.每次只能调用一个工具
    5.不能伪造数据
    6.严格按照以下xml格式生成响应文本：
    ```
    <思考过程>你的思考过程</思考过程>
    <工具调用>工具名称，不调用则为空</工具调用>
    <调用参数>工具输入参数{{json}}</调用参数>
    <最终答案>用户问题的最终结果（知道问题的最终答案时返回）</最终答案>
    ```
    '''
    history=[]
    history.append({"role":"system","content":system_prompt})
    history.append({"role":"user","content": user_query})

    round_num=1
    max_round=15

    while True:
        print(f"=== 第{round_num}轮对话 ===")
        ans= llm_call(history)
        if not ans:
            print("大模型返回None,结束")
            return

        # 打印大模型完整响应 (JSON形式或content都可以)
        print("## 大模型完整响应:", ans)

        # 把响应加入history
        history.append(ans)

        txt= ans.get("content","")
        result, done= react(txt)

        # 对工具调用结果做截断后也放进对话上下文
        short_result = shorten_tool_result(result, max_len=300)
        history.append({
            "role":"user",
            "content": f"<工具调用结果>: {short_result}"
        })

        if done:
            print("===最终输出===")
            print(result)
            return

        round_num+=1
        if round_num>max_round:
            print("超出上限")
            return



if __name__ == '__main__':
    # chat('你好')
    chat(
        '我想查询192.168.0.110这台主机这周星期一和上周星期一的cpu利用率的对比，并给出echarts折线图的完整html，并进行分析给出分析报告')
