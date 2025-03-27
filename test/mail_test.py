#! /usr/bin/env python3
import re
import json
import datetime
import requests
from django.conf import settings

AIOPS_BACKEND_DOMAIN = 'https://aiopsbackend.cstcloud.cn'
LLM_URL = 'http://10.16.1.16:58000/v1/chat/completions'

AUTH = ('chelseyyycheng@outlook.com', 'UofV1uwHwhVp9tcTue')

tools = [
    {
        "type": "function",
        "function": {
            "name": "请求智能运管后端Api，获取指标项的时序数据",
            "description": "请求智能运管后端Api，获取指标项的时序数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip": {
                        "type": "string",
                        "description": "要查询的ip"
                    },
                    "start": {
                        "type": "string",
                        "description": "日期，格式为 Y-%m-%d %H:%M:%S"
                    },
                    "end": {
                        "type": "string",
                        "description": "日期，格式为 Y-%m-%d %H:%M:%S"
                    },
                    "field": {
                        "type": "string",
                        "description": "监控项名称，只可在监控项列表中选择"
                    },
                },
                "required": ["ip", "start", "end", "field"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "请求智能运管后端Api，查询监控实例有哪些监控项",
            "description": "请求智能运管后端Api，查询监控实例有哪些监控项",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "要查询的系统名称"
                    },
                    "instance": {
                        "type": "string",
                        "description": "要查询的监控实例"
                    },
                },
                "required": ["service", "instance"],

            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "请求智能运管后端Api，查询监控实例之间的拓扑关联关系",
            "description": "监控实例上联了哪些监控实例列表，下联了哪些监控实例列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "要查询的系统名称"
                    },
                    "instance_ip": {
                        "type": "string",
                        "description": "要查询的监控实例的IP地址"
                    },
                },
                "required": ["service", 'instance_ip'],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "请求智能运管后端Api，查询监控服务的资产情况和监控实例",
            "description": "请求智能运管后端Api，查询监控服务的资产情况和监控实例",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "要查询的系统名称"
                    },
                },
                "required": ["service"],

            }
        }
    },

]


def monitor_item_list(ip):
    url = f'{AIOPS_BACKEND_DOMAIN}/api/v1/monitor/mail/machine/field/?instance={ip}'
    resp = requests.get(url=url, auth=AUTH)
    text = json.loads(resp.text)
    result = dict()
    if resp.status_code == 200:
        for item in text:
            result[item.get('field')] = item.get('purpose')
        return result
    else:
        return text


def get_service_asset(service):
    url = f'{AIOPS_BACKEND_DOMAIN}/api/v1/property/mail/?ordering=num_id&page=1&page_size=2000'
    resp = requests.get(url=url, auth=AUTH)
    text = json.loads(resp.text)
    results = text.get('results')
    item_list = []
    for _ in results:
        _['category'] = _.get('category').get('name')
        _["ip_set"] = [_.get("ip") for _ in _.get('ip_set')]
        _.pop('num_id')
        _.pop('creation')
        _.pop('modification')
        _.pop('remark')
        _.pop('sort_weight')
        _.pop('monitor_status')
        for k, v in _.copy().items():
            if not v or v == '无':
                _.pop(k)
        item_list.append(_)
    return item_list


def get_service_asset_edges(service, instance_ip):
    url = f'{AIOPS_BACKEND_DOMAIN}/api/v1/property/mail/topology/search?instance={instance_ip}'
    resp = requests.get(url=url, auth=AUTH)
    text = json.loads(resp.text)
    # print(text)
    return text


def get_monitor_metric_value(ip, start, end, field):
    metric_field_list = monitor_item_list(ip)
    if field not in metric_field_list.keys():
        return f"未知的监控项：{field}"
    # 查询监控指标情况
    start_timestamp = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S").timestamp()
    end_timestamp = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S").timestamp()
    url = f'{AIOPS_BACKEND_DOMAIN}/api/v1/monitor/mail/metric/format-value/?start={start_timestamp}&end={end_timestamp}&instance={ip}&field={field}'
    resp = requests.get(url=url, auth=AUTH)
    text = resp.text
    text = json.loads(text)
    return text


# ------------------------------------------------------------------------


def llm_call(messages):
    # for _ in messages:
    #     print(_)
    #     print('\n')
    data = {
        "model": "Qwen2.5-14B-Instruct",
        "temperature": 0.1,
        "messages": messages,
    }
    response = requests.post(LLM_URL, json=data)
    if response.status_code == 200:
        response_data = response.json()
        if 'choices' in response_data and len(response_data['choices']) > 0:
            generated_content = response_data['choices'][0]['message']['content']
            # print('##Token使用情况##:\n\n', response_data['usage'])
            # print('------------------\n\n')
            return response_data['choices'][0]['message']
        else:
            print(response_data)
            raise Exception("模型没有返回信息")
    else:
        print(f'Error: {response.status_code}')
        print(response.text)


def init_message_by_role(role, content):
    message = {
        'role': role,
        "content": content
    }
    return message


def parse_llm_response(llm_resp_content):
    invalid_value = ["空", '无']
    thought_match = re.search(r'<思考过程>(.*)</思考过程>', llm_resp_content, re.S)
    if thought_match and thought_match[0] not in invalid_value:
        thought = thought_match.group(1)
    else:
        thought = ""
    action_match = re.search(r'<工具调用>(.*)</工具调用>', llm_resp_content)
    if action_match and action_match[0] not in invalid_value:
        action = action_match.group(1)
    else:
        action = ""
    action_input_match = re.search(r'<调用参数>(.*)</调用参数>', llm_resp_content)
    if action_input_match and action_input_match[0] not in invalid_value:
        action_input = action_input_match.group(1)
    else:
        action_input = ""
    if "<最终答案>" in llm_resp_content and "</最终答案>" not in llm_resp_content:
        llm_resp_content += "</最终答案>"
    final_answer_match = re.search(r'<最终答案>(.*)</最终答案>', llm_resp_content, re.S)
    if final_answer_match and final_answer_match[0] not in invalid_value:
        final_answer = final_answer_match.group(1)
    else:
        final_answer = ""
    result = {
        'thought': thought,
        'action': action,
        'action_input': action_input,
        'final_answer': final_answer,
    }
    return result


def react(llm_resp_content):
    is_final = False
    llm_parsed_dict = parse_llm_response(llm_resp_content)
    # print("##解析后的参数", llm_parsed_dict)
    action = llm_parsed_dict.get('action')
    action_input = llm_parsed_dict.get('action_input')
    final_answer = llm_parsed_dict.get('final_answer')
    # print("当前的大模型解析", llm_parsed_dict)
    if action and action_input:
        # print("##调用函数##", action, action_input)
        action_input = json.loads(action_input)
        if action == '请求智能运管后端Api，获取指标项的时序数据':
            return get_monitor_metric_value(**action_input), is_final
        if action == '请求智能运管后端Api，查询监控实例有哪些监控项':
            return monitor_item_list(action_input.get('instance')), is_final
        if action == '请求智能运管后端Api，查询监控服务的资产情况和监控实例':
            return get_service_asset(action_input.get('service')), is_final
        if action == "请求智能运管后端Api，查询监控实例之间的拓扑关联关系":
            return get_service_asset_edges(**action_input), is_final
    if final_answer:
        is_final = True
        return final_answer, is_final
    else:
        result = """
生成的文本格式有误，严格按照以下指定格式生成响应：
```
<思考过程>你的思考过程</思考过程>
<工具调用>工具名称（必须是{tool_names}之一），如果不调用工具，则为空</工具调用>
<调用参数>工具输入参数（严格符合工具描述格式）</调用参数>
<最终答案>用户问题的最终结果（知道问题的最终答案时返回）</最终答案>
```
"""
        return result, is_final


def stream_response_format(category, content):
    data = f'data: {json.dumps({category: content}, ensure_ascii=False)}\n\n'
    return data


def chat(user_content):
    system_template = '''你是一个严格遵守格式规范的用于运维功能，运维数据可视化，运行于生产环境的ReAct智能体，你叫小助手，必须按以下格式处理请求：

    可用工具：
    {tools}

    处理规则：
    1.根据用户的问题来自行判断是否要调用工具以及调用哪个工具
    2.每次只能调用一个工具
    3.不能伪造数据
    3.严格按照以下xml格式生成响应文本：
    ```
    <思考过程>你的思考过程</思考过程>
    <工具调用>工具名称（必须是{tool_names}之一），如果不调用工具，则为空</工具调用>
    <调用参数>工具输入参数（严格符合工具描述格式）</调用参数>
    <最终答案>用户问题的最终结果（知道问题的最终答案时返回）</最终答案>
    ```
    '''
    history = list()
    history.append(init_message_by_role(
        role='system',
        content=system_template.format(
            tools=json.dumps(tools, ensure_ascii=False),
            tool_names=json.dumps([tool["function"]["name"] for tool in tools], ensure_ascii=False), )
    ))
    current_datetime = str(datetime.datetime.now()).split('.')[0]
    history.append(init_message_by_role(role='user', content=f'当前时间是 {current_datetime}'))
    history.append(init_message_by_role(role='user', content=user_content))
    count = 1
    while True:
        print(f'第{count}次循环')
        llm_resp_message = llm_call(history)
        print('##大模型响应##', llm_resp_message)
        history.append(llm_resp_message)
        response, is_final_flag = react(llm_resp_message.get('content'))
        history.append(init_message_by_role(role='user', content=f"<工具调用结果>: {response}</工具调用结果>"))
        if is_final_flag:
            print(response)
            return
        count += 1
        if count >= 15:
            response = llm_resp_message.get('content')
            print(response)
            return


if __name__ == '__main__':
    # chat('你好')
    chat(
        '我想查询邮件系统 192.168.0.110 这台主机今天1点到1点30分的cpu利用率，并给出echarts折线图的完整html，并进行分析给出分析报告')
