# utils/ts_cache.py
import os
import pickle
import hashlib
import json
import requests
import datetime
from typing import List, Tuple, Optional, Dict, Any

# 配置
CACHE_DIR = "cached_data"
os.makedirs(CACHE_DIR, exist_ok=True)

# 后端API相关配置
AIOPS_BACKEND_DOMAIN = 'https://aiopsbackend.cstcloud.cn'
AUTH = ('chelseyyycheng@outlook.com', 'UofV1uwHwhVp9tcTue')

def _cache_filename(ip: str, field: str, start_ts: int, end_ts: int) -> str:
    """
    生成缓存文件名
    
    参数:
        ip: 主机IP
        field: 指标字段
        start_ts: 开始时间戳
        end_ts: 结束时间戳
    
    返回:
        str: 缓存文件路径
    """
    key = f"{ip}_{field}_{start_ts}_{end_ts}"
    h = hashlib.md5(key.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.pkl")

def fetch_data_from_backend(ip: str, field: str, start_ts: int, end_ts: int) -> List[Tuple[int, float]]:
    """
    从后端API获取时序数据
    
    参数:
        ip: 主机IP
        field: 指标字段
        start_ts: 开始时间戳
        end_ts: 结束时间戳
    
    返回:
        List[Tuple[int, float]]: 时间戳-值的列表
    """
    url = f"{AIOPS_BACKEND_DOMAIN}/api/v1/monitor/mail/metric/format-value/?start={start_ts}&end={end_ts}&instance={ip}&field={field}"
    resp = requests.get(url, auth=AUTH)
    
    if resp.status_code != 200:
        raise Exception(f"后端请求失败: {resp.status_code} => {resp.text}")
    
    j = resp.json()
    results = j.get("results", [])
    if not results:
        return []
    
    vals = results[0].get("values", [])
    arr = []
    
    def parse_ts(s):
        try:
            dt = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            return int(dt.timestamp())
        except:
            return 0
    
    for row in vals:
        if len(row) >= 2:
            tstr, vstr = row[0], row[1]
            t = parse_ts(tstr)
            try:
                v = float(vstr)
            except:
                v = 0.0
            arr.append([t, v])
    
    return arr

def ensure_cache_file(ip: str, field: str, start: str, end: str) -> str:
    """
    确保缓存文件存在，如果不存在则从后端获取
    
    参数:
        ip: 主机IP
        field: 指标字段
        start: 开始时间 (格式: "YYYY-MM-DD HH:MM:SS")
        end: 结束时间 (格式: "YYYY-MM-DD HH:MM:SS")
    
    返回:
        str: 缓存文件路径
    """
    # 转换时间字符串为时间戳
    def to_int(s):
        dt = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    
    st_i = to_int(start)
    et_i = to_int(end)
    
    # 生成缓存文件路径
    fpath = _cache_filename(ip, field, st_i, et_i)
    
    # 检查缓存是否存在
    if os.path.exists(fpath):
        print(f"[缓存] 从本地缓存读取 {ip} {field} 数据")
        return fpath
    
    # 从后端获取数据
    try:
        print(f"[API] 从后端获取 {ip} {field} 数据")
        data = fetch_data_from_backend(ip, field, st_i, et_i)
        
        # 保存到缓存
        with open(fpath, "wb") as f:
            pickle.dump(data, f)
        
        print(f"[缓存] 数据已写入缓存 {fpath}")
        return fpath
    
    except Exception as e:
        print(f"[错误] 获取数据失败: {e}")
        return str(e)

def load_series_from_cache(ip: str, field: str, start: str, end: str) -> List[Tuple[int, float]]:
    """
    从缓存加载时序数据
    
    参数:
        ip: 主机IP
        field: 指标字段
        start: 开始时间 (格式: "YYYY-MM-DD HH:MM:SS")
        end: 结束时间 (格式: "YYYY-MM-DD HH:MM:SS")
    
    返回:
        List[Tuple[int, float]]: 时间戳-值的列表
    """
    # 确保缓存文件存在
    cache_file = ensure_cache_file(ip, field, start, end)
    
    # 检查路径是否为错误信息
    if not os.path.exists(cache_file):
        raise Exception(f"缓存文件不存在: {cache_file}")
    
    # 从缓存加载数据
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    
    return data