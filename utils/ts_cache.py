# utils/ts_cache.py
import os
import pickle
import hashlib
import json
import requests
import datetime
from typing import List, Tuple, Optional, Dict, Any

CACHE_DIR = "cached_data"
os.makedirs(CACHE_DIR, exist_ok=True)

AIOPS_BACKEND_DOMAIN = 'https://aiopsbackend.cstcloud.cn'
AUTH = ('chelseyyycheng@outlook.com', 'UofV1uwHwhVp9tcTue')

def _cache_filename(ip: str, field: str, start_ts: int, end_ts: int) -> str:

    key = f"{ip}_{field}_{start_ts}_{end_ts}"
    h = hashlib.md5(key.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.pkl")

def fetch_data_from_backend(ip: str, field: str, start_ts: int, end_ts: int) -> List[Tuple[int, float]]:
    #从后端API获取时序数据

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
    #查询缓存文件，不存在从后端获取

    def to_int(s):
        dt = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    
    st_i = to_int(start)
    et_i = to_int(end)
    
    fpath = _cache_filename(ip, field, st_i, et_i)
    
    if os.path.exists(fpath):
        print(f"[缓存] 从本地缓存读取 {ip} {field} 数据")
        return fpath
    
    try:
        print(f"[API] 从后端获取 {ip} {field} 数据")
        data = fetch_data_from_backend(ip, field, st_i, et_i)
        
        with open(fpath, "wb") as f:
            pickle.dump(data, f)
        
        print(f"[缓存] 数据已写入缓存 {fpath}")
        return fpath
    
    except Exception as e:
        print(f"[错误] 获取数据失败: {e}")
        return str(e)

def load_series_from_cache(ip: str, field: str, start: str, end: str) -> List[Tuple[int, float]]:

    
    cache_file = ensure_cache_file(ip, field, start, end)
    
    if not os.path.exists(cache_file):
        raise Exception(f"缓存文件不存在: {cache_file}")
    
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    
    return data