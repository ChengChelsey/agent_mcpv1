
import asyncio, json, datetime, random
from mcp_client import get_mcp_client

async def self_check():
    print("⌛ 连接 MCP / 启动 ADTK Server …")
    cli = await get_mcp_client()            # 首次调用会拉起 adtk_server.py
    det_meta = await cli.list_detectors()
    print(f"✅ 检测器数量: {len(det_meta)}，示例: {list(det_meta)[:5]}")

    # 生成 300 个点的演示序列并插入 3 个尖峰
    now = int(datetime.datetime.now().timestamp())
    series = [[now + 60*i, 0.35 + random.uniform(-.05, .05)] for i in range(300)]
    for idx in (50, 180, 250):
        series[idx][1] += 1.2               # 异常尖峰

    print("⌛ 执行 IQR异常检测 …")
    res = await cli.detect("IQR异常检测", series, {"c": 3.0})
    print("✅ IQR 结果片段:", json.dumps(res, ensure_ascii=False)[:200], "…")

asyncio.run(self_check())
