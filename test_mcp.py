import asyncio, datetime, random, json
from mcp_client import get_mcp_client

async def main():
    cli = await get_mcp_client()
    dets = await cli.list_detectors()
    print("✅ 检测器", list(dets))

    now = int(datetime.datetime.now().timestamp())
    series = [[now+i, 0.2+0.01*random.random()] for i in range(60)]
    series[30][1] = 1.0                                   # 异常尖峰

    res = await cli.detect("IQR异常检测", series, {"c": 2.5})
    print("检测结果片段:", json.dumps(res,ensure_ascii=False)[:120], "…")

asyncio.run(main())
