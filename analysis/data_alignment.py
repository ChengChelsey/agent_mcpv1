# 文件: analysis/data_alignment.py
import numpy as np
from scipy.interpolate import interp1d

def align_series(series1, series2, method="linear", fill_value="extrapolate"):
    """
    使用SciPy的interp1d进行插值，将series1和series2在相同的时间戳上对齐。
    
    Parameters
    ----------
    series1 : list of [int_ts, val]
    series2 : list of [int_ts, val]
    method  : str, 插值方法，可选 'linear', 'nearest', 'cubic' 等
    fill_value: str or float, 边界填充值,可 'extrapolate' 或一个常数

    Returns
    -------
    s1_aligned, s2_aligned : 两个对齐后的序列 (list of [int_ts, val])
        对齐后时间戳 = 两个序列所有时间戳的并集 + 指定插值方法
    """
    if not series1 or not series2:
        # 若其中一个为空，则原样返回
        return series1, series2

    # 1) 排序并拆成np.array
    s1_sorted = sorted(series1, key=lambda x: x[0])
    s2_sorted = sorted(series2, key=lambda x: x[0])

    t1 = np.array([row[0] for row in s1_sorted], dtype=np.float64)
    v1 = np.array([row[1] for row in s1_sorted], dtype=np.float64)
    t2 = np.array([row[0] for row in s2_sorted], dtype=np.float64)
    v2 = np.array([row[1] for row in s2_sorted], dtype=np.float64)

    # 2) 取并集时间戳
    all_ts = np.union1d(t1, t2)

    # 3) 建立插值函数
    f1 = interp1d(t1, v1, kind=method, fill_value=fill_value, bounds_error=False)
    f2 = interp1d(t2, v2, kind=method, fill_value=fill_value, bounds_error=False)

    # 4) 对并集时间戳插值
    new_v1 = f1(all_ts)
    new_v2 = f2(all_ts)

    # 5) 返回对齐结果
    s1_aligned = [[int(ts), float(val)] for ts, val in zip(all_ts, new_v1)]
    s2_aligned = [[int(ts), float(val)] for ts, val in zip(all_ts, new_v2)]

    return s1_aligned, s2_aligned
