# 文件: analysis/data_alignment.py
import numpy as np
from scipy.interpolate import interp1d

def align_series(series1, series2, method="linear", fill_value="extrapolate"):
    
    #使用SciPy的interp1d进行插值，将series1和series2在相同的时间戳上对齐。
    if not series1 or not series2:
        return series1, series2

    s1_sorted = sorted(series1, key=lambda x: x[0])
    s2_sorted = sorted(series2, key=lambda x: x[0])

    t1 = np.array([row[0] for row in s1_sorted], dtype=np.float64)
    v1 = np.array([row[1] for row in s1_sorted], dtype=np.float64)
    t2 = np.array([row[0] for row in s2_sorted], dtype=np.float64)
    v2 = np.array([row[1] for row in s2_sorted], dtype=np.float64)

    all_ts = np.union1d(t1, t2)

    f1 = interp1d(t1, v1, kind=method, fill_value=fill_value, bounds_error=False)
    f2 = interp1d(t2, v2, kind=method, fill_value=fill_value, bounds_error=False)

    new_v1 = f1(all_ts)
    new_v2 = f2(all_ts)

    s1_aligned = [[int(ts), float(val)] for ts, val in zip(all_ts, new_v1)]
    s2_aligned = [[int(ts), float(val)] for ts, val in zip(all_ts, new_v2)]

    return s1_aligned, s2_aligned
