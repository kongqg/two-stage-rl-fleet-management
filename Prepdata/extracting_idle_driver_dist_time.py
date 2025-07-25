import os
import pickle
from glob import glob

import numpy as np
import pandas as pd

# ── 用户配置 ───────────────────────────────────────────────
INPUT_PATTERN = "datasets/drivers_information/2016_11*.tar.gz"         # 匹配 2016.1101.csv ... 2016.1130.csv
OUTPUT_PKL    = "real_datasets/idle_driver_dist_time.pkl"

# 重采样频率：10 分钟，即一天 144 段
RESAMPLE_FREQ = "10min"


def compute_daily_idle_counts(filepath):
    """
    读取单日轨迹 CSV，返回长度为 144 的空闲司机数数组。
    """
    df = pd.read_csv(filepath)
    df['gps_time'] = pd.to_datetime(df['GPS时间'])

    # 每笔订单的服务开始/结束
    intervals = (
        df.groupby(['司机ID', '订单ID'])['gps_time']
          .agg(start='min', end='max')
          .reset_index()
    )

    # 当天起点（00:00:00）
    day = intervals['start'].dt.floor('D').iloc[0]
    # 构造 145 个分隔点，形成 144 个 10 分钟时段
    bins = pd.date_range(start=day, periods=145, freq=RESAMPLE_FREQ)

    # 统计服务中司机数
    busy = np.zeros(144, dtype=int)
    for _, row in intervals.iterrows():
        i_start = np.searchsorted(bins, row['start'], side='right') - 1
        i_end   = np.searchsorted(bins, row['end'],   side='right') - 1
        i_start = max(i_start, 0)
        i_end   = min(i_end,   143)
        if i_end >= i_start:
            busy[i_start:i_end+1] += 1

    total_drivers = df['司机ID'].nunique()
    idle_counts = total_drivers - busy
    return idle_counts


def main():
    # 收集所有文件路径并排序
    file_list = sorted(glob(INPUT_PATTERN))
    if not file_list:
        raise FileNotFoundError(f"No files match pattern {INPUT_PATTERN}")

    # 为每一天计算空闲计数
    daily_idle = []
    for fp in file_list:
        idle = compute_daily_idle_counts(fp)
        daily_idle.append(idle)

    # 堆叠成 (30, 144) 矩阵
    idle_matrix = np.stack(daily_idle, axis=0)

    # 计算每个时段的均值和标准差
    mu    = np.mean(idle_matrix, axis=0)  # 形状 (144,)
    sigma = np.std(idle_matrix,  axis=0)  # 形状 (144,)

    # 组合成 (144, 2)
    idle_driver_dist_time = np.vstack([mu, sigma]).T

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(idle_driver_dist_time, f)

    # 打印前 5 个时段的结果示例
    print("idle_driver_dist_time (前 5 个时段)：")
    for t in range(5):
        print(f"t={t:03d}: μ={mu[t]:.2f}, σ={sigma[t]:.2f}")
    print(f"\n已保存到 {OUTPUT_PKL}")


    '''
    idle_driver_dist_time (前 5 个时段)：
    t=000: μ=39070.30, σ=2051.41
    t=001: μ=38372.23, σ=1978.32
    t=002: μ=38021.20, σ=1967.73
    t=003: μ=38026.27, σ=1983.62
    t=004: μ=38140.53, σ=1985.52
    '''


if __name__ == "__main__":
    main()
