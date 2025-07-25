#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_idle_driver_location_mat_month.py
---------------------------------------
扫描 2016-11 全月司机轨迹 (tar.gz)，
计算每 10-min × 网格的平均空闲司机数，
保存为 real_datasets/idle_driver_location_mat.pkl

★ 读取 mapped_matrix_int.pkl → rows, cols
★ 自动推导 grid_size_lat / grid_size_lon
★ 因此无需手改任何常量即可兼容 18×28＝504 格
"""
import os, tarfile, math, pickle, sys
from glob import glob
from typing import Tuple

import numpy as np
import pandas as pd

# ───────────── 路径 & 参数 ─────────────────────────
MONTH_PATTERN = "../datasets/drivers_information/2016_1101.tar.gz"
MAPPED_PKL    = "../real_datasets/mapped_matrix_int.pkl"
OUTPUT_PKL    = "../real_datasets/idle_driver_location_mat.pkl"
RESAMPLE_FREQ = "10min"          # 144 槽/日
REQ_COLS      = ['司机ID','订单ID','GPS时间','轨迹点经度','轨迹点纬度']

# 成都 bbox
longitude_range = (102.989623, 104.896262)
latitude_range  = (30.090979,  31.437765)

# ═══════════════════════════════════════════════════
def load_grid() -> Tuple[np.ndarray, float, float]:
    """读取映射矩阵，并返还网格角度步长"""
    with open(MAPPED_PKL, 'rb') as f:
        mat = pickle.load(f)
    rows, cols = mat.shape
    gs_lat = (latitude_range[1]  - latitude_range[0]) / rows
    gs_lon = (longitude_range[1] - longitude_range[0]) / cols
    print(f"🗺  网格: {rows}×{cols}={rows*cols}, "
          f"lat_step={gs_lat:.6f}°, lon_step={gs_lon:.6f}°")
    return mat, gs_lat, gs_lon


def extract_csv(tar_path: str) -> pd.DataFrame:
    """tar.gz → DataFrame(REQ_COLS)"""
    with tarfile.open(tar_path, 'r:gz') as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith('.csv'))
        with tar.extractfile(member) as f:
            return pd.read_csv(f, usecols=REQ_COLS)


def daily_idle_matrix(df: pd.DataFrame,
                      mapped: np.ndarray,
                      gs_lat: float,
                      gs_lon: float,
                      node2col: dict[int,int]) -> np.ndarray:
    """单日空闲矩阵 (144, G)"""
    rows, cols = mapped.shape
    G = len(node2col)
    idle = np.zeros((144, G), int)

    # 1. 时间预处理
    df['gps_time'] = pd.to_datetime(df['GPS时间'])

    intervals = (df.groupby(['司机ID','订单ID'])['gps_time']
                   .agg(start='min', end='max')
                   .reset_index())
    day0 = intervals['start'].dt.floor('D').iloc[0]
    bins = pd.date_range(day0, periods=145, freq=RESAMPLE_FREQ)

    # 2. 每司机
    for drv_id, drv_traj in df.groupby('司机ID'):
        busy = np.zeros(144, bool)
        for _, r in intervals[intervals['司机ID']==drv_id].iterrows():
            i0 = np.searchsorted(bins, r.start, side='right')-1
            i1 = np.searchsorted(bins, r.end  , side='right')-1
            i0, i1 = max(i0,0), min(i1,143)
            if i1>=i0: busy[i0:i1+1] = True

        drv_traj = drv_traj.sort_values('gps_time')
        times = drv_traj['gps_time'].to_numpy('datetime64[ns]')

        for t in range(144):
            if busy[t]: continue
            ts = bins[t].to_datetime64()
            idx = np.argmin(np.abs(times - ts))
            lon, lat = drv_traj[['轨迹点经度','轨迹点纬度']].iloc[idx]

            x = int((lon - longitude_range[0]) / gs_lon)
            y = int((lat - latitude_range[0]) / gs_lat)
            if 0 <= x < cols and 0 <= y < rows:
                nid = mapped[y, x]
                if nid >= 0:
                    idle[t, node2col[nid]] += 1
    return idle


def main():
    mapped, gs_lat, gs_lon = load_grid()
    valid_nodes = np.unique(mapped[mapped >= 0])
    node2col = {nid:i for i, nid in enumerate(valid_nodes)}

    tar_paths = sorted(glob(MONTH_PATTERN))
    if not tar_paths:
        print("❌ 没找到文件"); sys.exit(1)
    print(f"共 {len(tar_paths)} 个日包，开始处理 …")

    sum_idle = np.zeros((144, len(valid_nodes)), float)
    for k, fp in enumerate(tar_paths, 1):
        print(f"[{k:02d}/{len(tar_paths)}] {os.path.basename(fp)} …", end="", flush=True)
        try:
            df_day  = extract_csv(fp)
            idle_d  = daily_idle_matrix(df_day, mapped, gs_lat, gs_lon, node2col)
            sum_idle += idle_d
            print("✓")
        except Exception as e:
            print("⚠️  跳过：", e)

    idle_avg = (sum_idle / len(tar_paths)).astype(np.float32)

    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(idle_avg, f)
    print(f"\n✅ 已保存 {OUTPUT_PKL}  shape={idle_avg.shape}")

    # 简单检查
    print("\n前 3 槽平均空闲司机数向量：")
    for t in range(3):
        print(f"slot {t:03d}", idle_avg[t])

if __name__ == "__main__":
    main()
