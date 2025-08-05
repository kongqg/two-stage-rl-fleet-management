#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate onoff_driver_location_mat.pkl
Shape: (144, G, 2) where G = number of valid grids

Steps:
Read mapped_matrix_int.pkl → get rows, cols
Scan driver trajectories from November 2016 (CSV files inside .tar.gz)
Count the number of idle drivers per 10-minute × grid node
For each day, compute Δ(t) = idle(t) − idle(t−1), then compute μ (mean) and σ (std) across days
"""
import os, tarfile, math, pickle, sys
from glob import glob
from typing import Tuple

import numpy as np
import pandas as pd

# ─────────── 配置 ───────────────────────────────────────
INPUT_PATTERN      = "datasets/drivers_information/2016_11*.tar.gz"
MAPPED_MATRIX_PKL  = "real_datasets/mapped_matrix_int.pkl"
OUTPUT_PKL         = "real_datasets/onoff_driver_location_mat.pkl"
RESAMPLE_FREQ      = "10min"      # => 144 槽/日

# 成都经纬度范围
longitude_range = (102.989623, 104.896262)
latitude_range  = (30.090979,  31.437765)

# ════════════════════════════════════════════════════════
def load_mapped_matrix() -> Tuple[np.ndarray, float, float]:
    with open(MAPPED_MATRIX_PKL, "rb") as f:
        mat = pickle.load(f)
    rows, cols     = mat.shape
    grid_size_lat  = (latitude_range[1]  - latitude_range[0]) / rows
    grid_size_lon  = (longitude_range[1] - longitude_range[0]) / cols
    print(f"网格: {rows}×{cols}={rows*cols}, "
          f"lat_step={grid_size_lat:.6f}°, lon_step={grid_size_lon:.6f}°")
    return mat, grid_size_lat, grid_size_lon


def read_one_tar(tar_path: str) -> pd.DataFrame:
    """Extract the first CSV file inside the tar.gz archive."""
    with tarfile.open(tar_path, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(".csv"))
        with tar.extractfile(member) as f:
            df = pd.read_csv(
                f,
                usecols=['司机ID', '订单ID', 'GPS时间', '轨迹点经度', '轨迹点纬度'],
            )
    df['gps_time'] = pd.to_datetime(df['GPS时间'])
    return df


def compute_idle_mat_for_one_day(
    df: pd.DataFrame,
    mapped_matrix: np.ndarray,
    grid_size_lat: float,
    grid_size_lon: float,
) -> np.ndarray:
    """→ idle_mat shape (144, valid_nodes)"""
    rows, cols = mapped_matrix.shape
    valid_nodes = np.unique(mapped_matrix[mapped_matrix >= 0])
    node2col = {nid: i for i, nid in enumerate(valid_nodes)}
    idle_mat = np.zeros((144, len(valid_nodes)), dtype=int)

    # Calculate the order time interval.
    interval_df = (
        df.groupby(['司机ID', '订单ID'])['gps_time']
          .agg(start='min', end='max')
          .reset_index()
    )
    day0 = interval_df['start'].dt.floor('D').iloc[0]
    bins = pd.date_range(day0, periods=145, freq=RESAMPLE_FREQ)

    # For each driver
    for drv_id, drv_traj in df.groupby('司机ID'):
        busy = np.zeros(144, bool)
        for _, row in interval_df[interval_df['司机ID'] == drv_id].iterrows():
            i0 = np.searchsorted(bins, row['start'], side='right') - 1
            i1 = np.searchsorted(bins, row['end'],   side='right') - 1
            i0, i1 = max(i0, 0), min(i1, 143)
            if i1 >= i0:
                busy[i0:i1+1] = True

        drv_traj = drv_traj.sort_values('gps_time')
        for t in range(144):
            if busy[t]:
                continue
            ts = bins[t]
            idx = np.argmin(np.abs(drv_traj['gps_time'] - ts).values)
            lon = drv_traj['轨迹点经度'].iat[idx]
            lat = drv_traj['轨迹点纬度'].iat[idx]

            x = int((lon - longitude_range[0]) / grid_size_lon)
            y = int((lat - latitude_range[0]) / grid_size_lat)
            if 0 <= x < cols and 0 <= y < rows:
                nid = mapped_matrix[y, x]
                if nid >= 0:
                    idle_mat[t, node2col[nid]] += 1
    return idle_mat


def main():
    mapped_matrix, gs_lat, gs_lon = load_mapped_matrix()

    tar_files = sorted(glob(INPUT_PATTERN))
    if not tar_files:
        print("❌ 未找到任何 tar.gz 文件"); sys.exit(1)

    daily_mats = []
    for k, fp in enumerate(tar_files, 1):
        print(f"[{k:02d}/{len(tar_files)}] {os.path.basename(fp)} …")
        df_day = read_one_tar(fp)
        idle   = compute_idle_mat_for_one_day(df_day,
                                              mapped_matrix,
                                              gs_lat, gs_lon)
        daily_mats.append(idle)

    idle_stack = np.stack(daily_mats, axis=0)          # (D,144,G)
    delta      = np.diff(idle_stack, axis=1, prepend=0)  # (D,144,G)
    mu         = delta.mean(axis=0)                    # (144,G)
    sigma      = delta.std(axis=0)                     # (144,G)

    onoff = np.stack([mu, sigma], axis=2)              # (144,G,2)
    print(f"✅ 生成矩阵 shape={onoff.shape}")

    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(onoff, f)
    print(f"已保存到 {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
