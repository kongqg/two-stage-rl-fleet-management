#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the spatiotemporal distribution of all orders in November 2016:

1. Read `mapped_matrix_int.pkl` (with arbitrary rows × cols, e.g., 18 × 28 = 504)
2. Map each order to its corresponding `node_id`
3. Count orders per 10-minute × node, then stack daily counts to compute mean (μ) and standard deviation (σ) for each slot
4. Save the result as `order_num_dist.pkl` (`len=144`, one `dict{node_id: [μ, σ]}` per time slot)

"""
import os, glob, math, tarfile, pickle, sys
from typing import Tuple

import numpy as np
import pandas as pd

# ─────────────── 路径 & 常量 ──────────────────────────────
GLOB_PATTERN  = "../datasets/orders_information/2016_11*.csv"
MAPPED_PKL    = "../real_datasets/mapped_matrix_int.pkl"      # 必须已由 504 网格脚本生成
OUTPUT_PKL    = "../real_datasets/order_num_dist.pkl"

longitude_range = (102.989623, 104.896262)
latitude_range  = (30.090979,  31.437765)

RESAMPLE_FREQ   = "10min"   # 144 段/天
REQ_COLS        = ['开始计费时间', '上车位置经度', '上车位置纬度']

# ══════════════════════════════════════════════════════════
def extract_csv(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path, usecols=REQ_COLS)

    if path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, 'r:gz') as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith('.csv'))
            with tar.extractfile(member) as f:
                return pd.read_csv(f, usecols=REQ_COLS)

    raise ValueError(f"未知文件格式: {path}")


def map_orders_to_nodes(df: pd.DataFrame,
                        mapped_matrix: np.ndarray,
                        grid_size_lat: float,
                        grid_size_lon: float,
                        ) -> pd.DataFrame:
    """Latitude & longitude → (row, col) → node_id, and return a DataFrame containing begin_time and node_id."""
    n_rows, n_cols = mapped_matrix.shape
    x_idx = ((df['上车位置经度'] - longitude_range[0]) / grid_size_lon).astype(int)
    y_idx = ((df['上车位置纬度'] - latitude_range[0]) / grid_size_lat).astype(int)

    mask = (x_idx >= 0) & (x_idx < n_cols) & (y_idx >= 0) & (y_idx < n_rows)
    node_id = mapped_matrix[y_idx[mask], x_idx[mask]]

    df_valid = df.loc[mask, ['开始计费时间']].copy()
    df_valid['node_id'] = node_id
    df_valid['begin_time'] = pd.to_datetime(df_valid['开始计费时间'])
    return df_valid[['begin_time', 'node_id']]


def main():
    # 0. Load the grid mapping matrix and automatically infer the grid angular width.
    with open(MAPPED_PKL, 'rb') as f:
        mapped_matrix = pickle.load(f)
    rows, cols     = mapped_matrix.shape
    grid_size_lat  = (latitude_range[1]  - latitude_range[0]) / rows
    grid_size_lon  = (longitude_range[1] - longitude_range[0]) / cols

    valid_node_ids = np.unique(mapped_matrix[mapped_matrix >= 0])
    print(f"网格尺寸: {rows} × {cols} = {rows*cols}（应为 504）")
    print(f"lat_step={grid_size_lat:.6f}°, lon_step={grid_size_lon:.6f}°")

    # 1. Locate all order files from the month of November.
    paths = sorted(glob.glob(GLOB_PATTERN))
    if not paths:
        print("❌ 未找到订单文件"); sys.exit(1)
    print(f"将处理 {len(paths)} 份订单文件 …")

    # Accumulator: node_id → [144 × list of counts]
    per_slot = {nid: [[] for _ in range(144)] for nid in valid_node_ids}

    # 2. Count file by file.
    for k, path in enumerate(paths, 1):
        print(f"[{k:02d}/{len(paths)}] {os.path.basename(path)} …", end="", flush=True)
        try:
            df_raw = extract_csv(path)
            df_map = map_orders_to_nodes(df_raw,
                                         mapped_matrix,
                                         grid_size_lat, grid_size_lon)
            # resample to 10-minute slots
            df_map = df_map.set_index('begin_time')
            grp    = df_map.groupby('node_id').resample(RESAMPLE_FREQ).size()
            counts = grp.unstack(fill_value=0)      # (node_id, 144)

            for nid, row in counts.iterrows():
                for t in range(144):
                    per_slot[nid][t].append(int(row.iloc[t]))
            print("✓")
        except Exception as e:
            print("⚠️  跳过，原因:", e)

    # 3. Compute μ (mean) and σ (standard deviation).
    order_num_dist: list[dict[int, list[float]]] = []
    for t in range(144):
        slot_dict = {}
        for nid in valid_node_ids:
            arr = np.array(per_slot[nid][t], dtype=float)
            if arr.size == 0:
                continue
            slot_dict[int(nid)] = [float(arr.mean()), float(arr.std(ddof=0))]
        order_num_dist.append(slot_dict)

    # 4. save
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(order_num_dist, f)
    print(f"\n✅ 已保存到 {OUTPUT_PKL} ，len={len(order_num_dist)} (144 槽/天)")

    print("\n=== t=0 前 6 个网格 μ/σ 示例 ===")
    for nid, ms in list(order_num_dist[0].items())[:100]:
        print(f"node {nid:>4}: μ={ms[0]:.2f}  σ={ms[1]:.2f}")


if __name__ == "__main__":
    main()
