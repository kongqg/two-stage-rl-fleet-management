#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 2016-11 全月订单在网格上的时-空分布：
1. 读取 mapped_matrix_int.pkl（任意行×列，例如 18 × 28 = 504）
2. 将订单映射到 node_id
3. 以 10min × node 计数，按天堆叠求每槽的 μ, σ
4. 保存 order_num_dist.pkl  (len=144，每槽一个 dict{node_id:[μ,σ]})
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
    """支持 .csv 或 .tar.gz（压缩里第一张 csv）"""
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
    """经纬度 → (row,col) → node_id，返回含 begin_time,node_id 的 DataFrame"""
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
    # 0. 载入网格映射矩阵，自动推导网格角度宽度
    with open(MAPPED_PKL, 'rb') as f:
        mapped_matrix = pickle.load(f)
    rows, cols     = mapped_matrix.shape
    grid_size_lat  = (latitude_range[1]  - latitude_range[0]) / rows
    grid_size_lon  = (longitude_range[1] - longitude_range[0]) / cols

    valid_node_ids = np.unique(mapped_matrix[mapped_matrix >= 0])
    print(f"🗺  网格尺寸: {rows} × {cols} = {rows*cols}（应为 504）")
    print(f" lat_step={grid_size_lat:.6f}°, lon_step={grid_size_lon:.6f}°")

    # 1. 找到 11 月全部订单文件
    paths = sorted(glob.glob(GLOB_PATTERN))
    if not paths:
        print("❌ 未找到订单文件"); sys.exit(1)
    print(f"将处理 {len(paths)} 份订单文件 …")

    # 累加器：node_id → [144 × list[counts]]
    per_slot = {nid: [[] for _ in range(144)] for nid in valid_node_ids}

    # 2. 逐文件统计
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

    # 3. 计算 μ/σ
    order_num_dist: list[dict[int, list[float]]] = []
    for t in range(144):
        slot_dict = {}
        for nid in valid_node_ids:
            arr = np.array(per_slot[nid][t], dtype=float)
            if arr.size == 0:
                continue
            slot_dict[int(nid)] = [float(arr.mean()), float(arr.std(ddof=0))]
        order_num_dist.append(slot_dict)

    # 4. 保存
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(order_num_dist, f)
    print(f"\n✅ 已保存到 {OUTPUT_PKL} ，len={len(order_num_dist)} (144 槽/天)")

    # 5. 打印前几行验算
    print("\n=== t=0 前 6 个网格 μ/σ 示例 ===")
    for nid, ms in list(order_num_dist[0].items())[:100]:
        print(f"node {nid:>4}: μ={ms[0]:.2f}  σ={ms[1]:.2f}")


if __name__ == "__main__":
    main()
