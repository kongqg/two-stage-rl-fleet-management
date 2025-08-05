#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Divide the region into exactly TARGET_CELLS grids based on the given latitude and longitude range.
Automatically select (rows, cols) such that rows * cols = TARGET_CELLS,
and ensure rows/cols closely matches the ratio of the latitude span to the longitude span.
Save the result as real_datasets/mapped_matrix_int.pkl.
"""
import os
import pickle
import math
import numpy as np

# --------------------------- 配   置 ---------------------------
TARGET_CELLS     = 504   # 目标网格总数
longitude_range  = (102.989623, 104.896262)
latitude_range   = (30.090979,  31.437765)
SAVE_DIR         = "../real_datasets"
# --------------------------------------------------------------


def choose_rows_cols(total_cells: int,
                     ratio_lat_lon: float) -> tuple[int, int]:
    """
        Among all factor pairs of total_cells, select rows and cols such that
        rows * cols == total_cells
        and rows / cols is closest to ratio_lat_lon
    """
    best_rows, best_cols = 1, total_cells
    best_err = float("inf")

    for r in range(1, int(math.sqrt(total_cells)) + 1):
        if total_cells % r != 0:
            continue
        c = total_cells // r
        for rows, cols in [(r, c), (c, r)]:   # 两种排列都试一下
            err = abs((rows / cols) - ratio_lat_lon)
            if err < best_err:
                best_rows, best_cols, best_err = rows, cols, err
    return best_rows, best_cols


def create_mapped_matrix(rows: int,
                         cols: int,
                         lon_min: float, lon_max: float,
                         lat_min: float, lat_max: float,
                         save_dir: str = SAVE_DIR) -> str:
    """
        Generate a row-major indexed mapping matrix and persist it.
    """
    mapped_matrix_int = np.arange(rows * cols, dtype=int).reshape((rows, cols))

    delta_lat = lat_max - lat_min
    delta_lon = lon_max - lon_min
    grid_size_lat_deg = delta_lat / rows
    grid_size_lon_deg = delta_lon / cols
    mean_lat = (lat_min + lat_max) / 2.0

    # ChatGPT 说：
    # Convert to kilometers.
    km_lat = grid_size_lat_deg * 110.574
    km_lon = grid_size_lon_deg * 111.320 * math.cos(math.radians(mean_lat))

    # print("------------- 网格参数 -------------")
    # print(f"行  (纬向) rows : {rows}")
    # print(f"列  (经向) cols : {cols}")
    # print(f"总格数 rows*cols: {rows*cols}")
    # print()
    # print(f"纬度跨度 Δlat   : {delta_lat:.6f}°")
    # print(f"经度跨度 Δlon   : {delta_lon:.6f}°")
    # print(f"单格纬度宽 ~    : {grid_size_lat_deg:.6f}°  ≈ {km_lat:.2f} km")
    # print(f"单格经度宽 ~    : {grid_size_lon_deg:.6f}°  ≈ {km_lon:.2f} km")
    # print("------------------------------------")

    # save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "mapped_matrix_int.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(mapped_matrix_int, f)

    print(f"映射矩阵已保存到: {out_path}")
    return out_path


def main():
    lon_min, lon_max = longitude_range
    lat_min, lat_max = latitude_range

    # Latitude-to-longitude span ratio
    ratio_extent = (lat_max - lat_min) / (lon_max - lon_min)

    # 1. choose rows, cols
    rows, cols = choose_rows_cols(TARGET_CELLS, ratio_extent)

    # 2. Create and save the mapping matrix.
    create_mapped_matrix(rows, cols,
                         lon_min, lon_max,
                         lat_min, lat_max)


if __name__ == "__main__":
    main()
