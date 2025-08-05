#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert all orders from November 2016 into the format:
[origin_node, dest_node, start_interval, duration, price]
and save to real_datasets/order_real.pkl.

★ Automatically read mapped_matrix_int.pkl → get rows, cols
★ Automatically infer grid_size_lat / grid_size_lon
★ Thus, directly supports 18×28 = 504 grids (or any other row-column configuration)
"""
import os, glob, pickle, math
import numpy as np
import pandas as pd

# ─────────── 路径与常量 ──────────────────────────
INPUT_DIR          = "../datasets/orders_information"
INPUT_PATTERN      = os.path.join(INPUT_DIR, "2016_11*.csv")
MAPPED_MATRIX_PKL  = "real_datasets/mapped_matrix_int.pkl"
OUTPUT_PKL         = "real_datasets/order_real.pkl"

TIME_INTERVAL      = 10   # min
L_MAX              = 9    # Maximum number of layers (discard those >9 as noise)


# Chengdu latitude and longitude range
longitude_range = (102.989623, 104.896262)
latitude_range  = (30.090979,  31.437765)

# —— Pricing parameters ————————————
DAY_START, DAY_END      = 6, 23
BASE_FARE_DAY,  PER_KM_DAY  = 8.0, 1.9
BASE_FARE_NIGHT, PER_KM_NIGHT = 8.0, 2.2
BASE_DIST, RETURN_SURCHARGE   = 2.0, 1.5
EARTH_R = 6371.2  # km

# ─────────── Pricing function & distance calculation ─────────────────────
def haversine(lat1, lon1, lat2, lon2):
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = φ2 - φ1
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(a))

def calc_price(dist_km, start_time):
    h = start_time.hour + start_time.minute/60.0
    if DAY_START <= h < DAY_END:                       # day
        base, per = BASE_FARE_DAY, PER_KM_DAY
    else:                                              # night
        base, per = BASE_FARE_NIGHT, PER_KM_NIGHT

    if dist_km <= BASE_DIST:
        return base
    elif dist_km <= 10:
        return base + (dist_km - BASE_DIST) * per
    return (base
            + (10 - BASE_DIST) * per
            + (dist_km - 10) * per * RETURN_SURCHARGE)

# ════════════════════════════════════════════════════════
def main():
    # 0. mapped_matrix → rows, cols → grid step
    with open(MAPPED_MATRIX_PKL, "rb") as f:
        mapped_matrix = pickle.load(f)
    rows, cols = mapped_matrix.shape
    grid_size_lat = (latitude_range[1]  - latitude_range[0]) / rows
    grid_size_lon = (longitude_range[1] - longitude_range[0]) / cols
    print(f"🗺  网格: {rows}×{cols}={rows*cols}，"
          f"lat_step={grid_size_lat:.6f}°, lon_step={grid_size_lon:.6f}°")

    # 1. Collect all CSV files.
    csv_files = sorted(glob.glob(INPUT_PATTERN))
    if not csv_files:
        raise FileNotFoundError(f"在 {INPUT_PATTERN} 找不到文件")
    print(f"读取 {len(csv_files)} 个CSV …")

    # 2. Read and merge.
    df = pd.concat((pd.read_csv(p) for p in csv_files), ignore_index=True)

    # 3. Preprocess time.
    df['start_time'] = pd.to_datetime(df['开始计费时间'])
    df['end_time']   = pd.to_datetime(df['结束计费时间'])
    minutes = df['start_time'].dt.hour * 60 + df['start_time'].dt.minute
    df['start_interval'] = (minutes // TIME_INTERVAL).astype(int)
    df['duration'] = np.ceil(
        (df['end_time'] - df['start_time']).dt.total_seconds() / 60 / TIME_INTERVAL
    ).astype(int)

    # 4. Latitude & longitude → grid index
    df['x1'] = ((df['上车位置经度'] - longitude_range[0]) / grid_size_lon).astype(int)
    df['y1'] = ((df['上车位置纬度'] - latitude_range[0]) / grid_size_lat).astype(int)
    df['x2'] = ((df['下车位置经度'] - longitude_range[0]) / grid_size_lon).astype(int)
    df['y2'] = ((df['下车位置纬度'] - latitude_range[0]) / grid_size_lat).astype(int)

    # 5. Out-of-bounds filtering
    mask = (
        (df.x1>=0)&(df.x1<cols)&(df.y1>=0)&(df.y1<rows) &
        (df.x2>=0)&(df.x2<cols)&(df.y2>=0)&(df.y2<rows)
    )
    df = df[mask].copy()

    # 6. Map to node_id
    df['origin_node'] = mapped_matrix[df['y1'], df['x1']]
    df['dest_node']   = mapped_matrix[df['y2'], df['x2']]

    # 7. Calculate distance & price
    df['dist_km'] = np.vectorize(haversine)(
        df['上车位置纬度'], df['上车位置经度'],
        df['下车位置纬度'], df['下车位置经度']
    )
    df['price'] = df.apply(lambda r: calc_price(r.dist_km, r.start_time), axis=1)

    # 8. Final filtering
    df = df[
        (df.origin_node>=0)&(df.dest_node>=0) &
        (df.start_interval>=0)               &
        (df.duration.between(1, L_MAX))
    ]

    order_real = df[['origin_node','dest_node',
                     'start_interval','duration','price']].astype(int)
    order_real = order_real.values.tolist()

    # 9. save
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(order_real, f)
    print(f"✅ 已保存 {len(order_real):,} 条 → {OUTPUT_PKL}")

    print("示例前 5 行：")
    for row in order_real[:5]:
        print(row)

if __name__ == "__main__":
    main()
