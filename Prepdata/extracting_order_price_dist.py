# build_order_price_dist_month.py  ★请直接覆盖原文件★
# =========================================================
"""
扫 2016-11 全月订单，生成 15×2 的 price μ/σ：
  • 每 10-min 为一层 (L_MAX = 15)
  • 每层 ≥ MIN_CNT（默认 30）才计入；否则 μ=σ=0
  • 先做 5–95% 分位截断，去掉离群高价
"""
import os, glob, tarfile, math, pickle, sys
import numpy as np
import pandas as pd
from typing import List

# ─────────── 配置 ─────────────────────────────────────────
GLOB_PATTERN   = "datasets/orders_information/2016_11*.*"  # *.csv / *.tar.gz
OUTPUT_PKL     = "real_datasets/order_price_dist.pkl"

TIME_INTERVAL  = 10   # min
L_MAX          = 9
MIN_CNT        = 30   # 每层最少样本；不足则视为噪声

# —— 成都计价参数（与单日脚本保持一致） ——————————
DAY_START, DAY_END      = 6, 23
BASE_FARE_DAY,  PER_KM_DAY  = 8.0, 1.9
BASE_FARE_NIGHT, PER_KM_NIGHT = 8.0, 2.2
BASE_DIST, RETURN_SURCHARGE   = 2.0, 1.5
EARTH_R = 6371.2  # km

REQ_COLS = [
    '开始计费时间','结束计费时间',
    '上车位置经度','上车位置纬度',
    '下车位置经度','下车位置纬度'
]

# ─────────── 工具函数 ─────────────────────────────────────
def calc_price(dist_km: float, start_time) -> float:
    """估算理论车费（保持原公式）"""
    h = start_time.hour + start_time.minute/60.0
    if DAY_START <= h < DAY_END:                       # 白天
        if dist_km <= BASE_DIST:
            return BASE_FARE_DAY
        elif dist_km <= 10:
            return BASE_FARE_DAY + (dist_km-BASE_DIST)*PER_KM_DAY
        return (BASE_FARE_DAY
                + (10-BASE_DIST)*PER_KM_DAY
                + (dist_km-10)*PER_KM_DAY*RETURN_SURCHARGE)
    else:                                              # 夜间
        if dist_km <= BASE_DIST:
            return BASE_FARE_NIGHT
        elif dist_km <= 10:
            return BASE_FARE_NIGHT + (dist_km-BASE_DIST)*PER_KM_NIGHT
        return (BASE_FARE_NIGHT
                + (10-BASE_DIST)*PER_KM_NIGHT
                + (dist_km-10)*PER_KM_NIGHT*RETURN_SURCHARGE)

def read_one(path: str) -> pd.DataFrame:
    """读取 csv / tar.gz → DataFrame(REQ_COLS)"""
    if path.endswith(".csv"):
        return pd.read_csv(path, usecols=REQ_COLS)
    if path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, 'r:gz') as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith(".csv"))
            with tar.extractfile(member) as f:
                return pd.read_csv(f, usecols=REQ_COLS)
    raise ValueError(f"未知格式: {path}")

def process_df(df: pd.DataFrame, buckets: List[list]) -> None:
    """批量计算距离→价格→分层，填充到 buckets"""
    # 解析时间列
    start = pd.to_datetime(df['开始计费时间'])
    end   = pd.to_datetime(df['结束计费时间'])

    # —— 向量化 Haversine —— (比 apply 快百倍)
    lat1 = np.radians(df['上车位置纬度'].to_numpy())
    lon1 = np.radians(df['上车位置经度'].to_numpy())
    lat2 = np.radians(df['下车位置纬度'].to_numpy())
    lon2 = np.radians(df['下车位置经度'].to_numpy())

    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    dist_km = 2*EARTH_R*np.arcsin(np.sqrt(a))

    # 估算价格（价格公式本身快，直接 apply）
    tmp = pd.DataFrame({'dist_km':dist_km,'start_time':start,'end_time':end})
    tmp['price'] = tmp.apply(lambda r: calc_price(r.dist_km, r.start_time), axis=1)

    # 层编号
    dur_min = (tmp['end_time']-tmp['start_time']).dt.total_seconds()/60.0
    layers  = np.ceil(dur_min/TIME_INTERVAL).astype(int)
    valid   = (layers>=1)&(layers<=L_MAX)

    for l, p in zip(layers[valid], tmp['price'][valid].to_numpy()):
        buckets[l-1].append(float(p))

# ════════════════════════════════════════════════════════
def main():
    paths = sorted(glob.glob(GLOB_PATTERN))
    if not paths:
        print("❌ 未找到任何订单文件"); sys.exit(1)

    print(f"共 {len(paths)} 份 11 月文件，开始处理 …\n")
    buckets = [[] for _ in range(L_MAX)]

    for k, path in enumerate(paths, 1):
        try:
            print(f"[{k:02d}/{len(paths)}] {os.path.basename(path)} …", end="", flush=True)
            df = read_one(path)
            process_df(df, buckets)
            print("✓")
        except Exception as e:
            print(f"⚠️  跳过（{e}）")

    # —— 计算 μ/σ with trimming & MIN_CNT ——
    stats = np.zeros((L_MAX,2), np.float32)
    for i, arr in enumerate(buckets):
        n = len(arr)
        if n < MIN_CNT:
            # 样本太少，视为噪声
            stats[i] = 0.0
            continue
        v = np.asarray(arr, dtype=float)
        # 5–95% 分位截断
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        v = v[(v>=lo)&(v<=hi)]
        stats[i,0] = v.mean()
        stats[i,1] = v.std(ddof=0)

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL,'wb') as f:
        pickle.dump(stats, f)
    print(f"\n✅ price μ/σ 已保存 → {OUTPUT_PKL} (shape={stats.shape})\n")

    # 打印检查
    print("Layer | Dur |  mean |  std | n_raw")
    for i,(μ,σ) in enumerate(stats,start=1):
        print(f"{i:5d} | {i*TIME_INTERVAL:3d}m | {μ:6.2f} | {σ:6.2f} | {len(buckets[i-1])}")

if __name__ == "__main__":
    main()
