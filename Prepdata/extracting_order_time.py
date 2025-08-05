# -*- coding: utf-8 -*-
"""
Build the duration distribution of 10-minute layers (entire November 2016)

Output: `real_datasets/order_time_dist.pkl`
Shape: `List[float]` with length = `L_MAX` (15)
Values: ∑pᵢ = 1, where pᵢ represents the probability of layer *i* (i × 10 minutes)

"""
import os, glob, tarfile, pickle, sys
import numpy as np
import pandas as pd

# ── 参数 ────────────────────────────────────────────────
GLOB_PATTERN       = "datasets/orders_information/2016_11*.csv"       # 支持 csv / tar.gz
OUTPUT_PKL    = "real_datasets/order_time_dist.pkl"
TIME_INTERVAL = 10      # min
L_MAX         = 9

REQ_COLS = ['开始计费时间', '结束计费时间']

def read_order_file(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path, usecols=REQ_COLS)
    if path.endswith((".tar.gz", ".tgz")):                  # tar 包只取里面第一份 csv
        with tarfile.open(path, 'r:gz') as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith('.csv'))
            with tar.extractfile(member) as f:
                return pd.read_csv(f, usecols=REQ_COLS)
    raise ValueError(f"无法识别文件格式 {path}")

# ════════════════════════════════════════════════════════
def main():
    paths = sorted(glob.glob(GLOB_PATTERN))
    if not paths:
        print("❌ 未找到任何订单文件"); sys.exit(1)

    layer_counts = np.zeros(L_MAX, dtype=np.int64)     # index 0 → layer-1

    print(f"将处理 {len(paths)} 份文件 …")
    for k, p in enumerate(paths, 1):
        print(f"[{k:02d}/{len(paths)}] {os.path.basename(p)}", end=" ", flush=True)
        try:
            df = read_order_file(p)
            # — Parse time —
            start = pd.to_datetime(df['开始计费时间'])
            end   = pd.to_datetime(df['结束计费时间'])
            duration_min = (end - start).dt.total_seconds() / 60.0
            layers = np.ceil(duration_min / TIME_INTERVAL).astype(np.int16)

            # — Filter values in the range 1…L\_MAX. —
            mask = (layers >= 1) & (layers <= L_MAX)
            for l in layers[mask]:
                layer_counts[l-1] += 1
            print("✓")
        except Exception as e:
            print(f"⚠️  跳过 ({e})")

    total = layer_counts.sum()
    if total == 0:
        print("⚠️  没有合法订单，退出"); sys.exit(1)

    order_time_dist = (layer_counts / total).tolist()

    # save
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(order_time_dist, f)
    print(f"\n✅ 已保存到 {OUTPUT_PKL}")

    print("\nLayer | Duration |  Count |   Prob")
    for i, (cnt, p) in enumerate(zip(layer_counts, order_time_dist), start=1):
        print(f"{i:5d} | {i*TIME_INTERVAL:8d}m | {cnt:6d} | {p:7.5f}")

if __name__ == "__main__":
    main()
