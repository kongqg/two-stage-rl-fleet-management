import os
import pickle
from glob import glob

import numpy as np
import pandas as pd

INPUT_PATTERN = "datasets/drivers_information/2016_11*.tar.gz"         # match 2016.1101.csv ... 2016.1130.csv
OUTPUT_PKL    = "real_datasets/idle_driver_dist_time.pkl"

# Resampling frequency: 10 minutes, which is 144 segments per day.
RESAMPLE_FREQ = "10min"


def compute_daily_idle_counts(filepath):
    """
    Read the single-day trajectory CSV and return an array of idle driver counts of length 144.
    """
    df = pd.read_csv(filepath)
    df['gps_time'] = pd.to_datetime(df['GPS时间'])

    # Service start/end for each order.
    intervals = (
        df.groupby(['司机ID', '订单ID'])['gps_time']
          .agg(start='min', end='max')
          .reset_index()
    )

    # Start of the day (00:00:00).
    day = intervals['start'].dt.floor('D').iloc[0]
    # Construct 145 separation points to form 144 10-minute time slots.
    bins = pd.date_range(start=day, periods=145, freq=RESAMPLE_FREQ)

    # Count the number of drivers in service.
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
    # Collect all file paths and sort them.
    file_list = sorted(glob(INPUT_PATTERN))
    if not file_list:
        raise FileNotFoundError(f"No files match pattern {INPUT_PATTERN}")

    # Calculate the idle count for each day.
    daily_idle = []
    for fp in file_list:
        idle = compute_daily_idle_counts(fp)
        daily_idle.append(idle)

    # Stack into a (30, 144) matrix.
    idle_matrix = np.stack(daily_idle, axis=0)

    # Calculate the mean and standard deviation for each time slot.
    mu    = np.mean(idle_matrix, axis=0)  # 形状 (144,)
    sigma = np.std(idle_matrix,  axis=0)  # 形状 (144,)

    # Combine into (144, 2)
    idle_driver_dist_time = np.vstack([mu, sigma]).T

    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(idle_driver_dist_time, f)

    print("idle_driver_dist_time (前 5 个时段)：")
    for t in range(5):
        print(f"t={t:03d}: μ={mu[t]:.2f}, σ={sigma[t]:.2f}")
    print(f"\n已保存到 {OUTPUT_PKL}")




if __name__ == "__main__":
    main()
