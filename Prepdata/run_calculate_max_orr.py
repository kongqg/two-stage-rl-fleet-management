import pickle
import numpy as np
from scipy.optimize import linprog
import os
import sys


# --- 辅助函数 (为了让脚本能独立运行) ---
def ids_1dto2d(idx, M, N):
    """将一维的网格ID转换为二维的坐标 (row, col)"""
    i = idx // N
    j = idx % N
    return (i, j)


def ids_2dto1d(i, j, M, N):
    """将二维的坐标 (row, col) 转换为一维的网格ID"""
    i = int(i)
    j = int(j)
    if not (0 <= i < M and 0 <= j < N):
        return -1
    return i * N + j


def get_neighbor_list_hex(i, j, M, N):
    """计算六边形网格的邻居坐标"""
    neighbors = []
    if i % 2 == 0:
        potential_neighbors = [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j), (i - 1, j - 1), (i + 1, j - 1)]
    else:
        potential_neighbors = [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j), (i - 1, j + 1), (i + 1, j + 1)]

    for ni, nj in potential_neighbors:
        if 0 <= ni < M and 0 <= nj < N:
            neighbors.append((ni, nj))
    return neighbors


def calculate_max_orr(data_dir='../real_datasets'):
    """
    通过将问题建模为最大流问题，计算理论上的最大订单应答率 (ORR)。
    """
    # --- 1. 数据加载与环境设置 ---
    print("Step 1: Loading data and setting up environment...")
    try:
        with open(os.path.join(data_dir, 'mapped_matrix_int.pkl'), 'rb') as f:
            mapped_matrix_int = pickle.load(f)
        with open(os.path.join(data_dir, 'order_num_dist.pkl'), 'rb') as f:
            order_num_dist = pickle.load(f)
        with open(os.path.join(data_dir, 'idle_driver_dist_time.pkl'), 'rb') as f:
            idle_driver_dist_time = pickle.load(f)
        with open(os.path.join(data_dir, 'idle_driver_location_mat.pkl'), 'rb') as f:
            idle_driver_location_mat = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}. Make sure all .pkl files are in the directory '{data_dir}'.")
        return

    M, N_cols = mapped_matrix_int.shape
    valid_grid_ids = sorted(list(set(mapped_matrix_int.flatten()) - {-1}))
    num_valid_grids = len(valid_grid_ids)
    grid_id_to_idx_map = {grid_id: i for i, grid_id in enumerate(valid_grid_ids)}
    print(f"Grid dimensions: {M}x{N_cols}. Found {num_valid_grids} valid grids.")

    adjacency_list = [[] for _ in range(num_valid_grids)]
    for i, grid_id in enumerate(valid_grid_ids):
        row, col = ids_1dto2d(grid_id, M, N_cols)
        adjacency_list[i].append(i)
        neighbor_coords = get_neighbor_list_hex(row, col, M, N_cols)
        for r, c in neighbor_coords:
            neighbor_id = ids_2dto1d(r, c, M, N_cols)
            if neighbor_id in grid_id_to_idx_map:
                neighbor_idx = grid_id_to_idx_map[neighbor_id]
                if neighbor_idx not in adjacency_list[i]:
                    adjacency_list[i].append(neighbor_idx)

    NUM_SIMULATION_DAYS = 30
    T = 144
    total_orr_list = []

    print(f"Starting simulation for {NUM_SIMULATION_DAYS} days...")

    for day in range(NUM_SIMULATION_DAYS):
        np.random.seed(day)

        # --- 2. 模拟生成单日供需数据 ---
        daily_orders = np.zeros((T, num_valid_grids))
        for t in range(T):
            if t < len(order_num_dist):
                dist_for_t = order_num_dist[t]
                for i in range(num_valid_grids):
                    grid_id = valid_grid_ids[i]
                    if grid_id in dist_for_t:
                        mean, std = dist_for_t[grid_id]
                        if mean > 0 or std > 0:
                            daily_orders[t, i] = max(0, np.round(np.random.normal(mean, std)))

        daily_drivers = np.zeros((T, num_valid_grids))
        for t in range(T):
            if t < len(idle_driver_dist_time):
                total_drivers_mean, total_drivers_std = idle_driver_dist_time[t]
                total_drivers = max(0, int(np.round(np.random.normal(total_drivers_mean, total_drivers_std))))

                if total_drivers > 0:
                    driver_dist_prob = idle_driver_location_mat[t].astype(np.float64)  # 使用更高精度
                    if driver_dist_prob.sum() > 1e-9:
                        # ***** 关键修复 *****
                        # 强制重新归一化以修正浮点数精度误差
                        driver_dist_prob /= driver_dist_prob.sum()

                        distributed_drivers = np.random.multinomial(total_drivers, driver_dist_prob)
                        daily_drivers[t, :] = distributed_drivers

        total_orders_generated_day = daily_orders.sum()
        total_orders_served_day = 0

        # --- 3. 建立并求解每时刻的最大流模型 ---
        for t in range(T):
            orders_t = daily_orders[t, :]
            drivers_t = daily_drivers[t, :]

            if orders_t.sum() == 0 or drivers_t.sum() == 0:
                continue

            num_driver_nodes = num_valid_grids
            num_order_nodes = num_valid_grids

            var_map = {}
            num_vars = 0
            for i in range(num_driver_nodes):
                if drivers_t[i] > 0:
                    for j in adjacency_list[i]:
                        if orders_t[j] > 0:
                            var_map[(i, j)] = num_vars
                            num_vars += 1

            if num_vars == 0:
                continue

            c = -np.ones(num_vars)
            A_ub = np.zeros((num_driver_nodes + num_order_nodes, num_vars))
            b_ub = np.zeros(num_driver_nodes + num_order_nodes)

            for i in range(num_driver_nodes):
                for j in adjacency_list[i]:
                    if (i, j) in var_map:
                        A_ub[i, var_map[(i, j)]] = 1
                b_ub[i] = drivers_t[i]

            for j in range(num_order_nodes):
                for i in range(num_driver_nodes):
                    if j in adjacency_list[i] and (i, j) in var_map:
                        A_ub[num_driver_nodes + j, var_map[(i, j)]] = 1
                b_ub[num_driver_nodes + j] = orders_t[j]

            bounds = (0, None)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

            if res.success:
                total_orders_served_day += -res.fun

        if total_orders_generated_day > 0:
            orr_day = total_orders_served_day / total_orders_generated_day
            total_orr_list.append(orr_day)
            print(
                f"Day {day + 1}/{NUM_SIMULATION_DAYS} -> Max ORR: {orr_day:.4f} ({int(total_orders_served_day)} / {int(total_orders_generated_day)})")
        else:
            print(f"Day {day + 1}/{NUM_SIMULATION_DAYS} -> No orders generated.")

    # --- 4. 汇总最终结果 ---
    if total_orr_list:
        max_orr_avg = np.mean(total_orr_list)
        max_orr_std = np.std(total_orr_list)
        print("\n--- Final Result ---")
        print(f"Theoretical Maximum Order Response Rate (averaged over {NUM_SIMULATION_DAYS} days):")
        print(f"Mean: {max_orr_avg:.4f}")
        print(f"Standard Deviation: {max_orr_std:.4f}")
    else:
        print("\n--- Final Result ---")
        print("Could not calculate ORR as no orders/drivers were generated in the simulation.")


if __name__ == '__main__':
    calculate_max_orr()