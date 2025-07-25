# -*- coding: utf-8 -*-
"""
Diffusion Baseline:
- 在每个决策点，每个有空闲司机的网格，会从其所有邻居中随机选择一个，
  然后将该网格内的所有空闲司机都派往该邻居。
- 如果一个网格没有邻居或没有空闲司机，则动作为“停留”。
"""
from __future__ import annotations
import os, sys, time, random, pickle
import numpy as np

# ───────────────────────── 项目相对路径 ─────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))

# ---------- 日志目录 & 文件 ----------
# 为了不与之前的日志混淆，创建一个新的目录
out_dir = f"dispatch_simulator/experiments/baseline_diffusion_{time.strftime('%Y%m%d_%H-%M')}"
os.makedirs(out_dir, exist_ok=True)
log_path = os.path.join(out_dir, "training_log.txt")
with open(log_path, "w") as f:               # 清空
    f.write("")

# ============ 数据加载 ============
data_dir = os.path.join("../", "real_datasets")
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"找不到数据集目录: {data_dir}")

load = lambda name: pickle.load(open(os.path.join(data_dir, name), "rb"))
mapped_matrix_int = load("mapped_matrix_int.pkl")
order_num_dist = load("order_num_dist.pkl")
idle_driver_dist_time = load("idle_driver_dist_time.pkl")
idle_driver_location = load("idle_driver_location_mat.pkl")
onoff_driver_location = load("onoff_driver_location_mat.pkl")
order_real = load("order_real.pkl")
order_time = load("order_time_dist.pkl")
order_price = load("order_price_dist.pkl")

# ============ 构造环境 ============
from simulator.envs import CityReal

M, N = mapped_matrix_int.shape
env = CityReal(mapped_matrix_int, order_num_dist,
               idle_driver_dist_time, idle_driver_location,
               order_time, order_price,
               l_max=9, M=M, N=N, n_side=6,
               probability=1 / 11.0,
               real_orders=order_real,
               onoff_driver_location_mat=onoff_driver_location)
print("CityReal ready – valid grids:", env.n_valid_grids)

# ============ 训练超参 ============
EP_LEN = 144

# ============ 主循环 ============
print("\n========== Running Diffusion Baseline ==========")
# 只跑后10个episode
for ep in range(15,25):
    # seed = ep + 50 - 10  # follow baseline research
    # seed = ep
    seed = ep + 1040
    random.seed(seed)
    np.random.seed(seed)
    env.reset_randomseed(seed)
    env.metrics.reset_step()
    env.metrics.unserved_demand_total = 0
    env.metrics.same_grid_contention_total = 0
    env.reset_episode_metrics()

    # 只需要生成订单和司机，不需要复杂的 contextual state
    curr_state = env.reset_clean(generate_order=1, ratio=0.4, city_time=0)
    # normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
    # s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0
    for t in range(EP_LEN):
        dispatch_actions = []
        for g_idx, node_id in enumerate(env.target_grids):
            total_idle_drivers = env.nodes[node_id].idle_driver_num
            if total_idle_drivers > 0:
                current_node = env.nodes[node_id]
                neighbor_node_objects = current_node.neighbors
                valid_neighbors_ids = [n.get_node_index() for n in neighbor_node_objects if n is not None]
                num_neighbors = len(valid_neighbors_ids)

                if num_neighbors > 0:
                    drivers_per_neighbor = total_idle_drivers // num_neighbors
                    remainder_drivers = total_idle_drivers % num_neighbors
                    for i in range(num_neighbors):
                        num_to_dispatch = drivers_per_neighbor
                        if i < remainder_drivers:
                            num_to_dispatch += 1
                        if num_to_dispatch > 0:
                            destination_node_id = valid_neighbors_ids[i]
                            action = (node_id, destination_node_id, num_to_dispatch)
                            dispatch_actions.append(action)
        save_remove_id = env.step_dispatch_invalid(dispatch_actions)
        env.step_add_dispatched_drivers(save_remove_id)
        env.step_assign_order_broadcast_neighbor_reward_update()

        env.step_increase_city_time()
        env.step_driver_status_control()
        moment = env.city_time % env.n_intervals
        moment = int(moment)
        env.step_bootstrap_order_real(env.day_orders[moment])
        env.step_driver_online_offline_nodewise()
        env.step_remove_unfinished_orders()
    # ==== 日志 ====
    resp_rate = env.episode_finished_orders / env.episode_total_orders if env.episode_total_orders > 0 else 0
    log_str = (f"[EP {ep:02d}/{ep}] "
               f"Total GMV: {env.episode_reward:.2f} | "
               f"Response Rate: {resp_rate:.4f} | "
               f"Total Orders: {env.episode_total_orders} | "
               f"Served: {env.episode_finished_orders}")
    print(log_str)
    with open(log_path, "a") as f:
        f.write(log_str + "\n")

print("\nDiffusion baseline run finished ✅")
print(f"Log file saved to: {log_path}")