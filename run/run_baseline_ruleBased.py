# -*- coding: utf-8 -*-
"""
Rule-based Baseline:
- 在每个决策点，对于有空闲司机的网格，会根据其邻居网格在当前时段的“历史平均订单量”
  构建一个概率分布。
- 历史订单量越高的邻居，被选为目的地的概率也越高。
- 相比 Diffusion 的纯随机，这是一个有目的、趋利性的启发式策略。
"""
from __future__ import annotations
import os
import sys
import time
import random
import pickle
import numpy as np



# ---------- 日志目录 & 文件 ----------
out_dir = f"dispatch_simulator/experiments/baseline_rule_based_{time.strftime('%Y%m%d_%H%M')}"
os.makedirs(out_dir, exist_ok=True)
log_path = os.path.join(out_dir, "training_log.txt")
with open(log_path, "w") as f:
    f.write("")

# ============ 数据加载 ============
data_dir = "../real_datasets/"

load = lambda name: pickle.load(open(os.path.join(data_dir, name), "rb"))
mapped_matrix_int = load("mapped_matrix_int.pkl")
order_num_dist = load("order_num_dist.pkl")  # Rule-based 策略需要此数据
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
EP_LEN = 144  # 一天 144 个 10-min slot

# ============ 主循环 ============
print("\n========== Running Rule-based Baseline ==========")
for ep in range(15,25):
    # --- 环境重置 ---
    # seed =  ep + 50 - 10  # follow baseline research
    # seed = ep
    seed = ep + 1040
    random.seed(seed)
    np.random.seed(seed)
    env.reset_randomseed(seed)
    env.metrics.reset_step()
    env.metrics.unserved_demand_total = 0
    env.metrics.same_grid_contention_total = 0
    env.reset_episode_metrics()

    curr_state = env.reset_clean(generate_order=1, ratio=0.4, city_time=0)

    for t in range(EP_LEN):
        source_nodes = []
        destination_nodes = []
        nums_of_drivers = []
        dispatch_commands = []

        current_time_slot = env.city_time
        historical_orders_for_slot = order_num_dist[current_time_slot]

        STAY_WEIGHT = 0.2
        EXPLORE_WEIGHT = 0.01

        for node_id in env.target_grids:
            idle_drivers_count = env.nodes[node_id].idle_driver_num
            if idle_drivers_count == 0:
                continue

            neighbor_node_objects = env.nodes[node_id].neighbors
            neighbor_ids = [n for n in neighbor_node_objects if n is not None]

            if not neighbor_ids:  # 如果过滤后没有有效邻居，则跳过
                continue
            neighbor_ids = [n.get_node_index() for n in neighbor_ids]
            action_options = [node_id] + neighbor_ids

            weights = [STAY_WEIGHT]
            for neighbor_id in neighbor_ids:
                mean_orders = historical_orders_for_slot.get(neighbor_id, [0, 0])[0]
                weights.append(mean_orders + EXPLORE_WEIGHT)

            probabilities = np.array(weights, dtype=np.float32)
            probabilities /= np.sum(probabilities)

            # 为该网格的所有司机做决策，并汇总结果
            # 使用一个字典来统计从当前node_id出发，到各个目的地的司机数量
            dispatch_counts = {}  # key: destination_id, value: count
            for _ in range(idle_drivers_count):
                chosen_destination = np.random.choice(action_options, p=probabilities)
                if chosen_destination != node_id:
                    dispatch_counts[chosen_destination] = dispatch_counts.get(chosen_destination, 0) + 1

            # 将汇总后的决策加入到最终的指令列表中
            for dest, num in dispatch_counts.items():
                dispatch_commands.append([node_id, dest, num])

        action_tuple = dispatch_commands
        save_remove_id = env.step_dispatch_invalid(action_tuple)
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

print("\nRule-based baseline run finished ✅")
print(f"Log file saved to: {log_path}")