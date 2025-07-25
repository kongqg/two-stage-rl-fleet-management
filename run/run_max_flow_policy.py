# -*- coding: utf-8 -*-
"""
Runs a simulation using a Myopic Optimal Policy based on a max-flow solver.
At each time step, it calculates the optimal dispatch action that maximizes
immediate order fulfillment and applies it to the environment.
"""

import pickle
import numpy as np
from scipy.optimize import linprog
import os
import sys
import time
import random

# --- 确保项目路径可被访问 ---
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, '..'))

from simulator.envs import CityReal
from simulator.utilities import ids_1dto2d, ids_2dto1d

base_dir = f"dispatch_simulator/experiments/max_flow_{time.strftime('%Y%m%d_%H-%M')}"
os.makedirs(base_dir, exist_ok=True)
total_log_path = os.path.join(base_dir, "all_episodes_log.txt")
with open(total_log_path, "w"):  # 清空总日志
    pass
def get_neighbor_ids_from_utilities(i, j, M, N, n, nodes):
    """
    完全复刻 utilities.py 中 get_neighbor_index 函数的逻辑。
    """
    neighbor_list = [None] * n
    if n == 6:
        # hexagonal
        if j % 2 == 0:
            if i - 1 >= 0:
                neighbor_list[0] = nodes[ids_2dto1d(i - 1, j, M, N)]
            if j + 1 < N:
                neighbor_list[1] = nodes[ids_2dto1d(i, j + 1, M, N)]
            if i + 1 < M and j + 1 < N:
                neighbor_list[2] = nodes[ids_2dto1d(i + 1, j + 1, M, N)]
            if i + 1 < M:
                neighbor_list[3] = nodes[ids_2dto1d(i + 1, j, M, N)]
            if i + 1 < M and j - 1 >= 0:
                neighbor_list[4] = nodes[ids_2dto1d(i + 1, j - 1, M, N)]
            if j - 1 >= 0:
                neighbor_list[5] = nodes[ids_2dto1d(i, j - 1, M, N)]
        elif j % 2 == 1:
            if i - 1 >= 0:
                neighbor_list[0] = nodes[ids_2dto1d(i - 1, j, M, N)]
            if i - 1 >= 0 and j + 1 < N:
                neighbor_list[1] = nodes[ids_2dto1d(i - 1, j + 1, M, N)]
            if j + 1 < N:
                neighbor_list[2] = nodes[ids_2dto1d(i, j + 1, M, N)]
            if i + 1 < M:
                neighbor_list[3] = nodes[ids_2dto1d(i + 1, j, M, N)]
            if j - 1 >= 0:
                neighbor_list[4] = nodes[ids_2dto1d(i, j - 1, M, N)]
            if i - 1 >= 0 and j - 1 >= 0:
                neighbor_list[5] = nodes[ids_2dto1d(i - 1, j - 1, M, N)]
    elif n == 4:
        # square
        if i - 1 >= 0:
            neighbor_list[0] = nodes[ids_2dto1d(i - 1, j, M, N)]
        if j + 1 < N:
            neighbor_list[1] = nodes[ids_2dto1d(i, j + 1, M, N)]
        if i + 1 < M:
            neighbor_list[2] = nodes[ids_2dto1d(i + 1, j, M, N)]
        if j - 1 >= 0:
            neighbor_list[3] = nodes[ids_2dto1d(i, j - 1, M, N)]
    return neighbor_list

class MaxFlowSolver:
    """
    一个封装了最大流问题建模与求解的类。
    """

    def __init__(self, mapped_matrix_int, nodes):
        print("Initializing MaxFlowSolver...")
        self.M, self.N_cols = mapped_matrix_int.shape
        self.mapped_matrix_int = mapped_matrix_int
        self.valid_grid_ids = sorted(list(set(mapped_matrix_int.flatten()) - {-1}))
        self.num_valid_grids = len(self.valid_grid_ids)
        self.grid_id_to_idx_map = {grid_id: i for i, grid_id in enumerate(self.valid_grid_ids)}
        self.nodes = nodes

        self.adjacency_list = self._build_adjacency_list()
        print("MaxFlowSolver initialized successfully.")

    def _build_adjacency_list(self):
        """构建邻接表"""
        adjacency_list = [[] for _ in range(self.num_valid_grids)]
        for i, grid_id in enumerate(self.valid_grid_ids):
            row, col = ids_1dto2d(grid_id, self.M, self.N_cols)

            adjacency_list[i].append(i)

            # 这个函数返回的是一维的邻居ID列表
            neighbor_nodes = get_neighbor_ids_from_utilities(row, col, self.M, self.N_cols, 6, self.nodes)
            for node in neighbor_nodes:
                if node is not None:
                    neighbor_id = node.get_node_index()
                    if self.mapped_matrix_int[ids_1dto2d(neighbor_id, self.M, self.N_cols)] != -1:
                        neighbor_idx = self.grid_id_to_idx_map[neighbor_id]
                        if neighbor_idx not in adjacency_list[i]:
                            adjacency_list[i].append(neighbor_idx)
        return adjacency_list

    def solve(self, drivers_t, orders_t):
        """
        为当前时刻的供需分布求解最大流。
        返回: (解向量, 变量映射字典)
        """
        if orders_t.sum() == 0 or drivers_t.sum() == 0:
            return None, None

        num_driver_nodes = self.num_valid_grids
        num_order_nodes = self.num_valid_grids

        var_map = {}
        num_vars = 0
        for i in range(num_driver_nodes):
            if drivers_t[i] > 0:
                for j in self.adjacency_list[i]:
                    if orders_t[j] > 0:
                        var_map[(i, j)] = num_vars
                        num_vars += 1

        if num_vars == 0:
            return None, None

        c = -np.ones(num_vars)
        A_ub = np.zeros((num_driver_nodes + num_order_nodes, num_vars))
        b_ub = np.zeros(num_driver_nodes + num_order_nodes)

        # ----------------------
        # 处理 A， B的约束
        for i in range(num_driver_nodes):
            for j in self.adjacency_list[i]:
                if (i, j) in var_map:
                    A_ub[i, var_map[(i, j)]] = 1
            b_ub[i] = drivers_t[i]

        for j in range(num_order_nodes):
            for i in range(num_driver_nodes):
                if j in self.adjacency_list[i] and (i, j) in var_map:
                    A_ub[num_driver_nodes + j, var_map[(i, j)]] = 1
            b_ub[num_driver_nodes + j] = orders_t[j]

        # ----------------------

        bounds = (0, None)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            # res.x 是最优解  var_map： 路径
            return res.x, var_map
        else:
            return None, None


def convert_solution_to_action(solution, var_map, valid_grid_ids):
    """
    将线性规划的解向量转换为环境可接受的 action_tuple 格式。
    """
    action_tuple = []
    if solution is None or var_map is None:
        return action_tuple

    for (driver_idx, order_idx), var_idx in var_map.items():
        num_drivers_to_dispatch = int(np.round(solution[var_idx]))
        source_id = valid_grid_ids[driver_idx]
        dest_id = valid_grid_ids[order_idx]
        # 避免自己向自己派发车辆的行为
        if num_drivers_to_dispatch > 0 and source_id != dest_id:
            action_tuple.append((source_id, dest_id, num_drivers_to_dispatch))
    return action_tuple


def run_simulation_with_max_flow():
    # --- 1. 数据加载 ---
    print("Loading data...")
    data_dir = "../real_datasets/"  # 假设数据在上一级目录的 real_datasets 中
    try:
        with open(os.path.join(data_dir, 'mapped_matrix_int.pkl'), 'rb') as f:
            mapped_matrix_int = pickle.load(f)
        with open(os.path.join(data_dir, 'order_num_dist.pkl'), 'rb') as f:
            order_num_dist = pickle.load(f)
        with open(os.path.join(data_dir, 'idle_driver_dist_time.pkl'), 'rb') as f:
            idle_driver_dist_time = pickle.load(f)
        with open(os.path.join(data_dir, 'idle_driver_location_mat.pkl'), 'rb') as f:
            idle_driver_location_mat = pickle.load(f)
        with open(os.path.join(data_dir, 'onoff_driver_location_mat.pkl'), 'rb') as f:
            onoff_driver_location_mat = pickle.load(f)
        with open(os.path.join(data_dir, 'order_real.pkl'), 'rb') as f:
            order_real = pickle.load(f)
        with open(os.path.join(data_dir, 'order_time_dist.pkl'), 'rb') as f:
            order_time = pickle.load(f)
        with open(os.path.join(data_dir, 'order_price_dist.pkl'), 'rb') as f:
            order_price = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Data loading failed: {e}. Please check the path.")
        return

    # --- 2. 初始化环境和求解器 ---
    M, N = mapped_matrix_int.shape
    env = CityReal(mapped_matrix_int, order_num_dist,
                   idle_driver_dist_time, idle_driver_location_mat,
                   order_time, order_price,
                   l_max=9, M=M, N=N, n_side=6,
                   probability=1 / 11.0,
                   real_orders=order_real,
                   onoff_driver_location_mat=onoff_driver_location_mat)

    solver = MaxFlowSolver(mapped_matrix_int, env.nodes)

    # --- 3. 模拟主循环 ---
    EP_LEN = 144

    episode_rewards = []
    episode_orrs = []
    final_reward = 0


    for ep in range(15,25):
        # 设置随机种子以保证可复现性
        # seed = ep + 50 - 10
        # seed = ep
        seed = ep + 1040
        random.seed(seed)
        np.random.seed(seed)
        env.reset_randomseed(seed)

        # 重置环境
        env.reset_episode_metrics()
        curr_state = env.reset_clean(generate_order=1, ratio=0.40, city_time=0)

        for t in range(EP_LEN):
            driver_map = curr_state[0]
            order_map = curr_state[1]

            drivers_t = np.array(
                [driver_map[ids_1dto2d(grid_id, solver.M, solver.N_cols)] for grid_id in solver.valid_grid_ids])
            orders_t = np.array(
                [order_map[ids_1dto2d(grid_id, solver.M, solver.N_cols)] for grid_id in solver.valid_grid_ids])

            solution, var_map = solver.solve(drivers_t, orders_t)
            # 3. 将解转换为环境可接受的 action
            action_tuple = convert_solution_to_action(solution, var_map, solver.valid_grid_ids)
            # print(f"action_tuple:{action_tuple}")
            '''**************************** T = 1 ****************************'''
            # Loop over all dispatch action, change the driver distribution
            save_remove_id = env.step_dispatch_invalid(action_tuple)
            # When the drivers go to invalid grid, set them offline.
            env.step_add_dispatched_drivers(save_remove_id)
            reward, reward_node = env.step_assign_order_broadcast_neighbor_reward_update()
            final_reward += reward

            '''**************************** T = 2 ****************************'''
            # increase city time t + 1
            env.step_increase_city_time()
            env.step_driver_status_control()  # drivers finish order become available again.

            # drivers dispatched at t, arrived at t + 1, become available at t+1


            # generate order at t + 1
            moment = env.city_time % env.n_intervals
            moment = int(moment)
            env.step_bootstrap_order_real(env.day_orders[moment])

            env.step_driver_online_offline_nodewise()
            env.step_remove_unfinished_orders()
            # get states S_{t+1}  [driver_dist, order_dist]
            next_state = env.get_observation()
            # 更新状态
            curr_state = next_state

        # 记录回合结束时的性能指标
        ep_reward = env.episode_reward
        ep_orr = env.episode_finished_orders / env.episode_total_orders if env.episode_total_orders > 0 else 0
        episode_rewards.append(ep_reward)
        episode_orrs.append(ep_orr)

        log = (f"Episode {ep} -> GMV: {ep_reward:,.2f}, "
              f"ORR: {ep_orr:.4f} "
              f"Orders_num:{env.episode_total_orders}")
        with open(total_log_path, "a") as f:
            f.write(log + "\n")
        print(log)  # update value network

    # --- 4. 打印最终平均结果 ---
    log = ("\n--- Final Results (Averaged over all episodes) --- "
            f"Mean GMV: {np.mean(episode_rewards):,.2f} "
            f"Mean Order Response Rate: {np.mean(episode_orrs):.4f} "
            f"Standard Deviation of ORR: {np.std(episode_orrs):.4f}")
    #
    with open(total_log_path, "a") as f:
        f.write(log + "\n")
    print(log)  # update value network


if __name__ == '__main__':
    run_simulation_with_max_flow()