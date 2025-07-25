# -*- coding: utf-8 -*-
"""
two stage‑Agent joint training script (Agent‑1 = A2C baseline / TensorFlow, Agent‑2 = Weighted neighbor RL / PyTorch).
--------------------------------------------------------------------------
目标
• 在**保持原有派单流程完全一致**的前提下，同步在线训练两套策略：
  ─ Stage‑1：Agent‑1（A2C，dispatch driver → greedy first‑round matching）
  ─ Stage‑2：Agent‑2（权重矩阵 w，加权邻居补单）
• 训练逻辑：
  1. 先由 CityReal.step_stage1() 调用 Agent‑1 完成调度 + 第一轮派单
  2. 若仍有待补单网格，则由 Agent‑2 输出 7‑维权重行向量，调用 step_stage2_weight_plus()
  3. 推进时间、生成下一时段订单，写入经验池，分别更新两套网络
"""
from __future__ import annotations
import os, sys, time, random, pickle
import numpy as np
import tensorflow as tf
import torch

# ───────────────────────── 项目相对路径 ─────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)                    # 工程根（算法包）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Init] Torch device: {device}")
# ---------- 日志目录 & 文件 ----------
out_dir = f"dispatch_simulator/experiments/dual_agent_{time.strftime('%Y%m%d_%H-%M')}"
os.makedirs(out_dir, exist_ok=True)
log_path = os.path.join(out_dir, "training_log.txt")
with open(log_path, "w") as f:               # 清空
    f.write("")

# ============ 2. 数据加载 ============
data_dir = "../real_datasets/"
load = lambda name: pickle.load(open(os.path.join(data_dir, name), "rb"))
mapped_matrix_int      = load("mapped_matrix_int.pkl")
order_num_dist         = load("order_num_dist.pkl")
idle_driver_dist_time  = load("idle_driver_dist_time.pkl")
idle_driver_location   = load("idle_driver_location_mat.pkl")
onoff_driver_location  = load("onoff_driver_location_mat.pkl")
order_real             = load("order_real.pkl")
order_time             = load("order_time_dist.pkl")
order_price            = load("order_price_dist.pkl")

# ============ 3. 构造环境 ============
from simulator.envs import CityReal
M, N = mapped_matrix_int.shape
env = CityReal(mapped_matrix_int, order_num_dist,
               idle_driver_dist_time, idle_driver_location,
               order_time, order_price,
               l_max=9, M=M, N=N, n_side=6,
               probability=1/11.0,  # 真实订单 1/11 抽样
               real_orders=order_real,
               onoff_driver_location_mat=onoff_driver_location)
print("CityReal ready – valid grids:", env.n_valid_grids)

# ============ 4. Agent‑1  ============
from algorithm.cA2C import Estimator, stateProcessor, ReplayMemory, policyReplayMemory
T = 144
STATE_DIM_1 = env.n_valid_grids * 3 + T
ACTION_DIM  = 7
sess = tf.compat.v1.Session()

# ---- 实例化 Estimator ----
agent1 = Estimator(sess=sess,
                   action_dim=ACTION_DIM,
                   state_dim=STATE_DIM_1,
                   env=env,
                   scope="q_estimator")
sess.run(tf.compat.v1.global_variables_initializer())

# TensorFlow Saver（全部 q_estimator 变量）
vars_agent1 = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="q_estimator")
saver1 = tf.compat.v1.train.Saver(var_list=vars_agent1, max_to_keep=None)

# ---- 状态处理器 & 经验池 ----
# target_id_states = driver_flat_ids + order_flat_ids
id_offset = np.array(env.target_grids) + env.M * env.N
TARGET_ID_STATES = env.target_grids + id_offset.tolist()
sp1 = stateProcessor(TARGET_ID_STATES, env.target_grids, env.n_valid_grids)

replay1        = ReplayMemory(memory_size=100000, batch_size=3000)
policy_replay1 = policyReplayMemory(memory_size=100000, batch_size=3000)

# ============ 5. Agent‑2  (PyTorch 权重策略) ============
from algorithm.neighbor_weight_a2c_agent import NeighborAgentW, NeighborStateProcessor, NeighborReplay
K_NEI, HIST = 6, 3
STATE_DIM_2 = env.n_valid_grids + env.n_valid_grids * K_NEI + 2 * HIST

agent2  = NeighborAgentW(state_dim=STATE_DIM_2, action_dim=6, lr=1e-5, rho_max=2.0, device=device)
sp2     = NeighborStateProcessor(env, n_neighbors=K_NEI, hist_len=HIST)
replay2 = NeighborReplay(cap=2_00_000, batch=2048, state_dim=STATE_DIM_2)

# ============ 6. 训练超参 ============
EPISODES      = 20
EP_LEN        = 144      # 一天 144 个 10‑min slot
GAMMA_1       = 0.90
GAMMA_2       = 0.95
LEARNING_RATE = 1e-3      # for Agent‑1
UPDATES_1     = 4000      # 每日批训练步 – Agent‑1
UPDATES_2     = 4000      # 每日批训练步 – Agent‑2
EPS_START, EPS_END = 0.5, 0.1
EPS_DECAY_EP       = 15    # 第 15 天衰减到 EPS_END

# ============ 7. 辅助函数 ============
def eps_schedule(ep: int) -> float:
    if ep >= EPS_DECAY_EP:
        return EPS_END
    frac = (EPS_DECAY_EP - ep) / EPS_DECAY_EP
    return EPS_END + (EPS_START - EPS_END) * frac

# ============ 8. 主训练循环 ============
for ep in range(25):

    ep_dir = os.path.join(out_dir, f"EP_{ep:03d}")
    os.makedirs(ep_dir, exist_ok=True)
    seed = ep + 50 - 10
    # seed = ep
    # seed = ep + 1040
    random.seed(seed); np.random.seed(seed); tf.compat.v1.set_random_seed(seed); env.reset_randomseed(seed)

    # --- 环境重置，一次性生成全天订单 & 初始司机分布 ---
    env.reset_episode_metrics(); env.metrics.reset_step()
    env.metrics.unserved_demand_total = 0;env.metrics.same_grid_contention_total = 0
    curr_state = env.reset_clean(generate_order=1, ratio=0.40, city_time=0)  # ndarray (2,M,N)
    # ---- Agent‑1 初始全局状态  ----
    info0     = env.step_pre_order_assigin(curr_state)
    context_1 = sp1.compute_context(info0)
    s1_conv   = sp1.utility_conver_states(curr_state)
    s1_norm   = sp1.utility_normalize_states(s1_conv)
    s1_grid   = sp1.to_grid_states(s1_norm, env.city_time)

    # ---- buffer for Agent‑1 上一个时间步 ----
    prev_s_grid, prev_valid_prob, prev_policy_state = None, None, None
    prev_action_mat = None
    prev_curr_state_value, prev_next_state_ids = None, None

    # ---- Agent‑2 历史缓存 ----
    sp2._hist.clear()

    eps_cur = eps_schedule(ep)
    print(f"\n========== Episode {ep:02d} / ε = {eps_cur:.3f} ==========")
    for t in range(EP_LEN):

        # ───── Stage‑1：调用（Agent‑1）─────

        (action_tuple,
         valid_prob_mat,
         policy_state_1,
         action_onehot,
         curr_state_value,
         neigh_mask,
         next_state_ids) = agent1.action(s1_grid, context_1, eps_cur)

        gmv1, pending_nodes, gmv_vec1 = env.step_stage1(action_tuple, epsilon=eps_cur,return_node_gmv=True)
        #
        # cache1 = agent1._cache
        # ───── Stage‑2：若有缺口，Agent‑2 输出权重矩阵─────
        if pending_nodes:
            # 构造 Agent‑2 的全局状态
            remain_vec = np.zeros(env.n_valid_grids, np.float32)
            for nid in pending_nodes:
                remain_vec[env.target_grids.index(nid)] = env.nodes[nid].order_num
            s2_global = sp2.get_state(remain_vec, env.neighbor_idle_snapshot, env.metrics.get_step())
            # 对每个待补单网格生成一行 w
            w_rows = []
            for _ in pending_nodes:
                w_row = agent2.action(s2_global, eps=eps_cur)
                w_rows.append(w_row)
            w_mat = np.vstack(w_rows)
            gmv2, gmv_vec2  = env.step_stage2_weight_plus(pending_nodes, w_mat,return_node_gmv=True)
            # Agent‑2 Replay
            s2_next = sp2.get_state(np.zeros_like(remain_vec), env.neighbor_idle_snapshot, env.metrics.get_step())
            for w_row in w_rows:
                replay2.add(s2_global, w_row, gmv2, s2_next)
        else:
            gmv2 = 0.0
        gmv_total = gmv1 + gmv2
        # ───── 时间推进到 t+1 ─────
        env.step_increase_city_time(); env.step_finish_interval(True)
        next_state = env.get_observation()

        # ───── Agent‑1 经验池写入 & 目标/优势计算 ─────
        # 构造均匀 node_reward 近似 (reward_local / G)
        node_gmv = gmv_vec1 + gmv_vec2
        info_reward = ([node_gmv], None)
        # r_grid     = sp1.to_grid_rewards(reward_vec)
        immediate_reward = sp1.reward_wrapper(info_reward, s1_conv)

        if prev_s_grid is not None:
            r_grid = sp1.to_grid_rewards(immediate_reward)
            # TD‑Target
            targets_batch = agent1.compute_targets(prev_valid_prob, s1_grid, r_grid, GAMMA_1)
            adv_batch     = agent1.compute_advantage(prev_curr_state_value, prev_next_state_ids, s1_grid, r_grid, GAMMA_1)
            replay1.add(prev_s_grid, prev_action_mat, targets_batch, s1_grid)
            policy_replay1.add(prev_policy_state, prev_action_onehot, adv_batch, prev_neigh_mask)

        # 更新 Agent‑1上一时刻缓存
        prev_s_grid = s1_grid
        prev_valid_prob = valid_prob_mat
        prev_policy_state = policy_state_1
        prev_action_mat = valid_prob_mat  # ← 如果 Critic replay 还想存 G×7，可保留
        prev_curr_state_value, prev_next_state_ids = curr_state_value, next_state_ids
        prev_action_onehot = action_onehot
        prev_neigh_mask = neigh_mask

        # ---- 切换到 t+1 全局状态 (Agent‑1) ----
        s1_conv   = sp1.utility_conver_states(next_state)
        s1_norm   = sp1.utility_normalize_states(s1_conv)
        s1_grid   = sp1.to_grid_states(s1_norm, env.city_time)
        context_1 = sp1.compute_context(env.step_pre_order_assigin(next_state))

    # ==== 日志 ====
    sg_total, ud_total = env.metrics.get_total()
    log_str = (f"[EP {ep:03d}] "
               f"reward={env.episode_reward} "
               f"resp_rate={(env.episode_finished_orders / env.episode_total_orders)} "
               f"total_orders={env.episode_total_orders}  "
               f"remain_orders={env.episode_total_orders - env.episode_finished_orders} "
               f"same-grid={sg_total}  unserved={ud_total}  ")
    print(log_str)
    with open(log_path, "a") as f:
        f.write(log_str + "\n")

    # ==== End‑of‑Episode：批量更新两套网络 ====
    # Agent‑1 (value)
    if replay1.curr_lens:
        for _ in range(UPDATES_1):
            bs, ba, br, ns = replay1.sample()
            agent1.update_value(bs, br, LEARNING_RATE, _)
    # Agent‑1 (policy)
    if policy_replay1.curr_lens:
        for _ in range(UPDATES_1):
            bs, ba, adv, mask = policy_replay1.sample()
            agent1.update_policy(bs, adv.reshape([-1,1]), ba, mask, LEARNING_RATE, _)
    global_step = 0
    # # Agent‑2
    if replay2.size >= replay2.batch:
        for _ in range(UPDATES_2):
            batch = replay2.sample()
            agent2.update(*batch, gamma=GAMMA_2)

        # ============ 保存模型 ============
    saver1.save(sess, os.path.join(ep_dir, "agent1.ckpt"))
    torch.save(agent2.actor.state_dict(),  os.path.join(out_dir, f"actor_ep{ep}.pth"))
    torch.save(agent2.critic.state_dict(), os.path.join(out_dir, f"critic_ep{ep}.pth"))


print("Training finished ✅")
