# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, time, random, pickle
import numpy as np
import tensorflow as tf
import torch

from run.run_Full_LP_matching import MaxFlowSolver
from simulator.utilities import build_valid_mask_for_source
from algorithm.neighbor_weight_a2c_agent import (
    NeighborAgentW, NeighborStateProcessor,
    Stage2ActorReplay, Stage2ValueReplay
)

# ───────────────────────── 项目相对路径 ─────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)                    # 工程根（算法包）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Init] Torch device: {device}")
# ---------- 日志目录 & 文件 ----------
out_dir = f"dispatch_simulator/experiments/dual_agent_with_max_flow{time.strftime('%Y%m%d_%H-%M')}"
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
SEED = 50 - 10
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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

vars_agent1 = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="q_estimator")
saver1 = tf.compat.v1.train.Saver(var_list=vars_agent1, max_to_keep=None)

# target_id_states = driver_flat_ids + order_flat_ids
id_offset = np.array(env.target_grids) + env.M * env.N
TARGET_ID_STATES = env.target_grids + id_offset.tolist()
sp1 = stateProcessor(TARGET_ID_STATES, env.target_grids, env.n_valid_grids)

replay1        = ReplayMemory(memory_size=100000, batch_size=3000)
policy_replay1 = policyReplayMemory(memory_size=100000, batch_size=3000)

K_NEI, HIST = 6, 3
STATE_DIM_2 = env.n_valid_grids + 6 + 3 + env.n_valid_grids

agent2  = NeighborAgentW(state_dim=STATE_DIM_2, action_dim=6, lr=1e-5, rho_max=2.0, device=device)
sp2     = NeighborStateProcessor(env, n_neighbors=K_NEI, hist_len=HIST)
actor_replay2 = Stage2ActorReplay(capacity=200_000)
value_replay2 = Stage2ValueReplay(capacity=200_000)

EPISODES      = 20
EP_LEN        = 144      # 一天 144 个 10‑min slot
GAMMA_1       = 0.90
GAMMA_2       = 0.95
LEARNING_RATE = 1e-3
UPDATES_1     = 4000
UPDATES_2     = 100
EPS_START, EPS_END = 0.5, 0.1
EPS_DECAY_EP       = 15


def eps_schedule(ep: int) -> float:
    if ep >= EPS_DECAY_EP:
        return EPS_END
    frac = (EPS_DECAY_EP - ep) / EPS_DECAY_EP
    return EPS_END + (EPS_START - EPS_END) * frac


# ============ 8. 主训练循环 ============
for ep in range(25):

    ep_dir = os.path.join(out_dir, f"EP_{ep:03d}")
    os.makedirs(ep_dir, exist_ok=True)




    # seed = ep + 50 - 10 # follow baseline research
    # seed = ep
    seed = ep + 1040
    random.seed(seed); np.random.seed(seed); tf.compat.v1.set_random_seed(seed); env.reset_randomseed(seed)

    env.reset_episode_metrics(); env.metrics.reset_step()
    env.metrics.unserved_demand_total = 0;env.metrics.same_grid_contention_total = 0
    curr_state = env.reset_clean(generate_order=1, ratio=0.40, city_time=0)  # ndarray (2,M,N)
    # ---- Agent‑1 ----
    info = env.step_pre_order_assigin(curr_state)
    context = sp1.compute_context(info)
    curr_s = sp1.utility_conver_states(curr_state)
    normalized_curr_s = sp1.utility_normalize_states(curr_s)
    #  (G, 3G+T)
    s_grid = sp1.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

    sp2._hist.clear()

    eps_cur = eps_schedule(ep)

    max_flow_orders = 0
    solver = MaxFlowSolver(mapped_matrix_int, env.nodes)
    print(f"\n========== Episode {ep:02d} / ε = {eps_cur:.3f} ==========")
    for t in range(EP_LEN):

        (action_tuple,
         valid_action_prob_mat,
         policy_state,
         action_choosen_mat,
         curr_state_value,
         curr_neighbor_mask,
         next_state_ids) = agent1.action(s_grid, context, eps_cur)

        # In the current environment, perform an additional LP calculation to check the performance of the upper bound.
        gmv1, pending_nodes, gmv_vec1,time_step_max_matching = env.step_stage1_with_max_flow(action_tuple, solver,return_node_gmv=True)

        max_flow_orders += time_step_max_matching
        # cache1 = agent1._cache
        if pending_nodes:
            remain_vec = np.array(
                [env.nodes[g].order_num for g in env.target_grids],
                dtype=np.float32
            )
            w_rows, traces = [], {}
            for s_id in pending_nodes:
                s_id_node = env.nodes[s_id]
                s2_s = sp2.get_state_for_source(
                    s_id_node,
                    remain_vec,
                    env.neighbor_idle_snapshot,
                    env.metrics.unserved_demand_step
                )  # 504 + 6 + 3 + 504 = 1020
                # 6, 6, 6
                mask_s, neigh_ids, idle_vec = build_valid_mask_for_source(env, s_id_node, require_has_idle=True)
                prob_s, order_s = agent2.action(s2_s, mask_s, eps=eps_cur, select='argmax')
                w_rows.append(prob_s.astype(np.float32))

                traces[s_id] = {
                    "state": s2_s.astype(np.float32),
                    "mask0": mask_s.astype(np.int64),
                    "neigh_ids": neigh_ids,
                    "order": order_s
                }
            w_mat = np.vstack(w_rows).astype(np.float32)
            gmv2, gmv_vec2, exec_traces, p_next = env.step_stage2_weight_plus(
                pending_nodes,
                w_mat,
                return_node_gmv=True,
                return_traces=True,
                return_next_state=True
            )
            for s_id in pending_nodes:
                r_s = float(gmv_vec2[s_id]) / 5000.0  # 缩放奖励，避免TD值因reward 太大
                s_id_node = env.nodes[s_id]
                idle_after = {nid: env.nodes[nid].idle_driver_num for nid in env.target_grids}
                s2_next_s = sp2.get_state_for_source(
                    s_id_node,
                    p_next,  # Stage-2 执行后的全局剩余 pending
                    idle_after,
                    env.metrics.unserved_demand_step
                )
                attempts = exec_traces[s_id]["attempts"]  # [j1, j2, ...] 0..5
                masks_seq = exec_traces[s_id]["masks"]  # [mask6_at_t, ...]
                actor_replay2.add(traces[s_id]["state"], attempts, masks_seq, r_s, s2_next_s)
                value_replay2.add(traces[s_id]["state"], r_s, s2_next_s)
        else:
            gmv2 = 0.0
        neighbor_reward = np.zeros((len(env.nodes)))
        gmv_total = gmv1 + gmv2
        env.step_increase_city_time();
        env.step_finish_interval(True)
        next_state = env.get_observation()
        context1 = env.step_pre_order_assigin(next_state)
        node_gmv = gmv_vec1 + neighbor_reward
        info_reward = ([node_gmv, neighbor_reward], context1)
        immediate_reward = sp1.reward_wrapper(info_reward, curr_s)

        if t != 0:
            r_grid = sp1.to_grid_rewards(immediate_reward)
            # TD‑Target
            targets_batch = agent1.compute_targets(action_mat_prev, s_grid, r_grid, GAMMA_1)
            advantage = agent1.compute_advantage(curr_state_value_prev, next_state_ids_prev, s_grid, r_grid, GAMMA_1)
            replay1.add(state_mat_prev, action_mat_prev, targets_batch, s_grid)
            policy_replay1.add(policy_state_prev, action_choosen_mat_prev, advantage, curr_neighbor_mask_prev)

        state_mat_prev = s_grid
        action_mat_prev = valid_action_prob_mat

        # for updating policy net
        action_choosen_mat_prev = action_choosen_mat
        curr_neighbor_mask_prev = curr_neighbor_mask
        policy_state_prev = policy_state
        # for computing advantage
        curr_state_value_prev = curr_state_value
        next_state_ids_prev = next_state_ids

        curr_state = next_state
        curr_s = sp1.utility_conver_states(next_state)
        normalized_curr_s = sp1.utility_normalize_states(curr_s)
        s_grid = sp1.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0
        context = sp1.compute_context(context1)
    # ==== log ====
    sg_total, ud_total = env.metrics.get_total()
    log_str = (f"[EP {ep:03d}] "
               f"reward={env.episode_reward} "
               f"resp_rate={(env.episode_finished_orders / env.episode_total_orders)} "
               f"full_LP_resp_rate={(max_flow_orders / env.episode_total_orders)} "
               f"total_orders={env.episode_total_orders}  "
               f"remain_orders={env.episode_total_orders - env.episode_finished_orders} "
               f"same-grid={sg_total}  unserved={ud_total}  ")
    print(log_str)
    with open(log_path, "a") as f:
        f.write(log_str + "\n")

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
    global_step = 0
    print("Agent2 value start training")
    val_hist = []
    if len(value_replay2) > 0:
        for u in range(UPDATES_2):
            vs, vr, vns = value_replay2.sample(batch_size=256)
            loss = agent2.update_value(vs, vr, vns, gamma=GAMMA_2)

    print("Agent2 policy start training")
    if len(actor_replay2) > 0:
        for u in range(UPDATES_2):
            s, attempts, masks, r, ns = actor_replay2.sample(batch_size=256)
            loss = agent2.update_policy(s, attempts, masks, r, ns, gamma=GAMMA_2, beta=0.01)

    saver1.save(sess, os.path.join(ep_dir, "agent1.ckpt"))
    torch.save(agent2.actor.state_dict(),  os.path.join(out_dir, f"actor_ep{ep}.pth"))
    torch.save(agent2.critic.state_dict(), os.path.join(out_dir, f"critic_ep{ep}.pth"))


print("Training finished ✅")
