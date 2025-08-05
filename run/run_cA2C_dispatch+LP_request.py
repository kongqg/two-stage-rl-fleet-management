from __future__ import annotations
import os, sys, time, random, pickle
import numpy as np
import tensorflow as tf
import torch

from run.run_Full_LP_matching import MaxFlowSolver

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Init] Torch device: {device}")
# ---------- log----------
out_dir = f"dispatch_simulator/experiments/cA2C_step_stage2_with_max_flow{time.strftime('%Y%m%d_%H-%M')}"
os.makedirs(out_dir, exist_ok=True)
log_path = os.path.join(out_dir, "training_log.txt")
with open(log_path, "w") as f:               # 清空
    f.write("")

# ============ 2. Data loading ============
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

# ============ 3. Building the environment ============
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

# ---- Estimator ----
q_estimator = Estimator(sess=sess,
                   action_dim=ACTION_DIM,
                   state_dim=STATE_DIM_1,
                   env=env,
                   scope="q_estimator")
sess.run(tf.compat.v1.global_variables_initializer())


# ---- State Processor & Replay Buffer ----
# target_id_states = driver_flat_ids + order_flat_ids
id_offset = np.array(env.target_grids) + env.M * env.N
TARGET_ID_STATES = env.target_grids + id_offset.tolist()
stateprocessor = stateProcessor(TARGET_ID_STATES, env.target_grids, env.n_valid_grids)

replay = ReplayMemory(memory_size=100000, batch_size=int(3e+3))
policy_replay = policyReplayMemory(memory_size=100000, batch_size=int(3e+3))
restore = True
saver = tf.compat.v1.train.Saver()

# ============ 6. Training Hyperparameters ============
EPISODES      = 20
EP_LEN        = 144      # 一天 144 个 10‑min slot
GAMMA_1       = 0.90
GAMMA_2       = 0.95
LEARNING_RATE = 1e-3      # for Agent‑1
UPDATES_1     = 4000
UPDATES_2     = 4000
EPS_START, EPS_END = 0.5, 0.1
EPS_DECAY_EP       = 15    # 第 15 天衰减到 EPS_END
global_step1 = 0
global_step2 = 0

def eps_schedule(ep: int) -> float:
    if ep >= EPS_DECAY_EP:
        return EPS_END
    frac = (EPS_DECAY_EP - ep) / EPS_DECAY_EP
    return EPS_END + (EPS_START - EPS_END) * frac

# ============ 8. main loop ============
for ep in range(25):

    ep_dir = os.path.join(out_dir, f"EP_{ep:03d}")
    os.makedirs(ep_dir, exist_ok=True)
    # seed = ep + 50 - 10
    seed = ep
    # seed = ep + 1040
    random.seed(seed); np.random.seed(seed); tf.compat.v1.set_random_seed(seed); env.reset_randomseed(seed)

    # --- Reset the environment, and generate all-day orders & the initial driver distribution at once. ---
    env.reset_episode_metrics(); env.metrics.reset_step()
    env.metrics.unserved_demand_total = 0;env.metrics.same_grid_contention_total = 0
    max_match = 0
    curr_state = env.reset_clean(generate_order=1, ratio=0.40, city_time=0)  # ndarray (2,M,N)
    # ---- Agent‑1 Initial global state  ----
    info = env.step_pre_order_assigin(curr_state)
    context = stateprocessor.compute_context(info)
    curr_s = stateprocessor.utility_conver_states(curr_state)
    normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
    # 形状 (G, 3G+T)
    s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

    eps_cur = eps_schedule(ep)
    print(f"\n========== Episode {ep:02d} / ε = {eps_cur:.3f} ==========")

    # max_flow solver
    solver = MaxFlowSolver(mapped_matrix_int, env.nodes)
    for t in range(EP_LEN):

        (action_tuple,
         valid_action_prob_mat,
         policy_state,
         action_choosen_mat,
         curr_state_value,
         curr_neighbor_mask,
         next_state_ids) = q_estimator.action(s_grid, context, eps_cur)

        gmv1, pending_nodes, gmv_vec1,match_result = env.step_stage1_with_max_flow(action_tuple, solver,return_node_gmv=True)
        gmv2, gmv_vec2 = env.step_stage2_with_max_flow(pending_nodes, solver,True)
        max_match += match_result
        neighbor_reward = np.zeros((len(env.nodes)))
        gmv_total = gmv1 + gmv2
        # ───── Advance the time to t+1. ─────
        env.step_increase_city_time()
        env.step_finish_interval(True)
        next_state = env.get_observation()
        context1 = env.step_pre_order_assigin(next_state)
        # Save transition to replay memory
        node_gmv = gmv_vec1 + neighbor_reward
        info_reward = ([node_gmv, neighbor_reward], context1)
        # r_grid     = sp1.to_grid_rewards(reward_vec)
        immediate_reward = stateprocessor.reward_wrapper(info_reward, curr_s)

        if t != 0:
            r_grid = stateprocessor.to_grid_rewards(immediate_reward)
            # TD‑Target
            targets_batch = q_estimator.compute_targets(action_mat_prev, s_grid, r_grid, GAMMA_1)
            advantage = q_estimator.compute_advantage(curr_state_value_prev, next_state_ids_prev, s_grid, r_grid, GAMMA_1)
            replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid)
            policy_replay.add(policy_state_prev, action_choosen_mat_prev, advantage, curr_neighbor_mask_prev)

        # Update Agent-1's previous moment cache.
        state_mat_prev = s_grid
        action_mat_prev = valid_action_prob_mat

        # for updating policy net
        action_choosen_mat_prev = action_choosen_mat
        curr_neighbor_mask_prev = curr_neighbor_mask
        policy_state_prev = policy_state
        # for computing advantage
        curr_state_value_prev = curr_state_value
        next_state_ids_prev = next_state_ids

        # ---- 切换到 t+1 全局状态 (Agent‑1) ----
        curr_state = next_state
        curr_s = stateprocessor.utility_conver_states(next_state)
        normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
        s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0
        context = stateprocessor.compute_context(context1)
    # ==== log ====
    sg_total, ud_total = env.metrics.get_total()
    log_str = (f"[EP {ep:03d}] "
               f"reward={env.episode_reward} "
               f"resp_rate={(env.episode_finished_orders / env.episode_total_orders)} "
               f"LP_resp_rate={(max_match / env.episode_total_orders)} "
               f"total_orders={env.episode_total_orders}  "
               f"remain_orders={env.episode_total_orders - env.episode_finished_orders} "
               f"same-grid={sg_total}  unserved={ud_total}  ")
    print(log_str)
    with open(log_path, "a") as f:
        f.write(log_str + "\n")

    # Agent‑1 (value)
    for _ in np.arange(4000):
        batch_s, _, batch_r, _ = replay.sample()
        iloss = q_estimator.update_value(batch_s, batch_r, 1e-3, global_step1)
        global_step1 += 1

    # training method 2
    # update policy network
    for _ in np.arange(4000):
        batch_s, batch_a, batch_r, batch_mask = policy_replay.sample()
        q_estimator.update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, LEARNING_RATE,
                                  global_step2)
        global_step2 += 1

    saver.save(sess, os.path.join(ep_dir, "agent1.ckpt"))

print("Training finished ✅")
