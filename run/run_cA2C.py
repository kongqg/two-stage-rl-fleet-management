import pickle, sys
import time

sys.path.append("../")

from simulator.utilities import *
from simulator.envs import *
from algorithm.cA2C import *
################## Load data ###################################
dir_prefix = "/mnt/research/linkaixi/AllData/dispatch/"
current_time = time.strftime("%Y%m%d_%H-%M")

data_dir = "../real_datasets/"
mapped_matrix_int = pickle.load(open(data_dir + "mapped_matrix_int.pkl", "rb"))
order_num_dist = pickle.load(open(data_dir + "order_num_dist.pkl", "rb"))
idle_driver_dist_time = pickle.load(open(data_dir + "idle_driver_dist_time.pkl", "rb"))
idle_driver_location_mat = pickle.load(open(data_dir + "idle_driver_location_mat.pkl", "rb"))
onoff_driver_location_mat = pickle.load(open(data_dir + "onoff_driver_location_mat.pkl", "rb"))
# [origin, destination, start_time, duration, price]
order_real = pickle.load(open(data_dir + "order_real.pkl", "rb"))
M, N = mapped_matrix_int.shape
order_time = pickle.load(open(data_dir + "order_time_dist.pkl", "rb"))
order_price = pickle.load(open(data_dir + "order_price_dist.pkl", "rb"))
print("finish load files")

# ---------- 日志目录 & 文件 ----------
base_dir = f"dispatch_simulator/experiments/A2C_{time.strftime('%Y%m%d_%H-%M')}"
os.makedirs(base_dir, exist_ok=True)
total_log_path = os.path.join(base_dir, "all_episodes_log2.txt")
with open(total_log_path, "w"):  # 清空总日志
    pass

################## Initialize env ###################################
n_side = 6
GAMMA = 0.9
l_max = 9

env = CityReal(mapped_matrix_int, order_num_dist,
               idle_driver_dist_time, idle_driver_location_mat,
               order_time, order_price,
               l_max, M, N, n_side, 1/11.0, order_real, onoff_driver_location_mat)



temp = np.array(env.target_grids) + env.M * env.N
target_id_states = env.target_grids + temp.tolist()


curr_s = np.array(env.reset_clean()).flatten()  # [0] driver dist; [1] order dist
curr_s = utility_conver_states(curr_s, target_id_states)
print ("******************* Finish generating one day order **********************")


print ("******************* Starting training Deep actor critic **********************")



MAX_ITER = 50
is_plot_figure = False
city_time_start = 0
EP_LEN = 144
temp_value = 10
city_time_end = city_time_start + EP_LEN
epsilon = 0.5
gamma = 0.9
learning_rate = 1e-3

prev_epsiode_reward = 0

all_rewards = []
order_response_rate_episode = []
value_table_sum = []
episode_rewards = []
episode_conflicts_drivers = []
record_all_order_response_rate = []

T = 144
action_dim = 7
state_dim = env.n_valid_grids * 3 + T



# tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()
q_estimator = Estimator(sess, action_dim,
                        state_dim,
                        env,
                        scope="q_estimator",
                        summaries_dir=base_dir)


sess.run(tf.compat.v1.global_variables_initializer())

replay = ReplayMemory(memory_size=100000, batch_size=int(3e+3))
policy_replay = policyReplayMemory(memory_size=100000, batch_size=int(3e+3))
stateprocessor = stateProcessor(target_id_states, env.target_grids, env.n_valid_grids)


restore = True
saver = tf.compat.v1.train.Saver()

save_random_seed = []
episode_dispatched_drivers = []
global_step1 = 0
global_step2 = 0
RATIO = 0.40
for n_iter in range(25):
    # -------- 子目录 ----------
    ep_dir = os.path.join(base_dir, f"EP_{n_iter:03d}")
    os.makedirs(ep_dir, exist_ok=True)
    log_path = os.path.join(ep_dir, "training_log.txt")
    with open(log_path, "w"):               # 清空本集日志
        pass

    # RANDOM_SEED = n_iter + 50 - 10
    # RANDOM_SEED = n_iter
    RANDOM_SEED = n_iter + 1040
    # RANDOM_SEED = n_iter + seed_list[1]
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.compat.v1.set_random_seed(RANDOM_SEED)
    env.reset_randomseed(RANDOM_SEED)
    save_random_seed.append(RANDOM_SEED)
    batch_s, batch_a, batch_r = [], [], []
    batch_reward_gmv = []
    epsiode_reward = 0
    num_dispatched_drivers = 0

    # reset env
    is_regenerate_order = 1
    env.metrics.reset_step()
    env.metrics.unserved_demand_total = 0
    env.metrics.same_grid_contention_total = 0
    env.reset_episode_metrics()
    curr_state = env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
    info = env.step_pre_order_assigin(curr_state)
    context = stateprocessor.compute_context(info)
    curr_s = stateprocessor.utility_conver_states(curr_state)
    normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
    # 形状 (G, 3G+T)
    s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

    # record rewards to update the value table
    episodes_immediate_rewards = []
    num_conflicts_drivers = []
    curr_num_actions = []
    order_response_rates = []
    for ii in np.arange(EP_LEN):
        # record_curr_state.append(curr_state)
        # INPUT: state,  OUTPUT: action
        action_tuple, valid_action_prob_mat, policy_state, action_choosen_mat, \
        curr_state_value, curr_neighbor_mask, next_state_ids = q_estimator.action(s_grid, context, epsilon)
        # a0
        # ONE STEP: r0
        next_state, r, info = env.step(action_tuple, 2)

        # r0
        immediate_reward = stateprocessor.reward_wrapper(info, curr_s)

        # Save transition to replay memory
        if ii != 0:
            # r1, c0
            r_grid = stateprocessor.to_grid_rewards(immediate_reward)
            # s0, a0, r1  for value newtwork
            targets_batch = q_estimator.compute_targets(action_mat_prev, s_grid, r_grid, gamma)

            # advantage for policy network.
            advantage = q_estimator.compute_advantage(curr_state_value_prev, next_state_ids_prev,
                                                      s_grid, r_grid, gamma)

            replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid)
            policy_replay.add(policy_state_prev, action_choosen_mat_prev, advantage, curr_neighbor_mask_prev)

        # for updating value network
        state_mat_prev = s_grid
        action_mat_prev = valid_action_prob_mat

        # for updating policy net
        action_choosen_mat_prev = action_choosen_mat
        curr_neighbor_mask_prev = curr_neighbor_mask
        policy_state_prev = policy_state
        # for computing advantage
        curr_state_value_prev = curr_state_value
        next_state_ids_prev = next_state_ids

        # s1
        curr_state = next_state
        curr_s = stateprocessor.utility_conver_states(next_state)
        normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
        s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

        # c1
        context = stateprocessor.compute_context(info[1])

        # training method 1.
        # #    # Sample a minibatch from the replay memory and update q network
        # if replay.curr_lens != 0:
        #     # update policy network
        #     for _ in np.arange(30):
        #         batch_s, batch_a, batch_r, batch_mask = policy_replay.sample()
        #         q_estimator.update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, learning_rate,
        #                                   global_step2)
        #         global_step2 += 1

        # Perform gradient descent update
        # book keeping
        global_step1 += 1
        global_step2 += 1
        all_rewards.append(r)
        batch_reward_gmv.append(r)
        order_response_rates.append(env.order_response_rate)
        curr_num_action = np.sum([aa[2] for aa in action_tuple]) if len(action_tuple) != 0 else 0
        curr_num_actions.append(curr_num_action)
        num_conflicts_drivers.append(collision_action(action_tuple))


    sg_t, ud_t = env.metrics.get_total()
    ep_reward = env.episode_reward
    resp_rate = env.episode_finished_orders / env.episode_total_orders
    log_str = (f"[EP {n_iter:03d}]  reward={ep_reward:,.0f} "
               f"response_rate={resp_rate:.6f} total_orders={env.episode_total_orders}  "
               f"remain_orders={env.episode_total_orders - env.episode_finished_orders} "
               f"same-grid={sg_t}  unserved={ud_t}  eps={epsilon:.3f}")
    # 写入本集日志 & 总日志
    with open(log_path, "a") as f: f.write(log_str + "\n")
    with open(total_log_path, "a") as f: f.write(log_str + "\n")
    print(log_str)    # update value network
    for _ in np.arange(4000):
        batch_s, _, batch_r, _ = replay.sample()
        iloss = q_estimator.update_value(batch_s, batch_r, 1e-3, global_step1)
        global_step1 += 1

    # training method 2
    # update policy network
    for _ in np.arange(4000):
        batch_s, batch_a, batch_r, batch_mask = policy_replay.sample()
        q_estimator.update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, learning_rate,
                                  global_step2)
        global_step2 += 1

    # -------- 保存模型参数 ----------
    model_path = os.path.join(ep_dir, "model.ckpt")
    saver.save(sess, model_path)
    print(f"  ✔ Saved model to {model_path}\n")
