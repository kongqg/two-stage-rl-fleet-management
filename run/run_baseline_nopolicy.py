
import pickle, sys
import time

sys.path.append("../")

from simulator.envs import *


################## Load data ###################################
data_dir = "../real_datasets/"

order_time_dist = []
order_price_dist = []
mapped_matrix_int = pickle.load(open(data_dir + "mapped_matrix_int.pkl", "rb"))
order_num_dist = pickle.load(open(data_dir + "order_num_dist.pkl", "rb"))
idle_driver_dist_time = pickle.load(open(data_dir + "idle_driver_dist_time.pkl", "rb"))
idle_driver_location_mat = pickle.load(open(data_dir + "idle_driver_location_mat.pkl", "rb"))
onoff_driver_location_mat = pickle.load(open(data_dir + "onoff_driver_location_mat.pkl", "rb"))
order_real = pickle.load(open(data_dir + "order_real.pkl", "rb"))
M, N = mapped_matrix_int.shape
print("finish load data")


################## Initialize env ###################################
n_side = 6
GAMMA = 0.9
l_max = 9

env = CityReal(mapped_matrix_int, order_num_dist,
               idle_driver_dist_time, idle_driver_location_mat,
               order_time_dist, order_price_dist,
               l_max, M, N, n_side, 1/11.0, order_real, onoff_driver_location_mat)


temp = np.array(env.target_grids) + env.M * env.N
target_id_states = env.target_grids + temp.tolist()
# ---------- 日志目录 & 文件 ----------
base_dir = f"dispatch_simulator/experiments/no_policy_{time.strftime('%Y%m%d_%H-%M')}"
os.makedirs(base_dir, exist_ok=True)
total_log_path = os.path.join(base_dir, "all_episodes_log2.txt")
with open(total_log_path, "w"):  # 清空总日志
    pass


print("******************* Finish generating one day order **********************")



print("******************* Starting runing no policy baseline **********************")


MAX_ITER = 50  # 10 iteration the Q-learning loss will converge.
is_plot_figure = False
city_time_start = 0
EP_LEN = 144
global_step = 0
city_time_end = city_time_start + EP_LEN
epsilon = 0.5
gamma = 0.9
learning_rate = 1e-3

prev_epsiode_reward = 0
curr_num_actions = []
all_rewards = []
order_response_rate_episode = []
value_table_sum = []
episode_rewards = []
num_conflicts_drivers = []
driver_numbers_episode = []
order_numbers_episode = []

T = 144
action_dim = 7
state_dim = env.n_valid_grids * 3 + T

record_all_order_response_rate = []


def compute_context(target_grids, info):

    context = info.flatten()
    context = [context[idx] for idx in target_grids]
    return context

RATIO = 0.40

print("Start Running ")
save_random_seed = []
episode_avaliables_vehicles = []
for n_iter in np.arange(15,25):
    # -------- 子目录 ----------
    ep_dir = os.path.join(base_dir, f"EP_{n_iter:03d}")
    os.makedirs(ep_dir, exist_ok=True)
    log_path = os.path.join(ep_dir, "training_log.txt")
    with open(log_path, "w"):  # 清空本集日志
        pass
    # RANDOM_SEED = n_iter + MAX_ITER - 10  # follow baseline research
    # RANDOM_SEED = n_iter
    RANDOM_SEED = n_iter + 1040
    env.reset_randomseed(RANDOM_SEED)
    save_random_seed.append(RANDOM_SEED)
    batch_s, batch_a, batch_r = [], [], []
    batch_reward_gmv = []
    epsiode_reward = 0
    num_dispatched_drivers = 0

    driver_numbers = []
    order_numbers = []
    is_regenerate_order = 1
    env.metrics.reset_step()
    env.metrics.unserved_demand_total = 0
    env.metrics.same_grid_contention_total = 0
    env.reset_episode_metrics()
    curr_state = env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
    driver_numbers.append(np.sum(curr_state[0]))
    order_numbers.append(np.sum(curr_state[1]))
    info = env.step_pre_order_assigin(curr_state)
    context = compute_context(env.target_grids, np.array(info))

    # record rewards to update the value table
    episodes_immediate_rewards = []
    order_response_rates = []
    available_drivers = []
    for ii in np.arange(EP_LEN):
        available_drivers.append(np.sum(context))
        # ONE STEP: r0
        next_state, r, info = env.step([], 2)
        driver_numbers.append(np.sum(next_state[0]))
        order_numbers.append(np.sum(next_state[1]))

        context = compute_context(env.target_grids, np.array(info[1]))
        # Perform gradient descent update
        # book keeping
        global_step += 1

    ep_reward = env.episode_reward
    episode_rewards.append(ep_reward)
    driver_numbers_episode.append(np.sum(driver_numbers[:-1]))
    order_numbers_episode.append(np.sum(order_numbers[:-1]))
    episode_avaliables_vehicles.append(np.sum(available_drivers[:-1]))


    total_orders = env.episode_total_orders
    remain_orders = env.episode_total_orders - env.episode_finished_orders
    log_str = (f"[EP {n_iter:03d}]  reward={ep_reward:,.0f} "
               f"response_rate={1-(remain_orders/total_orders):.6f} total_orders={total_orders}  "
               f"remain_orders={remain_orders} ")
    # 写入本集日志 & 总日志
    with open(log_path, "a") as f:
        f.write(log_str + "\n")
    with open(total_log_path, "a") as f:
        f.write(log_str + "\n")
    print(log_str)  # update value network

print("averaged available vehicles per time step: {}".format(np.mean(episode_avaliables_vehicles)/144.0))