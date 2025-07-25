import sys
import logging


sys.path.append("../")

from simulator.objects import *
from simulator.utilities import *
from simulator.conflict_metrics import ConflictMetrics

# from algorithm import *

# current_time = time.strftime("%Y%m%d_%H-%M")
# log_dir = "/nfs/private/linkaixiang_i/data/dispatch_simulator/experiments/"+current_time + "/"
# mkdir_p(log_dir)
# logging.basicConfig(filename=log_dir +'logger_env.log', level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_ch = logging.StreamHandler()
logger_ch.setLevel(logging.DEBUG)
logger_ch.setFormatter(logging.Formatter(
    '%(asctime)s[%(levelname)s][%(lineno)s:%(funcName)s]||%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(logger_ch)
RANDOM_SEED = 0  # unit test use this random seed.


class CityReal:
    '''A real city is consists of M*N grids '''

    def __init__(self, mapped_matrix_int, order_num_dist, idle_driver_dist_time, idle_driver_location_mat,
                 order_time_dist, order_price_dist,
                 l_max, M, N, n_side, probability=1.0 / 30, real_orders="", onoff_driver_location_mat="",
                 global_flag="global", time_interval=10):
        """
        :param mapped_matrix_int: 2D matrix: each position is either -100 or grid id from order in real data.
        :param order_num_dist: 144 [{node_id1: [mu, std]}, {node_id2: [mu, std]}, ..., {node_idn: [mu, std]}]
                            node_id1 is node the index in self.nodes
        :param idle_driver_dist_time: [[mu1, std1], [mu2, std2], ..., [mu144, std144]] mean and variance of idle drivers in
        the city at each time
        :param idle_driver_location_mat: 144 x num_valid_grids matrix.
        :param order_time_dist: [ 0.27380797,..., 0.00205766] The probs of order duration = 1 to 9
        :param order_price_dist: [[10.17, 3.34],   # mean and std of order's price, order durations = 10 minutes.
                                   [15.02, 6.90],  # mean and std of order's price, order durations = 20 minutes.
                                   ...,]
        :param onoff_driver_location_mat: 144 x 504 x 2: 144 total time steps, num_valid_grids = 504.
        mean and std of online driver number - offline driver number
        onoff_driver_location_mat[t] = [[-0.625       2.92350389]  <-- Corresponds to the grid in target_node_ids
                                        [ 0.09090909  1.46398452]
                                        [ 0.09090909  2.36596622]
                                        [-1.2         2.05588586]...]
        :param M:
        :param N:
        :param n_side:
        :param time_interval:
        :param l_max: The max-duration of an order
        :return:
        """
        # City.__init__(self, M, N, n_side, time_interval)
        self.M = M  # row numbers
        self.N = N  # column numbers
        self.nodes = [Node(i) for i in range(M * N)]  # a list of nodes: node id start from 0
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        self.n_offline_drivers = 0  # total number of offline drivers.
        self.construct_map_simulation(M, N, n_side)
        self.city_time = 0
        # self.idle_driver_distribution = np.zeros((M, N))
        self.n_intervals = 1440 / time_interval
        self.n_nodes = self.M * self.N
        self.n_side = n_side
        self.order_response_rate = 0

        self.RANDOM_SEED = RANDOM_SEED
        self.set_global_seeds(self.RANDOM_SEED)

        self.l_max = l_max  # Start from 1. The max number of layers an order can across.
        assert l_max <= M - 1 and l_max <= N - 1
        assert 1 <= l_max <= 15  # Ignore orders less than 10 minutes and larger than 1.5 hours

        self.target_grids = []
        self.n_valid_grids = 0  # num of valid grid
        self.nodes = [None for _ in np.arange(self.M * self.N)]
        self.construct_node_real(mapped_matrix_int)
        self.mapped_matrix_int = mapped_matrix_int

        self.construct_map_real(n_side)
        self.order_num_dist = order_num_dist
        self.distribution_name = "Poisson"
        self.idle_driver_dist_time = idle_driver_dist_time
        self.idle_driver_location_mat = idle_driver_location_mat

        self.order_time_dist = order_time_dist[:l_max] / np.sum(order_time_dist[:l_max])
        self.order_price_dist = order_price_dist

        target_node_ids = []
        target_grids_sorted = np.sort(mapped_matrix_int[np.where(mapped_matrix_int > 0)])
        for item in target_grids_sorted:
            x, y = np.where(mapped_matrix_int == item)
            target_node_ids.append(ids_2dto1d(x, y, M, N))
        self.target_node_ids = target_node_ids
        # store valid note id. Sort by number of orders emerged. descending.

        self.node_mapping = {}
        self.construct_mapping()

        self.real_orders = real_orders  # 4 weeks' data
        self.p = probability  # sample probability
        self.time_keys = [int(dt.strftime('%H%M')) for dt in
                          datetime_range(datetime(2017, 9, 1, 0), datetime(2017, 9, 2, 0),
                                         timedelta(minutes=time_interval))]
        self.day_orders = []  # one day's order.

        self.onoff_driver_location_mat = onoff_driver_location_mat

        # Stats
        self.all_grids_on_number = 0  # current online # drivers.
        self.all_grids_off_number = 0

        self.out_grid_in_orders = np.zeros((int(self.n_intervals), len(self.target_grids)))
        self.global_flag = global_flag
        self.weights_layers_neighbors = [1.0, np.exp(-1), np.exp(-2)]

        self.metrics = ConflictMetrics()
        self._reset_step_counters()
        self.neighbor_idle_snapshot = {}

        # Episode 级指标
        self.episode_reward = 0.0
        self.episode_total_orders = 0
        self.episode_finished_orders = 0
        self.episode_same_grid = 0

        self._dispatch_buf = []  # 保存 (dest_node_id, driver_id)

    def reset_episode_metrics(self):
        """在每个新 Ep 开始时调用，清零所有 Episode 指标"""
        self.episode_reward = 0.0
        self.episode_total_orders = 0
        self.episode_finished_orders = 0
        self.episode_same_grid = 0

    def set_global_seeds(self,seed):
        import numpy as np, random, tensorflow as tf
        np.random.seed(seed)
        random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

    def _reset_step_counters(self):
        self.metrics.reset_step()

    def construct_map_simulation(self, M, N, n):
        """Connect node to its neighbors based on a simulated M by N map
            :param M: M row index matrix
            :param N: N column index matrix
            :param n: n - sided polygon
        """
        for idx, current_node in enumerate(self.nodes):
            if current_node is not None:
                i, j = ids_1dto2d(idx, M, N)
                current_node.set_neighbors(get_neighbor_list(i, j, M, N, n, self.nodes))

    def construct_mapping(self):
        """
        :return:
        """
        target_grid_id = self.mapped_matrix_int[np.where(self.mapped_matrix_int > 0)]
        for g_id, n_id in zip(target_grid_id, self.target_grids):
            self.node_mapping[g_id] = n_id

    def construct_node_real(self, mapped_matrix_int):
        """ Initialize node, only valid node in mapped_matrix_in will be initialized.
        """

        # ── 新写法：只取 >0 的格子 ──
        row_inds, col_inds = np.where(mapped_matrix_int >= 0)

        target_ids = []
        for x, y in zip(row_inds, col_inds):
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id] = Node(node_id)
            target_ids.append(node_id)

        for x, y in zip(row_inds, col_inds):
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id].get_layers_neighbors(self.l_max, self.M, self.N, self)

        self.target_grids = target_ids
        self.n_valid_grids = len(target_ids)
        # ---------- 额外：为每个有效网格缓存 6-邻居 ----------
        self.valid_neighbor_node_id = []  # 每行 6 个 node_id (None→-1)
        self.valid_action_mask = np.ones((len(target_ids), 7), np.int8)

        for g_idx, nid in enumerate(self.target_grids):
            neigh_ids = []
            for k in range(6):
                nb = self.nodes[nid].neighbors[k] if k < len(self.nodes[nid].neighbors) else None
                if nb is None:
                    neigh_ids.append(-1)  # 占位
                    self.valid_action_mask[g_idx, k] = 0  # 该方向无效
                else:
                    neigh_ids.append(nb.get_node_index())
            self.valid_neighbor_node_id.append(neigh_ids)

    def construct_map_real(self, n_side):
        """Build node connection.
        """
        for idx, current_node in enumerate(self.nodes):
            i, j = ids_1dto2d(idx, self.M, self.N)
            if current_node is not None:
                current_node.set_neighbors(get_neighbor_list(i, j, self.M, self.N, n_side, self.nodes))

    def initial_order_random(self, distribution_all, dis_paras_all):
        """ Initialize order distribution
        :param distribution: 'Poisson', 'Gaussian'
        :param dis_paras:     lambda,    mu, sigma
        """
        for idx, node in enumerate(self.nodes):
            if node is not None:
                node.order_distribution(distribution_all[idx], dis_paras_all[idx])

    def get_observation(self):
        next_state = np.zeros((2, self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[0, row_id, column_id] = _node.idle_driver_num
                next_state[1, row_id, column_id] = _node.order_num

        return next_state

    def get_num_idle_drivers(self):
        """ Compute idle drivers
        :return:
        """
        temp_n_idle_drivers = 0
        for _node in self.nodes:
            if _node is not None:
                temp_n_idle_drivers += _node.idle_driver_num
        return temp_n_idle_drivers

    def get_observation_driver_state(self):
        """ Get idle driver distribution, computing #drivers from node.
        :return:
        """
        next_state = np.zeros((self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[row_id, column_id] = _node.get_idle_driver_numbers_loop()

        return next_state

    def reset_randomseed(self, random_seed):
        self.RANDOM_SEED = int(random_seed)
        self.set_global_seeds(self.RANDOM_SEED)

    def reset(self):
        """ Return initial observation: get order distribution and idle driver distribution

        """

        _M = self.M
        _N = self.N
        assert self.city_time == 0
        # initialization drivers according to the distribution at time 0
        num_idle_driver = self.utility_get_n_idle_drivers_real()
        self.step_driver_online_offline_control(num_idle_driver)

        # generate orders at first time step
        distribution_name = [self.distribution_name] * (_M * _N)
        distribution_param_dictionary = self.order_num_dist[self.city_time]
        distribution_param = [0] * (_M * _N)
        for key, value in distribution_param_dictionary.items():
            if self.distribution_name == 'Gaussian':
                mu, sigma = value
                distribution_param[key] = mu, sigma
            elif self.distribution_name == 'Poisson':
                mu = value[0]
                distribution_param[key] = mu
            else:
                print("Wrong distribution")

        self.initial_order_random(distribution_name, distribution_param)
        self.step_generate_order_real()

        return self.get_observation()

    def reset_clean(self, generate_order=1, ratio=1, city_time=""):
        """ 1. bootstrap oneday's order data.
            2. clean current drivers and orders, regenerate new orders and drivers.
            can reset anytime
        :return:
        """
        if city_time != "":
            self.city_time = city_time

        # clean orders and drivers
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        self.n_offline_drivers = 0  # total number of offline drivers.
        for node in self.nodes:
            if node is not None:
                node.clean_node()

        # Generate one day's order.
        if generate_order == 1:
            self.utility_bootstrap_oneday_order()

        # Init orders of current time step
        moment = self.city_time % self.n_intervals
        moment = int(moment)
        self.step_bootstrap_order_real(self.day_orders[moment])

        # Init current driver distribution
        if self.global_flag == "global":
            num_idle_driver = self.utility_get_n_idle_drivers_real()
            num_idle_driver = int(num_idle_driver * ratio)
            print(f"num_idle_driver:{num_idle_driver}")
        else:
            num_idle_driver = self.utility_get_n_idle_drivers_nodewise()
        self.step_driver_online_offline_control_new(num_idle_driver)
        self.neighbor_idle_snapshot = {}
        return self.get_observation()

    def utility_collect_offline_drivers_id(self):
        """count how many drivers are offline
        :return: offline_drivers: a list of offline driver id
        """
        count = 0  # offline driver num
        offline_drivers = []  # record offline driver id
        for key, _driver in self.drivers.items():
            if _driver.online is False:
                count += 1
                offline_drivers.append(_driver.get_driver_id())
        return offline_drivers

    def utility_get_n_idle_drivers_nodewise(self):
        """ compute idle drivers.
        :return:
        """
        time = self.city_time % self.n_intervals
        idle_driver_num = np.sum(self.idle_driver_location_mat[time])
        return int(idle_driver_num)

    def utility_add_driver_real_new(self, num_added_driver):
        curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution_resort = np.array(
            [int(curr_idle_driver_distribution.flatten()[index]) for index in
             self.target_node_ids])

        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]

        idle_diff = idle_driver_distribution.astype(int) - curr_idle_driver_distribution_resort
        idle_diff[np.where(idle_diff <= 0)] = 0

        node_ids = np.random.choice(self.target_node_ids, size=[num_added_driver],
                                    p=idle_diff / float(np.sum(idle_diff)))

        n_total_drivers = len(self.drivers.keys())
        for ii, node_id in enumerate(node_ids):
            added_driver_id = n_total_drivers + ii
            self.drivers[added_driver_id] = Driver(added_driver_id)
            self.drivers[added_driver_id].set_position(self.nodes[node_id])
            self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])

        self.n_drivers += num_added_driver

    def utility_add_driver_real_new_offlinefirst(self, num_added_driver):

        # curr_idle_driver_distribution = self.get_observation()[0][np.where(self.mapped_matrix_int > 0)]
        curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution_resort = np.array(
            [int(curr_idle_driver_distribution.flatten()[index]) for index in
             self.target_node_ids])

        idle_driver_distribution = self.idle_driver_location_mat[
            int(self.city_time % self.n_intervals), self.target_node_ids
        ]

        idle_diff = idle_driver_distribution.astype(int) - curr_idle_driver_distribution_resort
        idle_diff[np.where(idle_diff <= 0)] = 0

        if float(np.sum(idle_diff)) == 0:
            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(self.target_node_ids, size=[num_added_driver],
                                    p=idle_diff / float(np.sum(idle_diff)))

        for ii, node_id in enumerate(node_ids):

            if self.nodes[node_id].offline_driver_num > 0:
                self.nodes[node_id].set_offline_driver_online()
                self.n_drivers += 1
                self.n_offline_drivers -= 1
            else:

                n_total_drivers = len(self.drivers.keys())
                added_driver_id = n_total_drivers
                self.drivers[added_driver_id] = Driver(added_driver_id)
                self.drivers[added_driver_id].set_position(self.nodes[node_id])
                self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])
                self.n_drivers += 1

    def utility_add_driver_real_nodewise(self, node_id, num_added_driver):

        while num_added_driver > 0:
            if self.nodes[node_id].offline_driver_num > 0:
                self.nodes[node_id].set_offline_driver_online()
                self.n_drivers += 1
                self.n_offline_drivers -= 1
            else:

                n_total_drivers = len(self.drivers.keys())
                added_driver_id = n_total_drivers
                self.drivers[added_driver_id] = Driver(added_driver_id)
                self.drivers[added_driver_id].set_position(self.nodes[node_id])
                self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])
                self.n_drivers += 1
            num_added_driver -= 1

    def utility_set_drivers_offline_real_nodewise(self, node_id, n_drivers_to_off):

        while n_drivers_to_off > 0:
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random()
                self.n_drivers -= 1
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1
                self.all_grids_off_number += 1
            else:
                break

    def utility_set_drivers_offline_real_new(self, n_drivers_to_off):

        curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution_resort = np.array([int(curr_idle_driver_distribution.flatten()[index])
                                                         for index in self.target_node_ids])

        # historical idle driver distribution
        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % int(self.n_intervals), :]

        # ——— 对齐两者长度，避免广播错误 ———
        if idle_driver_distribution.shape[0] != curr_idle_driver_distribution_resort.shape[0]:
            idle_driver_distribution = idle_driver_distribution[:curr_idle_driver_distribution_resort.shape[0]]

        # diff of curr idle driver distribution and history
        idle_diff = curr_idle_driver_distribution_resort - idle_driver_distribution.astype(int)
        idle_diff[np.where(idle_diff <= 0)] = 0

        n_drivers_can_be_off = int(np.sum(curr_idle_driver_distribution_resort[np.where(idle_diff >= 0)]))
        if n_drivers_to_off > n_drivers_can_be_off:
            n_drivers_to_off = n_drivers_can_be_off

        sum_idle_diff = np.sum(idle_diff)
        if sum_idle_diff == 0:
            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(self.target_node_ids, size=[n_drivers_to_off],
                                    p=idle_diff / float(sum_idle_diff))

        for ii, node_id in enumerate(node_ids):
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random()
                self.n_drivers -= 1
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1

    def utility_bootstrap_oneday_order(self):
        np.random.seed(self.RANDOM_SEED)
        num_all_orders = len(self.real_orders)

        # 这行代码用来从 num_all_orders 个订单中，以概率 self.p 随机抽取订单，并返回这些被抽取订单的索引。
        index_sampled_orders = np.where(np.random.binomial(1, self.p, num_all_orders) == 1)[0]
        one_day_orders = [self.real_orders[i] for i in index_sampled_orders]

        print(f"generate_order=1时，进行1/28比例之后的订单采样数量：{len(one_day_orders)}")

        self.out_grid_in_orders = np.zeros((int(self.n_intervals), len(self.target_grids)))

        day_orders = [[] for _ in np.arange(self.n_intervals)]
        for iorder in one_day_orders:
            #  iorder: [92, 300, 143, 2, 13.2]
            start_time = int(iorder[2])
            if iorder[0] not in self.node_mapping.keys() and iorder[1] not in self.node_mapping.keys():
                continue
            start_node = self.node_mapping.get(iorder[0], -100)
            end_node = self.node_mapping.get(iorder[1], -100)
            duration = int(iorder[3])
            price = iorder[4]

            if start_node == -100:
                column_index = self.target_grids.index(end_node)
                self.out_grid_in_orders[int((start_time + duration) % self.n_intervals), column_index] += 1

                continue

            day_orders[start_time].append([start_node, end_node, start_time, duration, price])

        # 打印整天的总订单数量
        self.day_orders = day_orders
        self.episode_total_orders = sum(len(slot) for slot in self.day_orders)

    def step_driver_status_control(self):
        # Deal with orders finished at time T=1, check driver status. finish order, set back to off service
        for key, _driver in self.drivers.items():
            _driver.status_control_eachtime(self)
        moment = self.city_time % self.n_intervals
        moment = int(moment)
        orders_to_on_drivers = self.out_grid_in_orders[moment, :]
        for idx, item in enumerate(orders_to_on_drivers):
            if item != 0:
                node_id = self.target_grids[idx]
                self.utility_add_driver_real_nodewise(node_id, int(item))

    def step_driver_online_offline_nodewise(self):
        """ node wise control driver online offline
        :return:
        """
        np.random.seed(self.RANDOM_SEED)
        moment = self.city_time % self.n_intervals
        moment = int(moment)
        curr_onoff_distribution = self.onoff_driver_location_mat[moment]

        self.all_grids_on_number = 0
        self.all_grids_off_number = 0
        for idx, target_node_id in enumerate(self.target_node_ids):
            curr_mu = curr_onoff_distribution[idx, 0]
            curr_sigma = curr_onoff_distribution[idx, 1]
            on_off_number = np.round(np.random.normal(curr_mu, curr_sigma, 1)[0]).astype(int)

            if on_off_number > 0:
                self.utility_add_driver_real_nodewise(target_node_id, on_off_number)
                self.all_grids_on_number += on_off_number
            elif on_off_number < 0:
                self.utility_set_drivers_offline_real_nodewise(target_node_id, abs(on_off_number))
            else:
                pass

    def step_driver_online_offline_control_new(self, n_idle_drivers):
        """ control the online offline status of drivers

        :param n_idle_drivers: the number of idle drivers expected at current moment
        :return:
        """

        offline_drivers = self.utility_collect_offline_drivers_id()
        self.n_offline_drivers = len(offline_drivers)

        if n_idle_drivers > self.n_drivers:

            self.utility_add_driver_real_new_offlinefirst(n_idle_drivers - self.n_drivers)

        elif n_idle_drivers < self.n_drivers:
            self.utility_set_drivers_offline_real_new(self.n_drivers - n_idle_drivers)
        else:
            pass

    def step_driver_online_offline_control(self, n_idle_drivers):
        """ control the online offline status of drivers

        :param n_idle_drivers: the number of idle drivers expected at current moment
        :return:
        """

        offline_drivers = self.utility_collect_offline_drivers_id()
        self.n_offline_drivers = len(offline_drivers)
        if n_idle_drivers > self.n_drivers:
            # bring drivers online.
            while self.n_drivers < n_idle_drivers:
                if self.n_offline_drivers > 0:
                    for ii in np.arange(self.n_offline_drivers):
                        self.drivers[offline_drivers[ii]].set_online()
                        self.n_drivers += 1
                        self.n_offline_drivers -= 1
                        if self.n_drivers == n_idle_drivers:
                            break

                self.utility_add_driver_real_new(n_idle_drivers - self.n_drivers)

        elif n_idle_drivers < self.n_drivers:
            self.utility_set_drivers_offline_real_new(self.n_drivers - n_idle_drivers)
        else:
            pass

    def utility_get_n_idle_drivers_real(self):
        """ control the number of idle drivers in simulator;
        :return:
        """
        time = self.city_time % self.n_intervals
        mean, std = self.idle_driver_dist_time[int(time)]
        np.random.seed(self.city_time)
        return np.round(np.random.normal(mean, std, 1)[0]).astype(int)

    def utility_set_neighbor_weight(self, weights):
        self.weights_layers_neighbors = weights

    def step_generate_order_real(self):
        # generate order at t + 1
        for node in self.nodes:
            if node is not None:
                node_id = node.get_node_index()
                # generate orders start from each node
                random_seed = node.get_node_index() + self.city_time
                node.generate_order_real(self.l_max, self.order_time_dist, self.order_price_dist,
                                         self.city_time, self.nodes, random_seed)

    def step_bootstrap_order_real(self, day_orders_t):
        for iorder in day_orders_t:
            start_node_id = iorder[0]
            end_node_id = iorder[1]
            start_node = self.nodes[start_node_id]

            if end_node_id in self.target_grids:
                end_node = self.nodes[end_node_id]
            else:
                end_node = None
            start_node.add_order_real(self.city_time, end_node, iorder[3], iorder[4])

    def step_assign_order(self):

        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        for node in self.nodes:
            if node is not None:
                node.remove_unfinished_order(self.city_time)
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time,
                                                                                                         self)
                reward += reward_node
                all_order_num += all_order_num_node
                finished_order_num += finished_order_num_node
        if all_order_num != 0:
            self.order_response_rate = finished_order_num / float(all_order_num)
        else:
            self.order_response_rate = -1
        return reward

    def step_assign_order_broadcast_neighbor_reward_update(self):
        """ Consider the orders whose destination or origin is not in the target region
        :param num_layers:
        :param weights_layers_neighbors: [1, 0.5, 0.25, 0.125]
        :return:
        """

        node_reward = np.zeros((len(self.nodes)))
        # neighbor_reward = np.zeros((len(self.nodes)))
        # First round broadcast
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0

        print(f"[t={self.city_time:03d}] 本时段产生订单数：{len(self.day_orders[self.city_time])}")
        idle_drivers = self.get_num_idle_drivers()
        print(f"[t={self.city_time:03d}] 空闲司机总数：{idle_drivers}")
        finished_order_num_node1 = 0
        for node in self.nodes:
            if node is not None:
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time,
                                                                                                         self)
                finished_order_num_node1 += finished_order_num_node
                reward += reward_node
                self.episode_reward += reward_node
                all_order_num += all_order_num_node
                finished_order_num += finished_order_num_node
                node_reward[node.get_node_index()] += reward_node

        print(f"第一轮派单完成数量:{finished_order_num_node1}")
        # 记录广播开始前的邻居空车快照
        # self.neighbor_idle_snapshot = {
        #     node.get_node_index(): node.idle_driver_num
        #     for node in self.nodes if node is not None
        # }
        # # 重置 step 级别的冲突计数
        # self.metrics.reset_step()
        # # Second round broadcast
        # finished_order_num_node_broadcast1 = 0
        #
        # for node in self.nodes:
        #     if node is not None:
        #         if node.order_num != 0:
        #             reward_node_broadcast, finished_order_num_node_broadcast \
        #                 = node.simple_order_assign_broadcast_update(self, neighbor_reward)
        #             finished_order_num_node_broadcast1 += finished_order_num_node_broadcast
        #             reward += reward_node_broadcast
        #             self.episode_reward += reward_node_broadcast
        #             finished_order_num += finished_order_num_node_broadcast
        #
        # print(f"第二轮派单完成数量:{finished_order_num_node_broadcast1}")
        #
        # node_reward = node_reward + neighbor_reward
        node_reward = node_reward
        if all_order_num != 0:
            self.order_response_rate = finished_order_num / float(all_order_num)
        else:
            self.order_response_rate = -1

        self.episode_finished_orders += finished_order_num
        # sg_s, ud_s = self.metrics.same_grid_contention_step, self.metrics.unserved_demand_step
        # print(f"[metric] step {self.city_time:03d}: "
        #       f"same_grid_contention={sg_s}, unserved_demand={ud_s}")


        # return reward, [node_reward, neighbor_reward]
        return reward, [node_reward]
    def step_remove_unfinished_orders(self):
        for node in self.nodes:
            if node is not None:
                node.remove_unfinished_order(self.city_time)

    def step_pre_order_assigin(self, next_state):

        remain_drivers = next_state[0] - next_state[1]
        remain_drivers[remain_drivers < 0] = 0

        remain_orders = next_state[1] - next_state[0]
        remain_orders[remain_orders < 0] = 0

        if np.sum(remain_orders) == 0 or np.sum(remain_drivers) == 0:
            context = np.array([remain_drivers, remain_orders])
            return context

        remain_orders_1d = remain_orders.flatten()
        remain_drivers_1d = remain_drivers.flatten()

        for node in self.nodes:
            if node is not None:
                curr_node_id = node.get_node_index()
                if remain_orders_1d[curr_node_id] != 0:
                    for neighbor_node in node.neighbors:
                        if neighbor_node is not None:
                            neighbor_id = neighbor_node.get_node_index()
                            a = remain_orders_1d[curr_node_id]
                            b = remain_drivers_1d[neighbor_id]
                            remain_orders_1d[curr_node_id] = max(a - b, 0)
                            remain_drivers_1d[neighbor_id] = max(b - a, 0)
                        if remain_orders_1d[curr_node_id] == 0:
                            break

        context = np.array([remain_drivers_1d.reshape(self.M, self.N),
                            remain_orders_1d.reshape(self.M, self.N)])
        return context

    def step_dispatch_invalid(self, dispatch_actions):
        """ If a
        :param dispatch_actions:
        :return:
        """
        save_remove_id = []
        for action in dispatch_actions:

            start_node_id, end_node_id, num_of_drivers = action
            if self.nodes[start_node_id] is None or num_of_drivers == 0:
                continue  # not a feasible action

            if self.nodes[start_node_id].get_driver_numbers() < num_of_drivers:
                num_of_drivers = self.nodes[start_node_id].get_driver_numbers()

            if end_node_id < 0:
                for _ in np.arange(num_of_drivers):
                    self.nodes[start_node_id].set_idle_driver_offline_random()
                    self.n_drivers -= 1
                    self.n_offline_drivers += 1
                    self.all_grids_off_number += 1
                continue

            if self.nodes[end_node_id] is None:
                for _ in np.arange(num_of_drivers):
                    self.nodes[start_node_id].set_idle_driver_offline_random()
                    self.n_drivers -= 1
                    self.n_offline_drivers += 1
                    self.all_grids_off_number += 1
                continue

            if self.nodes[end_node_id] not in self.nodes[start_node_id].neighbors:
                raise ValueError('City:step(): not a feasible dispatch')

            for _ in np.arange(num_of_drivers):
                # t = 1 dispatch start, idle driver decrease
                remove_driver_id = self.nodes[start_node_id].remove_idle_driver_random()
                save_remove_id.append((end_node_id, remove_driver_id))
                self.drivers[remove_driver_id].set_position(None)
                self.drivers[remove_driver_id].set_offline_for_start_dispatch()
                self.n_drivers -= 1

        return save_remove_id

    def step_add_dispatched_drivers(self, save_remove_id):
        # drivers dispatched at t, arrived at t + 1
        for destination_node_id, arrive_driver_id in save_remove_id:
            self.drivers[arrive_driver_id].set_position(self.nodes[destination_node_id])
            self.drivers[arrive_driver_id].set_online_for_finish_dispatch()
            self.nodes[destination_node_id].add_driver(arrive_driver_id, self.drivers[arrive_driver_id])
            self.n_drivers += 1

    def step_increase_city_time(self):
        self.city_time += 1
        # set city time of drivers
        for driver_id, driver in self.drivers.items():
            driver.set_city_time(self.city_time)

    def step(self, dispatch_actions, generate_order=1): # action: [source, destination, nums]
        info = []
        print(f"dispatch_actions:{dispatch_actions}")

        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        save_remove_id = self.step_dispatch_invalid(dispatch_actions)
        # When the drivers go to invalid grid, set them offline.

        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        self.step_increase_city_time()
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        if generate_order == 1:
            self.step_generate_order_real()
        else:
            moment = self.city_time % self.n_intervals
            moment = int(moment)
            self.step_bootstrap_order_real(self.day_orders[moment])

        # offline online control;
        self.step_driver_online_offline_nodewise()
        self.step_remove_unfinished_orders()
        # get states S_{t+1}  [driver_dist, order_dist]
        next_state = self.get_observation()
        print(f"next_state:{next_state.shape}")
        sys.exit()
        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_state, reward, info

    def step_stage1(self, action_tuple, epsilon=0.0,return_node_gmv: bool = False):
        """
        第 1 阶段：
          • 若提供 estimator+stateProcessor，则调用 Agent-1 产生调度动作并实际移动司机
          • 之后每个网格各自 simple_order_assign_real()
        返回:
          reward_local : float
          pending_nodes: List[int]  — 仍有未完成订单的网格 id
        """
        reward_local = 0.0
        node_gmv_vec = np.zeros(self.M * self.N, dtype=np.float32)
        total_local_finished = 0
        pending_nodes = []
        # print(f"[t={self.city_time:03d}] 本时段产生订单数：{len(self.day_orders[self.city_time])}")
        # idle_drivers = self.get_num_idle_drivers()
        # print(f"[t={self.city_time:03d}] 空闲司机总数：{idle_drivers}")

        # ---------- ① (可选) 调度司机 ----------
        if action_tuple:  # 真正移动司机
            save_remove_id = self.step_dispatch_invalid(action_tuple)
            # 暂存在缓冲区，下一时间步回收
            self._dispatch_buf.extend(save_remove_id)

        print(f"[t={self.city_time:03d}] 本时段产生订单数：{len(self.day_orders[self.city_time])}")
        idle_drivers = self.get_num_idle_drivers()
        print(f"[t={self.city_time:03d}] 空闲司机总数：{idle_drivers}")
        # ---------- ② 本格子自己接单 ----------
        for node in self.nodes:
            if node is None: continue
            r_loc, _, fin = node.simple_order_assign_real(self.city_time, self)
            reward_local += r_loc
            total_local_finished += fin
            node_gmv_vec[node.get_node_index()] += r_loc
            if node.order_num > 0:
                pending_nodes.append(node.get_node_index())
        print(f"== 本轮完成汇总 == local_total finished = {total_local_finished}")

        # ---------- ③ 保存邻居空车快照 ----------
        self.neighbor_idle_snapshot = {
            node.get_node_index(): node.idle_driver_num
            for node in self.nodes if node is not None
        }

        # 第一段的reward的累加
        self.episode_reward += reward_local
        # 第一轮完成的订单数量
        self.episode_finished_orders += total_local_finished
        if return_node_gmv:
            return reward_local, pending_nodes, node_gmv_vec
        else:
            return reward_local, pending_nodes

    def step_stage2_weight_plus(self, pending_nodes, weights_mat,return_node_gmv: bool = False):

        reward_total, finished_total = 0.0, 0
        node_gmv_vec2 = np.zeros(self.M * self.N, dtype=np.float32)
        self.metrics.reset_step()
        num_finished_orders = 0
        idle_init = dict(self.neighbor_idle_snapshot)  # 广播开始前的邻居 idle
        for node_id, w in zip(pending_nodes, weights_mat):
            node = self.nodes[node_id]
            pending0 = int(node.order_num)
            remain = pending0
            if remain == 0:
                continue

            # ---------- 1. 选邻居并排序 ----------
            neigh_ids = self.valid_neighbor_node_id[self.target_grids.index(node_id)]
            # (6,)
            idle_vec = np.array(
                [self.nodes[n].idle_driver_num for n in neigh_ids[:6]], dtype=np.int64
            )
            # (6,)
            # score = idle_vec * w[:6]
            score = idle_vec * w
            order = np.argsort(-score)  # 邻居排序，idle×weight 降序

            total_shortage_init = 0  # baseline 用
            taken_cnt = {int(n): 0 for n in neigh_ids[:6]}  # 记录借走量

            # ---------- 2. 第一轮：排序扫描 ----------
            for j in order:
                neigh_id = neigh_ids[j]
                neigh = self.nodes[neigh_id]
                if neigh is None:
                    continue
                avail = int(neigh.idle_driver_num)
                if avail == 0:
                    continue

                need = remain
                given = min(need, avail)

                # 如果 需求大于 空车 那么产生空缺
                short = need - given


                # 这里只有当 初始 idle_init.get(neigh_id, 0) > 0 代表有车的时候，才会走下面这一步
                if short > 0 and idle_init.get(neigh_id, 0) > 0:
                    total_shortage_init += short

                if given > 0:
                    rr = node.utility_assign_orders_neighbor(self, neigh, given)  # 返回该批 GMV
                    reward_total += rr
                    node_gmv_vec2[node_id] += rr
                    num_finished_orders += given

                    # 第二轮派单reward - 第一轮 按照权重取车
                    remain -= given
                    taken_cnt[int(neigh_id)] += given

                if remain == 0:
                    break


            # ---------- 3. 第二轮：原顺序再次贪心 ----------
            if remain > 0:
                rr2, finished2 = node.simple_order_assign_broadcast_update(
                    self,  # city 实例
                    node_gmv_vec2  # ← 传入用来累加 GMV 的向量
                )
                reward_total += rr2
                num_finished_orders += finished2
                # simple_order_assign_broadcast_update 已经自己删掉 node.orders 中已完成的单
                remain = node.order_num

                # ---------- 4. 计算真实 same-grid 抢车量 ----------
            excess_sum = 0
            for nid, taken in taken_cnt.items():
                excess_sum += max(0, taken - idle_init.get(nid, 0))

            # ---------- 5. 写冲突指标 ----------
            if remain == 0:  # 全部补齐
                if excess_sum:
                    self.metrics.add_same_grid_contention(excess_sum)
            else:

                # 如果一开始邻居都没车，那么total_shortage_init == 0，那么conflict也只会加 0
                conflicts_add = min(remain, total_shortage_init)  # baseline 方式
                samegrid_add = max(0, excess_sum - conflicts_add)
                if samegrid_add:
                    self.metrics.add_same_grid_contention(samegrid_add)
                if conflicts_add:
                    self.metrics.add_unserved_demand(conflicts_add)
                # 余下 (remain - conflicts_add) 照 baseline 口径忽略

        print(f"== 本轮完成汇总 == neighbor finished = {num_finished_orders}")
        self.episode_reward += reward_total
            # 第二轮完成的订单数量
        self.episode_finished_orders += num_finished_orders

        if return_node_gmv:
            return reward_total, node_gmv_vec2
        else:
            return reward_total


    def step_finish_interval(self,
                             inject_next_orders: bool = False):
        """
        • ① 司机调度到达 → 回到目的地并变 idle
        • ② 服务中的司机检查是否完单
        • ③ （可选）注入 t+1 的订单
        • ④ （可选）根据统计分布做司机上下线
        • ⑤ 清理超时未服务订单
        """
        self.step_driver_status_control()
        # ---------- ① 调度司机到达 ----------
        if self._dispatch_buf:
            self.step_add_dispatched_drivers(self._dispatch_buf)
            self._dispatch_buf = []


        # ---------- ③ 注入下一步订单 ----------
        if inject_next_orders:
            moment = int(self.city_time % self.n_intervals)  # **注意：此时 city_time 已在外层 +1**
            self.step_bootstrap_order_real(self.day_orders[moment])

        self.step_driver_online_offline_nodewise()

        self.step_remove_unfinished_orders()

    def apply_manual_dispatch(self, action_tuple):
        """
        直接应用一个外部计算出的调度方案 (action_tuple)，并更新环境状态。
        这个函数绕过了环境内部的 dispatch_orders 逻辑。

        Args:
            action_tuple (list): [(start_node_id, end_node_id, num_drivers), ...]

        Returns:
            float: 由本次调度产生的总 GMV。
        """
        total_gmv_this_step = 0.0

        # 1. 首先验证调度的可行性
        driver_counts_to_dispatch = {}
        for start_node_id, _, num_drivers in action_tuple:
            driver_counts_to_dispatch[start_node_id] = \
                driver_counts_to_dispatch.get(start_node_id, 0) + num_drivers

        for start_node_id, total_dispatch in driver_counts_to_dispatch.items():
            if self.nodes[start_node_id].idle_driver_num < total_dispatch:
                # 理论上，如果求解器输入正确，不应该发生此错误
                raise ValueError(
                    f"City:apply_manual_dispatch(): not a feasible dispatch. "
                    f"Grid {start_node_id} has {self.nodes[start_node_id].idle_driver_num} drivers, "
                    f"but trying to dispatch {total_dispatch}."
                )

        # 2. 执行调度并更新状态
        for start_node_id, end_node_id, num_drivers in action_tuple:
            # a. 减少源头网格的空闲司机
            self.nodes[start_node_id].idle_driver_num -= num_drivers

            # b. 匹配订单并计算 GMV
            orders_to_serve = self.nodes[end_node_id].order_num
            matched_orders = min(num_drivers, orders_to_serve)

            if matched_orders > 0:
                # 减少目的地网格的订单
                self.nodes[end_node_id].order_num -= matched_orders

                # 更新统计指标
                self.episode_finished_orders += matched_orders

                # 为这些匹配的订单计算并累加 GMV
                # 我们需要从订单池中“取出”订单来获取其价值
                gmv_from_match = 0
                for _ in range(matched_orders):
                    if self.nodes[end_node_id].order_pool:
                        order = self.nodes[end_node_id].order_pool.pop(0)
                        gmv_from_match += order.price

                self.episode_reward += gmv_from_match
                total_gmv_this_step += gmv_from_match

        return total_gmv_this_step

    def step_stage1_with_max_flow(self, action_tuple, solver=None, return_node_gmv: bool = False):
        """
        第 1 阶段：
          • 若提供 estimator+stateProcessor，则调用 Agent-1 产生调度动作并实际移动司机
          • 之后每个网格各自 simple_order_assign_real()
        返回:
          reward_local : float
          pending_nodes: List[int]  — 仍有未完成订单的网格 id
        """
        reward_local = 0.0
        node_gmv_vec = np.zeros(self.M * self.N, dtype=np.float32)
        total_local_finished = 0
        pending_nodes = []
        # ---------- ① (可选) 调度司机 ----------
        if action_tuple:  # 真正移动司机
            save_remove_id = self.step_dispatch_invalid(action_tuple)
            # 暂存在缓冲区，下一时间步回收
            self._dispatch_buf.extend(save_remove_id)

        # -------------额外计算最大流问题------------------------
        state = self.get_observation()
        driver_map = state[0]
        order_map = state[1]

        drivers_t = np.array(
            [driver_map[ids_1dto2d(grid_id, solver.M, solver.N_cols)] for grid_id in solver.valid_grid_ids])
        orders_t = np.array(
            [order_map[ids_1dto2d(grid_id, solver.M, solver.N_cols)] for grid_id in solver.valid_grid_ids])
        solution, var_map = solver.solve(drivers_t, orders_t)



        print(f"[t={self.city_time:03d}] 本时段产生订单数：{len(self.day_orders[self.city_time])}")
        idle_drivers = self.get_num_idle_drivers()
        print(f"[t={self.city_time:03d}] 空闲司机总数：{idle_drivers}")
        # ---------- ② 本格子自己接单 ----------
        for node in self.nodes:
            if node is None: continue
            r_loc, _, fin = node.simple_order_assign_real(self.city_time, self)
            reward_local += r_loc
            total_local_finished += fin
            node_gmv_vec[node.get_node_index()] += r_loc
            if node.order_num > 0:
                pending_nodes.append(node.get_node_index())
        print(f"== 本轮完成汇总 == local_total finished = {total_local_finished}")

        # ---------- ③ 保存邻居空车快照 ----------
        self.neighbor_idle_snapshot = {
            node.get_node_index(): node.idle_driver_num
            for node in self.nodes if node is not None
        }

        # 第一段的reward的累加
        self.episode_reward += reward_local
        # 第一轮完成的订单数量
        self.episode_finished_orders += total_local_finished
        if return_node_gmv:
            return reward_local, pending_nodes, node_gmv_vec,np.sum(solution)
        else:
            return reward_local, pending_nodes
