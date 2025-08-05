import sys
import logging


sys.path.append("../")

from simulator.objects import *
from simulator.utilities import *
from simulator.conflict_metrics import ConflictMetrics
import numpy as np


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
        """Call at the beginning of each new episode to reset all episode-level metrics."""
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

        # ——— Align the lengths of both to prevent broadcasting errors. ———
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

        print(f"When generate_order = 1, the number of sampled orders after applying the {self.p} ratio:{len(one_day_orders)}")

        self.out_grid_in_orders = np.zeros((int(self.n_intervals), len(self.target_grids))) # 144,504

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
        neighbor_reward = np.zeros((len(self.nodes)))
        # First round broadcast
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0

        print(f"[t={self.city_time:03d}] time orders:{len(self.day_orders[self.city_time])}")
        idle_drivers = self.get_num_idle_drivers()
        print(f"[t={self.city_time:03d}] idle drivers: {idle_drivers}")
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

        print(f"first round finished orders:{finished_order_num_node1}")
        # Record a snapshot of neighbor idle vehicles before broadcasting begins.
        self.neighbor_idle_snapshot = {
            node.get_node_index(): node.idle_driver_num
            for node in self.nodes if node is not None
        }
        # Reset step-level conflict counters.
        self.metrics.reset_step()
        # Second round broadcast
        finished_order_num_node_broadcast1 = 0
        before_order_num_node = 0

        for node in self.nodes:
            if node is not None:
                if node.order_num != 0:
                    before_order_num_node += node.order_num
                    reward_node_broadcast, finished_order_num_node_broadcast \
                        = node.simple_order_assign_broadcast_update(self, neighbor_reward)
                    finished_order_num_node_broadcast1 += finished_order_num_node_broadcast
                    reward += reward_node_broadcast
                    self.episode_reward += reward_node_broadcast
                    finished_order_num += finished_order_num_node_broadcast
        print(f"second round finished orders:{finished_order_num_node_broadcast1}")
        node_reward = node_reward + neighbor_reward
        if all_order_num != 0:
            self.order_response_rate = finished_order_num / float(all_order_num)
        else:
            self.order_response_rate = -1

        self.episode_finished_orders += finished_order_num
        sg_s, ud_s = self.metrics.same_grid_contention_step, self.metrics.unserved_demand_step
        print(f"[metric] step {self.city_time:03d}: "
              f"same_grid_contention={sg_s}, unserved_demand={ud_s}")
        return reward, [node_reward, neighbor_reward]
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
        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_state, reward, info

    def step_stage1(self, action_tuple, epsilon=0.0,return_node_gmv: bool = False):
        """
            Stage 1:

            If both `estimator` and `stateProcessor` are provided, Agent-1 is called to generate dispatch actions and actually move the drivers.
            fter that, each grid independently runs `simple_order_assign_real()`.

            Returns:

            reward_local`: *float* — local reward for this stage
            pending_nodes`: List[int] — list of grid IDs that still have unassigned orders

        """
        reward_local = 0.0
        node_gmv_vec = np.zeros(self.M * self.N, dtype=np.float32)
        total_local_finished = 0
        pending_nodes = []

        # ---------- ① Dispatch drivers ----------
        if action_tuple:
            save_remove_id = self.step_dispatch_invalid(action_tuple)
            # Temporarily store in buffer and reclaim in the next time step.
            self._dispatch_buf.extend(save_remove_id)

        print(f"[t={self.city_time:03d}] time orders:{len(self.day_orders[self.city_time])}")
        idle_drivers = self.get_num_idle_drivers()
        print(f"[t={self.city_time:03d}] idle drivers: {idle_drivers}")
        # ---------- ② This grid handles its own orders. ----------
        for node in self.nodes:
            if node is None: continue
            r_loc, _, fin = node.simple_order_assign_real(self.city_time, self)
            reward_local += r_loc
            total_local_finished += fin
            node_gmv_vec[node.get_node_index()] += r_loc
            if node.order_num > 0:
                pending_nodes.append(node.get_node_index())
        print(f"== local_total finished = {total_local_finished}")

        # ---------- ③ Save a snapshot of neighbor idle vehicles. ----------
        self.neighbor_idle_snapshot = {
            node.get_node_index(): node.idle_driver_num
            for node in self.nodes if node is not None
        }

        # Accumulation of rewards from the first stage.
        self.episode_reward += reward_local
        # Number of orders completed in the first round.
        self.episode_finished_orders += total_local_finished
        if return_node_gmv:
            return reward_local, pending_nodes, node_gmv_vec
        else:
            return reward_local, pending_nodes

    def step_stage2_weight_plus(
            self,
            pending_nodes,
            weights_mat,
            select_mode: str = 'argmax',
            return_node_gmv: bool = False,
            return_traces: bool = True,
            return_next_state: bool = True
    ):
        reward_total, finished_total = 0.0, 0
        G = self.M * self.N
        node_gmv_vec2 = np.zeros(G, dtype=np.float32)

        # Used for measurement and masking.
        traces = {}  # {s_id: {'attempts':[j...], 'masks':[mask6_at_t,...]}}
        self.metrics.reset_step()
        num_finished_orders = 0

        # Record neighbor idle states at the start of Stage-2 (for conflict statistics).
        idle_init = dict(self.neighbor_idle_snapshot)



        # ---------- Main loop: iterate over each source grid.  ----------
        for node_id, w in zip(pending_nodes, weights_mat):
            node = self.nodes[node_id]
            pending0 = int(node.order_num)
            remain = pending0
            if remain == 0:
                continue

            # === 1) Fetch the **aligned** 6-neighbor indices (consistent with the weight dimensions). ===
            neigh_ids = []
            for neigh in node.neighbors:
                neigh_ids.append(neigh.get_node_index()) # 长度6，可能含 -1


            # Construct `idle_vec` and initial mask `mask0` (set to 0 for invalid or no-vehicle neighbors).
            idle_vec = np.zeros(6, dtype=np.int64)
            mask0 = np.zeros(6, dtype=np.int64)
            for j in range(6):
                nid = neigh_ids[j]
                if nid == -1:
                    idle_vec[j] = 0
                    mask0[j] = 0
                else:
                    avail = int(self.nodes[nid].idle_driver_num)
                    idle_vec[j] = avail
                    mask0[j] = 1 if avail > 0 else 0

            # === 2) Compute attempt order. ===
            #`argmax`: Sort by `idle * weight`;
            #`sample`: Sample **without replacement** from the support set defined by the mask, weighted by `weight`.

            if select_mode == 'argmax':
                score = idle_vec * w
                order_all = np.argsort(-score).tolist()
            else:
                # Perform normalized sampling **without replacement** from `w`, constrained by the `mask`, to generate a sequence.
                p = np.array(w, dtype=np.float64) * mask0
                s = p.sum()
                if s <= 0:
                    order_all = [i for i in range(6)]  # 退化：随便一个顺序
                else:
                    p = p / s
                    order_all = []
                    mask_tmp = mask0.copy()
                    for _ in range(6):
                        # Sample only from the remaining candidates.
                        p_tmp = p * mask_tmp
                        st = p_tmp.sum()
                        if st <= 1e-12:
                            break
                        p_tmp = p_tmp / st
                        j = int(np.random.choice(6, p=p_tmp))
                        order_all.append(j)
                        mask_tmp[j] = 0

            # === 3) Attempt neighbors one by one (record mask and selection; stop early if successful). ===
            attempts = []  # a_{s,1:m}
            masks_per_step = []  # R_{s,t}，
            taken_cnt = {int(nid): 0 for nid in neigh_ids if nid != -1}

            total_shortage_init = 0  # baseline conflict
            visited = np.zeros(6, dtype=np.int8)

            for j in order_all:
                # At each step, construct the current candidate mask: **valid** + not yet visited + still has vehicles.
                mask_t = np.zeros(6, dtype=np.int64)
                for k in range(6):
                    nid_k = neigh_ids[k]
                    if nid_k == -1 or visited[k]:
                        mask_t[k] = 0
                    else:
                        avail_k = int(self.nodes[nid_k].idle_driver_num)
                        mask_t[k] = 1 if avail_k > 0 else 0
                if mask_t.sum() == 0:
                    break
                # Record the candidate mask.
                masks_per_step.append(mask_t.copy())

                # If neighbor `j` is no longer selectable (no vehicles / invalid / already visited), skip it.
                if mask_t[j] == 0:
                    visited[j] = 1
                    continue

                # Execute the attempt.
                neigh_id = neigh_ids[j]
                neigh = self.nodes[neigh_id]
                avail = int(neigh.idle_driver_num)
                need = remain
                given = min(need, avail)

                short = need - given
                if short > 0 and idle_init.get(neigh_id, 0) > 0:
                    total_shortage_init += short

                if given > 0:
                    rr = node.utility_assign_orders_neighbor(self, neigh, given)
                    reward_total += rr
                    node_gmv_vec2[node_id] += rr
                    num_finished_orders += given
                    remain -= given
                    taken_cnt[int(neigh_id)] += given

                # Record the selected neighbor index (relative to the 6-dimensional order).
                attempts.append(int(j))
                visited[j] = 1

                if remain == 0:
                    break


            # === 5)  Conflict statistics. ===
            excess_sum = 0
            for nid, taken in taken_cnt.items():
                excess_sum += max(0, taken - idle_init.get(nid, 0))

            if remain == 0:
                if excess_sum:
                    self.metrics.add_same_grid_contention(excess_sum)
            else:
                conflicts_add = min(remain, total_shortage_init)
                samegrid_add = max(0, excess_sum - conflicts_add)
                if samegrid_add:
                    self.metrics.add_same_grid_contention(samegrid_add)
                if conflicts_add:
                    self.metrics.add_unserved_demand(conflicts_add)

            if return_traces:
                traces[node_id] = {
                    "attempts": attempts,  # [j1, j2, ...] 相对 6 维
                    "masks": masks_per_step,  # [mask6_at_t, ...] 二值
                    "neigh_ids": neigh_ids  # 如需从 6 维映射回真实 grid id
                }

        print(f"== neighbor finished = {num_finished_orders}")
        self.episode_reward += reward_total
        self.episode_finished_orders += num_finished_orders

        # Global pending orders after Stage-2
        p_next = None
        if return_next_state:
            p_next = np.array([self.nodes[g].order_num for g in range(G)], dtype=np.int32)

        if return_traces and return_next_state and return_node_gmv:
            return reward_total, node_gmv_vec2, traces, p_next
        if return_traces and return_next_state:
            return reward_total, traces, p_next
        if return_node_gmv:
            return reward_total, node_gmv_vec2
        return reward_total

    def step_finish_interval(self,
                             inject_next_orders: bool = False):
        """
        * Drivers finish dispatch, return to destinations, and become idle
        * Check if drivers currently serving have completed orders
        * (Optional) Inject orders for time t+1
        * (Optional) Manage driver online/offline status based on statistical distributions
        * Clear orders that have timed out without service

        """
        self.step_driver_status_control()
        # ---------- Dispatch drivers to their destinations. ----------
        if self._dispatch_buf:
            self.step_add_dispatched_drivers(self._dispatch_buf)
            self._dispatch_buf = []


        # ----------  Inject orders for the next time step. ----------
        if inject_next_orders:
            moment = int(self.city_time % self.n_intervals)
            self.step_bootstrap_order_real(self.day_orders[moment])

        self.step_driver_online_offline_nodewise()

        self.step_remove_unfinished_orders()

    def step_stage1_with_max_flow(self, action_tuple, solver=None, return_node_gmv: bool = False):
        """
            Stage 1:

            If both `estimator` and `stateProcessor` are provided, Agent-1 is called to generate dispatch actions and actually move the drivers.
            Then each grid independently runs `simple_order_assign_real()`.

            Returns:

            `reward_local`: *float* — local reward obtained in this stage
            pending_nodes`: *List\[int]* — list of grid IDs that still have uncompleted orders

        """
        reward_local = 0.0
        node_gmv_vec = np.zeros(self.M * self.N, dtype=np.float32)
        total_local_finished = 0
        pending_nodes = []


        if action_tuple:
            save_remove_id = self.step_dispatch_invalid(action_tuple)
            self._dispatch_buf.extend(save_remove_id)

        # -------------Additionally, compute the Linear programming problem.------------------------
        state = self.get_observation()
        driver_map = state[0]
        order_map = state[1]

        drivers_t = np.array(
            [driver_map[ids_1dto2d(grid_id, solver.M, solver.N_cols)] for grid_id in solver.valid_grid_ids])
        orders_t = np.array(
            [order_map[ids_1dto2d(grid_id, solver.M, solver.N_cols)] for grid_id in solver.valid_grid_ids])
        solution, var_map = solver.solve(drivers_t, orders_t)

        print(f"[t={self.city_time:03d}] time orders: {len(self.day_orders[self.city_time])}")
        idle_drivers = self.get_num_idle_drivers()
        print(f"[t={self.city_time:03d}] idle drivers：{idle_drivers}")
        max_match = 0
        # ---------- ② self grid ----------
        for node in self.nodes:
            if node is None: continue
            r_loc, _, fin = node.simple_order_assign_real(self.city_time, self)
            reward_local += r_loc
            total_local_finished += fin
            max_match += fin
            node_gmv_vec[node.get_node_index()] += r_loc
            if node.order_num > 0:
                pending_nodes.append(node.get_node_index())
        print(f"== local_total finished = {total_local_finished}")

        self.neighbor_idle_snapshot = {
            node.get_node_index(): node.idle_driver_num
            for node in self.nodes if node is not None
        }
        self.episode_reward += reward_local
        self.episode_finished_orders += total_local_finished
        if return_node_gmv:
            return reward_local, pending_nodes, node_gmv_vec,np.sum(solution)
        else:
            return reward_local, pending_nodes

    def step_stage2_with_max_flow(self, pending_nodes, solver, return_node_gmv: bool = False):
        """
            Stage 2: Use maximum flow to complete the “neighbor borrowing” dispatch in one step.

            `pending_nodes`: list of grid IDs with remaining orders after Stage 1
            `solver`: your instance of `MaxFlowSolver(mapped_matrix_int, self.nodes)`

            Returns:

            If `return_node_gmv=False`: `reward_total`
            If `return_node_gmv=True`: `(reward_total, node_gmv_vec2)`

        """

        reward_total = 0.0
        node_gmv_vec2 = np.zeros(self.M * self.N, dtype=np.float32)
        self.metrics.reset_step()

        # If there are no pending orders or no idle vehicles, return immediately.
        if not pending_nodes or not hasattr(self, "neighbor_idle_snapshot") or len(self.neighbor_idle_snapshot) == 0:
            print("second round finished:0")
            return (reward_total, node_gmv_vec2) if return_node_gmv else reward_total

        # --- 1) Assemble `drivers_t` and `orders_t` following the order of `solver`’s `valid_grid_ids`. ---
        num_valid = solver.num_valid_grids
        drivers_t = np.zeros(num_valid, dtype=np.int64)
        orders_t = np.zeros(num_valid, dtype=np.int64)

        # Supply: use the snapshot of neighbor idle vehicles saved after Stage-1.
        for gid in solver.valid_grid_ids:
            idx = solver.grid_id_to_idx_map[gid]
            drivers_t[idx] = int(self.neighbor_idle_snapshot.get(gid, 0))


        # Demand: assign values only for `pending_nodes`.
        for nid in pending_nodes:
            j = solver.grid_id_to_idx_map.get(nid, None)
            if j is None:
                continue
            orders_t[j] = int(self.nodes[nid].order_num)

        print(f"orders_t:{orders_t.shape}")

        total_demand_before = int(orders_t.sum())
        print(f"total_demand_before:{total_demand_before}")

        if total_demand_before == 0 or int(drivers_t.sum()) == 0:
            print("second round finished:0")
            return (reward_total, node_gmv_vec2) if return_node_gmv else reward_total

        # --- 2) Call your max\_flow solver. ---
        solution, var_map = solver.solve(drivers_t, orders_t)
        if solution is None or var_map is None:
            print("second round finished:0")
            return (reward_total, node_gmv_vec2) if return_node_gmv else reward_total
        # --- 3) Convert the solution vector into actual “borrow vehicle and assign order” actions. ---
        # Note: solver's var_map looks like {(i_idx, j_idx): var_idx}
        # For each variable, we take the floor to avoid fractional values from LP;
        # rounding can be used instead if desired.

        num_finished_orders = 0

        for (i_idx, j_idx), var_idx in var_map.items():
            amount = int(np.floor(solution[var_idx]))
            if amount <= 0:
                continue

            src_gid = solver.valid_grid_ids[i_idx]  # Grid(s) lending vehicles.
            dst_gid = solver.valid_grid_ids[j_idx]  # Grid(s) receiving orders (pending).

            if src_gid == dst_gid:
                continue

            src_node = self.nodes[src_gid]
            dst_node = self.nodes[dst_gid]
            if (src_node is None) or (dst_node is None):
                continue
            if dst_node.order_num <= 0:
                continue
            if src_node.idle_driver_num <= 0:
                continue

            # Borrow `amount` drivers from `src` to fulfill `amount` orders at `dst`.
            # This function will:
            # Assign `assigned_time` to orders,
            # Decrease `dst.orders`,
            # Deduct from `src.idle_driver_num`,
            # Accumulate GMV, etc.
            given = min(amount, src_node.idle_driver_num, dst_node.order_num)
            if given <= 0:
                continue

            rr = dst_node.utility_assign_orders_neighbor(self, src_node, given)
            reward_total += rr
            node_gmv_vec2[dst_gid] += rr
            num_finished_orders += given

        unserved = max(0, total_demand_before - num_finished_orders)
        if unserved > 0:
            self.metrics.add_unserved_demand(unserved)

        self.episode_reward += reward_total
        self.episode_finished_orders += num_finished_orders

        return (reward_total, node_gmv_vec2) if return_node_gmv else reward_total

    def step_w_o_request(self, dispatch_actions, generate_order=1): # action: [source, destination, nums]
        info = []

        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        save_remove_id = self.step_dispatch_invalid(dispatch_actions)
        # When the drivers go to invalid grid, set them offline.

        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update_w_o_request()

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
        context = self.step_pre_order_assigin(next_state)
        info = [reward_node, context]
        return next_state, reward, info

    def step_assign_order_broadcast_neighbor_reward_update_w_o_request(self):
        """ Consider the orders whose destination or origin is not in the target region
        :param num_layers:
        :param weights_layers_neighbors: [1, 0.5, 0.25, 0.125]
        :return:
        """

        node_reward = np.zeros((len(self.nodes)))
        neighbor_reward = np.zeros((len(self.nodes)))
        # First round broadcast
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0

        print(f"[t={self.city_time:03d}] time orders: {len(self.day_orders[self.city_time])}")
        idle_drivers = self.get_num_idle_drivers()
        print(f"[t={self.city_time:03d}] idle drivers: {idle_drivers}")
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

        print(f"the first round finished:{finished_order_num_node1}")
        self.neighbor_idle_snapshot = {
            node.get_node_index(): node.idle_driver_num
            for node in self.nodes if node is not None
        }
        self.metrics.reset_step()
        # Second round broadcast
        finished_order_num_node_broadcast1 = 0
        before_order_num_node = 0

        # remove the second stage request processing
        # for node in self.nodes:
        #     if node is not None:
        #         if node.order_num != 0:
        #             before_order_num_node += node.order_num
        #             reward_node_broadcast, finished_order_num_node_broadcast \
        #                 = node.simple_order_assign_broadcast_update(self, neighbor_reward)
        #             finished_order_num_node_broadcast1 += finished_order_num_node_broadcast
        #             reward += reward_node_broadcast
        #             self.episode_reward += reward_node_broadcast
        #             finished_order_num += finished_order_num_node_broadcast
        # print(f"第二轮派单前订单数量:{before_order_num_node}")
        # print(f"第二轮派单完成数量:{finished_order_num_node_broadcast1}")
        # print(f"剩余订单数量:{before_order_num_node - finished_order_num_node_broadcast1}")
        node_reward = node_reward + neighbor_reward
        if all_order_num != 0:
            self.order_response_rate = finished_order_num / float(all_order_num)
        else:
            self.order_response_rate = -1

        self.episode_finished_orders += finished_order_num
        sg_s, ud_s = self.metrics.same_grid_contention_step, self.metrics.unserved_demand_step
        print(f"[metric] step {self.city_time:03d}: "
              f"same_grid_contention={sg_s}, unserved_demand={ud_s}")
        return reward, [node_reward, neighbor_reward]

