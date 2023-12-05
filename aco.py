import random
import sys
import threading
from typing import Any

import numpy
import numpy as np
import dataset_prepare as dp
import matplotlib.pyplot as plt
from tqdm import tqdm


def greedy_find(graph):
    ret = [0]
    cost = 0
    while len(ret) < len(graph.data):
        cursor_index = 1
        while True:
            small_index = np.argsort(graph[ret[-1]])[cursor_index]
            if small_index in ret:
                cursor_index = cursor_index + 1
                continue
            small = np.sort(graph[ret[-1]])[cursor_index]
            ret.append(small_index)
            cost = cost + small
            break
    return ret, cost


class ACO(object):
    def __init__(
        self,
        a,
        b,
        p,
        c,
        ants,
        graph,
        greedy_sum,
        epochs=500,
        great_ant_weight=5.0,
        reward_num=0.2,
        cool_batch_limit=30,
        cool_batch_weight_offset=4,
    ):
        self.a = a  # 阿尔法参数
        self.b = b  # 贝塔参数
        self.p = p  # ρ参数
        self.c = c  # 常数参数
        self.ants = ants  # 蚂蚁数量
        self.cities_distance: numpy.ndarray[Any, numpy.dtype] = graph  # 城市矩阵图
        self.greedy_sum = greedy_sum  # 贪婪初始化总代价
        self.count = len(self.cities_distance.data)
        self.pheromone_matrix = np.zeros([self.count, self.count])  # 信息素矩阵
        self.epochs = epochs
        self.generate_default_pheromone_matrix()
        self.batch_no = 1
        self.great_ant_weight = great_ant_weight
        self.reward_num = reward_num
        self.cool_batch_limit = cool_batch_limit  # 冷却 batch size
        self.cool_batch_weight_offset = cool_batch_weight_offset  # 冷冻后赋权值

    def print_search_result(self, cost, map_route):
        print(f"batch: {self.batch_no} | cost:{cost} | map:{map_route}")

    def generate_default_pheromone_matrix(self):
        default_value = self.count / self.greedy_sum
        self.pheromone_matrix = np.full((self.count, self.count), default_value)
        np.fill_diagonal(self.pheromone_matrix, 0)

    def solve(self):
        min_cost = sys.maxsize
        min_route = []
        pbar = tqdm(range(self.epochs), desc="ACO Search")
        cool_batch_size = 0
        cool_wight = 1
        for _ in pbar:
            threads = []
            routes = []
            costs = []

            for _ in range(self.ants):
                thread = threading.Thread(target=self.run_wrapped, args=(routes, costs))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            self.update_pheromones(routes, costs)
            min_index = np.argsort(np.array(costs))[0]
            if costs[min_index] < min_cost:
                min_cost = costs[min_index]
                min_route = routes[min_index]
                cool_batch_size = 0
                cool_wight = 1
            else:
                cool_batch_size = cool_batch_size + 1
            self.batch_no = self.batch_no + 1

            count = int(self.reward_num * len(costs))
            np_costs = np.argsort(np.array(costs))
            reward_routes = []
            reward_costs = []
            if cool_batch_size > self.cool_batch_limit:
                cool_wight = 2
            for _, arg in enumerate(
                np_costs[count * (cool_wight - 1) : count * cool_wight]
            ):
                reward_routes.append(routes[arg])
                reward_costs.append(costs[arg])
            self.update_pheromones(
                reward_routes,
                reward_costs,
                weight=self.great_ant_weight
                ** cool_wight
                ** self.cool_batch_weight_offset,
            )
            if cool_wight > 1:
                cool_wight = 1
                cool_batch_size = 0
            pbar.set_postfix({"cost": min_cost, "cool_batch": cool_batch_size})
        pbar.set_postfix({"cost": min_cost})
        return min_route, min_cost

    def update_pheromones(self, routes, costs, weight=1.0):
        self.pheromone_matrix *= 1 - self.p
        for route, cost in zip(routes, costs):
            pheromone_deposit = self.c / cost
            for i in range(len(route) - 1):
                self.pheromone_matrix[route[i]][route[i + 1]] += (
                    pheromone_deposit * weight
                )
                self.pheromone_matrix[route[i + 1]][route[i]] += (
                    pheromone_deposit * weight
                )

    def run_wrapped(self, routes, costs):
        lock = threading.Lock()
        route, cost = self.run(random.randint(0, self.count - 1))
        with lock:
            routes.append(route)
            costs.append(cost)

    def run(self, start_city):
        route = [start_city]
        cost = 0
        visited = set(route)

        while len(route) < self.count:
            current_city = route[-1]
            probabilities = []

            for i in range(len(self.cities_distance)):
                if i not in visited:
                    pheromone_contribution = (
                        self.pheromone_matrix[current_city][i] ** self.a
                    )
                    distance_contribution = (
                        10 / self.cities_distance[current_city][i]
                    ) ** self.b
                    probabilities.append(pheromone_contribution * distance_contribution)
                else:
                    probabilities.append(0)

            probabilities = np.array(probabilities)
            probabilities /= np.sum(probabilities)

            # 轮盘赌算法
            choose = random.uniform(0.0, 1.0)
            cumulative_prob = 0.0
            next_city = None
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if cumulative_prob > choose:
                    next_city = i
                    break

            route.append(next_city)
            visited.add(next_city)
            cost += self.cities_distance[current_city][next_city]

        # 添加初始的路径和代价
        cost += self.cities_distance[route[-1]][start_city]
        route.append(start_city)

        return route, cost


dp.prepare_dataset("st70.tsp")
array = np.array(dp.generate_graph())
(greedy_map, init_cost) = greedy_find(array)
print(f"以贪婪搜索的方式初始化路径，代价为：{init_cost}，路线为：{greedy_map}")
print("下面开始进行搜索...")
print(f"cost:{init_cost} | map:{greedy_map}")
aco = ACO(
    1,
    2,
    0.62,
    100,
    100,
    graph=array,
    greedy_sum=init_cost,
    great_ant_weight=1.5,
    reward_num=0.3,
    epochs=400,
    cool_batch_limit=30,
    cool_batch_weight_offset=4,
)
aco.update_pheromones([greedy_map], [init_cost], weight=5)
route, cost = aco.solve()

point_maps = dp.get_point_maps()
for city in point_maps:
    plt.scatter(city[0], city[1], color="red")
for i in range(len(route) - 1):
    start_city = point_maps[route[i]]
    end_city = point_maps[route[i + 1]]
    plt.plot([start_city[0], end_city[0]], [start_city[1], end_city[1]], color="blue")
plt.title(f"st70:ACO Result For:{cost}")
plt.show()
