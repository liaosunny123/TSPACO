import math

tsp = {}


def prepare_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                tsp[parts[0]] = (float(parts[1]), float(parts[2]))


def get_distance_between(a, b):
    a, b = str(a), str(b)
    return math.sqrt((tsp[a][0] - tsp[b][0]) ** 2 + (tsp[a][1] - tsp[b][1]) ** 2)


def get_all_distance_map(a):
    ret = []
    for key in tsp:
        if key != a:
            ret.append(get_distance_between(a, key))
    return ret


def get_all_distance_map_include_self(a):
    ret = []
    for key in tsp:
        ret.append(get_distance_between(a, key))
    return ret


def generate_graph():
    graph = []
    for i in range(get_count()):
        graph.append(get_all_distance_map_include_self(i + 1))
    return graph


def get_count():
    return len(tsp.items())


def get_point(no):
    return tsp[no][0], tsp[no][1]


def get_point_maps():
    ret = []
    for i in range(get_count()):
        ret.append(get_point(str(i + 1)))
    return ret
