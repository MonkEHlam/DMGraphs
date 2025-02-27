import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
from sympy import *
import COLORS


def find_excentries(metr):
    """2 лаба. Возвращает список эксцентрисететов графа."""
    output = []
    for line in metr:
        output.append(max(line))
    return output


def print_matrix(matrix: list[list[object]]):
    """Вывод матрицы в консоль."""
    for line in matrix:
        print(*line)


def matrix_addition(m1, m2):
    """Складывает две матрицы.
    Не имеет проверки на соответсвие размерностей матриц."""
    output_matrix = [[0 for i in range(len(m1))] for i in range(len(m1[0]))]
    for i in range(len(m1)):
        for j in range(len(m1)):
            output_matrix[i][j] = m1[i][j] + m2[i][j]
    return output_matrix


def matrix_multiplication(m1, m2):
    """Перемножает две матрицы.
    Не имеет проверки на соответсвие размерностей матриц."""
    output_matrix = [[0 for j in range(len(m1))] for i in range(len(m2[0]))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
                output_matrix[i][j] += m1[i][k] * m2[k][j]
    return output_matrix


def is_matrixes_equals(m1, m2):
    """Проверяет равенство двух матриц."""
    try:
        for i in range(len(m1)):
            for j in range(len(m1[i])):
                if m1[i][j] != m2[i][j]:
                    return False
        return True
    except Exception:
        return False


def find_difference(size, exp):
    """3 лаба. Инвертирует упрощенное выражение из алгоритма Магу-Вейсмана и
    возвращает список списков, в котором каждый список - неинцендентные вершины графа"""
    output = []
    vari = set(range(size))
    exp = str(exp).replace("x", "").split(" | ")
    for i in range(len(exp)):
        exp[i] = exp[i].replace("(", "").replace(")", "")
        exp[i] = set(int(j) for j in exp[i].split(" & "))
        exp[i] = vari.difference(exp[i])


    while len(exp) != 0:
        output.append(max(exp))
        max_len = max(exp)
        for i in range(len(exp)):
            exp[i] = exp[i].difference(max_len)
        exp = [i for i in exp if i]
    return output


def create_skeleton_sm_matrix(sm_matrix: list[list[int]]):
    """3 лаба. Создет матрицу смежности, где все ребра еденичные"""
    skeleton_sm_matrix = sm_matrix.copy()
    for i in range(len(sm_matrix)):
        for j in range(len(sm_matrix[i])):
            if sm_matrix[i][j] > 1:
                skeleton_sm_matrix[i][j] = 1
    return skeleton_sm_matrix


def create_sm_matrix(amount_of_nodes: int, arguments: list):
    """Создает матрицу смежности.
    В списке arguments олжны находиться строковые коды типа графа:
    Полный граф - "1"
    Граф с петлями - "2"
    Граф с кратными ребрами - "3"
    При отсутствии аргументов в arguments формируется матрица смежности простого графа.
    Возвращает матрицу смежности и количество ребер в графе."""
    sm_matrix = [[0 for j in range(amount_of_nodes)] for i in range(amount_of_nodes)]
    amount_of_edges = 0
    for i in range(amount_of_nodes):
        for j in range(i + 1, amount_of_nodes):
            num = random.randint(
                1 if "1" in arguments else 0, 5 if "3" in arguments else 1
            )
            sm_matrix[i][j] = num
            sm_matrix[j][i] = num
            amount_of_edges += num
        sm_matrix[i][i] = (
            0
            if "2" not in arguments
            else random.randint(0, 5 if "3" in arguments else 1)
        )
        amount_of_edges += sm_matrix[i][i]

    return sm_matrix, amount_of_edges


def create_in_matrix(
    sm_matrix: list[list[int]], amount_of_nodes: int, amount_of_edges: int
):
    """Создает матриицу инцедентности.
    Принимает в качестве параметров матрицу смежности, количество вершин, количество ребер
    Возвращает Матрицу инцедентности"""
    in_matrix = [[0 for j in range(amount_of_edges)] for i in range(amount_of_nodes)]
    k = 0
    for i in range(amount_of_nodes):
        for j in range(i, amount_of_nodes):
            if sm_matrix[i][j]:
                for m in range(sm_matrix[i][j]):
                    in_matrix[i][k] += 1
                    in_matrix[j][k] += 1
                    k += 1
    return in_matrix


def create_metric_matrix(sm_matrix):
    """Создает матрицу метрики на основе матрицы смежности."""
    metric_matrix = [
        [0 if i == j else None for i in range(len(sm_matrix))]
        for j in range(len(sm_matrix))
    ]
    S_matrix = matrix_addition(
        sm_matrix,
        [
            [1 if i == j else 0 for j in range(len(sm_matrix))]
            for i in range(len(sm_matrix))
        ],
    )

    k = 1
    while any(None in x for x in metric_matrix):
        temp_metric_matrix = copy.deepcopy(metric_matrix)
        Sk_matrix = S_matrix
        for i in range(1, k):
            Sk_matrix = matrix_multiplication(Sk_matrix, S_matrix)
        for i in range(len(metric_matrix)):
            for j in range(i + 1, len(metric_matrix[i])):
                if metric_matrix[i][j] is None and Sk_matrix[i][j] != 0:
                    metric_matrix[i][j], metric_matrix[j][i] = k, k
        if is_matrixes_equals(temp_metric_matrix, metric_matrix):
            break
        k += 1

    for line in metric_matrix:
        for j in range(len(line)):
            if line[j] is None:
                line[j] = float("inf")
    return metric_matrix


def create_weighted_matrix(amount_of_nodes):
    """Создает матрицу смежности взвешанного графа."""
    weighted_matrix = [
        [0 for j in range(amount_of_nodes)] for i in range(amount_of_nodes)
    ]
    for i in range(amount_of_nodes):
        for j in range(i + 1, amount_of_nodes):
            weighted_matrix[i][j] = random.randint(10, 30)
            weighted_matrix[i][j] = weighted_matrix[i][j] * random.randint(0, 1)
            weighted_matrix[j][i] = weighted_matrix[i][j]
    return weighted_matrix


def find_combinations(sm_matrix) -> list[list[int]]:
    # Преобразует символьное выражения для алгоритма Магу-Вейсмана
    skelet = create_skeleton_sm_matrix(sm_matrix)
    exp = ""
    variables = symbols(
        "x0:%d" % len(skelet)
    )  # получаем вершины графа в качестве переменных SimPy в виде х1, х2, xn
    # Создаем выражение формата (x1|x2)&(x2|x3)... где в каждой скобке нахлдятся вершины, которые пересекаются
    for i in range(len(skelet)):
        for j in range(i + 1, len(skelet)):
            if skelet[i][j] == 1:
                exp = (
                    exp + f" & ({variables[i]} | {variables[j]})"
                    if exp != ""
                    else f"({variables[i]} | {variables[j]})"
                )
    fin = to_dnf(exp, simplify=True, force=True)  # Все преобразовывается и упрощается.
    print(exp)
    print(fin)
    return find_difference(len(variables), fin)


def get_colors(sm_matrix):
    """3 лаба. Создаёт словарь, где каждому набору вершин соответствует свой цвет."""
    color_map = {}
    combinations = find_combinations(sm_matrix)
    for i in range(len(combinations)):
        for node in combinations[i]:
            color_map[node + 1] = color_map.get(node + 1, COLORS.COLORS[i])
    print(len(combinations))
    return color_map


def find_graph_radius(metric_matrix):
    """2 лаба. Возвращает радиус графа."""
    return min(find_excentries(metric_matrix))


def find_graph_diametr(metric_matrix):
    """2 лаба. Возвращает диаметр графа."""
    return max(find_excentries(metric_matrix))


def find_graph_center(metric_matrix):
    """2 лаба. Возвращает множество индексов вершин графа, которые являются радиусами"""
    center = set()
    radius = find_graph_radius(metric_matrix)
    for i in metric_matrix:
        if radius in i:
            center.add(str(i.index(radius) + 1))
    return center


def find_graph_distant_points(metric_matrix):
    """2 лаба. Возвращает множество индексов вершин графа, которые являются диаметрами"""
    distant_points = set()
    diametr = find_graph_diametr(metric_matrix)
    for i in metric_matrix:
        if diametr in i:
            distant_points.add(str(i.index(diametr) + 1))
    return distant_points


def dijkstra(sm_matrix, first_node):
    """Реализует алгоритм Дейкстры.
    Возращает список расстояний от вершины start до остальных вершин и
    словарь, в котором ключ -- вершина, до которой простроен маршрут, а объект -- список вершин, соответствующих маршруту.
    {aim_node: [node_num0, node_num2, ..., node_num_N]}"""

    amount_of_nodes = len(sm_matrix)
    visited = [False] * amount_of_nodes
    distances = [
        float("inf")
    ] * amount_of_nodes
    distances[first_node] = 0
    ways = {first_node: [first_node]}

    while True:
        min_distance = float("inf")
        min_node = None

        for i in range(amount_of_nodes):
            if not visited[i] and distances[i] < min_distance:
                min_distance = distances[i]
                min_node = i

        if min_node is None:
            break
        visited[min_node] = True

        for i in range(amount_of_nodes):
            if not visited[i] and sm_matrix[min_node][i] != 0:
                new_distance = distances[min_node] + sm_matrix[min_node][i]
                if new_distance < distances[i]:
                    distances[i] = new_distance
                    ways[i] = ways[min_node] + [i]

    return distances, ways


def draw_matrix(sm_matrix, is_oriented=False):
    """Выводит граф, полученный из матрицы смежности, через GUI matplotlib"""
    g = nx.convert_node_labels_to_integers(nx.Graph(np.array(sm_matrix)), first_label=1)
    pos = nx.circular_layout(g)
    if not is_oriented:
        colors = get_colors(sm_matrix)  # Получаем цвета
        nx.draw(
            g,
            pos=pos,
            with_labels=True,
            font_weight="bold",
            node_color=[colors[node] for node in g.nodes()],  # Используем цвета
        )
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels=nx.get_edge_attributes(g, name="weight"),
            label_pos=0.5,
            font_color="red",
            font_size=8,
            font_weight="bold",
        )
    else:
        nx.draw(g, pos, with_labels=True, font_weight="bold")
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels=nx.get_edge_attributes(g, name="weight"),
            label_pos=0.5,
            font_color="red",
            font_size=8,
            font_weight="bold",
        )
    plt.show()


amount_of_nodes = int(input("Введите количество вершин графа: "))
args = input(
    "Введите коды особенностей графа\nПолный граф: 1\nГраф с петлями: 2\nГраф с кратными ребрами: 3\nВзвешенный граф: 4\n"
)
if args:
    args = args.split()
else:
    args = []
if "4" in args:
    weighted_matrix = create_weighted_matrix(amount_of_nodes)
    l, r = dijkstra(weighted_matrix, int(input("Введите начальную вершину: ")) - 1)
    for i in range(len(l)):
        if l[i] == float("inf"):
            l[i] = "None"
    print(*l)
    for key in sorted(r.keys()):
        print(f"{key + 1}:{[x + 1 for x in r[key]]}")

    draw_matrix(weighted_matrix, True)

else:
    sm_matrix, amount_of_edges = create_sm_matrix(amount_of_nodes, args)
    metric_matrix = create_metric_matrix(sm_matrix)
    print(sum(sm_matrix[1]))
    in_matrix = create_in_matrix(sm_matrix, amount_of_nodes, amount_of_edges)
    print("Матрица Смежности")
    print_matrix(sm_matrix)
    print("\nМатрица Инцедентности")
    print_matrix(in_matrix)
    print("\nМатрица Метрики")
    print_matrix(metric_matrix)
    print("Радиус: " + str(find_graph_radius(metric_matrix)))
    print("Диаметр: ", str(find_graph_diametr(metric_matrix)))
    print(
        "Перифирийные точки\n"
        + " ".join(list(find_graph_distant_points(metric_matrix)))
    )
    print("Центры\n" + " ".join(list(find_graph_center(metric_matrix))))
    draw_matrix(sm_matrix)

    for i in range(1001):
        if i % 3 == 0 and i % 5 != 0 and sum([int(ch) for ch in str(i)]) < 10:
            print(i)
