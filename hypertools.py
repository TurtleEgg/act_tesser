import enum
import json
import matplotlib.pyplot as plt

from collections import Counter
from contextlib import suppress
from itertools import chain, combinations, compress, permutations, product, repeat
from math import cos, sin, radians
from random import choice, sample

from trivia import CUBES, FULL_INTERSECTIONS


class IncorrectNodeTransformationError(Exception):
    """По заданным точкам нельзя восстановить преобразование."""


class Label(str, enum.Enum):
    has_cubes = "has_cubes"
    has_no_cubes = "has_no_cubes"


class Hypertools:
    def __init__(self):
        self.DIMENSIONS = 4
        self.LEN = 1
        self.MAX_COOR = self.LEN

        self.nodes, self.coors = self._create_nodes()
        self.full_edges = self._create_full_edges()
        self.full_directed_edges = self._get_directed_edges(self.full_edges)
        self.all_transformations = self._get_all_transformations()

        self.directed_intersections = self._get_directed_intersections(
            FULL_INTERSECTIONS
        )

    def _create_nodes(self) -> tuple[dict, dict]:
        """Определение координат всех вершин гиперкуба."""
        nodes = {}
        coors = {}
        for index, coordinates in enumerate(product(range(self.LEN + 1), repeat=4)):
            x, y, z, t = coordinates
            nodes[index] = (x, y, z, t)
            coors[(x, y, z, t)] = index

        return nodes, coors

    def _create_full_edges(self) -> list[tuple]:
        """Получение всех рёбер гиперкуба."""
        full_edges = set()
        for index, node in self.nodes.items():
            for x_i in range(self.DIMENSIONS):
                neighbour = list(node)
                if node[x_i] > 0:
                    neighbour[x_i] -= 1
                elif node[x_i] < self.MAX_COOR:
                    neighbour[x_i] += 1
                neighbour_index = self.coors[tuple(neighbour)]
                full_edges.add(tuple(sorted([index, neighbour_index])))

        return sorted(full_edges)

    def _get_directed_edges(self, edges: list[tuple]) -> dict:
        """Получение направленного графа из ненаправленного."""
        directed_edges = {node: [] for node in self.nodes}
        for V1, V2 in edges:
            directed_edges[V1].append(V2)
            directed_edges[V2].append(V1)
        return directed_edges

    def _get_directed_intersections(self, intersections: list[tuple]) -> dict:
        """Получение для каждого ребра списка рёбер, с которыми оно
        пересекается на плоской проекции."""
        directed_intersections = {edge: [] for edge in self.full_edges}
        for E1, E2 in intersections:
            directed_intersections[E1].append(E2)
            directed_intersections[E2].append(E1)
        return directed_intersections

    def _go_down_one_layer(
        self,
        prev_layer: list,
        prev_layer_destination: list,
        this_layer: list,
        node_transformation: dict,
    ) -> tuple[set, dict]:
        """Проход одного слоя при восстановлении полного преобразования."""
        next_layer = set()
        for source1, source2 in combinations(this_layer, 2):
            for source_neighbour in set(self.full_directed_edges[source1]) & set(
                self.full_directed_edges[source2]
            ):
                if (
                    source_neighbour not in prev_layer
                    and source_neighbour not in node_transformation
                ):
                    for destination_neighbour in set(
                        self.full_directed_edges[node_transformation[source1]]
                    ) & set(self.full_directed_edges[node_transformation[source2]]):
                        if destination_neighbour not in prev_layer_destination:
                            node_transformation[
                                source_neighbour
                            ] = destination_neighbour
                            next_layer.add(source_neighbour)
                            break
                    break

        return next_layer, node_transformation

    def _get_node_transformations(self, main_node_transformation: dict):
        """Получение всех преобразований с одной зафиксированной вершиной."""
        transformations = []
        for source, destination in main_node_transformation.items():
            fellow_sources = self.full_directed_edges[source]
            fellow_destinations = self.full_directed_edges[destination]
            for fellow_destinations_option in permutations(fellow_destinations):
                transformation = main_node_transformation.copy()
                for fellow_source, fellow_destination in zip(
                    fellow_sources, fellow_destinations_option
                ):
                    transformation[fellow_source] = fellow_destination
                transformations.append(self.get_full_transformation(transformation))

        return transformations

    def _get_all_transformations(self):
        """Получение всех возможных преобразований гиперкуба
        как твёрдого тела (повороты, отражения)."""
        transformations = []
        for node1, node2 in product(self.nodes[0], self.nodes):
            transformations.extend(self._get_node_transformations({node1: node2}))

        return transformations

    def _get_xy(self, coordinates: tuple) -> tuple:
        """Вычисление координат вершины гиперкуба на плоской проекции."""
        fi0 = 90  # deg
        dfi = -180 / self.DIMENSIONS  # deg
        fi = [fi0 + i * dfi for i in range(self.DIMENSIONS)]

        x, y = 0, 0
        for i in range(self.DIMENSIONS):
            x = x + cos(radians(fi[-i - 1])) * coordinates[i]
            y = y + sin(radians(fi[-i - 1])) * coordinates[i]

        return x, y

    @staticmethod
    def _trim_by_nones(elements: list) -> list:
        """Удаление None-элементов в конце списка."""
        trimming_element = next(
            (
                len(elements) - i
                for i, element in enumerate(elements[::-1])
                if element or element == 0
            ),
            len(elements),
        )
        return elements[:trimming_element]

    @staticmethod
    def _sort_edges(edges: list[tuple]) -> list[tuple]:
        """Упорядочивание списка рёбер."""
        return sorted([tuple(sorted([V1, V2])) for V1, V2 in edges])

    def _count_intersections(self, edges: list[tuple]) -> int:
        """Подсчёт числа пересечений на плоской проекции
        для заданного набора рёбер."""
        count = 0
        edges = self._sort_edges(edges)
        for edge in edges:
            for counterpart in self.directed_intersections[edge]:
                if counterpart in edges:
                    count += 1
        return count // 2

    @staticmethod
    def _int_to_bools(num: int):
        bin_string = format(num, "032b")
        return [x == "1" for x in bin_string[::-1]]

    @staticmethod
    def _bools_to_int(bools: list[bool]) -> int:
        chars = [str(int(element)) for element in bools]
        return int("".join(reversed(chars)), 2)

    def _bools_to_edges(self, bools: list[bool]) -> list[tuple]:
        return [edge for edge, exists in zip(self.full_edges, bools) if exists]

    def _edges_to_bools(self, edges: list[tuple]) -> list[bool]:
        return [edge in edges for edge in self.full_edges]

    def int_to_edges(self, index: int) -> list:
        bools = self._int_to_bools(index)
        return list(compress(self.full_edges, bools))

    def edges_to_int(self, edges: list[tuple]) -> int:
        return self._bools_to_int(self._edges_to_bools(edges))

    def _take_into_account(
        self, index, edges: list[tuple], counter: Counter, checked: set
    ) -> tuple:
        """Учёт карты (запись в счётчик и файл)."""
        label = Label.has_cubes if self.has_cubes(edges) else Label.has_no_cubes
        checked.add(index)
        with open(f"{label}_unique.txt", "a") as file:
            file.write(f"{index}\n")
        with open(f"{label}.txt", "a") as file:
            for transformation in self.all_transformations:
                index_transformed = self.edges_to_int(
                    self.apply_transformation(edges, transformation)
                )
                if index_transformed not in checked:
                    counter[label.value] += 1
                    file.write(f"{index_transformed}\n")
                    checked.add(index_transformed)

        return counter, checked

    def plot_map(self, edges: list[tuple]):
        """Отображение гиперкуба на экране."""
        plt.figure()
        for edge in edges:
            point1 = self.nodes[edge[0]]
            point2 = self.nodes[edge[1]]
            x1, y1 = self._get_xy(point1)
            x2, y2 = self._get_xy(point2)
            plt.plot((x1, x2), (y1, y2), marker="o")

    def export_map(
        self, file_name: str, edges: list[tuple], special_group: list[int] | None = None
    ):
        """Экспорт карты в файл."""
        directed_edges = self._get_directed_edges(edges)
        pointarrs = {}
        if not special_group:
            special_group = sample(range(15), 12)
            special_group.append(choice(range(1, 28)))
        target_group = special_group[:10]
        exclamation = special_group[10]
        question_node = special_group[11]
        question_number = special_group[12]

        for index, node in self.nodes.items():
            bool_neighbours = []
            index_neighbours = []
            dimensions_range = range(self.DIMENSIONS - 1, -1, -1)
            additions_iterator = chain(
                repeat(1, self.DIMENSIONS), repeat(-1, self.DIMENSIONS)
            )
            for x_i, addition in zip(
                chain(dimensions_range, dimensions_range), additions_iterator
            ):
                neighbour = list(node)
                neighbour[x_i] += addition
                neighbour_index = self.coors.get(tuple(neighbour), None)
                index_neighbours.append(neighbour_index)
                if neighbour_index in directed_edges[index]:
                    bool_neighbours.append(0)
                else:
                    bool_neighbours.append(1)

            index_neighbours = self._trim_by_nones(index_neighbours)
            pointarrs[index] = [
                target_group.index(index) + 1 if index in target_group else 0,
                bool_neighbours,
                index_neighbours,
                question_number if index == question_node else None,
                0 if index == exclamation else None,
            ]
        dimensions_value = 4
        dimensions_w_value = 2
        maze_is_finished = False
        target_value = 1
        way_shift = 4
        way_shift_aim = 4
        player_point = 0

        with open(file_name, "w") as file:
            json.dump(
                [
                    dimensions_value,
                    dimensions_w_value,
                    maze_is_finished,
                    target_value,
                    way_shift,
                    way_shift_aim,
                    player_point,
                    target_group,
                    list(pointarrs.values()),
                ],
                file,
            )

    def import_map(self, file_name: str) -> tuple:
        """Импорт карты из файла."""
        with open(file_name, "r") as file:
            data = json.load(file)
        special_group = data[7]
        pointarrs = data[8]

        edges = set()
        exclamation = None
        question_node = None
        question_number = None
        for index, element in enumerate(pointarrs):
            bool_neighbours = element[1]
            dimensions_range = range(self.DIMENSIONS - 1, -1, -1)
            dimensions_range_1 = range(2 * self.DIMENSIONS - 1, self.DIMENSIONS - 1, -1)
            dimensions = chain(dimensions_range, dimensions_range)
            bool_numbers = chain(dimensions_range, dimensions_range_1)
            additions = chain(repeat(1, self.DIMENSIONS), repeat(-1, self.DIMENSIONS))
            for x_i, bool_i, addition in zip(dimensions, bool_numbers, additions):
                neighbour = list(self.nodes[index])
                neighbour[-x_i - 1] += addition
                neighbour_index = self.coors.get(tuple(neighbour), None)
                if not bool_neighbours[bool_i]:
                    edges.add(tuple(sorted([index, neighbour_index])))
            if element[3]:
                question_node = index
                question_number = element[3]
            if element[4] is not None:
                exclamation = index

        special_group.append(exclamation)
        special_group.append(question_node)
        special_group.append(question_number)

        return list(edges), special_group

    def get_full_transformation(self, node_transformation: dict):
        """Восстановление преобразования карты по преобразованию вершины
        и 4 её смежных вершин. При этом гиперкуб преобразуется как
        твёрдое тело (допустимы повороты, отражения).
        """
        prev_layer = list(node_transformation.copy())
        prev_layer_destination = list(node_transformation.values())
        this_layer = list(node_transformation.copy())
        something_left = True
        while something_left:
            next_layer, node_transformation = self._go_down_one_layer(
                prev_layer, prev_layer_destination, this_layer, node_transformation
            )
            if not next_layer:
                something_left = False
            this_layer = next_layer
            prev_layer = list(node_transformation.keys())
            prev_layer_destination = list(node_transformation.values())
        if len(node_transformation) < 16:
            raise IncorrectNodeTransformationError
        return node_transformation

    @staticmethod
    def apply_transformation(edges: list[tuple], transformation: dict) -> list:
        """Получение результата преобразования для заданной карты."""
        return [
                tuple(sorted([transformation[V1], transformation[V2]]))
                for V1, V2 in edges
            ]


    def unravel(self, edges: list[tuple]) -> list:
        """Распутывание карты (минимизация числа пересечений
        на плоской проекции)."""
        min_count = self._count_intersections(edges)
        unraveling = self.all_transformations[0]
        for transformation in self.all_transformations:
            count = self._count_intersections(
                self.apply_transformation(edges, transformation)
            )
            if count < min_count:
                min_count = count
                unraveling = transformation

        return self.apply_transformation(edges, unraveling)

    def unravel_file(self, file_name: str):
        """Распутывание карты из файла с сохранением в другой файл
        (минимизация числа пересечений на плоской проекции)."""
        edges, special_group = self.import_map(file_name)
        self.plot_map(edges)
        self.plot_map(self.unravel(edges))
        plt.show()
        self.export_map(
            f"{file_name[:-4]}_unraveled{file_name[-4:]}", edges, special_group
        )

    def is_spanning_tree(self, edges: list[tuple]) -> bool:
        """Является ли карта деревом."""
        directed_edges = self._get_directed_edges(edges)
        visited = []

        def go_down(node, previous):
            if node in visited:
                return False
            visited.append(node)
            for adjacent in directed_edges[node]:
                if adjacent == previous:
                    continue
                if not go_down(adjacent, node):
                    return False
            return True

        return len(visited) >= len(self.nodes) if go_down(0, None) else False

    def has_cubes(self, edges: list[tuple]) -> bool:
        """Есть ли в карте подграфы, покрывающие куб."""
        for cube in CUBES:
            cube_edges = list(filter(lambda x: x[0] in cube and x[1] in cube, edges))
            if len(cube_edges) == 7:
                return True
        return False

    def search_for_trees(self, search_range: range):
        """Поиск карт-покрывающих деревьев гиперкуба.

        Максимальный индекс, соответствующий дереву - 3981265480."""
        counter = Counter()
        checked = set()
        for label in Label:
            with suppress(FileExistsError):
                with open(f"{label}.txt", "x"):
                    pass
            with open(f"{label}.txt", "r") as file:
                data = list(map(lambda x: int(x[:-1]), file.readlines()))
            checked.update(data)
            counter[label.value] = len(data)

        print(counter)
        for index in search_range:
            if index in checked:
                continue
            bools = self._int_to_bools(index)
            if bools.count(True) != 15:
                continue
            edges = self.int_to_edges(index)
            if self.is_spanning_tree(edges):
                counter, checked = self._take_into_account(
                    index, edges, counter, checked
                )
        print(counter)


def sort_files():
    """Упорядочивание индексов в файлах."""
    with open("has_cubes.txt", "r") as file_has_cubes:
        has_cubes = set(list(map(lambda x: int(x[:-1]), file_has_cubes.readlines())))
    with open("has_no_cubes.txt", "r") as file_has_no_cubes:
        has_no_cubes = set(map(lambda x: int(x[:-1]), file_has_no_cubes.readlines()))

    with open("has_cubes.txt", "w") as file_has_cubes:
        file_has_cubes.writelines(
            map(lambda x: f"{x}\n", sorted(list(has_cubes), reverse=True))
        )
    with open("has_no_cubes.txt", "w") as file_has_no_cubes:
        file_has_no_cubes.writelines(
            map(lambda x: f"{x}\n", sorted(list(has_no_cubes), reverse=True))
        )
