import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')

from hypertools import Hypertools
from trivia import DEMO_MAPS


def demo_show_full_hypercube():
    hyper = Hypertools()
    print(hyper.nodes)
    print(hyper.full_edges)
    hyper.plot_map(hyper.full_edges)
    plt.show()


def demo_plot_map():
    hyper = Hypertools()
    edges = DEMO_MAPS["save2"]
    hyper.plot_map(edges)
    plt.show()


def demo_transformations():
    hyper = Hypertools()
    print(f"trivial transformation: {hyper.all_transformations[0]}")
    print(f"all transformations: {len(hyper.all_transformations)}")
    transformation = hyper.all_transformations[100]
    edges = DEMO_MAPS["star"]
    hyper.plot_map(edges)
    transformed_edges = hyper.apply_transformation(edges, transformation)
    hyper.plot_map(transformed_edges)
    transformed_edges = hyper.apply_transformation(
        transformed_edges, transformation
    )
    hyper.plot_map(transformed_edges)
    transformed_edges = hyper.apply_transformation(
        transformed_edges, transformation
    )
    hyper.plot_map(transformed_edges)

    node_transformation = {1: 3, 3: 7, 5: 11, 9: 1, 0: 2}
    transformation = hyper.get_full_transformation(node_transformation)
    transformed_edges = hyper.apply_transformation(
        edges, transformation
    )
    hyper.plot_map(transformed_edges)
    transformed_edges = hyper.apply_transformation(
        transformed_edges, transformation
    )
    hyper.plot_map(transformed_edges)

    node_transformation = {1: 3, 3: 1, 5: 11, 9: 7, 0: 2}
    transformation = hyper.get_full_transformation(node_transformation)
    transformed_edges = hyper.apply_transformation(
        edges, transformation
    )
    hyper.plot_map(transformed_edges)

    plt.show()


def demo_rotations_reflections():
    hyper = Hypertools()
    edges = DEMO_MAPS["save2"]
    hyper.plot_map(edges)

    node_transformation = {3: 3, 2: 7, 1: 1, 7: 2, 11: 11}
    transformation = hyper.get_full_transformation(node_transformation)
    transformed_edges = hyper.apply_transformation(edges, transformation)
    hyper.plot_map(transformed_edges)

    node_transformation = {3: 3, 2: 2, 1: 11, 7: 7, 11: 1}
    transformation = hyper.get_full_transformation(node_transformation)
    transformed_edges = hyper.apply_transformation(edges, transformation)
    hyper.plot_map(transformed_edges)

    node_transformation = {3: 3, 2: 11, 1: 1, 7: 7, 11: 2}
    transformation = hyper.get_full_transformation(node_transformation)
    transformed_edges = hyper.apply_transformation(edges, transformation)
    hyper.plot_map(transformed_edges)

    plt.show()


def demo_unraveling():
    hyper = Hypertools()

    edges = DEMO_MAPS["star"]
    hyper.plot_map(edges)

    unraveled = hyper.unravel(edges)
    hyper.plot_map(unraveled)

    plt.show()


def demo_export():
    hyper = Hypertools()
    edges = DEMO_MAPS["snake_0"]
    hyper.plot_map(edges)
    plt.show()
    hyper.export_map("snake_0.txt", edges)


def demo_index():
    hyper = Hypertools()
    hyper.plot_map(hyper.int_to_edges(1521293122))
    hyper.plot_map(hyper.int_to_edges(1494517717))
    hyper.plot_map(hyper.int_to_edges(1531766980))
    hyper.plot_map(hyper.int_to_edges(1562028834))

    hyper.plot_map(hyper.int_to_edges(3976206173))
    hyper.plot_map(hyper.int_to_edges(3925890938))
    hyper.plot_map(hyper.int_to_edges(3960017373))
    hyper.plot_map(hyper.int_to_edges(3927130616))
    hyper.plot_map(hyper.int_to_edges(3893423913))
    plt.show()
    index = 1562028834
    print(f"index: {index}")
    bools = hyper._int_to_bools(index)
    edges = hyper._bools_to_edges(bools)
    print(edges)
    print(hyper.edges_to_int(edges))


def search():
    hyper = Hypertools()
    for start, stop in zip(range(3919920000, 3850000000, -200000),
                           range(3919920000 - 200000, 3850000000 - 200000, -200000)):
        print(start, stop)
        hyper.search_for_trees(range(start, stop, -1))

if __name__ == "__main__":
    demo_transformations()
