import json
import pygraphblas as gb
import numpy as np
import networkx as nx

__all__ = [
    "read_dot_file",
    "digraph_to_bool_matrix_gb",
    "digraph_to_float_matrix_gb",
    "load_test_json_bool_matrix",
    "load_test_json_float_matrix",
    "check_matrix_graph_size_and_type",
]

shift = -1


def read_dot_file(file_path: str) -> nx.DiGraph:
    """
    Read dot file and get MultiGraph from networkx
    :param file_path: name of the dot file
    :return: DiGraph (directed graph) from networkx
    """

    return nx.DiGraph(nx.nx_pydot.read_dot(file_path))


def digraph_to_bool_matrix_gb(graph: nx.DiGraph) -> gb.Matrix:
    """
    Convert DiGraph from networkx with integer verticies to boolean adjacency matrix from pygraphblas
    :param graph: DiGraph (directed graph) from networkx
    :return: boolean adjacency matrix from pygraphblas
    """

    size = graph.number_of_nodes() + shift
    Matrix = gb.Matrix.sparse(gb.BOOL, size, size)
    for source, target in graph.edges():
        Matrix[int(source) - 1, int(target) - 1] = True

    return Matrix


def digraph_to_float_matrix_gb(graph: nx.DiGraph) -> gb.Matrix:
    """
    Convert DiGraph from networkx with integer verticies to float adjacency matrix from pygraphblas
    label has to be named "weight"
    :param graph: DiGraph (directed graph) from networkx
    :return: boolean adjacency matrix from pygraphblas
    """

    size = graph.number_of_nodes() + shift
    Matrix = gb.Matrix.sparse(gb.FP64, size, size)
    for source, target, labels in graph.edges(data=True):
        Matrix[int(source) - 1, int(target) - 1] = float(labels["weight"])

    return Matrix


def load_test_json_bool_matrix(path: str):
    matrix, expected, extra = [], [], {}
    extra["source"] = []

    with open(path, "r") as file:
        json_dict = json.load(file)
        data_chunks = json_dict["test_data"]

    for data in data_chunks:
        matrix.append(gb.Matrix.from_lists(*data["matrix"], V=True, typ=gb.BOOL))
        extra["source"].append(data.get("source"))
        expected.append(data["expected"])

    return (list(zip(matrix, expected)), extra)


def load_test_json_float_matrix(path: str):
    matrix, expected, extra = [], [], {}
    extra["source"] = []

    with open(path, "r") as file:
        json_dict = json.load(file)
        data_chunks = json_dict["test_data"]

    for data in data_chunks:
        matrix.append(
            gb.Matrix.from_lists(
                data["matrix"][0], data["matrix"][1], V=data["matrix"][2], typ=gb.FP64
            )
        )
        extra["source"].append(data.get("source"))
        expected.append(data["expected"])

    return (list(zip(matrix, expected)), extra)


def check_matrix_graph_size_and_type(Graph: gb.Matrix, expected_type: type) -> int:
    if Graph.type != expected_type:
        raise ValueError(
            f"Adjacency matrix must have {expected_type} type, provided type: {Graph.type}"
        )
    if Graph.square:
        return Graph.ncols
    else:
        raise ValueError("Matrix is not square")


def check_matrix_graph_unoriented(Graph: gb.Matrix):
    lower_mtx = Graph.tril(-1)
    upper_mtx = Graph.triu(1)
    if (
        lower_mtx.transpose().iseq(upper_mtx)
        and np.size(Graph.diag().nonzero().npV) == 0
    ):
        return
    else:
        raise ValueError("Matrix is not symmetric, so graph isn't unoriented")
