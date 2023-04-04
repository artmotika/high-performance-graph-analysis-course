import json
import pygraphblas as gb
import networkx as nx

__all__ = [
    "read_dot_file",
    "digraph_to_matrix_gb",
    "load_test_json",
    "check_matrix_graph_size",
]

shift = -1


def read_dot_file(file_path: str) -> nx.DiGraph:
    """
    Read dot file and get MultiGraph from networkx
    :param file_path: name of the dot file
    :return: DiGraph (directed graph) from networkx
    """

    return nx.DiGraph(nx.nx_pydot.read_dot(file_path))


def digraph_to_matrix_gb(graph: nx.DiGraph) -> gb.Matrix:
    """
    Convert DiGraph from networkx with integer verticies to boolean adjacency matrix from pygraphblas
    :param graph: DiGraph (directed graph) from networkx
    :return: boolean adjacency matrix from pygraphblas
    """

    size = graph.number_of_nodes() + shift
    Matrix = gb.Matrix.sparse(gb.types.BOOL, size, size)
    for source, target in graph.edges():
        Matrix[int(source) - 1, int(target) - 1] = True

    return Matrix


def load_test_json(path: str):
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


def check_matrix_graph_size(Graph: gb.Matrix) -> int:
    if Graph.square:
        return Graph.ncols
    else:
        raise ValueError("Matrix is not square")
