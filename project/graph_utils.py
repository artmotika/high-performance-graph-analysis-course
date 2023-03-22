import json
import pygraphblas as gb
import networkx as nx

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
    Convert DiGraph from networkx to boolean adjacency matrix from pygraphblas
    :param graph: DiGraph (directed graph) from networkx
    :return: boolean adjacency matrix from pygraphblas
    """

    node_to_idx = {}
    size = graph.number_of_nodes() + shift
    Matrix = gb.Matrix.sparse(gb.types.BOOL, size, size)
    for i, s in enumerate(graph.nodes, start=shift):
        node_to_idx[s] = i
    for source, target in graph.edges():
        Matrix[node_to_idx[source], node_to_idx[target]] = True

    return Matrix


def load_test_json(path: str):
    matrix, source, expected = [], [], []

    with open(path, "r") as file:
        json_dict = json.load(file)
        data_chunks = json_dict["test_data"]

        for data in data_chunks:
            matrix.append(gb.Matrix.from_lists(*data["matrix"], V=True, typ=gb.BOOL))
            source.append(data["source"])
            expected.append(data["expected"])

    return list(zip(matrix, source, expected))
