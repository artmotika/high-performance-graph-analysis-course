import networkx as nx
import pygraphblas as gb

__all__ = ["read_dot_file", "digraph_to_matrix_gb"]


def read_dot_file(file_path: str) -> nx.DiGraph:
    """
    Read dot file and get MultiGraph from networkx

    :param file_name: name of the dot file
    :return: DiGraph (directed graph) from networkx
    """

    return nx.DiGraph(nx.nx_pydot.read_dot(file_path))


def digraph_to_matrix_gb(graph: nx.DiGraph) -> gb.Matrix:
    """
    Convert DiGraph from networkx to boolean adjacency matrix from pygraphblas

    :param graph: DiGraph (directed graph) from networkx
    :return: boolean adjacency matrix from pygraphblas
    """

    size = graph.number_of_nodes()
    Matrix = gb.sparse(gb.types.BOOL, size, size)
    for i, s in enumerate(graph.nodes):
        node_to_idx[s] = i
    for source, target in graph.edges():
        Matrix[node_to_idx[source], node_to_idx[target]] = True

    return Matrix
