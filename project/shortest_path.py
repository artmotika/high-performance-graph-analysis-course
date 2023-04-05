import pygraphblas as gb
import numpy as np
from project.graph_utils import check_matrix_graph_size_and_type

__all__ = ["bellman_ford", "msbellman_ford", "floyd_warshall"]


def bellman_ford(Graph: gb.Matrix, source: int) -> np.array:
    """
        Belman-Ford algorithm on matrices

        :param Graph: the adjacency matrix of a graph with float weights values
        :param source: the start vertex to calculate the shortest path to the remaining vertices
        :return: a numpy array where the value at index n is equal to the length of the shortest path
         from source vertex to the vertex with number n (If the vertex is unreachable, then the vertex
    of the corresponding cell is float('inf'))
    """
    size = check_matrix_graph_size_and_type(Graph, gb.FP64)
    dist = gb.Vector.sparse(gb.FP64, size)
    Graph += gb.Matrix.identity(gb.FP64, size, value=0.0)
    dist[source] = 0.0
    for i in range(size - 1):
        dist.vxm(
            Graph,
            semiring=gb.FP64.min_plus,
            out=dist,
            desc=gb.descriptor.R,
        )
    temp_dist = dist.dup()
    dist.vxm(
        Graph,
        semiring=gb.FP64.min_plus,
        out=dist,
        desc=gb.descriptor.R,
    )
    if temp_dist.isne(dist):
        raise ValueError("Matrix (graph) has loops with negative numbers")

    dist.assign_scalar(float("inf"), mask=dist, desc=gb.descriptor.S & gb.descriptor.C)
    return dist.npV


def msbellman_ford(Graph: gb.Matrix, source: list[int]) -> list[tuple[int, np.array]]:
    """
    Multy-source Belman-Ford algorithm on matrices

    :param Graph: the adjacency matrix of a graph with float weights values
    :param source: the list of start vertices to calculate the shortest path to the remaining vertices
    :return: an array of pairs: a vertex, and an array where for each vertex the distance to it
     from the specified one. If the vertex is not reachable, then the value of the corresponding cell is float('inf')
    """
    size = check_matrix_graph_size_and_type(Graph, gb.FP64)
    n_ms = len(source)
    dist = gb.Matrix.sparse(gb.FP64, n_ms, size)
    Graph += gb.Matrix.identity(gb.FP64, size, value=0.0)
    for i in range(n_ms):
        dist[source[i], source[i]] = 0.0
    for i in range(size - 1):
        dist.mxm(
            Graph,
            semiring=gb.FP64.min_plus,
            out=dist,
            desc=gb.descriptor.R,
        )
    temp_dist = dist.dup()
    dist.mxm(
        Graph,
        semiring=gb.FP64.min_plus,
        out=dist,
        desc=gb.descriptor.R,
    )
    if temp_dist.isne(dist):
        raise ValueError("Matrix (graph) has loops with negative numbers")

    dist.assign_scalar(float("inf"), mask=dist, desc=gb.descriptor.S & gb.descriptor.C)
    return [(source[i], dist[i].npV) for i in range(n_ms)]


def floyd_warshall(Graph: gb.Matrix) -> list[tuple[int, np.array]]:
    """
    Floyd-Warshall algorithm on matrices where is the shortest distance from each vertex to each

    :param Graph: the adjacency matrix of a graph with float weights values
    :return: an array of pairs: a vertex, and an array where for each vertex the distance to it
     from the specified one. If the vertex is not reachable, then the value of the corresponding cell is float('inf')
    """
    size = check_matrix_graph_size_and_type(Graph, gb.FP64)
    Graph += gb.Matrix.identity(gb.FP64, size, value=0.0)
    result_mtx = Graph.dup()

    for k in range(size):
        col = result_mtx.extract_matrix(col_index=k)
        row = result_mtx.extract_matrix(row_index=k)
        col.mxm(row, semiring=gb.FP64.min_plus, out=result_mtx, accum=gb.FP64.min)

    if result_mtx.diag().reduce_float(accum=gb.FP64.min) < 0:
        raise ValueError("Matrix (graph) has loops with negative numbers")

    result_mtx.assign_scalar(
        float("inf"), mask=result_mtx, desc=gb.descriptor.S & gb.descriptor.C
    )
    return [(i, result_mtx[i].npV) for i in range(size)]
