import pygraphblas as gb
import numpy as np
from project.graph_utils import check_matrix_graph_size

__all__ = ["bfs_level", "msbfs_parents"]


def bfs_level(Graph: gb.Matrix, source: int) -> np.array:
    """
    Bfs implementation on matrices

    :param Graph: the boolean adjacency matrix of a graph
    :param source: the start node in the bfs
    front: the vector of vertices reachable after the n-th iteration of bfs
    :return: an array where for each vertex it is indicated at which step it is reachable (for not reachable -1)
    """

    size = check_matrix_graph_size(Graph)
    visited_step = gb.Vector.sparse(gb.types.INT64, size)
    front = gb.Vector.sparse(gb.types.BOOL, size)
    front[source] = True
    step = 0
    while front.nvals > 0:
        visited_step[front] = step
        front.vxm(
            Graph,
            semiring=gb.INT64.any_pair,
            out=front,
            mask=visited_step,
            desc=gb.descriptor.RSC,
        )
        step += 1
    visited_step.assign_scalar(
        -1, mask=visited_step, desc=gb.descriptor.S & gb.descriptor.C
    )
    return visited_step.npV


def msbfs_parents(Graph: gb.Matrix, source: list[int]) -> list[tuple[int, np.array]]:
    """
    Multy source bfs implementation on matrices

    :param Graph: the boolean adjacency matrix of a graph
    :param source: ab array of the start nodes in the bfs
    front: the vector of vertices reachable after the n-th iteration of bfs
    parents: array (multy source) of arrays of the parent nodes while bfs where did node with
     index i come from array[i] in bfs
    :return: list of pairs: starting vertex, and array (parents) where for the starting vertex itself,
     take this value equal to -1, and for unreachable vertices, take it equal to -2
    """

    size = check_matrix_graph_size(Graph)
    n_ms = len(source)
    idx_row = gb.Vector.from_list(range(size))
    idx = gb.Matrix.sparse(gb.types.INT64, n_ms, size)
    front = gb.Matrix.sparse(gb.types.INT64, n_ms, size)
    parents = gb.Matrix.sparse(gb.types.INT64, n_ms, size)
    for i in range(n_ms):
        v = source[i]
        idx.assign_row(i, idx_row)
        front[i, v] = v
        parents[i, v] = -1
    while True:
        front.mxm(
            Graph,
            semiring=gb.INT64.min_first,
            out=front,
            mask=parents,
            desc=gb.descriptor.RSC,
        )
        if front.nvals == 0:
            break
        parents += front
        front.assign_matrix(idx, mask=front, desc=gb.descriptor.S)
    parents.assign_scalar(-2, mask=parents, desc=gb.descriptor.S & gb.descriptor.C)
    return [(source[i], parents[i].npV) for i in range(n_ms)]
