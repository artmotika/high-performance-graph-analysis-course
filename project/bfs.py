import array
import pygraphblas as gb
import numpy as np

__all__ = ["bfs"]


def bfs(Graph: gb.Matrix, source: int) -> np.array:
    """
    Bfs implementation on matrices

    :param Graph: the boolean adjacency matrix of a graph
    :param source: the start node in the bfs
    front: the vector of vertices reachable after the n-th iteration of bfs
    :return: an array where for each vertex it is indicated at which step it is reachable (for not reachable -1)
    """

    size = Graph.shape[0]
    visited_step = gb.Vector.sparse(gb.types.INT64, size)
    front = gb.Vector.sparse(gb.types.BOOL, size)
    front[source] = True
    step = 0
    while front.nvals > 0:
        visited_step[front] = step
        front.vxm(Graph, out=front, mask=visited_step, desc=gb.descriptor.RSC)
        step += 1
    visited_step.assign_scalar(
        -1, mask=visited_step, desc=gb.descriptor.S & gb.descriptor.C
    )
    return visited_step.npV
