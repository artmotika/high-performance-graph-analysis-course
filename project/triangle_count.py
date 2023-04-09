import pygraphblas as gb
import numpy as np
from project.graph_utils import (
    check_matrix_graph_size_and_type,
    check_matrix_graph_unoriented,
)

__all__ = ["triangle_count_cohen", "triangle_count_sandia", "triangle_count"]


def triangle_count_cohen(Graph: gb.Matrix) -> int:
    """
    Counts the number of cycles of length 3 with Cohen's method

    :param Graph: the boolean adjacency matrix of a graph
    :return: the number of cycles of length 3
    """
    size = check_matrix_graph_size_and_type(Graph, gb.BOOL)
    check_matrix_graph_unoriented(Graph)
    Graph.assign_scalar(1, mask=Graph, desc=gb.descriptor.S)
    result_mtx = gb.Matrix.sparse(gb.INT64, size, size)
    lower_mtx = Graph.tril()
    upper_mtx = Graph.triu()
    lower_mtx.mxm(
        upper_mtx,
        out=result_mtx,
        mask=Graph,
        desc=gb.descriptor.S,
    )
    return result_mtx.reduce_int() / 2


def triangle_count_sandia(Graph: gb.Matrix) -> int:
    """
    Counts the number of cycles of length 3 with Sandia's method

    :param Graph: the boolean adjacency matrix of a graph
    :return: the number of cycles of length 3
    """
    size = check_matrix_graph_size_and_type(Graph, gb.BOOL)
    check_matrix_graph_unoriented(Graph)
    Graph.assign_scalar(1, mask=Graph, desc=gb.descriptor.S)
    result_mtx = gb.Matrix.sparse(gb.INT64, size, size)
    lower_mtx = Graph.tril()
    lower_mtx.mxm(
        lower_mtx,
        out=result_mtx,
        mask=lower_mtx,
        desc=gb.descriptor.S,
    )
    return result_mtx.reduce_int()


def triangle_count(Graph: gb.Matrix) -> np.array:
    """
    Counts the number of cycles of length 3 for each vertex in which it participates

    :param Graph: the boolean adjacency matrix of a graph
    :return: numpy array, where the value at index n is equal to the number of cycles of length 3,
     in which the vertex at number n participates
    """
    size = check_matrix_graph_size_and_type(Graph, gb.BOOL)
    check_matrix_graph_unoriented(Graph)
    result_mtx = gb.Matrix.sparse(gb.INT64, size, size)
    Graph.mxm(
        Graph,
        out=result_mtx,
        mask=Graph,
        desc=gb.descriptor.RS,
    )
    result_vector = result_mtx.reduce_vector().apply_second(gb.INT64.DIV, 2)
    result_vector.assign_scalar(
        0, mask=result_vector, desc=gb.descriptor.S & gb.descriptor.C
    )
    return result_vector.npV
