import pygraphblas as gb
import numpy as np
from project.graph_utils import check_matrix_graph_size

__all__ = ["triangle_count_cohen", "triangle_count_sandia", "triangle_count"]


def triangle_count_cohen(Graph: gb.Matrix) -> int:
    size = check_matrix_graph_size(Graph)
    Graph.assign_scalar(1, mask=Graph, desc=gb.descriptor.S)
    result_mtx = gb.Matrix.sparse(gb.types.INT64, size, size)
    lower_mtx = Graph.tril(-1)
    upper_mtx = Graph.triu(1)
    lower_mtx.mxm(
        upper_mtx,
        out=result_mtx,
        mask=Graph,
        desc=gb.descriptor.S,
    )
    return result_mtx.reduce_int() / 2


def triangle_count_sandia(Graph: gb.Matrix) -> int:
    size = check_matrix_graph_size(Graph)
    Graph.assign_scalar(1, mask=Graph, desc=gb.descriptor.S)
    result_mtx = gb.Matrix.sparse(gb.types.INT64, size, size)
    lower_mtx = Graph.tril(-1)
    lower_mtx.mxm(
        lower_mtx,
        out=result_mtx,
        mask=lower_mtx,
        desc=gb.descriptor.S,
    )
    return result_mtx.reduce_int()


def triangle_count(Graph: gb.Matrix) -> np.array:
    size = check_matrix_graph_size(Graph)
    result_mtx = gb.Matrix.sparse(gb.types.INT64, size, size)
    Graph.mxm(
        Graph,
        out=result_mtx,
        mask=Graph,
        desc=gb.descriptor.RS,
    )
    result_mtx.assign_scalar(0, mask=result_mtx, desc=gb.descriptor.S & gb.descriptor.C)
    result_vector = result_mtx.reduce_vector().apply_second(gb.types.INT64.DIV, 2)
    result_vector.assign_scalar(
        0, mask=result_vector, desc=gb.descriptor.S & gb.descriptor.C
    )
    return result_vector.npV
