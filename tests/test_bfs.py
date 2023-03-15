import pytest
import pygraphblas as gb
import project
import networkx as nx


def test_bfs_loop():
    assert bfs(
        digraph_to_matrix_gb(read_dot_file("data/graph_loop.dot")), 0
    ) == np.array([0, 1])
