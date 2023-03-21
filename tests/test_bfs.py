import pathlib
from project.graph_utils import load_test_json, read_dot_file, digraph_to_matrix_gb
from project.bfs import bfs
import numpy as np

path_json = pathlib.Path(__file__).parent / "data" / "test_bfs.json"
test_data_json = load_test_json(path_json)

path_dot1 = pathlib.Path(__file__).parent / "data" / "test_bfs1.dot"
path_dot2 = pathlib.Path(__file__).parent / "data" / "test_bfs2.dot"


def test_bfs_json1():
    matrix, source, expected = test_data_json[0]
    assert np.array_equal(bfs(matrix, source), np.asarray(expected))


def test_bfs_json2():
    matrix, source, expected = test_data_json[1]
    assert np.array_equal(bfs(matrix, source), np.asarray(expected))


def test_bfs_dot1():
    assert np.array_equal(
        bfs(digraph_to_matrix_gb(read_dot_file(path_dot1)), 0), np.array([0, 1])
    )


def test_bfs_dot2():
    assert np.array_equal(
        bfs(digraph_to_matrix_gb(read_dot_file(path_dot2)), 0),
        np.array([0, 1, 2, 2, -1, -1]),
    )
