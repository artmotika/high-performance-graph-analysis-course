import pathlib
from project.graph_utils import load_test_json, read_dot_file, digraph_to_matrix_gb
from project.bfs import bfs_level
import numpy as np

path_json = pathlib.Path(__file__).parent / "data" / "test_bfs_level.json"
test_data_json = load_test_json(path_json)

path_dot1 = pathlib.Path(__file__).parent / "data" / "test_bfs_level1.dot"
path_dot2 = pathlib.Path(__file__).parent / "data" / "test_bfs_level2.dot"


def test_bfs_level_json1():
    matrix, source, expected = test_data_json[0]
    assert np.array_equal(bfs_level(matrix, source), np.asarray(expected))


def test_bfs_level_json2():
    matrix, source, expected = test_data_json[1]
    assert np.array_equal(bfs_level(matrix, source), np.asarray(expected))


def test_bfs_level_dot1():
    assert np.array_equal(
        bfs_level(digraph_to_matrix_gb(read_dot_file(path_dot1)), 0), np.array([0, 1])
    )


def test_bfs_level_dot2():
    assert np.array_equal(
        bfs_level(digraph_to_matrix_gb(read_dot_file(path_dot2)), 0),
        np.array([0, 1, 2, 2, -1, -1]),
    )
