import pathlib
from project.graph_utils import load_test_json, read_dot_file, digraph_to_matrix_gb
from project.bfs import bfs_level, msbfs_parents
import numpy as np

import pygraphblas as gb

path_json = pathlib.Path(__file__).parent / "data" / "test_msbfs_parents.json"
test_data_json = load_test_json(path_json)

path_dot1 = pathlib.Path(__file__).parent / "data" / "test_msbfs_parents1.dot"
path_dot2 = pathlib.Path(__file__).parent / "data" / "test_msbfs_parents2.dot"


def test_msbfs_parents_json1():
    matrix, source, expected = test_data_json[0]
    parents = msbfs_parents(matrix, source)
    keys = []
    expected_keys = expected.keys()
    for i in range(len(parents)):
        keys.append(str(parents[i][0]))
    for key in expected_keys:
        if key not in keys:
            assert False
    for key in keys:
        if key not in expected_keys:
            assert False

    for i in range(len(parents)):
        if not (np.array_equal(parents[i][1], expected[str(parents[i][0])])):
            assert False
    assert True


def test_msbfs_parents_json2():
    matrix, source, expected = test_data_json[1]
    parents = msbfs_parents(matrix, source)
    keys = []
    expected_keys = expected.keys()
    for i in range(len(parents)):
        keys.append(str(parents[i][0]))
    for key in expected_keys:
        if key not in keys:
            assert False
    for key in keys:
        if key not in expected_keys:
            assert False

    for i in range(len(parents)):
        if not (np.array_equal(parents[i][1], expected[str(parents[i][0])])):
            assert False
    assert True


def test_msbfs_parents_dot1():
    parents = msbfs_parents(digraph_to_matrix_gb(read_dot_file(path_dot1)), [0, 1])
    expected = [(0, [-1, 0]), (1, [1, -1])]
    for i in range(len(parents)):
        if not (
            np.array_equal(parents[i][1], expected[i][1])
            and parents[i][0] == expected[i][0]
        ):
            assert False
    assert True


def test_msbfs_parents_dot2():
    parents = msbfs_parents(digraph_to_matrix_gb(read_dot_file(path_dot2)), [0, 3, 4])
    expected = [
        (0, [-1, 0, 1, 1, -2, -2]),
        (3, [-2, 3, 1, -1, -2, -2]),
        (4, [-2, -2, -2, -2, -1, 4]),
    ]
    for i in range(len(parents)):
        if not (
            np.array_equal(parents[i][1], expected[i][1])
            and parents[i][0] == expected[i][0]
        ):
            assert False
    assert True
