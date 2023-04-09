import pathlib
from project.graph_utils import (
    load_test_json_bool_matrix,
    read_dot_file,
    digraph_to_bool_matrix_gb,
)
from project.bfs import bfs_level, msbfs_parents
import numpy as np

pathfile = pathlib.Path(__file__).parent
path_json = pathfile / "data" / "test_bfs_level.json"
test_data_json = load_test_json_bool_matrix(path_json)

path_dot1 = pathfile / "data" / "test_bfs_level1.dot"
path_dot2 = pathfile / "data" / "test_bfs_level2.dot"

path_json_ms = pathfile / "data" / "test_msbfs_parents.json"
test_data_json_ms = load_test_json_bool_matrix(path_json_ms)

path_dot1_ms = pathfile / "data" / "test_msbfs_parents1.dot"
path_dot2_ms = pathfile / "data" / "test_msbfs_parents2.dot"


def test_bfs_level_json1():
    matrix, expected = test_data_json[0][0]
    extra = test_data_json[1]
    source = extra["source"][0]
    assert np.array_equal(bfs_level(matrix, source), expected)


def test_bfs_level_json2():
    matrix, expected = test_data_json[0][1]
    extra = test_data_json[1]
    source = extra["source"][1]
    assert np.array_equal(bfs_level(matrix, source), expected)


def test_bfs_level_dot1():
    assert np.array_equal(
        bfs_level(digraph_to_bool_matrix_gb(read_dot_file(path_dot1)), 0), [0, 1]
    )


def test_bfs_level_dot2():
    assert np.array_equal(
        bfs_level(digraph_to_bool_matrix_gb(read_dot_file(path_dot2)), 0),
        [0, 1, 2, 2, -1, -1],
    )


def test_msbfs_parents_json1():
    matrix, expected = test_data_json_ms[0][0]
    extra = test_data_json_ms[1]
    source = extra["source"][0]
    parents = msbfs_parents(matrix, source)
    keys = []
    expected_keys = list(expected.keys())
    for i in range(len(parents)):
        keys.append(str(parents[i][0]))
    if expected_keys != keys:
        assert False
    for i in range(len(parents)):
        if not (np.array_equal(parents[i][1], expected[str(parents[i][0])])):
            assert False
    assert True


def test_msbfs_parents_json2():
    matrix, expected = test_data_json_ms[0][1]
    extra = test_data_json_ms[1]
    source = extra["source"][1]
    parents = msbfs_parents(matrix, source)
    keys = []
    expected_keys = list(expected.keys())
    for i in range(len(parents)):
        keys.append(str(parents[i][0]))
    if expected_keys != keys:
        assert False
    for i in range(len(parents)):
        if not (np.array_equal(parents[i][1], expected[str(parents[i][0])])):
            assert False
    assert True


def test_msbfs_parents_dot1():
    parents = msbfs_parents(
        digraph_to_bool_matrix_gb(read_dot_file(path_dot1_ms)), [0, 1]
    )
    expected = [(0, [-1, 0]), (1, [1, -1])]
    if len(parents) != len(expected):
        assert False
    for i in range(len(parents)):
        if not (
            np.array_equal(parents[i][1], expected[i][1])
            and parents[i][0] == expected[i][0]
        ):
            assert False
    assert True


def test_msbfs_parents_dot2():
    parents = msbfs_parents(
        digraph_to_bool_matrix_gb(read_dot_file(path_dot2_ms)), [0, 3, 4]
    )
    expected = [
        (0, [-1, 0, 1, 1, -2, -2]),
        (3, [-2, 3, 1, -1, -2, -2]),
        (4, [-2, -2, -2, -2, -1, 4]),
    ]
    if len(parents) != len(expected):
        assert False
    for i in range(len(parents)):
        if not (
            np.array_equal(parents[i][1], expected[i][1])
            and parents[i][0] == expected[i][0]
        ):
            assert False
    assert True
