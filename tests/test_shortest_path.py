import pathlib
from project.graph_utils import (
    load_test_json_float_matrix,
    read_dot_file,
    digraph_to_float_matrix_gb,
)
from project.shortest_path import bellman_ford, msbellman_ford, floyd_warshall
import numpy as np

pathfile = pathlib.Path(__file__).parent
path_json_bellman_ford = pathfile / "data" / "test_bellman_ford.json"
test_data_json_bellman_ford = load_test_json_float_matrix(path_json_bellman_ford)

path_dot1_bellman_ford = pathfile / "data" / "test_bellman_ford1.dot"

path_json_msbellman_ford = pathfile / "data" / "test_msbellman_ford.json"
test_data_json_msbellman_ford = load_test_json_float_matrix(path_json_msbellman_ford)

path_dot1_msbellman_ford = pathfile / "data" / "test_msbellman_ford1.dot"

path_json_floyd_warshall = pathfile / "data" / "test_floyd_warshall.json"
test_data_json_floyd_warshall = load_test_json_float_matrix(path_json_floyd_warshall)

path_dot1_floyd_warshall = pathfile / "data" / "test_floyd_warshall1.dot"


def test_bellman_ford_json1():
    matrix0, expected0 = test_data_json_bellman_ford[0][0]
    matrix1, expected1 = test_data_json_bellman_ford[0][1]
    matrix2, expected2 = test_data_json_bellman_ford[0][2]
    matrix3, expected3 = test_data_json_bellman_ford[0][3]
    source = test_data_json_bellman_ford[1]["source"]
    assert all(
        (
            np.array_equal(bellman_ford(matrix0, source[0]), expected0),
            np.array_equal(bellman_ford(matrix1, source[1]), expected1),
            np.array_equal(bellman_ford(matrix2, source[2]), expected2),
            np.array_equal(bellman_ford(matrix3, source[3]), expected3),
        )
    )


def test_bellman_ford_dot1():
    matrix = digraph_to_float_matrix_gb(read_dot_file(path_dot1_bellman_ford))
    assert all(
        (
            np.array_equal(bellman_ford(matrix, 0), [0, 3, 7, 5]),
            np.array_equal(bellman_ford(matrix, 1), [2, 0, 6, 4]),
            np.array_equal(bellman_ford(matrix, 2), [3, 1, 0, 5]),
            np.array_equal(bellman_ford(matrix, 3), [5, 3, 2, 0]),
        )
    )


def test_msbellman_ford_json1():
    matrix, expected = test_data_json_msbellman_ford[0][0]
    source = test_data_json_msbellman_ford[1]["source"]
    actual = msbellman_ford(matrix, source[0])
    actual_keys = []
    expected_keys = list(expected.keys())
    for i in range(len(actual)):
        actual_keys.append(str(actual[i][0]))
    if expected_keys != actual_keys:
        assert False
    for i in range(len(actual)):
        if not (np.array_equal(actual[i][1], expected[str(actual[i][0])])):
            assert False
    assert True


def test_msbellman_ford_dot1():
    matrix = digraph_to_float_matrix_gb(read_dot_file(path_dot1_msbellman_ford))
    actual = msbellman_ford(matrix, [0, 1, 2, 3])
    expected = [
        (0, [0, 3, 7, 5]),
        (1, [2, 0, 6, 4]),
        (2, [3, 1, 0, 5]),
        (3, [5, 3, 2, 0]),
    ]
    if len(actual) != len(expected):
        assert False
    for i in range(len(actual)):
        if not (
            np.array_equal(actual[i][1], expected[i][1])
            and actual[i][0] == expected[i][0]
        ):
            assert False
    assert True


def test_floyd_warshall_json1():
    matrix, expected = test_data_json_floyd_warshall[0][0]
    actual = floyd_warshall(matrix)
    actual_keys = []
    expected_keys = list(expected.keys())
    for i in range(len(actual)):
        actual_keys.append(str(actual[i][0]))
    if expected_keys != actual_keys:
        assert False
    for i in range(len(actual)):
        if not (np.array_equal(actual[i][1], expected[str(actual[i][0])])):
            assert False
    assert True


def test_floyd_warshall_dot1():
    matrix = digraph_to_float_matrix_gb(read_dot_file(path_dot1_floyd_warshall))
    actual = floyd_warshall(matrix)
    expected = [
        (0, [0, 3, 7, 5]),
        (1, [2, 0, 6, 4]),
        (2, [3, 1, 0, 5]),
        (3, [5, 3, 2, 0]),
    ]
    if len(actual) != len(expected):
        assert False
    for i in range(len(actual)):
        if not (
            np.array_equal(actual[i][1], expected[i][1])
            and actual[i][0] == expected[i][0]
        ):
            assert False
    assert True
