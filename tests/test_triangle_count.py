import pathlib
from project.graph_utils import (
    load_test_json_bool_matrix,
    read_dot_file,
    digraph_to_bool_matrix_gb,
)
from project.triangle_count import (
    triangle_count,
    triangle_count_cohen,
    triangle_count_sandia,
)
import numpy as np

pathfile = pathlib.Path(__file__).parent

path_json = pathfile / "data" / "test_triangle_count_array.json"
test_data_json = load_test_json_bool_matrix(path_json)

path_dot1 = pathfile / "data" / "test_triangle_count1.dot"
path_dot2 = pathfile / "data" / "test_triangle_count2.dot"

path_json_cohen = pathfile / "data" / "test_triangle_count_number.json"
test_data_json_cohen = load_test_json_bool_matrix(path_json_cohen)

path_dot1_cohen = pathfile / "data" / "test_triangle_count1.dot"
path_dot2_cohen = pathfile / "data" / "test_triangle_count2.dot"

path_json_sandia = pathfile / "data" / "test_triangle_count_number.json"
test_data_json_sandia = load_test_json_bool_matrix(path_json_sandia)

path_dot1_sandia = pathfile / "data" / "test_triangle_count1.dot"
path_dot2_sandia = pathfile / "data" / "test_triangle_count2.dot"


def test_triangle_count_json1():
    matrix, expected = test_data_json[0][0]
    assert np.array_equal(
        triangle_count(matrix),
        expected,
    )


def test_triangle_count_json2():
    matrix, expected = test_data_json[0][1]
    assert np.array_equal(
        triangle_count(matrix),
        expected,
    )


def test_triangle_count_dot1():
    assert np.array_equal(
        triangle_count(digraph_to_bool_matrix_gb(read_dot_file(path_dot1))),
        [1, 3, 2, 4, 1, 1, 3],
    )


def test_triangle_count_dot2():
    assert np.array_equal(
        triangle_count(digraph_to_bool_matrix_gb(read_dot_file(path_dot2))),
        [0, 1, 1, 1, 0, 0],
    )


def test_triangle_count_cohen_json1():
    matrix, expected = test_data_json_cohen[0][0]
    assert np.array_equal(
        triangle_count_cohen(matrix),
        expected,
    )


def test_triangle_count_cohen_json2():
    matrix, expected = test_data_json_cohen[0][1]
    assert np.array_equal(
        triangle_count_cohen(matrix),
        expected,
    )


def test_triangle_count_cohen_dot1():
    assert (
        triangle_count_cohen(digraph_to_bool_matrix_gb(read_dot_file(path_dot1_cohen)))
        == 5
    )


def test_triangle_count_cohen_dot2():
    assert (
        triangle_count_cohen(digraph_to_bool_matrix_gb(read_dot_file(path_dot2_cohen)))
        == 1
    )


def test_triangle_count_sandia_json1():
    matrix, expected = test_data_json_sandia[0][0]
    assert np.array_equal(
        triangle_count_sandia(matrix),
        expected,
    )


def test_triangle_count_sandia_json2():
    matrix, expected = test_data_json_sandia[0][1]
    assert np.array_equal(
        triangle_count_sandia(matrix),
        expected,
    )


def test_triangle_count_sandia_dot1():
    assert (
        triangle_count_sandia(
            digraph_to_bool_matrix_gb(read_dot_file(path_dot1_sandia))
        )
        == 5
    )


def test_triangle_count_sandia_dot2():
    assert (
        triangle_count_sandia(
            digraph_to_bool_matrix_gb(read_dot_file(path_dot2_sandia))
        )
        == 1
    )
