import pathlib
from project.graph_utils import load_test_json, read_dot_file, digraph_to_matrix_gb
from project.triangle_count import triangle_count_cohen
import numpy as np

path_json = pathlib.Path(__file__).parent / "data" / "test_triangle_count_number.json"
test_data_json = load_test_json(path_json)

path_dot1 = pathlib.Path(__file__).parent / "data" / "test_triangle_count1.dot"
path_dot2 = pathlib.Path(__file__).parent / "data" / "test_triangle_count2.dot"


def test_triangle_count_cohen_json1():
    matrix, expected = test_data_json[0][0]
    assert np.array_equal(
        triangle_count_cohen(matrix),
        np.array(expected),
    )


def test_triangle_count_cohen_json2():
    matrix, expected = test_data_json[0][1]
    assert np.array_equal(
        triangle_count_cohen(matrix),
        np.array(expected),
    )


def test_triangle_count_cohen_dot1():
    assert triangle_count_cohen(digraph_to_matrix_gb(read_dot_file(path_dot1))) == 5


def test_triangle_count_cohen_dot2():
    assert triangle_count_cohen(digraph_to_matrix_gb(read_dot_file(path_dot2))) == 1
