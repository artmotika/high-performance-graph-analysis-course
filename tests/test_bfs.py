import pathlib
from project.graph_utils import load_test_data
from project.bfs import bfs
import numpy as np

path = pathlib.Path(__file__).parent / "data" / "test_bfs.json"
test_data = load_test_data(path)


def test_bfs1():
    matrix, source, expected = test_data[0]
    assert np.array_equal(bfs(matrix, source), np.asarray(expected))


def test_bfs2():
    matrix, source, expected = test_data[1]
    assert np.array_equal(bfs(matrix, source), np.asarray(expected))
