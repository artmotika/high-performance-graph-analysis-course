import json
import pygraphblas as pg


def load_test_data(path: str):
    matrix = []
    source = []
    expected = []

    with open(path, "r") as file:
        json_dict = json.load(file)
        data_chunks = json_dict["test_data"]

        for data in data_chunks:
            matrix.append(pg.Matrix.from_lists(*data["matrix"], V=True, typ=pg.BOOL))

            source.append(data["source"])

            expected.append(data["expected"])

    return list(zip(matrix, source, expected))
