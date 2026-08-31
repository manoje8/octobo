import pandas as pd


def data_generator(filepath, chunk_size=1000, target="target"):
    for chunk in pd.read_csv(filepath, chunksize=chunk_size, encoding="utf-8"):
        X = chunk.drop(target, axis=1)
        y = chunk[target]
        yield X, y
