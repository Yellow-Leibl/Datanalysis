import numpy as np
import pandas as pd


class ReaderDatas:
    def __init__(self) -> None:
        pass

    def read_from_text(self, text: list[str]):
        return self.read_vectors_from_txt(text)

    def read_from_file(self, filename: str):
        last_dot_index = -filename[::-1].index('.')
        ext_name = filename[last_dot_index:]
        if ext_name == 'csv':
            return self.read_from_csv(filename)
        else:
            return self.read_from_txt(filename)

    def read_from_txt(self, filename: str) -> np.ndarray:
        with open(filename, 'r') as file:
            text = file.readlines()
        return self.read_vectors_from_txt(text)

    def read_vectors_from_txt(self, text: list[str]):
        if ',' in text[0]:
            for i, line in enumerate(text):
                text[i] = line.replace(',', '.')

        n = len(np.fromstring(text[0], dtype=float, sep=' '))

        vectors = np.empty((len(text), n), dtype=float)
        for j, line in enumerate(text):
            vectors[j] = np.fromstring(line, dtype=float, sep=' ')
        return vectors.transpose()

    def read_from_csv(self, filename: str) -> np.ndarray:
        data = pd.read_csv(filename)
        for column in data.columns:
            if type(data[column][0]) is not str:
                continue
            unique_data = data[column].unique()
            data[column] = data[column].replace(unique_data,
                                                range(len(unique_data)))
        arr = data.to_numpy()
        return arr.transpose()
