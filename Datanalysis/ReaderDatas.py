import numpy as np
import pandas as pd


class ReaderDatas:
    def read_from_text(self, text: list[str]):
        return self.read_vectors_from_txt("from_text", text)

    def read_from_file(self, path: str):
        last_dot_index = -path[::-1].index('.')
        ext_name = path[last_dot_index:]
        if ext_name == 'csv':
            return self.read_from_csv(path)
        else:
            return self.read_from_txt(path)

    def read_from_txt(self, filename: str):
        with open(filename, 'r') as file:
            text = file.readlines()
        return self.read_vectors_from_txt(filename, text)

    def read_vectors_from_txt(self,
                              path: str,
                              text: list[str]):
        filename_with_ext = path.split('/')[-1]
        filename = filename_with_ext.split('.')[0]
        if ',' in text[0]:
            for i, line in enumerate(text):
                text[i] = line.replace(',', '.')

        n = len(np.fromstring(text[0], dtype=float, sep=' '))

        vectors = np.empty((len(text), n), dtype=float)
        for j, line in enumerate(text):
            vectors[j] = np.fromstring(line, dtype=float, sep=' ')
        names = [f"{filename}_{i}" for i in range(n)]
        return names, vectors.transpose()

    def read_from_csv(self, filename: str):
        data = pd.read_csv(filename)
        names = [column for column in data.columns]
        for i in range(len(data.columns)):
            column = data.columns[i]
            if type(data[column][0]) is not str:
                continue
            unique_data = data[column].unique()
            data[column] = data[column].replace(unique_data,
                                                range(len(unique_data)))
            names[i] = f"{names[i]}:{unique_data}"
        arr = data.to_numpy()
        return names, arr.transpose()
