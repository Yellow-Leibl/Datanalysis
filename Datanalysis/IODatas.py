import numpy as np
import pandas as pd
from Datanalysis.SamplingData import SamplingData


class IODatas:
    def read_from_text(self, text: list[str]):
        names, vectors = self.read_vectors_from_text("from_text", text)
        ticks = [None for _ in names]
        return self.make_samples(names, ticks, vectors)

    def read_from_file(self, path: str):
        last_dot_index = -path[::-1].index('.')
        ext_name = path[last_dot_index:]
        if ext_name == 'csv':
            (names, ticks), arr = self.read_from_file_csv(path)
        else:
            names, arr = self.read_from_file_txt(path)
            ticks = [None for _ in names]
        return self.make_samples(names, ticks, arr)

    def read_from_file_txt(self, filename: str):
        with open(filename, 'r') as file:
            text = file.readlines()
        return self.read_vectors_from_text(filename, text)

    def read_vectors_from_text(self,
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

    def read_from_file_csv(self, filename: str):
        data = pd.read_csv(filename)
        names = [column for column in data.columns]
        ticks = [None for _ in names]
        for i in range(len(data.columns)):
            column = data.columns[i]
            if type(data[column][0]) is not str:
                continue
            unique_data = data[column].unique()
            data[column] = data[column].replace(unique_data,
                                                range(len(unique_data)))
            ticks[i] = unique_data
        arr = data.to_numpy()
        return (names, ticks), arr.transpose()

    def make_samples(self, names: list[str], ticks, vectors: np.ndarray):
        datas = []
        for name, ticksi, v in zip(names, ticks, vectors):
            datas.append(SamplingData(v, move_data=True,
                                      name=name, ticks=ticksi))
        return datas

    def save_to_csv(self,
                    filename: str,
                    names: list[str],
                    ticks: list,
                    vectors: np.ndarray):
        df = pd.DataFrame(vectors, columns=names)
        for name, ticksi in zip(names, ticks):
            if ticksi is None:
                continue
            df[name] = df[name].replace(range(len(ticksi)), ticksi)

        if filename.split('.')[-1] != 'csv':
            filename += '.csv'
        df.to_csv(filename, index=False)
