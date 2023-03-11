from Datanalysis.SamplingData import SamplingData
from Datanalysis.SamplesCriteria import SamplesCriteria
from Datanalysis.DoubleSampleData import DoubleSampleData
from time import time
import numpy as np

SPLIT_CHAR = ' '


def splitAndRemoveEmpty(s: str) -> list:
    return list(filter(lambda x: x != '\n' and x != '',
                       s.split(SPLIT_CHAR)))


def readVectors(text: list[str]) -> list:
    def strToFloat(x: str): return float(x.replace(',', '.'))
    split_float_data = [[strToFloat(j) for j in splitAndRemoveEmpty(i)]
                        for i in text]
    return [[vector[i] for vector in split_float_data]
            for i in range(len(split_float_data[0]))]


class SamplingDatas(SamplesCriteria):
    def __init__(self):
        self.samples: list[SamplingData] = []

    def appendSample(self, s: SamplingData):
        self.samples.append(s)

    def append(self, not_ranked_series_str: list[str]):
        t1 = time()
        vectors = readVectors(not_ranked_series_str)

        def rankAndCalc(s: SamplingData):
            s.toRanking()
            s.toCalculateCharacteristic()
        for v in vectors:
            s = SamplingData(v)
            rankAndCalc(s)
            self.samples.append(s)
        print(f"Reading vector time = {time() - t1} sec")

    def __len__(self) -> int:
        return len(self.samples)

    def pop(self, i: int) -> SamplingData:
        return self.samples.pop(i)

    def __getitem__(self, i: int) -> SamplingData:
        return self.samples[i]

    def getMaxDepthRangeData(self) -> int:
        if len(self.samples) == 0:
            return 0
        return max([len(i._x) for i in self.samples])

    def getMaxDepthRawData(self) -> int:
        if len(self.samples) == 0:
            return 0
        return max([len(i.getRaw()) for i in self.samples])

    def toCalculateCharacteristic(self, s: list[SamplingData]):
        n = len(s)
        DC = [[0.0 for j in range(n)] for i in range(n)]
        for i in range(n):
            DC[i][i] = s[i].Sigma ** 2

        for i in range(n):
            for j in range(i + 1, n):
                d2 = DoubleSampleData(s[i], s[j])
                d2.pearsonCorrelationСoefficient()
                cor = s[i].Sigma * s[j].Sigma * d2.r
                DC[i][j] = cor
                DC[j][i] = cor

        print(np.array(DC))
