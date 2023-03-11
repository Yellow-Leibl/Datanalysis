# from Datanalysis.SamplingData import SamplingData
# from Datanalysis.SamplesCriteria import SamplesCriteria
# from Datanalysis.DoubleSampleData import DoubleSampleData
from Datanalysis.SamplingDatas import SamplingDatas


def readFile(file_name: str) -> list[str]:
    with open(file_name, 'r') as file:
        return file.readlines()


datas = SamplingDatas()
datas.append(readFile("data/self/parable.txt"))

datas.toCalculateCharacteristic([datas[0], datas[1], datas[2]])
