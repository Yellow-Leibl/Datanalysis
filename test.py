# from Datanalysis.SamplingData import SamplingData
# from Datanalysis.SamplesCriteria import SamplesCriteria
from Datanalysis.DoubleSampleData import DoubleSampleData
from Datanalysis.SamplingDatas import SamplingDatas
from GenerationSeries import generateSample


def readFile(file_name: str) -> list[str]:
    with open(file_name, 'r') as file:
        return file.readlines()


datas = SamplingDatas()


def test_calc_characteristic():
    datas.append(readFile("data/self/parable.txt"))
    datas.toCalculateCharacteristic()


def test_identPar():
    n = 3
    k = 3
    datas.append(generateSample(number_sample=7, vec_n=k*n, n=500).split('\n'))
    datas.toCalculateCharacteristic()
    datas.identAvr([datas.samples[i * n * 2:(i+1) * n * 2] for i in range(k)])
    datas.identDC([datas.samples[i * n * 2:(i+1) * n * 2] for i in range(k)])


def test_coeficientCorrelation():
    n = 3
    k = 3
    # test_calc_characteristic()
    datas.append(generateSample(number_sample=7, vec_n=k*n, n=500).split('\n'))
    datas.toCalculateCharacteristic()
    print(datas.coeficientOfCorrelation(0, 1, [2]))
    d2 = DoubleSampleData(datas[0], datas[1])
    d2.pearsonCorrelationСoefficient()
    print(d2.r)
    print(datas.coeficientOfRangeCorrelation(0, 1, [2]))
    d2 = DoubleSampleData(datas[0], datas[1])
    d2.rangeCorrelation()
    print(d2.teta_k)


def test_multiplyCoeficient():
    # datas.append(readFile("data/self/line.txt"))
    datas.append(readFile("data/6har.dat"))
    datas.samples = datas.samples[1:]
    [s.toLogarithmus10() for s in datas.samples]
    datas.toCalculateCharacteristic()
    for i in range(len(datas.samples)):
        print(datas.multipleCorrelationCoefficient(i))


# test_calc_characteristic()
# test_coeficientCorrelation()
test_multiplyCoeficient()
