from Datanalysis import DoubleSampleData, SamplingDatas
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
    datas.identAvrIfNotIdentDC(
        [datas.samples[i * n * 2:(i+1) * n * 2] for i in range(k)])
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


def test_find_el_from_1d_to_nd():
    import numpy as np
    t_a = np.arange(3 * 5 * 4 * 2)
    t_a_r = t_a.reshape((3, 5, 4, 2))
    print(t_a_r)
    print(t_a)
    i = 35
    print(t_a_r[i // (5 * 4 * 2) % 3, i // (4 * 2) % 5, i // (2) % 4, i % 2])


# test_calc_characteristic()
# test_coeficientCorrelation()
test_multiplyCoeficient()
