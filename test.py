from Datanalysis import (
    DoubleSampleData, SamplingDatas, IODatas, PolynomialRegressionModel)
from GenerationSeries import generate_sample, generate_parable
import numpy as np
import matplotlib.pyplot as plt


def readFile(file_name: str) -> list[str]:
    with open(file_name, 'r') as file:
        return file.readlines()


reader = IODatas()


def test_calc_characteristic():
    datas = SamplingDatas()
    data = reader.read_from_file("data/self/parable.txt")
    datas.append(data)
    datas.toCalculateCharacteristic()


def test_identPar():
    datas = SamplingDatas()
    n = 3
    k = 3
    datas.append(generate_sample(number_sample=7, vec_n=k*n, n=500).split('\n'))
    datas.toCalculateCharacteristic()
    datas.identAvrIfNotIdentDC(
        [datas.samples[i * n * 2:(i+1) * n * 2] for i in range(k)])
    datas.identDC([datas.samples[i * n * 2:(i+1) * n * 2] for i in range(k)])


def test_coeficientCorrelation():
    datas = SamplingDatas()
    n = 3
    k = 3
    # test_calc_characteristic()
    data = reader.read_from_text(
        generate_sample(number_sample=7, vec_n=k*n, n=500).split('\n'))
    datas.append(data)
    datas.toCalculateCharacteristic()
    print(datas.coeficientOfCorrelation(0, 1, [2]))
    d2 = DoubleSampleData(datas[0], datas[1])
    d2.pearson_correlation_coefficient()
    print(d2.r)
    print(datas.coeficientOfRangeCorrelation(0, 1, [2]))
    d2 = DoubleSampleData(datas[0], datas[1])
    d2.rangeCorrelation()
    print(d2.teta_k)


def test_multiplyCoeficient():
    datas = SamplingDatas()
    series = reader.read_from_file("data/6har.dat")
    datas.append(series)
    datas.samples = datas.samples[1:]
    [s.to_log10() for s in datas.samples]
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


def test_ai_train_example():
    linear_model = PolynomialRegressionModel(1)
    parab_model = PolynomialRegressionModel(2)
    poly6_model = PolynomialRegressionModel(6)

    s_res_test_linear = np.empty(1000)
    s_res_train_linear = np.empty(1000)

    s_res_train_parab = np.empty(1000)
    s_res_test_parab = np.empty(1000)

    s_res_train_poly6 = np.empty(1000)
    s_res_test_poly6 = np.empty(1000)

    a, b, c = 1, 2, 3
    sigma = 1
    low, high = -5, 5

    N = [10, 40, 100, 400, 1000]

    for n in N:
        for i in range(1000):
            x_all, y_all = generate_parable(low=low, high=high,
                                            a=a, b=b, c=c, sigma=sigma, N=n)
            x_train, y_train, x_test, y_test = train_test_split(x_all, y_all,
                                                                0.8)

            acc_train, acc_test = get_accuracy(linear_model, x_train, y_train,
                                               x_test, y_test)
            s_res_train_linear[i] = acc_train
            s_res_test_linear[i] = acc_test

            acc_train, acc_test = get_accuracy(parab_model, x_train, y_train,
                                               x_test, y_test)
            s_res_train_parab[i] = acc_train
            s_res_test_parab[i] = acc_test

            acc_train, acc_test = get_accuracy(poly6_model, x_train, y_train,
                                               x_test, y_test)
            s_res_train_poly6[i] = acc_train
            s_res_test_poly6[i] = acc_test

        print(f"\nn = {n}")
        print(f"linear: train={s_res_train_linear.mean()}")
        print(f"linear: test={s_res_test_linear.mean()}")
        print(f"parab: train={s_res_train_parab.mean()}")
        print(f"parab: test={s_res_test_parab.mean()}")
        print(f"poly6: train={s_res_train_poly6.mean()}")
        print(f"poly6: test={s_res_test_poly6.mean()}")


def train_test_split(x, y, test_size: float):
    n = int(len(x) * test_size)
    return x[:n], y[:n], x[n:], y[n:]


def get_accuracy(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    acc_train = model.score(x_train, y_train)
    acc_test = model.score(x_test, y_test)
    return acc_train, acc_test


def test_classificator():
    sd = SamplingDatas()
    sd.append(reader.read_from_file('data/iris_fish.txt'))

    sd.toCalculateCharacteristic()
    cluster1 = list(sd.clusters[0])
    cluster2 = list(sd.clusters[1])
    x1 = sd.samples[2].raw[cluster1]
    y1 = sd.samples[1].raw[cluster1]

    x2 = sd.samples[2].raw[cluster2]
    y2 = sd.samples[1].raw[cluster2]
    # plt.scatter(x1, y1)
    # plt.scatter(x2, y2)
    # plt.show()

    cluster1 = sd.clusters2[0]
    cluster2 = sd.clusters2[1]
    x1 = cluster1.T[2]
    y1 = cluster1.T[1]

    x2 = cluster2.T[2]
    y2 = cluster2.T[1]
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.show()


# test_calc_characteristic()
# test_coeficientCorrelation()
# test_multiplyCoeficient()
# test_ai_train_example()
# test_classificator()


# read lines from file and save with numeration as (1 line\n2 line\n...)

import pandas as pd

df = pd.read_csv('data/course/5_CO2_zx_rm_tail.csv')
df['Engine Power'] = df['Engine Size(L)'] * df['Cylinders'] * df['Fuel Consumption Comb']
df.drop(['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb'], axis=1)
df.to_csv('data/course/6_CO2_pw_eng.csv', index=False)
