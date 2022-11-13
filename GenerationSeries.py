import random as r
from time import time
import numpy as np
import math
from main import applicationLoadFromStr
from Datanalysis.SamplingData import SamplingData


def setCharacteristicForSample(sample, m, sigma) -> list:
    x = SamplingData(sample)
    x.toRanking()
    x.toCalculateCharacteristic()
    x.toStandardization()
    x.toTransform(lambda z: m + sigma * z)
    return x.getRaw()


def generateNormal(m=0, sigma=1, N: int = 1000):
    sample = []
    for i in range(N):
        sample.append(sum([r.random() for i in range(10)]))
    return setCharacteristicForSample(sample, m, sigma)


def generateUniform(a, b, N: int = 1000):
    sample = []
    for i in range(N):
        sample.append(r.randint(a, b))
    return sample


def generateExp(alpha, N: int = 1000):
    sample = []
    for i in range(N):
        sample.append(alpha * math.log(1 / (1 - r.random())))
    return sample


def generateWeibulla(N: int = 1000):
    sample = []
    a = r.random()
    b = 0.2 + r.random() * 10
    for i in range(N):
        sample.append((-a * math.log(1 - r.random())) ** (1 / b))
    return sample


def generateArcsin(b, N: int = 1000):
    sample = []
    for i in range(N):
        x = np.average([r.random() * b for i in range(5000)])
        sample.append(x * math.sin(math.pi * (r.random() - 0.5)))
    return sample


def binaryData(N: int = 1000):
    sample = generateNormal(N=N)
    x = SamplingData(sample)
    x.toRanking()
    x.toCalculateCharacteristic()
    x.toBinarization(x.x_)
    return x.getRaw()


def generateLine(N: int = 1000):
    x = generateNormal(N=N)
    e = generateNormal(N=N)
    y = [x[i] + e[i] * 0.3 for i in range(N)]
    return x, y


def generateParable(N: int = 1000):
    x = generateNormal(N=N)
    e = generateNormal(N=N)
    y = [x[i] ** 2 + e[i] * 0.3 for i in range(N)]
    return x, y


def generateSample(number_sample: int = 1,
                   a: float = 0, b: float = 1,
                   n: int = 1000, vec_n: int = 2) -> str:
    rozp = []
    for i in range(vec_n):
        if number_sample == 1:
            rozp.append(generateNormal(a, b, n))
        if number_sample == 2:
            rozp.append(generateUniform(a, b, n))
        if number_sample == 3:
            rozp.append(generateExp(a, n))
        if number_sample == 4:
            rozp.append(generateWeibulla(n))
        if number_sample == 5:
            rozp.append(generateArcsin(b, n))
        if number_sample == 6:
            rozp.append(binaryData(n))
        if number_sample == 7:
            x, y = generateLine(n)
            rozp.append(x)
            rozp.append(y)
        if number_sample == 8:
            x, y = generateParable(n)
            rozp.append(x)
            rozp.append(y)

    return '\n'.join(
        [''.join([str(rozp[j][i]).ljust(24) for j in range(len(rozp))])
         for i in range(n)])


if __name__ == "__main__":
    while (True):
        t1 = time()
        all_file = generateSample(number_sample=7)
        # with open("norm5n.txt", 'w') as f:
        #     f.write(all_file)
        print(f"generation time={time() - t1}")
        res = applicationLoadFromStr(all_file)
