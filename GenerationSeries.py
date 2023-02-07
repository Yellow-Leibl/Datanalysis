import random as r
from time import time
import numpy as np
import math
from main import applicationLoadFromStr
from Datanalysis.SamplingData import SamplingData


def generateNormal(m=0, sigma=1, N: int = 1000):
    sample = []
    for i in range(N):
        sample.append(sum([r.random() for i in range(10)]))

    x = SamplingData(sample)
    x.toRanking()
    x.toCalculateCharacteristic()
    x.toStandardization()
    x.toTransform(lambda z: m + sigma * z)
    return x.getRaw()


def generateUniform(N: int = 1000):
    sample = []
    for i in range(N):
        sample.append(r.random())
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


def generateLine(m1: float = 0, m2: float = 0,
                 sigma1: float = 1, sigma2: float = 1,
                 r_x_y: float = 0.5, N: int = 1000):
    z1 = generateNormal(N=N)
    z2 = generateNormal(N=N)

    x = [m1 + sigma1 * z1[i] for i in range(N)]
    y = [m2 + sigma2 * (z2[i] * (1 - r_x_y ** 2) ** 0.5 + z1[i] * r_x_y)
         for i in range(N)]

    return x, y


def generateParable(r: float = 0.01, N: int = 1000):
    x = generateUniform(N=N)
    xx = SamplingData(x)
    xx.toSlide(-0.5)
    x = xx.getRaw()
    e = generateNormal(N=N)
    y = [x[i] ** 2 + r * e[i] - 0.5 for i in range(N)]
    return x, y


def generateExp2D(a, b, N: int = 1000):
    x = generateUniform(N=N)
    y = [a * (math.exp(b * xi + r.random()) + r.random()) for xi in x]
    return x, y


def generateSample(number_sample: int = 1,
                   a: float = 0, b: float = 1,
                   n: int = 1000, vec_n: int = 2) -> str:
    rozp = []
    for i in range(vec_n):
        if number_sample == 1:
            rozp.append(generateNormal(a, b, n))
        if number_sample == 2:
            rozp.append(generateUniform(n))
        if number_sample == 3:
            rozp.append(generateExp(a, n))
        if number_sample == 4:
            rozp.append(generateWeibulla(n))
        if number_sample == 5:
            rozp.append(generateArcsin(b, n))
        if number_sample == 6:
            rozp.append(binaryData(n))
        if number_sample == 7:
            x, y = generateLine(N=n, m2=4)
            rozp.append(x)
            rozp.append(y)
        if number_sample == 8:
            x, y = generateParable(N=n)
            rozp.append(x)
            rozp.append(y)
        if number_sample == 9:
            x, y = generateExp2D(1, 5, n)
            rozp.append(x)
            rozp.append(y)

    return '\n'.join(
        [''.join([str(rozp[j][i]).ljust(24) for j in range(len(rozp))])
         for i in range(n)])


if __name__ == "__main__":
    while (True):
        t1 = time()
        all_file = generateSample(number_sample=8, vec_n=1)
        # with open("norm5n.txt", 'w') as f:
        #     f.write(all_file)
        print(f"generation time={time() - t1}")
        res = applicationLoadFromStr(all_file)
