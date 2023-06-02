import random as r
from time import time
import numpy as np
import math
from main import applicationLoadFromStr
from Datanalysis.SamplingData import SamplingData


def generateNormal(m=0, sigma=1, N: int = 1000):
    sample = [sum(r.random() for j in range(10)) for i in range(N)]
    x = SamplingData(sample)
    x.toRanking()
    x.toCalculateCharacteristic()
    x.toStandardization()
    x.toTransform(lambda z: m + sigma * z)
    return x.raw


def generateUniform(N: int = 1000):
    sample = []
    for i in range(N):
        sample.append(r.random())
    return sample


def generateExp(alpha=1, N: int = 1000):
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


def generateArcsin(b=1, N=1000):
    sample = []
    for i in range(N):
        x = np.average([r.random() * b for i in range(5000)])
        sample.append(x * math.sin(math.pi * (r.random() - 0.5)))
    return sample


def binaryData(N=1000):
    sample = generateNormal(N=N)
    x = SamplingData(sample)
    x.toRanking()
    x.toCalculateCharacteristic()
    x.toBinarization(x.x_)
    return x.raw


def generateLine(m1=0.0, m2=0.0, sigma1=1.0, sigma2=1.0, r_x_y=0.75,
                 N=1000):
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
    x = xx.raw
    e = generateNormal(N=N)
    y = [x[i] ** 2 + r * e[i] - 0.5 for i in range(N)]
    return x, y


def generateExp2D(a=1, b=1, N: int = 1000):
    x = generateUniform(N=N)
    y = [a * (math.exp(b * xi + r.random()) + r.random()) for xi in x]
    return x, y


def generateMultivariateNormal(E=None, DC=None, n: int = 3, N: int = 1000):
    if DC is None:
        DC = np.diag([1 + r.random() * 0.1 for i in range(n)])
        for i in range(n):
            for j in range(i + 1, n):
                DC[i, j] = DC[j, i] = (0.89 + r.random() * 0.1) \
                    * DC[i, i] ** 0.5 * DC[j, j] ** 0.5
    A = np.zeros((n, n))
    samples = []
    for i in range(n):
        samples.append(generateNormal(N=N))
    U = np.array(samples)
    for i in range(n):
        A[i, i] = (DC[i, i] - sum(A[i, w] ** 2 for w in range(i))) ** 0.5
        for j in range(i):
            A[i, j] = (DC[i, j] - sum(
                A[i, w] * A[j, w] for w in range(j))) / A[j, j]
    X = A @ U
    return X


def generateSample(number_sample: int = 1,
                   n: int = 1000, vec_n: int = 2,
                   parameters=[]) -> str:
    rozp = []
    for i in range(vec_n):
        if number_sample == 1:
            rozp.append(generateNormal(N=n))
        elif number_sample == 2:
            rozp.append(generateUniform(N=n))
        elif number_sample == 3:
            rozp.append(generateExp(N=n))
        elif number_sample == 4:
            rozp.append(generateWeibulla(N=n))
        elif number_sample == 5:
            rozp.append(generateArcsin(N=n))
        elif number_sample == 6:
            rozp.append(binaryData(N=n))
        elif number_sample == 7:
            if len(parameters) == 5:
                x, y = generateLine(parameters[0], parameters[1],
                                    parameters[2], parameters[3],
                                    parameters[4], N=n)
            else:
                x, y = generateLine(N=n)
            rozp.append(x)
            rozp.append(y)
        elif number_sample == 8:
            if len(parameters) == 1:
                x, y = generateParable(parameters[0], N=n)
            else:
                x, y = generateParable(N=n)
            rozp.append(x)
            rozp.append(y)
        elif number_sample == 9:
            x, y = generateExp2D(1, 5, n)
            rozp.append(x)
            rozp.append(y)
        elif number_sample == 10:
            x = generateMultivariateNormal(n=vec_n, N=n)
            for i in range(len(x)):
                rozp.append(x[i])
            break

    return '\n'.join(
        [''.join([str(rozp[j][i]).ljust(24) for j in range(len(rozp))])
         for i in range(n)])


if __name__ == "__main__":
    t1 = time()
    all_file = generateSample(number_sample=1, vec_n=1, n=27000)
    # with open("norm5n.txt", 'w') as f:
    #     f.write(all_file)
    print(f"generation time={time() - t1}")
    res = applicationLoadFromStr(all_file)
