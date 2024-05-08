import random as r
import numpy as np
import math
from main import applicationLoadFromStr
from Datanalysis import SamplingData
from Datanalysis.SamplesTools import timer


def generate_normal(m=0, sigma=1, N: int = 1000):
    return np.random.normal(m, sigma, N)


def generate_uniform(low, high, N: int = 1000):
    return np.random.uniform(low, high, N)


def generateExp(alpha=1, N: int = 1000):
    sample = []
    for _ in range(N):
        sample.append(alpha * math.log(1 / (1 - r.random())))
    return sample


def generateWeibulla(N: int = 1000):
    sample = []
    a = r.random()
    b = 0.2 + r.random() * 10
    for _ in range(N):
        sample.append((-a * math.log(1 - r.random())) ** (1 / b))
    return sample


def generateArcsin(b=1, N=1000):
    sample = []
    for i in range(N):
        x = np.average([r.random() * b for i in range(5000)])
        sample.append(x * math.sin(math.pi * (r.random() - 0.5)))
    return sample


def binaryData(N=1000):
    sample = generate_normal(N=N)
    x = SamplingData(sample)
    x.toRanking()
    x.toCalculateCharacteristic()
    x.toBinarization(x.x_)
    return x.raw


def generate_line(m1=0.0, m2=0.0, sigma1=1.0, sigma2=1.0, r_x_y=0.75,
                  N=1000):
    z1 = generate_normal(N=N)
    z2 = generate_normal(N=N)

    x = [m1 + sigma1 * z1[i] for i in range(N)]
    y = [m2 + sigma2 * (z2[i] * (1 - r_x_y ** 2) ** 0.5 + z1[i] * r_x_y)
         for i in range(N)]

    return x, y


def generate_parable(low, high, a, b, c, sigma, N=1000):
    x = generate_uniform(low, high, N)
    e = generate_normal(sigma=sigma, N=N)
    y = a + b * x + c * x ** 2 + e
    return x, y


def generateExp2D(a=1, b=1, N: int = 1000):
    x = generate_uniform(N=N)
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
    for _ in range(n):
        samples.append(generate_normal(N=N))
    U = np.array(samples)
    for i in range(n):
        A[i, i] = (DC[i, i] - sum(A[i, w] ** 2 for w in range(i))) ** 0.5
        for j in range(i):
            A[i, j] = (DC[i, j] - sum(
                A[i, w] * A[j, w] for w in range(j))) / A[j, j]
    X = A @ U
    return X


def generate_time_series(N: int):
    norm = generate_normal(N=N)
    time_ser = np.empty(N)
    time_ser[0] = norm[0]
    for i in range(1, N):
        time_ser[i] = time_ser[i - 1] + norm[i]
    return time_ser


def generate_sin_series(N: int):
    x = np.linspace(-5, 5 * math.pi, N)
    norm = generate_normal(N=N, m=10, sigma=0.1)
    y = np.sin(x) + np.cos(x + 2) + norm
    return y


def generate_several_normal(parameters):
    norm_x = []
    norm_y = []
    for i in range(5, len(parameters)+1, 5):
        m1, m2, sigma1, sigma2, N = parameters[i-5:i]
        x, y = generate_line(m1, m2, sigma1, sigma2, 0, N)
        norm_x.append(x)
        norm_y.append(y)

    x = np.concatenate(norm_x)
    y = np.concatenate(norm_y)

    return x, y


@timer
def generate_sample(**parameters) -> str:
    n = parameters.get('n', 1000)
    vec_n = parameters.get('vec_n', 2)
    number_sample = parameters.get('number_sample', 1)

    rozp = []
    for i in range(vec_n):
        if number_sample == 1:
            rozp.append(generate_normal(N=n))
        elif number_sample == 2:
            rozp.append(generate_uniform(N=n))
        elif number_sample == 3:
            rozp.append(generateExp(N=n))
        elif number_sample == 4:
            rozp.append(generateWeibulla(N=n))
        elif number_sample == 5:
            rozp.append(generateArcsin(N=n))
        elif number_sample == 6:
            rozp.append(binaryData(N=n))
        elif number_sample == 7:
            n = parameters.get('n', 1000)
            m1 = parameters.get('m1', 0)
            m2 = parameters.get('m2', 0)
            sigma1 = parameters.get('sigma1', 1)
            sigma2 = parameters.get('sigma2', 1)
            r_x_y = parameters.get('r_x_y', 0.75)
            x, y = generate_line(m1, m2, sigma1, sigma2,
                                 r_x_y, N=n)
            rozp.append(x)
            rozp.append(y)
        elif number_sample == 8:
            n = parameters.get('n', 1000)
            low = parameters.get('low', -5)
            high = parameters.get('high', 5)
            a = parameters.get('a', 1)
            b = parameters.get('b', 1)
            c = parameters.get('c', 1)
            sigma = parameters.get('sigma', 1)
            x, y = generate_parable(low, high, a, b, c, sigma, n)
            rozp += [x, y]
        elif number_sample == 9:
            x, y = generateExp2D(1, 5, n)
            rozp += [x, y]
        elif number_sample == 10:
            x = generateMultivariateNormal(n=vec_n, N=n)
            for i in range(len(x)):
                rozp.append(x[i])
            break
        elif number_sample == 11:
            rozp.append(generate_sin_series(N=n))

    return sample_to_str(rozp)


def sample_to_str(sample):
    text = []
    for i in range(len(sample[0])):
        line = ''
        for j in range(len(sample)):
            line += str(sample[j][i]).ljust(24)
        text.append(line)
    return '\n'.join(text)


if __name__ == "__main__":
    x, y = generate_several_normal([0, 0, 1, 1, 100,
                                    -5, 5, 1, 1, 100,
                                    10, 7, 1, 1, 100,
                                    15, -15, 1, 1, 100,
                                    -20, 20, 1, 1, 100])
    all_file = sample_to_str([x, y])
    applicationLoadFromStr(all_file, range(2))
