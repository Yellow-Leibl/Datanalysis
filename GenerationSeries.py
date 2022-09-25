import random
from time import time
import numpy as np
import math
from main import application


def generateNormal():
    sample = []
    for i in range(N):
        x = sum([random.random() * b for i in range(10)])
        sample.append(str(x))
    return sample


def generateUniform():
    sample = []
    for i in range(N):
        sample.append(str(random.random()))
    return sample


def generateExp():
    sample = []
    for i in range(N):
        alpha = sum([random.random() * b for i in range(10)])
        sample.append(str(alpha * math.log(1 / (1 - random.random()))))
    return sample


def generateWeibulla():
    sample = []
    a = random.random()
    b = 0.2 + random.random() * 10
    for i in range(N):
        sample.append(str((-a * math.log(1 - random.random())) ** (1 / b)))
    return sample


def generateArcsin():
    sample = []
    for i in range(N):
        x = np.average([random.random() * b for i in range(5000)])
        sample.append(str(x * math.sin(math.pi * (random.random() - 0.5))))
    return sample


a = 0
b = 1
N = 1000


def generateSample(file_name: str, number_sample: int = 1,
                   start: float = 0, end: float = 1,
                   n: int = 1000, vec_n: int = 2):
    global a
    global b
    global N
    a = start
    b = end
    N = n
    with open(file_name, "w") as f:
        rozp = [[] for i in range(vec_n)]
        for i in range(vec_n):
            if number_sample == 1:
                rozp[i] = generateNormal()
            if number_sample == 2:
                rozp[i] = generateUniform()
            if number_sample == 3:
                rozp[i] = generateExp()
            if number_sample == 4:
                rozp[i] = generateWeibulla()
            if number_sample == 5:
                rozp[i] = generateArcsin()

        f.write('\n'.join(
            [''.join([rozp[j][i].ljust(24) for j in range(vec_n)])
             for i in range(N)]))


if __name__ == "__main__":
    while (True):
        t1 = time()
        generateSample("out.txt", vec_n=3)
        print(f"generation time={time() - t1}")
        res = application("out.txt")
