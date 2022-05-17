import random
from time import time
import numpy as np
import math
from main import application

a = 0
b = 1
N = 5000

def generateNormal():
    l = []
    for i in range(N):
        rand_list = [random.random() * b for i in range(N)]
        avr = sum(rand_list) / len(rand_list)
        l.append(str(avr))
    return l

def generateUniform():
    l = []
    for i in range(N):
        l.append(str(random.random()))
    return l

def generateExp():
    l = []
    for i in range(N):
        avr = np.average([random.random() * b for i in range(N)])
        l.append(str(avr * math.log(1 / (1 - random.random()))))
    return l

def generateWeibulla():
    l = []
    a = random.random()
    b = 0.2 + random.random() * 10
    print("a=", a)
    print("b=", b)
    for i in range(N):
        l.append(str((-a * math.log(1 - random.random())) ** (1 / b)))
    return l

def generateArcsin():
    l = []
    for i in range(N):
        avr = np.average([random.random() * b for i in range(N)])
        l.append(str(avr * math.sin(math.pi * (random.random() - 0.5))))
    return l

t1 = time()
print("Тип розподілу\n\
    1. Нормальний\n\
    2. Рівномірний\n\
    3. Експоненціальний\n\
    4. Вейбулла\n\
    5. Арксінус")
# n = int(input())
n = 5
with open("out.txt", "w") as f:
    rozp = []
    if n == 1:
        rozp = generateNormal()
    if n == 2:
        rozp = generateUniform()
    if n == 3:
        rozp = generateExp()
    if n == 4:
        rozp = generateWeibulla()
    if n == 5:
        rozp = generateArcsin()
    f.write('\n'.join(rozp))
print(f"generation time={time() - t1}")

application("out.txt")