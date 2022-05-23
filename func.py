import math

# reproduction one-two parametr

def DFOneParametr(dF_d_theta, D_theta):
    return dF_d_theta ** 2 * D_theta

def DFTwoParametr(dF_d_theta1, dF_d_theta2, D_theta1, D_theta2, cov_theta12):
    return dF_d_theta1 ** 2 * D_theta1 + dF_d_theta2 ** 2 * D_theta2 \
        + 2 * dF_d_theta1 * dF_d_theta2 * cov_theta12

# kvant

def QuantileNorm(alpha):
    p = alpha
    t = math.sqrt(math.log(1 / p ** 2))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.1892659
    d3 = 0.001308
    Ea = 4.5 * 10 ** -4
    return t - (c0 + c1 * t + c2 * t ** 2) /\
        (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3) + Ea

def QuantileTStudent(alpha, nu):
    u = FNorm(alpha)
    g1 = 1 / 4 * (u ** 3 + u)
    g2 = 1 / 96 * (5 * u ** 5 + 16 * u ** 3 + 3 * u)
    g3 = 1 / 384 * (3 * u ** 7 + 19 * u ** 5 + 17 * u ** 3 - 15 * u)
    g4 = 1 / 92160 * (79 * u ** 9 + 779 * u ** 7 + 1482 * u ** 5 - 1920 * u ** 3 - 945 * u)
    return u + 1 / nu * g1 + 1 / nu ** 2 * g2 + 1 / nu ** 3 * g3 + 1 / nu ** 4 * g4

def QuantileXiXi(alpha, nu):
    return nu * (1 - 2 / (9 * nu) + QuantileNorm(alpha) * math.sqrt(2 / (9 * nu))) ** 3

# probability functions

def FNorm(x, m = 0, sigma = 1):
    P = 0.2316419
    B1 = 0.31938153
    B2 = -0.356563782
    B3 = 1.781477937
    B4 = -1.821255978
    B5 = 1.330274429
    EU = 7.8 * 10 ** -8
    u = abs((x - m) / sigma)
    t = 1 / (1 + P * u)
    Fu = 1 - 1 / math.sqrt(2 * math.pi) * math.exp(-(u ** 2) / 2) \
        * (B1 * t + B2 * t ** 2 + B3 * t ** 3 + B4 * t ** 4 + B5 * t ** 5) + EU
    if (x - m) / sigma < 0:
        Fu = 1 - Fu
    return Fu

def FUniform(x, a, b):
    if x < a:
        return 0
    elif x > b:
        return 1
    return (x - a) / (b - a)

def FExp(x, lambd_a):
    if x < 0:
        return 0
    return 1 - math.exp(-lambd_a * x)
    
def FWeibull(x, alpha, beta):
    return 1 - math.exp(-(x ** beta) / alpha)

def FArcsin(x, a):
    if x < -a:
        return 0
    elif x > a:
        return 1.0
    return 0.5 + 1 / math.pi * math.asin(x / a)

# probability density functions

def fNorm(x, m = 0, sigma = 1):
    return 1 / (sigma * math.sqrt(2 * math.pi)) \
        * math.exp(-((x - m) ** 2) / (2 * sigma ** 2))

def fUniform(a, b):
    return 1 / (b - a)

def fExp(x, lambd_a):
    if x < 0:
        return None
    try:
        return lambd_a * math.exp(-lambd_a * x)
    except OverflowError:
        return 1

def fWeibull(x, alpha, beta):
    return beta / alpha * x ** (beta - 1) * math.exp(-(x ** beta) / alpha)

def fArcsin(x, a):
    if not -a < x < a:
        return None
    return 1 / (math.pi * math.sqrt(a ** 2 - x ** 2))

# Derivative functions

def fNorm_d_m(x, m, sigma):
    return -1 / (sigma * math.sqrt(2 * math.pi)) \
        * math.exp(-((x - m) ** 2) / (2 * sigma ** 2))

def fNorm_d_sigma(x, m, sigma):
    return -(x - m) / (sigma ** 2 * math.sqrt(2 * math.pi)) \
        * math.exp(-(x - m) ** 2 / (2 * sigma ** 2))

def fExp_d_lamda(x, lambd_a):
    return x * math.exp(-lambd_a * x)

def fWeibull_d_alpha(x, alpha, beta):
    return -(x ** beta) / alpha ** 2 * math.exp(-(x ** beta) / alpha)

def fWeibull_d_beta(x, alpha, beta):
    return x ** beta / alpha * math.log(x) * math.exp(-(x ** beta) / alpha)
