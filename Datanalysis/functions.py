import math
import numpy as np
np.seterr(all='raise')


# Matrix operations


def gauss_method(a: np.ndarray,
                 b: np.ndarray) -> np.ndarray:
    a = a.copy()
    b = b.copy()
    n = a.shape[0]
    for i in range(n):
        b[i] /= a[i, i]
        a[i] /= a[i, i]

        for k in range(n):
            if k != i and a[k, i] != 0:
                b[k] -= b[i] * a[k, i]
                a[k] -= a[i] * a[k, i]
    return b


def sort_evects_n_evals(vals: np.ndarray, vects: np.ndarray):
    n = len(vals)
    sorted_by_disp = sorted([[vals[i], i] for i in range(n)],
                            key=lambda i: i[0], reverse=True)
    indexes_sort_DC = [i[1] for i in sorted_by_disp]
    vals = vals[indexes_sort_DC]
    vects = vects[:, indexes_sort_DC]
    return vals, vects


def EigenvalueJacob(A: np.ndarray, eps=0.00001):
    n = len(A)

    def phi(i, j):
        if A[i, i] == A[j, j]:
            return math.pi / 4
        return 1 / 2 * math.atan(2 * A[i, j] / (A[i, i] - A[j, j]))

    def U(i, j):
        u = np.identity(n)
        phi_k = phi(i, j)
        u[i, i] = math.cos(phi_k)
        u[i, j] = -math.sin(phi_k)
        u[j, i] = math.sin(phi_k)
        u[j, j] = math.cos(phi_k)
        return u

    def S(A):
        sum = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    sum += A[i, j] ** 2
        return sum

    def max(A):
        mi, mj, max_a = 0, 1, abs(A[0, 1])
        for i in range(n):
            for j in range(n):
                if abs(A[i, j]) > max_a and i != j:
                    mi, mj, max_a = i, j, abs(A[i, j])
        return mi, mj

    def diff_eigenvalue(A: np.ndarray, prev_eigenvalue: np.ndarray):
        return abs(A.diagonal() - prev_eigenvalue).max()

    e_vect = np.identity(n)
    prev_eigenvalue = A.diagonal() + 2 * eps
    while S(A) > eps and diff_eigenvalue(A, prev_eigenvalue) > eps:
        i, j = max(A)
        U_ = U(i, j)
        e_vect = e_vect @ U_
        prev_eigenvalue = A.diagonal()
        A = U_.transpose() @ A @ U_
    return A.diagonal(), e_vect


if __name__ == "__main__":
    val, vect = EigenvalueJacob(np.array([[3, 2, 1],
                                          [2, 2, 5],
                                          [1, 5, 1]]))
    print(val)
    print(vect)


# Reproduction one-two parametr

def DF1Parametr(dF_d_theta, D_theta):
    return dF_d_theta ** 2 * D_theta


def DF2Parametr(dF_d_theta1, dF_d_theta2, D_theta1, D_theta2, cov_theta12):
    return dF_d_theta1 ** 2 * D_theta1 + dF_d_theta2 ** 2 * D_theta2 \
        + 2 * dF_d_theta1 * dF_d_theta2 * cov_theta12


# Kvant
c0 = 2.515517
c1 = 0.802853
c2 = 0.010328
d1 = 1.432788
d2 = 0.1892659
d3 = 0.001308
Ea = 4.5 * 10 ** -4


def QuantileNorm(alpha) -> float:
    if alpha <= 0.5:
        p = alpha
    else:
        p = 1 - alpha
    t = math.log(1 / p ** 2) ** 0.5
    u = t - (c0 + c1 * t + c2 * t ** 2) / (
        1 + d1 * t + d2 * t ** 2 + d3 * t ** 3) + Ea
    if alpha <= 0.5:
        u = -u
    return u


def QuantileTStudent(alpha, nu) -> float:
    if nu <= 0:
        return 0.0
    u = QuantileNorm(alpha)
    g1 = 1 / 4 * (u ** 3 + u)
    g2 = 1 / 96 * (5 * u ** 5 + 16 * u ** 3 + 3 * u)
    g3 = 1 / 384 * (3 * u ** 7 + 19 * u ** 5 + 17 * u ** 3 - 15 * u)
    g4 = 1 / 92160 * (79 * u ** 9 + 779 * u ** 7 +
                      1482 * u ** 5 - 1920 * u ** 3 - 945 * u)
    return u + 1 / nu * g1 + 1 / nu ** 2 * g2 + \
        1 / nu ** 3 * g3 + 1 / nu ** 4 * g4


def QuantilePearson(alpha, nu) -> float:
    if nu <= 0:
        return 0.0
    return nu * (1 - 2 / (9 * nu) +
                 QuantileNorm(alpha) * math.sqrt(2 / (9 * nu))) ** 3


def QuantileFisher(alpha, nu1, nu2) -> float:
    dod1 = 0 if nu1 == 0 else 1 / nu1
    dod2 = 0 if nu2 == 0 else 1 / nu2
    sigma = dod1 + dod2
    delta = dod1 - dod2
    u = QuantileNorm(alpha)

    z = u * (sigma / 2) ** 0.5 - 1 / 6 * delta * (u ** 2 + 2) + \
        (sigma / 2) ** 0.5 * (sigma / 24 * (u ** 2 + 3 * u) +
                              1 / 72 * delta ** 2 / sigma * (u ** 3 + 11 * u))\
        - delta * sigma / 120 * (u ** 4 + 9 * u ** 2 + 8) + \
        delta ** 3 / (3240 * sigma) * (3 * u ** 4 + 7 * u ** 2 - 16) + \
        (sigma / 2) ** 0.5 * \
        (sigma ** 2 / 1920 * (u ** 5 + 20 * u ** 3 + 15 * u) +
         delta ** 4 / 2880 * (u ** 5 + 44 * u ** 3 + 183 * u) +
         delta ** 4 / (155520 * sigma ** 2) *
         (9 * u ** 5 - 284 * u ** 3 - 1513 * u))

    return math.exp(2 * z)


# Probability functions

def L(z, N):
    return 1 - math.e ** (-2 * z ** 2) * (
        1 - 2 * z / (3 * N ** 0.5) +
        2 * z ** 2 / (3 * N) * (1 - 2 * z ** 2 / 3) +
        4 * z / (9 * N ** 1.5) * (1 / 5 - 19 * z ** 2 / 15 + 2 * z ** 4 / 3)
        + z ** 12 / N ** 2)


def FNorm(x, m=0, sigma=1):
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


# Probability density functions

def fNorm(x, m=0, sigma=1):
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
