from time import time
import logging
import numpy as np
import math


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NM_SMBLS = 48
VL_SMBLS = 16

PROTOCOL_TITLE_SEP = ((NM_SMBLS + VL_SMBLS * 4 - 8) // 2) * "-"

PROTOCOL_TITLE = PROTOCOL_TITLE_SEP + "ПРОТОКОЛ" + PROTOCOL_TITLE_SEP + "\n"


def format_name(n: str) -> str:
    return n.ljust(NM_SMBLS)


def padding_value(v: str) -> str:
    return v.center(VL_SMBLS)


def format_value(val) -> str:
    if type(val) is str:
        return padding_value(val)
    if np.issubdtype(type(val), np.integer):
        return padding_value(f"{val}")
    if np.issubdtype(type(val), np.floating):
        if abs(val) < 0.00001:
            val = 0.0
        return padding_value(f"{val:.5}")

    return padding_value(f"{val}")


def formRowNV(name: str, *args) -> str:
    row = format_name(name)
    for arg in args:
        row += format_value(arg)
    return row


def timer(function):
    def wrapper(*args):
        t = time()
        if len(args) == 0:
            ret = function()
        else:
            ret = function(*args)
        logger.debug(f"{function.__qualname__}"
                     f"={time() - t}sec")
        return ret
    return wrapper


def toCalcRankSeries(x):  # (xl, rx)
    N_G = len(x)
    prev = x[0][0]
    x[0][1] = 1
    i = 1
    avr_r = 0
    avr_i = 0
    for i in range(1, N_G):
        if prev == x[i][0]:
            avr_r += i
            avr_i += 1
        else:
            x[i][1] = i + 1
            if avr_r != 0:
                avr_r = avr_r / avr_i
                j = i - 1
                while x[j][1] != 0:
                    x[j][1] = avr_r
                    j -= 1
                avr_r = 0
                avr_i = 0
        prev = x[i][0]
    return x


def MED(r):
    N = len(r)
    k = N // 2
    if 2 * k == N:
        med = (r[k - 1] + r[k]) / 2
    else:
        med = r[k]
    return med


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def calculate_m(n: int) -> int:
    if n == 2:
        return 2
    elif n < 100:
        m = math.floor(math.sqrt(n))
    else:
        m = math.floor(n ** (1 / 3))
    m -= 1 - m % 2
    return m
