from time import time
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NM_SMBLS = 32
VL_SMBLS = 16


def formatName(n: str) -> str:
    return n.ljust(NM_SMBLS)


def formatValue(v: str) -> str:
    return v.center(VL_SMBLS)


def formRowNV(name: str, *args) -> str:
    row = formatName(name)
    for arg in args:
        if type(arg) is str:
            row += formatValue(arg)
        else:
            row += formatValue(f"{arg:.5}")
    return row


def timer(function):
    def wrapper(*args):
        t = time()
        if len(args) == 0:
            function()
        else:
            function(*args)
        logger.debug(f"{function.__qualname__}"
                     f"={time() - t}sec")
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
    if N == 1:
        med = r[0]
    elif N == 2:
        med = (r[0] + r[1]) / 2
    else:
        k = N // 2
        if 2 * k == N:
            med = (r[k] + r[k + 1]) / 2
        else:
            med = r[k + 1]

    return med
