from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *


@proc
def rotm_flag_neg_one(n: size, x: [R][n], y: [R][n], H: R[2, 2]):
    for i in seq(0, n):
        xReg: R
        xReg = x[i]
        x[i] = H[0, 0] * xReg + H[0, 1] * y[i]
        y[i] = H[1, 0] * xReg + H[1, 1] * y[i]


@proc
def rotm_flag_zero(n: size, x: [R][n], y: [R][n], H: R[2, 2]):
    for i in seq(0, n):
        xReg: R
        xReg = x[i]
        x[i] = xReg + H[0, 1] * y[i]
        y[i] = H[1, 0] * xReg + y[i]


@proc
def rotm_flag_one(n: size, x: [R][n], y: [R][n], H: R[2, 2]):
    for i in seq(0, n):
        xReg: R
        xReg = x[i]
        x[i] = H[0, 0] * xReg + y[i]
        y[i] = -xReg + H[1, 1] * y[i]


for proc in rotm_flag_neg_one, rotm_flag_zero, rotm_flag_one:
    variants_generator(optimize_level_1)(proc, "i", 4, globals=globals())
