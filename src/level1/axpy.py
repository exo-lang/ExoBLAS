from __future__ import annotations

from exo import *

from codegen_helpers import *
from blaslib import *


@proc
def axpy(n: size, alpha: R, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        y[i] += alpha * x[i]


@proc
def axpy_alpha_1(n: size, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        y[i] += x[i]


for proc in axpy, axpy_alpha_1:
    variants_generator(optimize_level_1)(proc, "i", 8, globals=globals())
