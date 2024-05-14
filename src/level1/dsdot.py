from __future__ import annotations

from exo import *
from exo.platforms.x86 import *

from codegen_helpers import *
from blaslib import *
import exo_blas_config as C


@proc
def dsdot(n: size, x: [f32][n], y: [f32][n], result: f64):
    d_dot: f64
    d_dot = 0.0
    for i in seq(0, n):
        d_x: f64
        d_x = x[i]
        d_y: f64
        d_y = y[i]
        d_dot += d_x * d_y
    result = d_dot


@proc
def sdsdot(n: size, sb: f32, x: [f32][n], y: [f32][n], result: f32):
    d_result: f64
    d_dot: f64
    d_dot = 0.0
    for i in seq(0, n):
        d_x: f64
        d_x = x[i]
        d_y: f64
        d_y = y[i]
        d_dot += d_x * d_y
    d_result = d_dot
    d_result += sb
    result = d_result


for proc in dsdot, sdsdot:
    export_exo_proc(globals(), generate_stride_any_proc(proc))
    stride_1 = generate_stride_1_proc(proc)
    if C.Machine.name == "avx2":
        loop = stride_1.find_loop("i")
        stride_1 = optimize_level_1(stride_1, loop, "f64", C.Machine, 4)
    export_exo_proc(globals(), stride_1)
