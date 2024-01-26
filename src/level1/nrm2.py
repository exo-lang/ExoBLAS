from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import *
from codegen_helpers import *
from parameters import *

### EXO_LOC ALGORITHM START ###
@proc
def nrm2_template(
    n: size,
    x: [f32][n],
    t_sml: f32,
    t_big: f32,
    s_sml: f32,
    s_big: f32,
    a_sml: f32,
    a_med: f32,
    a_big: f32,
):
    zero: f32
    zero = 0.0
    a_sml = 0.0
    a_med = 0.0
    a_big = 0.0
    for i in seq(0, n):
        x_abs: f32
        x_abs = select(zero, x[i], x[i], -x[i])
        a_big += select(t_big, x_abs, (x[i] * s_big) * (x[i] * s_big), zero)
        a_sml += select(x_abs, t_sml, (x[i] * s_sml) * (x[i] * s_sml), zero)
        a_med += select(t_big, x_abs, zero, select(x_abs, t_sml, zero, x[i] * x[i]))


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
for precision in ("f32", "f64"):
    proc_stride_any = generate_stride_any_proc(nrm2_template, precision)
    proc_stride_any = bind_builtins_args(
        proc_stride_any, proc_stride_any.body(), precision
    )
    export_exo_proc(globals(), proc_stride_any)

    params = Level_1_Params(precision=precision)
    proc_stride_1 = generate_stride_1_proc(nrm2_template, params.precision)
    proc_stride_1 = bind_builtins_args(proc_stride_1, proc_stride_1.body(), precision)
    print(proc_stride_1)
    export_exo_proc(globals(), proc_stride_1)
