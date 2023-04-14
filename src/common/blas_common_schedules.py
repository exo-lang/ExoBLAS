from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *


"""
NOTE: this pattern of bind_expr, expand_dim, lift_alloc, fission is
pretty common, but can't use stage_mem. I think something like Samir's
stage_expr might be useful as a stdlib function still??
"""


def neon_stage_expr(proc, loop_var, expr, new_name, n_lifts=1):
    proc = bind_expr(proc, expr, new_name)
    proc = set_precision(proc, new_name, "f32")
    proc = set_memory(proc, new_name, Neon)
    proc = expand_dim(proc, new_name, 4, loop_var)
    proc = lift_alloc(proc, new_name, n_lifts=n_lifts)
    proc = autofission(proc, proc.find(f"{new_name}[_] = _").after(), n_lifts=n_lifts)
    # NOTE: before when I used fission + remove_loop instead of autofission,
    # remove_loop was kinda slow depending on the order I ran it in
    return proc


def neon_broadcast_constant(proc, const, loop_var, n_lifts=1):
    const_vec = f"{const}_vec"
    print(f"\tbroadcasting {const} to {const_vec}")
    proc = neon_stage_expr(proc, loop_var, const, const_vec, n_lifts=n_lifts)
    # NOTE: for neon, we need to convert const from f32 to f32[1]
    const_temp = f"{const}_temp"
    proc = stage_mem(proc, f"for {loop_var} in _:_", const, const_temp)
    proc = expand_dim(proc, const_temp, 1, "0")
    proc = replace(proc, f"for {loop_var} in _:_", neon_broadcast_4xf32)
    return proc


"""
Reuse the same pattern for scheduling the first
```
for i in seq(0, m):
  y[i] = beta * y[i]
```
loop in a lot of procedures.
"""


def neon_vectorize_beta_times_y(proc):
    print("vectorizing y[_] = beta * y[_]")

    proc = neon_broadcast_constant(proc, "beta", "ii", n_lifts=2)

    proc = neon_stage_expr(proc, "ii", "y[_]", "old_y_vec")
    proc = replace(proc, "for ii in _:_", neon_vld_4xf32)
    # NOTE: need two registers for y because no duplicate arguments allowed
    proc = simplify(stage_mem(proc, "for ii in _:_", "y[4*io:4*io+4]", "new_y_vec"))
    proc = set_memory(proc, "new_y_vec", Neon)
    proc = replace(proc, "for ii in _:_", neon_vmul_4xf32)
    proc = replace(proc, "for i0 in _:_", neon_vst_4xf32)

    return proc
