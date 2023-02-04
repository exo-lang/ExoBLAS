from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *

"""
Possible cases:
  - transpose or not
  - column or row major storage
"""

@proc
def gemv(
  alpha: f32,
  beta: f32,
  n: size,
  m: size,
  a: f32[m, n],
  x: f32[n],
  y: f32[m],
):
  for i in seq(0, m):
    y[i] = beta * y[i]
    for j in seq(0, n):
      y[i] += alpha * x[j] * a[i, j]


neon_gemv = gemv

# NOTE: is there a way to do additive/multiplicative associativity? for now I'm just changing the original proc...
# e.g if i want to bind_expr (b*c) in a*b*c = (a*b)*c, then (b*c) is not found

neon_gemv = divide_loop(neon_gemv, "i", 4, ["io", "ii"], tail="cut_and_guard")
neon_gemv = fission(neon_gemv, neon_gemv.find("y[_] = _").after(), n_lifts=2)

# NOTE: this pattern of bind_expr, expand_dim, lift_alloc, fission is pretty common,
# but can't use stage_mem. I think something like Samir's stage_expr might be useful as
# a stdlib function still??
def neon_stage_expr(proc, loop_var, expr, new_name, n_lifts=1):
  proc = bind_expr(proc, expr, new_name)
  proc = set_precision(proc, new_name, "f32")
  proc = set_memory(proc, new_name, Neon4f)
  proc = expand_dim(proc, new_name, 4, loop_var)
  proc = lift_alloc(proc, new_name, n_lifts=n_lifts)
  proc = autofission(proc, proc.find(f"{new_name}[_] = _").after(), n_lifts=n_lifts)
  # NOTE: before when I used fission + remove_loop instead of autofission,
  # remove_loop was kinda slow depending on the order I ran it in
  return proc

print("vectorizing y[_] = beta * y[_]")

print("\tbroadcasting beta to beta_vec")
neon_gemv = neon_stage_expr(neon_gemv, "ii", "beta", "beta_vec", n_lifts=2)
# neon_gemv = remove_loop(neon_gemv, "io #0")
# NOTE: needed to convert beta from f32 to f32[1]
neon_gemv = stage_mem(neon_gemv, "for ii in _:_", "beta", "beta_temp")
neon_gemv = expand_dim(neon_gemv, "beta_temp", 1, "0")
neon_gemv = replace(neon_gemv, "for ii in _:_", neon_broadcast_4xf32)

print("\tmaking y[_] a vector")
neon_gemv = neon_stage_expr(neon_gemv, "ii", "y[_]", "old_y_vec")
neon_gemv = replace(neon_gemv, "for ii in _:_", neon_vld_4xf32)
# NOTE: need two registers for y because no duplicate arguments allowed
neon_gemv = simplify(stage_mem(neon_gemv, "for ii in _:_", "y[4*io:4*io+4]", "new_y_vec"))
neon_gemv = set_memory(neon_gemv, "new_y_vec", Neon4f)
neon_gemv = replace(neon_gemv, "for ii in _:_", neon_vmul_4xf32)
neon_gemv = replace(neon_gemv, "for i0 in _:_", neon_vst_4xf32)

print("splitting j loop...")
neon_gemv = divide_loop(neon_gemv, "j", 4, ["jo", "ji"], tail="cut_and_guard")
neon_gemv = fission(neon_gemv, neon_gemv.find("for jo in _:_").after())
neon_gemv = reorder_loops(neon_gemv, "ii jo")

print("vectorizing y[_] += alpha * x[_] * a[_]")

print("\tbroadcasting alpha to alpha_vec")
neon_gemv = neon_stage_expr(neon_gemv, "ji", "alpha", "alpha_vec", n_lifts=4)
# NOTE: need to convert alpha from f32 to f32[1]
neon_gemv = stage_mem(neon_gemv, "for ji in _:_", "alpha", "alpha_temp")
neon_gemv = expand_dim(neon_gemv, "alpha_temp", 1, "0")
neon_gemv = replace(neon_gemv, "for ji in _:_", neon_broadcast_4xf32)

print("\tmaking y a vector")
neon_gemv = simplify(stage_mem(neon_gemv, "for jo in _:_", "y[4*io:4*io+4]", "y_vec"))
neon_gemv = set_memory(neon_gemv, "y_vec", Neon4f)
neon_gemv = replace(neon_gemv, "for i0 in _:_", neon_vld_4xf32)
neon_gemv = replace(neon_gemv, "for i0 in _:_", neon_vst_4xf32)

print("\tvectorizing alpha[_] * x[_]")
neon_gemv = neon_stage_expr(neon_gemv, "ji", "alpha_vec[_] * x[_]", "alpha_times_x", n_lifts=2)
neon_gemv = simplify(stage_mem(neon_gemv, "for ji in _:_", "x[4*jo:4*jo+4]", "x_vec"))
neon_gemv = set_memory(neon_gemv, "x_vec", Neon4f)
neon_gemv = replace(neon_gemv, "for i0 in _:_", neon_vld_4xf32)
neon_gemv = replace(neon_gemv, "for ji in _:_", neon_vmul_4xf32)

print("\tvectorizing alpha_times_x[_] * a[_]")
neon_gemv = simplify(stage_mem(neon_gemv, "for ii in _:_", "a[4*io:4*io+4, 4*jo:4*jo+4]", "a_vecs"))
neon_gemv = set_memory(neon_gemv, "a_vecs", Neon4f)
neon_gemv = simplify(stage_mem(neon_gemv, "for i0 in _:_", "a[4*io:4*io+4, 4*jo:4*jo+4]", "a_transposed"))
neon_gemv = rearrange_dim(neon_gemv, "a_transposed", [1, 0])
neon_gemv = rearrange_dim(neon_gemv, "a_vecs", [1, 0])
# TODO: how to do i0 i1 #1? it doesn't seem to work
neon_gemv = reorder_loops(neon_gemv, neon_gemv.find("for i0 in _:_ #1"))
neon_gemv = replace(neon_gemv, "for i0 in _:_ #1", neon_vld_4xf32)

neon_gemv = reorder_loops(neon_gemv, "ii ji")
neon_gemv = commute_expr(neon_gemv, "alpha_times_x[_] * a_vecs[_]")
neon_gemv = replace(neon_gemv, "for ii in _:_", neon_vfmla_4xf32_4xf32)

neon_gemv = simplify(neon_gemv)

# TODO: check that the load to a_vecs is right
# NOTE: I think we need unroll_memory for this to actually compile...

if __name__ == "__main__":
  print("="*50)
  print("Original proc:")
  print(gemv)
  print("="*50)
  print("Final scheduled proc:")
  print(neon_gemv)

__all__ = ["gemv", "neon_gemv"]
