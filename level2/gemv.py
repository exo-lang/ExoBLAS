from __future__ import annotations
import os
from pathlib import Path

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API import compile_procs


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
      # NOTE: is there a way to do additive/multiplicative associativity? for now I'm just changing the original proc...
      # e.g if i want to bind_expr (b*c) in a*b*c = (a*b)*c, then (b*c) is not found


@proc
def gemv_transpose(
  alpha: f32,
  beta: f32,
  n: size,
  m: size,
  a: f32[n, m],
  x: f32[n],
  y: f32[m],
):
  for i in seq(0, m):
    y[i] = beta * y[i]
    for j in seq(0, n):
      y[i] += alpha * x[j] * a[j, i]


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

def vectorize_beta_times_y(proc):
  print("vectorizing y[_] = beta * y[_]")

  proc = neon_broadcast_constant(proc, "beta", "ii", n_lifts=2)

  print("\tmaking y[_] a vector")
  proc = neon_stage_expr(proc, "ii", "y[_]", "old_y_vec")
  proc = replace(proc, "for ii in _:_", neon_vld_4xf32)
  # NOTE: need two registers for y because no duplicate arguments allowed
  proc = simplify(stage_mem(proc, "for ii in _:_", "y[4*io:4*io+4]", "new_y_vec"))
  proc = set_memory(proc, "new_y_vec", Neon4f)
  proc = replace(proc, "for ii in _:_", neon_vmul_4xf32)
  proc = replace(proc, "for i0 in _:_", neon_vst_4xf32)

  return proc

def schedule_gemv_transpose_on_neon():
  proc = gemv_transpose
  proc = divide_loop(proc, "i", 4, ["io", "ii"], tail="cut_and_guard")
  proc = fission(proc, proc.find("y[_] = _").after(), n_lifts=2)

  proc = vectorize_beta_times_y(proc)

  print("splitting j loop...")
  proc = divide_loop(proc, "j", 4, ["jo", "ji"], tail="cut_and_guard")
  proc = fission(proc, proc.find("for jo in _:_").after())
  proc = reorder_loops(proc, "ii jo")

  print("vectorizing y[_] += alpha * x[_] * a[_]")

  proc = neon_broadcast_constant(proc, "alpha", "ji", n_lifts=4)

  print("\tmaking y a vector")
  proc = simplify(stage_mem(proc, "for jo in _:_", "y[4*io:4*io+4]", "y_vec"))
  proc = set_memory(proc, "y_vec", Neon4f)
  proc = replace(proc, "for i0 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for i0 in _:_", neon_vst_4xf32)

  print("\tvectorizing alpha[_] * x[_]")
  proc = neon_stage_expr(proc, "ji", "alpha_vec[_] * x[_]", "alpha_times_x", n_lifts=2)
  proc = simplify(stage_mem(proc, "for ji in _:_", "x[4*jo:4*jo+4]", "x_vec"))
  proc = set_memory(proc, "x_vec", Neon4f)
  proc = replace(proc, "for i0 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for ji in _:_", neon_vmul_4xf32)

  print("\tvectorizing alpha_times_x[_] * a[_]")
  # NOTE: swapped some indices, removed the rearrange_dims
  proc = simplify(stage_mem(proc, "for ii in _:_", "a[4*jo:4*jo+4, 4*io:4*io+4]", "a_vecs"))
  proc = set_memory(proc, "a_vecs", Neon4f)
  proc = replace(proc, "for i1 in _:_", neon_vld_4xf32)
  proc = reorder_loops(proc, "ii ji")
  proc = commute_expr(proc, "alpha_times_x[_] * a_vecs[_]")
  proc = replace(proc, "for ii in _:_", neon_vfmla_4xf32_4xf32)

  return proc

def schedule_gemv_on_neon():
  proc = gemv
  proc = divide_loop(proc, "i", 4, ["io", "ii"], tail="cut_and_guard")
  proc = fission(proc, proc.find("y[_] = _").after(), n_lifts=2)

  proc = vectorize_beta_times_y(proc)

  print("splitting j loop...")
  proc = divide_loop(proc, "j", 4, ["jo", "ji"], tail="cut_and_guard")
  proc = fission(proc, proc.find("for jo in _:_").after())
  proc = reorder_loops(proc, "ii jo")

  print("vectorizing y[_] += alpha * x[_] * a[_]")

  proc = neon_broadcast_constant(proc, "alpha", "ji", n_lifts=4)

  print("\tmaking y a vector")
  proc = simplify(stage_mem(proc, "for jo in _:_", "y[4*io:4*io+4]", "y_vec"))
  proc = set_memory(proc, "y_vec", Neon4f)
  proc = replace(proc, "for i0 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for i0 in _:_", neon_vst_4xf32)

  print("\tvectorizing alpha[_] * x[_]")
  proc = neon_stage_expr(proc, "ji", "alpha_vec[_] * x[_]", "alpha_times_x", n_lifts=2)
  proc = simplify(stage_mem(proc, "for ji in _:_", "x[4*jo:4*jo+4]", "x_vec"))
  proc = set_memory(proc, "x_vec", Neon4f)
  proc = replace(proc, "for i0 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for ji in _:_", neon_vmul_4xf32)

  print("\tvectorizing alpha_times_x[_] * a[_]")
  proc = simplify(stage_mem(proc, "for ii in _:_", "a[4*io:4*io+4, 4*jo:4*jo+4]", "a_vecs"))
  proc = set_memory(proc, "a_vecs", Neon4f)
  proc = simplify(stage_mem(proc, "for i0 in _:_", "a[4*io:4*io+4, 4*jo:4*jo+4]", "a_transposed"))
  proc = rearrange_dim(proc, "a_transposed", [1, 0])
  proc = rearrange_dim(proc, "a_vecs", [1, 0])
  # TODO: how to do i0 i1 #1? it doesn't seem to work
  proc = reorder_loops(proc, proc.find("for i0 in _:_ #1"))
  proc = replace(proc, "for i0 in _:_ #1", neon_vld_4xf32)
  proc = reorder_loops(proc, "ii ji")
  proc = commute_expr(proc, "alpha_times_x[_] * a_vecs[_]")
  proc = replace(proc, "for ii in _:_", neon_vfmla_4xf32_4xf32)

  proc = simplify(proc)
  return proc

if __name__ == "__main__":
  # final_proc = schedule_gemv_on_neon()
  final_proc = schedule_gemv_transpose_on_neon()

  print("="*50)
  print("Original proc:")
  print(gemv)
  print("="*50)
  print("Final scheduled proc:")
  print(final_proc)

  compile_procs([final_proc], Path(os.path.expanduser("~/Documents/BLAS/temp")), "proc.c", "proc.h")
  print("Compiled to C!")