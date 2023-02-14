from __future__ import annotations
import os
import sys
from pathlib import Path

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API import compile_procs

from blas_common_schedules import *

@proc
def sgemv(
  alpha: f32,
  beta: f32,
  m: size,
  n: size,
  lda: size,
  a: f32[m, lda],
  x: f32[n],
  y: f32[m],
):
  assert n <= lda
  for i in seq(0, m):
    y[i] = beta * y[i]
    for j in seq(0, n):
      y[i] += alpha * x[j] * a[i, j]
      # NOTE: is there a way to do additive/multiplicative associativity? for now I'm just changing the original proc...
      # e.g if i want to bind_expr (b*c) in a*b*c = (a*b)*c, then (b*c) is not found


@proc
def sgemv_transpose(
  alpha: f32,
  beta: f32,
  n: size,
  m: size,
  lda: size,
  a: f32[n, lda],
  x: f32[n],
  y: f32[m],
):
  assert m <= lda
  for i in seq(0, m):
    y[i] = beta * y[i]
    for j in seq(0, n):
      y[i] += alpha * x[j] * a[j, i]


def shared_schedule(proc):
  proc = divide_loop(proc, "i", 4, ["io", "ii"], tail="cut_and_guard")
  proc = fission(proc, proc.find("y[_] = _").after(), n_lifts=2)

  proc = neon_vectorize_beta_times_y(proc)

  print("splitting j loop...")
  proc = divide_loop(proc, "j", 4, ["jo", "ji"], tail="cut_and_guard")
  proc = fission(proc, proc.find("for jo in _:_").after())
  proc = reorder_loops(proc, "ii jo")

  print("vectorizing y[_] += alpha * x[_] * a[_]")

  proc = neon_broadcast_constant(proc, "alpha", "ji", n_lifts=4)

  print("\tvectorizing alpha[_] * x[_]")
  proc = neon_stage_expr(proc, "ji", "alpha_vec[_] * x[_]", "alpha_times_x", n_lifts=2)
  proc = simplify(stage_mem(proc, "for ji in _:_", "x[4*jo:4*jo+4]", "x_vec"))
  proc = set_memory(proc, "x_vec", Neon4f)
  proc = replace(proc, "for i0 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for ji in _:_", neon_vmul_4xf32)

  print("\tmaking y a vector")
  proc = simplify(stage_mem(proc, "for jo in _:_", "y[4*io:4*io+4]", "y_vec"))
  proc = set_memory(proc, "y_vec", Neon4f)
  proc = replace(proc, "for i0 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for i0 in _:_", neon_vst_4xf32)

  return proc


def schedule_sgemv_transpose_on_neon():
  proc = sgemv_transpose
  proc = shared_schedule(proc)

  print("\tvectorizing alpha_times_x[_] * a[_]")
  # NOTE: swapped some indices, removed the rearrange_dims since I don't need to transpose
  # prior to loading into the a vector registers
  proc = simplify(stage_mem(proc, "for ii in _:_", "a[4*jo:4*jo+4, 4*io:4*io+4]", "a_vecs"))
  proc = set_memory(proc, "a_vecs", Neon4f)
  proc = replace(proc, "for i1 in _:_", neon_vld_4xf32)
  proc = reorder_loops(proc, "ii ji")
  proc = commute_expr(proc, "alpha_times_x[_] * a_vecs[_]")
  proc = replace(proc, "for ii in _:_", neon_vfmla_4xf32_4xf32)
  proc = unroll_loop(proc, "ji")

  proc = simplify(proc)
  return proc


def schedule_sgemv_on_neon():
  proc = sgemv
  proc = shared_schedule(proc)

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
  proc = unroll_loop(proc, "ji")

  proc = lift_alloc(proc, "a_transposed", n_lifts=2)

  proc = simplify(proc)
  return proc


def schedule_sgemv_on_neon_dot_product():
  proc = sgemv
  proc = fission(proc, proc.find("y[_] = _").after(), n_lifts=2)
  proc = divide_loop(proc, "i", 4, ["io", "ii"], tail="cut_and_guard")

  proc = neon_vectorize_beta_times_y(proc)

  print("splitting j loop...")
  proc = divide_loop(proc, "j", 4, ["jo", "ji"], tail="cut_and_guard")
  proc = fission(proc, proc.find("for jo in _:_").after())

  print("vectorizing y[_] += alpha * x[_] * a[_]")
  proc = divide_loop(proc, "i", 8, ["io", "ii"], tail="cut_and_guard")
  proc = neon_broadcast_constant(proc, "alpha", "ji", n_lifts=3)

  print("\tstaging y_partial_sums")
  proc = reorder_loops(proc, "jo ji")
  proc = simplify(stage_mem(proc, "for jo in _:_", "y[8*io+ii]", "y_partial_sums_vec", accum=True))
  proc = expand_dim(proc, "y_partial_sums_vec", 4, "ji")
  proc = expand_dim(proc, "y_partial_sums_vec", 8, "ii")
  proc = lift_alloc(proc, "y_partial_sums_vec", 2)
  proc = fission(proc, proc.find("for jo in _:_").before(), n_lifts=2)
  proc = fission(proc, proc.find("for jo in _:_").after(), n_lifts=2)
  proc = set_memory(proc, "y_partial_sums_vec", Neon4f)
  proc = replace(proc, "for ji in _:_", neon_zero_4xf32)
  proc = reorder_loops(proc, "ji jo")

  # TODO: can probably be vectorized to accumulate to across i
  print("\treducing partial sums")
  proc = simplify(stage_mem(proc, "for ji in _:_ #1", "y_partial_sums_vec[ii, 0:4]", "y_partial_sums"))
  proc = set_memory(proc, "y_partial_sums", DRAM)
  proc = replace(proc, "for i0 in _:_", neon_vst_4xf32)
  proc = lift_alloc(proc, "y_partial_sums")

  print("\tvectorizing alpha[_] * x[_]")
  proc = reorder_loops(proc, "ii jo")
  proc = neon_stage_expr(proc, "ji", "alpha_vec[_] * x[_]", "alpha_times_x", n_lifts=2)
  proc = simplify(stage_mem(proc, "for ji in _:_", "x[4*jo:4*jo+4]", "x_vec"))
  proc = set_memory(proc, "x_vec", Neon4f)
  proc = replace(proc, "for i0 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for ji in _:_", neon_vmul_4xf32)

  print("\tmaking a[i] a vector")
  proc = simplify(stage_mem(proc, "for ii in _:_ #2", "a[8*io:8*io+8, 4*jo:4*jo+4]", "a_vec"))
  proc = set_memory(proc, "a_vec", Neon4f)
  proc = replace(proc, "for i1 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for ji in _:_", neon_vfmadd_4xf32_4xf32)

  return proc

new_neon_sgemv = rename(schedule_sgemv_on_neon_dot_product(), "sgemv_exo_v2")
neon_sgemv = rename(schedule_sgemv_on_neon(), "sgemv_exo")
neon_sgemv_transpose = rename(schedule_sgemv_transpose_on_neon(), "sgemv_transpose_exo")

print("="*50)
print("Original GEMV:")
print(sgemv)
print("="*50)
print("Neon GEMV:")
print(new_neon_sgemv)
print("="*50)
# print("Original GEMV Transpose:")
# print(sgemv_transpose)
# print("="*50)
# print("Neon GEMV Transpose:")
# print(neon_sgemv_transpose)
# print("="*50)


__all__ = ["neon_sgemv", "new_neon_sgemv", "neon_sgemv_transpose"]
