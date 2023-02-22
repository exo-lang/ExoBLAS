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

import exo_blas_config as C

@proc
def sgemv(
  alpha: f32,
  beta: f32,
  m: size,
  n: size,
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
def sgemv_simple(
  alpha: f32,
  beta: f32,
  m: size,
  n: size,
  a: f32[m, n],
  x: f32[n],
  y: f32[m],
):
  for i in seq(0, m):
    for j in seq(0, n):
      y[i] += a[i, j] * x[j]

@proc
def sgemv_transpose(
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


@instr("{dst_data} = vld4q_f32(&{src_data});")
def neon_vld_4x4xf32(
    dst: [f32][4, 4] @ Neon4x4f,
    src: [f32][16] @ DRAM,
):
    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = src[4*j+i]


@instr("{dst_data} = vaddq_f32({dst_data}, {rhs_data});")
def neon_vadd_4xf32_alias_hack(
    dst: [f32][4] @ Neon4f, rhs: [f32][4] @ Neon4f
):
    assert stride(dst, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] += rhs[i]


@instr("vst1q_f32(&{dst_data}, vld1q_f32(&{src_data}));")
def neon_copy_4xf32(dst: [f32][4] @ DRAM, src: [f32][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr("{dst_data} = vmulq_laneq_f32({lhs_data}, {rhs_data}, {lane});")
def neon_vmul_lane_4xf32_4xf32_hack(
    dst: [f32][4] @ Neon4f, lhs: [f32][4] @ Neon4x4f, rhs: [f32][4] @ Neon4f, lane: index
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert lane >= 0
    assert lane < 4
    for i in seq(0, 4):
        dst[i] = lhs[i] * rhs[lane]


def schedule_sgemv_on_neon(i_tile=1):
  proc = sgemv_simple

  proc = divide_loop(proc, "i", 4, ["io", "ii"], tail="cut_and_guard")
  # proc = fission(proc, proc.find("y[_] = _").after(), n_lifts=2)
  # proc = neon_vectorize_beta_times_y(proc)

  print("splitting j loop...")
  proc = divide_loop(proc, "j", 4, ["jo", "ji"], tail="cut_and_guard")
  proc = fission(proc, proc.find("for jo in _:_").after())
  proc = reorder_loops(proc, "ii jo")

  print("making y a vector")
  proc = simplify(stage_mem(proc, "for jo in _:_", "y[4*io:4*io+4]", "y_vec", accum=True))
  proc = set_memory(proc, "y_vec", Neon4f)
  proc = replace(proc, "for i0 in _:_", neon_zero_4xf32)
  proc = simplify(stage_mem(proc, "for i0 in _:_", "y[4*io:4*io+4]", "y_vec_final"))
  proc = set_memory(proc, "y_vec_final", Neon4f)
  proc = replace(proc, "for i0 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for i0 in _:_", neon_vadd_4xf32_alias_hack)
  proc = replace(proc, "for i0 in _:_", neon_vst_4xf32)

  # proc = neon_broadcast_constant(proc, "alpha", "ji", n_lifts=4)

  print("\tvectorizing x[_] * a[_]")
  proc = reorder_loops(proc, "ii ji")
  proc = neon_stage_expr(proc, "ii", "a[_] * x[_]", "a_times_x", n_lifts=1)

  print("\tintroducing transpose")
  proc = simplify(stage_mem(proc, "for ji in _:_", "a[4*io:4*io+4, 4*jo:4*jo+4]", "a_transposed"))
  proc = set_memory(proc, "a_transposed", Neon4x4f)
  proc = rearrange_dim(proc, "a_transposed", [1, 0])

  print("\tloading A to 4x4 scratch")
  proc = simplify(stage_mem(proc, "for i0 in _:_", "a[4*io:4*io+4, 4*jo:4*jo+4]", "a_vecs"))
  proc = mult_dim(proc, "a_vecs : _", 0, 1)
  proc = lift_alloc(proc, "a_vecs", 2)
  proc = replace(proc, "for i1 in _:_", neon_copy_4xf32)
  proc = reorder_loops(proc, "i0 i1")
  proc = replace(proc, "for i1 in _:_", neon_vld_4x4xf32)

  print("\tmaking x a vector")
  proc = simplify(stage_mem(proc, "for ji in _:_", "x[4*jo:4*jo+4]", "x_vec"))
  proc = set_memory(proc, "x_vec", Neon4f)
  proc = replace(proc, "for i0 in _:_ #1", neon_vld_4xf32)
  proc = replace(proc, "for ii in _:_", neon_vmul_lane_4xf32_4xf32_hack)
  proc = replace(proc, "for ii in _:_", neon_vadd_4xf32_alias_hack)

  if i_tile > 1:
    print("tiling i")
    proc = divide_loop(proc, "io", i_tile, ["ioo", "ioi"], tail="cut_and_guard")
    proc = expand_dim(proc, "y_vec", i_tile, "ioi")
    proc = expand_dim(proc, "a_transposed", i_tile, "ioi")
    proc = expand_dim(proc, "x_vec", i_tile, "ioi")
    proc = expand_dim(proc, "a_times_x", i_tile, "ioi")
    proc = expand_dim(proc, "y_vec_final", i_tile, "ioi")

  print("lifting allocs")
  proc = lift_alloc(proc, "y_vec", 1)
  proc = lift_alloc(proc, "a_transposed", 2)
  proc = lift_alloc(proc, "x_vec", 2)
  proc = lift_alloc(proc, "a_times_x", 3)
  proc = lift_alloc(proc, "y_vec_final", 1)

  if i_tile > 1:
    print(proc)
    proc = fission(proc, proc.find("for jo in _:_").before())
    proc = fission(proc, proc.find("for jo in _:_").after())
    proc = reorder_loops(proc, "ioi jo")

  print("unrolling loops")
  proc = unroll_loop(proc, "i0")
  proc = unroll_loop(proc, "ji")
  if i_tile > 1:
    for i in range(3):
      proc = unroll_loop(proc, "ioi")

  return simplify(proc)


def sgemv_on_neon_row_product(i_tile=8):
  proc = sgemv
  proc = fission(proc, proc.find("y[_] = _").after(), n_lifts=2)
  proc = divide_loop(proc, "i", 4, ["io", "ii"], tail="cut_and_guard")

  proc = neon_vectorize_beta_times_y(proc)

  print("splitting j loop...")
  proc = divide_loop(proc, "j", 4, ["jo", "ji"], tail="cut_and_guard")
  proc = fission(proc, proc.find("for jo in _:_").after())

  print("vectorizing y[_] += alpha * x[_] * a[_]")
  proc = divide_loop(proc, "i", i_tile, ["io", "ii"], tail="cut_and_guard")
  proc = neon_broadcast_constant(proc, "alpha", "ji", n_lifts=4)

  print("\tstaging y_partial_sums")
  proc = reorder_loops(proc, "jo ji")
  proc = simplify(stage_mem(proc, "for jo in _:_", f"y[{i_tile}*io+ii]", "y_partial_sums_vec", accum=True))
  proc = expand_dim(proc, "y_partial_sums_vec", 4, "ji")
  proc = expand_dim(proc, "y_partial_sums_vec", i_tile, "ii")
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
  proc = simplify(stage_mem(proc, "for ii in _:_ #2", f"a[{i_tile}*io:{i_tile}*(io+1), 4*jo:4*jo+4]", "a_vec"))
  proc = set_memory(proc, "a_vec", Neon4f)
  proc = replace(proc, "for i1 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for ji in _:_", neon_vfmadd_4xf32_4xf32)

  proc = unroll_loop(proc, "ii #2")
  proc = unroll_loop(proc, "i0")

  # proc = unroll_loop(proc, "ji")
  # proc = unroll_loop(proc, "ii #2")

  return proc


def sgemv_on_neon_tiled_row_product(i_tile=8, j_tile=2):
  proc = sgemv
  proc = fission(proc, proc.find("y[_] = _").after(), n_lifts=2)
  proc = divide_loop(proc, "i", 4, ["io", "ii"], tail="cut_and_guard")

  proc = neon_vectorize_beta_times_y(proc)

  print("splitting j loop...")
  proc = divide_loop(proc, "j", 4, ["jo", "ji"], tail="cut_and_guard")
  proc = fission(proc, proc.find("for jo in _:_").after())

  proc = divide_loop(proc, "i", i_tile, ["io", "ii"], tail="cut_and_guard")
  proc = neon_broadcast_constant(proc, "alpha", "ji", n_lifts=4)

  print("\tstaging y_partial_sums")
  proc = reorder_loops(proc, "jo ji")
  proc = simplify(stage_mem(proc, "for jo in _:_", f"y[{i_tile}*io+ii]", "y_partial_sums_vec", accum=True))
  proc = expand_dim(proc, "y_partial_sums_vec", 4, "ji")
  proc = expand_dim(proc, "y_partial_sums_vec", i_tile, "ii")
  proc = lift_alloc(proc, "y_partial_sums_vec", 2)
  proc = fission(proc, proc.find("for jo in _:_").before(), n_lifts=2)
  proc = fission(proc, proc.find("for jo in _:_").after(), n_lifts=2)
  proc = set_memory(proc, "y_partial_sums_vec", Neon4f)
  proc = replace(proc, "for ji in _:_", neon_zero_4xf32)
  proc = reorder_loops(proc, "ji jo")

  print("\treducing partial sums")
  proc = simplify(stage_mem(proc, "for ji in _:_ #1", "y_partial_sums_vec[ii, 0:4]", "y_partial_sums"))
  proc = set_memory(proc, "y_partial_sums", DRAM)
  proc = replace(proc, "for i0 in _:_", neon_vst_4xf32)
  proc = lift_alloc(proc, "y_partial_sums")

  print("\tvectorizing alpha[_] * x[_]")
  proc = reorder_loops(proc, "ii jo")
  proc = neon_stage_expr(proc, "ji", "alpha_vec[_] * x[_]", "alpha_times_x", n_lifts=1)
  proc = simplify(stage_mem(proc, "for ji in _:_", "x[4*jo:4*jo+4]", "x_vec"))
  proc = replace(proc, "for i0 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for ji in _:_", neon_vmul_4xf32)
  proc = simplify(stage_mem(proc, "for ii in _:_ #2", f"a[{i_tile}*io:{i_tile}*(io+1), 4*jo:4*jo+4]", "a_vec"))
  proc = replace(proc, "for i1 in _:_", neon_vld_4xf32)
  proc = replace(proc, "for ji in _:_", neon_vfmadd_4xf32_4xf32)

  proc = set_memory(proc, "x_vec", Neon4f)
  proc = set_memory(proc, "a_vec", Neon4f)

  proc = lift_alloc(proc, "alpha_times_x")
  proc = lift_alloc(proc, "x_vec")
  proc = autofission(proc, proc.find("neon_vld_4xf32(_) #2").after())
  proc = autofission(proc, proc.find("neon_vmul_4xf32(_) #1").after())

  proc = divide_loop(proc, "jo", j_tile, ["jo", "jm"], tail="cut_and_guard")

  print("\texpanding dims...")
  proc = expand_dim(proc, "alpha_times_x", j_tile, "jm")
  proc = lift_alloc(proc, "alpha_times_x")
  proc = expand_dim(proc, "x_vec", j_tile, "jm")
  proc = lift_alloc(proc, "x_vec")
  proc = expand_dim(proc, "a_vec", j_tile, "jm")
  proc = lift_alloc(proc, "a_vec")
  proc = rearrange_dim(proc, "a_vec : _", [1, 0, 2])

  proc = fission(proc, proc.find("for i0 in _:_").after())
  proc = reorder_loops(proc, "jm i0")
  proc = fission(proc, proc.find("neon_vld_4xf32(_) #2").after())
  proc = fission(proc, proc.find("neon_vmul_4xf32(_) #1").after())

  # proc = unroll_loop(proc, "jm")
  # proc = unroll_loop(proc, "i0")
  # proc = unroll_loop(proc, "jm")
  # proc = unroll_loop(proc, "jm")
  # proc = unroll_loop(proc, "ii #2")
  # proc = unroll_loop(proc, "jm")

  return simplify(proc)


neon_sgemv_new = rename(schedule_sgemv_on_neon(i_tile=1), "sgemv_exo")
print(neon_sgemv_new)
neon_sgemv_8_rows_tiled = rename(sgemv_on_neon_tiled_row_product(i_tile=8, j_tile=8), "sgemv_exo_8_rows_tiled")
neon_sgemv_8_rows = rename(sgemv_on_neon_row_product(i_tile=8), "sgemv_exo_8_rows")

__all__ = ["neon_sgemv_8_rows", "neon_sgemv_8_rows_tiled", "neon_sgemv_new"]