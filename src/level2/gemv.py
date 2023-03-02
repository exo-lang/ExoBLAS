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
def sgemv_v2(
  alpha: f32,
  beta: f32,
  m: size,
  n: size,
  a: f32[m, n],
  x: f32[n],
  y: f32[m],
):
  for i in seq(0, m):
    y_tmp : f32 @ DRAM
    y_tmp = 0.0
    for j in seq(0, n):
      y_tmp += a[i, j] * x[j]
    y[i] = beta * y[i] + alpha * y_tmp


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


def replace_all_neon_instrs(proc):
  return replace_all(proc, [
    neon_zero_4xf32,
    neon_vld_4xf32,
    neon_vld1_4x4xf32,
    neon_vadd_4xf32_alias_hack,
    neon_vfmadd_4xf32_4xf32,
    neon_vfmadd_4xf32_4xf32_hack,
    neon_vaddv_4xf32,
  ])


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


@instr("{dst_data} = vaddvq_f32({src_data});")
def neon_vaddv_4xf32(dst: [f32][1] @ DRAM, src: [f32][4] @ Neon4f):
  dst[0] = src[0] + src[1] + src[2] + src[3]


@instr("{dst_data} = vld4q_f32(&{src_data});")
def neon_vld4_4x4xf32(
    dst: [f32][4, 4] @ Neon4x4f,
    src: [f32][16] @ DRAM,
):
    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = src[4*j+i]


@instr("{dst_data} = vld1q_f32_x4(&{src_data});")
def neon_vld1_4x4xf32(
    dst: [f32][4, 4] @ Neon4x4f,
    src: [f32][16] @ DRAM,
):
    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = src[4*i+j]


@instr("vst1q_f32(&{dst_data}, vld1q_f32(&{src_data}));")
def neon_copy_4xf32(dst: [f32][4] @ DRAM, src: [f32][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr("{dst_data} = vaddq_f32({dst_data}, {rhs_data});")
def neon_vadd_4xf32_alias_hack(
    dst: [f32][4] @ Neon4f, rhs: [f32][4] @ Neon4f
):
    assert stride(dst, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] += rhs[i]


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


@instr("{dst_data} = vmlaq_f32({dst_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_4xf32_4xf32_hack(
    dst: [f32][4] @ Neon4f, lhs: [f32][4] @ Neon4x4f, rhs: [f32][4] @ Neon4x4f
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] += lhs[i] * rhs[i]


@instr("{dst_data} = vld1q_f32(&{src_data});")
def neon_vld_4xf32_hack(dst: [f32][4] @ Neon4x4f, src: [f32][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


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
  proc = replace(proc, "for i1 in _:_", neon_vld4_4x4xf32)

  print("\tmaking x a vector")
  proc = simplify(stage_mem(proc, "for ji in _:_", "x[4*jo:4*jo+4]", "x_vec"))
  proc = set_memory(proc, "x_vec", Neon4f)
  proc = replace(proc, "for i0 in _:_ #1", neon_vld_4xf32)
  proc = replace(proc, "for ii in _:_", neon_vmul_lane_4xf32_4xf32_hack)
  proc = replace(proc, "for ii in _:_", neon_vadd_4xf32_alias_hack)

  proc = unroll_loop(proc, "i0")
  proc = unroll_loop(proc, "ji")

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
    print("Loop organization for i tiling")
    proc = fission(proc, proc.find("for jo in _:_").before())
    proc = fission(proc, proc.find("for jo in _:_").after())
    proc = reorder_loops(proc, "ioi jo")
    for i in range(3):
      proc = unroll_loop(proc, "ioi")

  return simplify(proc)


def neon_sgemv_row_product(i_tile):
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

  print("\treducing partial sums")
  proc = simplify(stage_mem(proc, "for ji in _:_ #1", f"y[{i_tile}*io+ii]", "y_acc", accum=True))
  proc = expand_dim(proc, "y_acc", i_tile, "ii")
  proc = lift_alloc(proc, "y_acc", 2)
  proc = unroll_loop(proc, "ji #1")
  proc = merge_writes(proc, "y_acc[ii] = 0.0; y_acc[ii] += y_partial_sums_vec[ii, 0]")
  proc = simplify(proc)
  proc = merge_writes(proc, "y_acc[ii] = y_partial_sums_vec[ii, 0]; y_acc[ii] += y_partial_sums_vec[ii, 1]")
  proc = merge_writes(proc, "y_acc[ii] = y_partial_sums_vec[ii, 0] + y_partial_sums_vec[ii, 1]; y_acc[ii] += y_partial_sums_vec[ii, 2]")
  proc = merge_writes(proc, "y_acc[ii] = y_partial_sums_vec[ii, 0] + y_partial_sums_vec[ii, 1] + y_partial_sums_vec[ii, 2]; y_acc[ii] += y_partial_sums_vec[ii, 3]")
  proc = replace(proc, "y_acc[_] = _", neon_vaddv_4xf32)

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

  proc = unroll_loop(proc, "i0")
  for i in range(3):
    proc = unroll_loop(proc, "ii #1")

  return proc

neon_sgemv_rows = rename(neon_sgemv_row_product(i_tile=8), "sgemv_exo_8_rows")


def schedule_neon_sgemv_rows_load1():
  proc = sgemv_v2

  # TODO: schedule beta * y[i] + alpha * y_tmp part
  proc = expand_dim(proc, "y_tmp", 1, 0) # TODO: hack because f32 != f32[1]

  print("vectorizing")
  proc = divide_loop(proc, "j", 4, ["jo", "jv"], tail="cut_and_guard")

  print("\tvectorizing y")
  proc = reorder_loops(proc, "jo jv")
  proc = simplify(stage_mem(proc, "for jo in _:_", "y_tmp[0]", "y_vec", accum=True))
  proc = expand_dim(proc, "y_vec", 4, "jv")
  proc = lift_alloc(proc, "y_vec")
  proc = fission(proc, proc.find("for jo in _:_").before())
  proc = fission(proc, proc.find("for jo in _:_").after())
  for i in range(3):
    proc = reorder_stmts(proc, proc.find("y_tmp[0] = 0.0").expand(1))
  proc = unroll_loop(proc, "for jv in _:_ #2")
  for i in range(4):
    proc = merge_writes(proc, proc.find("y_tmp[0] = _").expand(1))
  proc = reorder_loops(proc, "jv jo")

  print("\tvectorizing x and a")
  proc = simplify(stage_mem(proc, "for jv in _:_ #1", "x[4*jo:4*(jo+1)]", "x_vec"))
  proc = simplify(stage_mem(proc, "for jv in _:_ #1", "a[i, 4*jo:4*(jo+1)]", "a_vec"))

  print("loading 4 vectors")
  proc = divide_loop(proc, "jo", 4, ["jo", "ji"], tail="cut_and_guard")

  print("\texpanding x_vec and a_vec")
  for vec in ["x_vec", "a_vec"]:
    proc = expand_dim(proc, vec, 4, "ji")
    proc = lift_alloc(proc, f"{vec} : _", 1)
  for i in range(2):
    proc = fission(proc, proc.find("for ji in _:_").body()[-1].before())

  for mem in ["y_vec", "y_vec_1", "y_final", "x_vec", "a_vec"]:
    proc = set_memory(proc, mem, Neon4f)

  return simplify(proc)


# HACK: Manually modified memory types to Neon4x4f
# neon_sgemv_load1 = rename(schedule_neon_sgemv_rows_load1(), "sgemv_exo_row_load1")
@proc
def neon_sgemv_exo_row_load1_hack(alpha: f32 @ DRAM, beta: f32 @ DRAM, m: size, n: size,
                        a: f32[m, n] @ DRAM, x: f32[n] @ DRAM,
                        y: f32[m] @ DRAM):
    for i in seq(0, m):
        y_tmp: f32[1] @ DRAM
        y_vec: f32[4] @ Neon4f
        for jv in seq(0, 4):
            y_vec[jv] = 0.0
        for jo in seq(0, n / 4 / 4):
            x_vec: f32[4, 4] @ Neon4x4f
            a_vec: f32[4, 4] @ Neon4x4f
            for ji in seq(0, 4):
                for i0 in seq(0, 4):
                    x_vec[ji, i0] = x[i0 + 4 * ji + 16 * jo]
            for ji in seq(0, 4):
                for i0 in seq(0, 4):
                    a_vec[ji, i0] = a[i, i0 + 4 * ji + 16 * jo]
            for ji in seq(0, 4):
                for jv in seq(0, 4):
                    y_vec[jv] += a_vec[ji, jv] * x_vec[ji, jv]
        if n / 4 % 4 > 0:
            for ji in seq(0, n / 4 % 4):
                x_vec: f32[4] @ Neon4f
                for i0 in seq(0, 4):
                    x_vec[i0] = x[i0 + 4 * (ji + n / 4 / 4 * 4)]
                a_vec: f32[4] @ Neon4f
                for i0 in seq(0, 4):
                    a_vec[i0] = a[i, i0 + 4 * (ji + n / 4 / 4 * 4)]
                for jv in seq(0, 4):
                    y_vec[jv] += a_vec[jv] * x_vec[jv]
        y_tmp[0] = y_vec[0] + y_vec[1] + y_vec[2] + y_vec[3]
        if n % 4 > 0:
            for jv in seq(0, n % 4):
                y_tmp[0] += a[i, jv + n / 4 * 4] * x[jv + n / 4 * 4]
        y[i] = beta * y[i] + alpha * y_tmp[0]


def schedule_neon_sgemv_rows_load1():
  proc = neon_sgemv_exo_row_load1_hack
  proc = replace_all_neon_instrs(proc)
  proc = unroll_loop(proc, "for ji in _:_")
  return simplify(proc)
neon_sgemv_load1 = rename(schedule_neon_sgemv_rows_load1(), "sgemv_exo_row_load1")
# print(neon_sgemv_load1)

def schedule_neon_sgemv_rows_load1_tiled(j_tile=2):
  proc = sgemv_v2

  # TODO: schedule beta * y[i] + alpha * y_tmp part
  proc = expand_dim(proc, "y_tmp", 1, 0) # TODO: hack because f32 != f32[1]

  print("vectorizing")
  proc = divide_loop(proc, "j", 4, ["jo", "jv"], tail="cut_and_guard")
  proc = divide_loop(proc, "jo", 4, ["jo", "ji"], tail="cut_and_guard")

  print("\tvectorizing y")
  proc = reorder_loops(proc, "jo ji")
  proc = reorder_loops(proc, "jo jv")
  proc = simplify(stage_mem(proc, "for jo in _:_", "y_tmp[0]", "y_vec", accum=True))
  proc = expand_dim(proc, "y_vec", 4, "jv")
  proc = expand_dim(proc, "y_vec", 4, "ji")
  proc = lift_alloc(proc, "y_vec", 2)
  proc = fission(proc, proc.find("for jo in _:_").before(), 2)
  proc = fission(proc, proc.find("for jo in _:_").after(), 2)
  proc = reorder_loops(proc, proc.find("for ji in _:_ #2"))

  proc = simplify(stage_mem(proc, "for ji in _:_ #2", "y_tmp[0]", "y_final", accum=True))
  proc = expand_dim(proc, "y_final", 4, "jv")
  proc = lift_alloc(proc, "y_final")
  proc = fission(proc, proc.find("for ji in _:_ #2").before())
  proc = fission(proc, proc.find("for ji in _:_ #2").after())
  proc = reorder_loops(proc, "jv ji")
  for i in range(6):
    proc = reorder_stmts(proc, proc.find("y_tmp[_] = _").expand(1))
  proc = unroll_loop(proc, "for jv in _:_ #4")
  for i in range(4):
    proc = merge_writes(proc, proc.find("y_tmp[_] = _").expand(1))
  proc = reorder_loops(proc, "jv jo")
  proc = reorder_loops(proc, "ji jo")

  print("\tvectorizing x and a")
  proc = simplify(stage_mem(proc, "for jv in _:_ #1", "x[16*jo+4*ji:16*jo+4*(ji+1)]", "x_vec"))
  proc = simplify(stage_mem(proc, "for jv in _:_ #1", "a[i, 16*jo+4*ji:16*jo+4*(ji+1)]", "a_vec"))

  print("\texpanding x_vec and a_vec")
  for vec in ["x_vec", "a_vec"]:
    proc = expand_dim(proc, vec, 4, "ji")
    proc = lift_alloc(proc, f"{vec} : _", 1)
  for i in range(2):
    proc = fission(proc, proc.find("for ji in _:_ #1").body()[-1].before())

  print("replacing with Neon instructions")
  for mem in ["y_vec", "y_final"]:
    proc = set_memory(proc, mem, Neon4f)
  for mem in ["x_vec", "a_vec", "x_vec_1", "a_vec_1"]:
    proc = set_memory(proc, mem, Neon4x4f)
  proc = replace_all_neon_instrs(proc)

  for i in range(3):
    proc = unroll_loop(proc, "for ji in _:_")

  proc = divide_loop(proc, "jo", j_tile, ["joo", "joi"], tail="cut_and_guard")
  proc = unroll_loop(proc, "joi")

  return simplify(proc)



def schedule_neon_sgemv_rows_load1_multi_rows(i_tile=4):
  proc = neon_sgemv_exo_row_load1_hack

  print("dividing i loop")
  proc = divide_loop(proc, "i", i_tile, ["io", "ii"], tail="cut_and_guard")
  for mem in ["y_tmp", "y_vec"]:
    proc = expand_dim(proc, mem, i_tile, "ii")
  proc = mult_dim(proc, "y_tmp", 0, 1)
  for mem in ["y_tmp", "y_vec"]:
    proc = lift_alloc(proc, mem)
  for i in range(5):
    proc = fission(proc, proc.find("for ii in _:_").body()[-1].before())
  proc = reorder_loops(proc, "ii jo")
  for mem in ["a_vec", "x_vec"]:
    proc = lift_alloc(proc, mem)
  proc = autofission(proc, proc.find("for ii in _:_ #1").body()[0].after())

  print("replacing with Neon instructions")
  proc = replace_all(proc, [
    neon_zero_4xf32,
    neon_vld_4xf32,
    neon_vld1_4x4xf32,
    neon_vfmadd_4xf32_4xf32,
    neon_vfmadd_4xf32_4xf32_hack,
    neon_vaddv_4xf32,
  ])

  print("unrolling loop")
  proc = unroll_loop(proc, "for ji in _:_")
  proc = unroll_loop(proc, "for ji in _:_ #1")

  return simplify(proc)


neon_sgemv_load1_tiled = rename(schedule_neon_sgemv_rows_load1_tiled(j_tile=2), "sgemv_exo_row_load1_tiled")
print(neon_sgemv_load1_tiled)
neon_sgemv_load1_multi_rows = rename(schedule_neon_sgemv_rows_load1_multi_rows(i_tile=4), "sgemv_exo_row_load1_multi_rows")

__all__ = ["neon_sgemv_rows", "neon_sgemv_load1", "neon_sgemv_load1_multi_rows", "neon_sgemv_load1_tiled"]