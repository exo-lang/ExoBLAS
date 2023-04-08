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
    result: f32
    result = 0.0
    for j in seq(0, n):
      result += x[j] * a[i, j]
    y[i] = beta * y[i] + alpha * result
      # NOTE: is there a way to do additive/multiplicative associativity? for now I'm just changing the original proc...
      # e.g if i want to bind_expr (b*c) in a*b*c = (a*b)*c, then (b*c) is not found


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


def schedule_sgemv_on_neon_dot_product(i_tile=8):
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

  return proc

# neon_sgemv = rename(schedule_sgemv_on_neon_dot_product(i_tile=8), "sgemv_exo")
# neon_sgemv_transpose = rename(schedule_sgemv_transpose_on_neon(), "sgemv_transpose_exo")

def schedule_sdot_stride_1_interleaved(VEC_W, INTERLEAVE_FACTOR, memory, instructions):
    stride_1 = rename(sgemv, sgemv.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")
    stride_1 = stride_1.add_assertion("stride(y, 0) == 1")
        
    stride_1 = divide_loop(stride_1, "for j in _:_", VEC_W * INTERLEAVE_FACTOR, ("jo", "ji"), tail = "cut")
    stride_1 = divide_loop(stride_1, "for ji in _:_", VEC_W, ("jm", "ji"), perfect=True)
    
    stride_1 = reorder_loops(stride_1, "jo jm")
    stride_1 = reorder_loops(stride_1, "jo ji")
    stride_1 = simplify(stage_mem(stride_1, "for jo in _:_", "result", "resultReg", accum=True))
    stride_1 = expand_dim(stride_1, "resultReg :_ ", VEC_W, "ji")
    stride_1 = expand_dim(stride_1, "resultReg :_ ", INTERLEAVE_FACTOR, "jm")
    stride_1 = lift_alloc(stride_1, "resultReg : _", n_lifts=2)
    stride_1 = fission(stride_1, stride_1.find("for jo in _:_").before(), n_lifts=2)
    stride_1 = fission(stride_1, stride_1.find("for jo in _:_").after(), n_lifts=2)
    stride_1 = reorder_loops(stride_1, "ji jo")
    stride_1 = reorder_loops(stride_1, "jm jo")    
    
    lower_bound = f"{VEC_W * INTERLEAVE_FACTOR} * jo + jm * {VEC_W}"
    stride_1 = stage_mem(stride_1, "for ji in _:_ #1", f"x[{lower_bound}: {lower_bound} + {VEC_W}]", "xReg")
    stride_1 = stage_mem(stride_1, "for ji in _:_ #1", f"a[i, {lower_bound}: {lower_bound} + {VEC_W}]", "aReg")
    
    for buffer in ["xReg", "aReg", "resultReg"]:
        stride_1 = set_memory(stride_1, buffer, memory)
        stride_1 = set_precision(stride_1, buffer, "f32")

    stride_1 = simplify(stride_1)
    stride_1 = replace_all(stride_1, instructions)

    stride_1 = expand_dim(stride_1, "xReg", INTERLEAVE_FACTOR, "jm")
    stride_1 = lift_alloc(stride_1, "xReg : _")
    stride_1 = expand_dim(stride_1, "aReg", INTERLEAVE_FACTOR, "jm")
    stride_1 = lift_alloc(stride_1, "aReg : _")
    def interleave_instructions(proc, iter):
        while True:
            main_loop = proc.find(f"for {iter} in _:_")
            if len(main_loop.body()) == 1:
                break
            proc = fission(proc, main_loop.body()[0].after())
            proc = unroll_loop(proc, f"for {iter} in _:_")
        proc = unroll_loop(proc, "for jm in _:_")
        return proc
    stride_1 = interleave_instructions(stride_1, "jm")
    stride_1 = interleave_instructions(stride_1, "jm")
    stride_1 = interleave_instructions(stride_1, "jm")

    return simplify(stride_1)

def schedule_multiple_rows(VEC_W, INTERLEAVE_FACTOR, ROW_FACTOR, memory, instructions):
    
    stride_1 = rename(sgemv, sgemv.name() + "_stride_1")
    # stride_1 = stride_1.add_assertion("stride(x, 0) == 1")
    # stride_1 = stride_1.add_assertion("stride(y, 0) == 1")
        
    stride_1 = divide_loop(stride_1, "for j in _:_", VEC_W * INTERLEAVE_FACTOR, ("jo", "ji"), tail = "cut")
    stride_1 = divide_loop(stride_1, "for ji in _:_", VEC_W, ("jm", "ji"), perfect=True)
    
    stride_1 = reorder_loops(stride_1, "jo jm")
    stride_1 = reorder_loops(stride_1, "jo ji")
    stride_1 = simplify(stage_mem(stride_1, "for jo in _:_", "result", "resultReg", accum=True))
    stride_1 = expand_dim(stride_1, "resultReg :_ ", VEC_W, "ji")
    stride_1 = expand_dim(stride_1, "resultReg :_ ", INTERLEAVE_FACTOR, "jm")
    stride_1 = lift_alloc(stride_1, "resultReg : _", n_lifts=2)
    stride_1 = fission(stride_1, stride_1.find("for jo in _:_").before(), n_lifts=2)
    stride_1 = fission(stride_1, stride_1.find("for jo in _:_").after(), n_lifts=2)
    stride_1 = reorder_loops(stride_1, "ji jo")
    stride_1 = reorder_loops(stride_1, "jm jo")    
    
    lower_bound = f"{VEC_W * INTERLEAVE_FACTOR} * jo + jm * {VEC_W}"
    stride_1 = stage_mem(stride_1, "for ji in _:_ #1", f"x[{lower_bound}: {lower_bound} + {VEC_W}]", "xReg")
    stride_1 = stage_mem(stride_1, "for ji in _:_ #1", f"a[i, {lower_bound}: {lower_bound} + {VEC_W}]", "aReg")
    
    for buffer in ["xReg", "aReg", "resultReg"]:
        stride_1 = set_memory(stride_1, buffer, memory)
        stride_1 = set_precision(stride_1, buffer, "f32")

    stride_1 = simplify(stride_1)

    stride_1 = expand_dim(stride_1, "xReg", INTERLEAVE_FACTOR, "jm")
    stride_1 = lift_alloc(stride_1, "xReg : _")
    stride_1 = expand_dim(stride_1, "aReg", INTERLEAVE_FACTOR, "jm")
    stride_1 = lift_alloc(stride_1, "aReg : _")
    def interleave_instructions(proc, iter):
        while True:
            main_loop = proc.find(f"for {iter} in _:_")
            if len(main_loop.body()) == 1:
                break
            proc = fission(proc, main_loop.body()[0].after())
            proc = unroll_loop(proc, f"for {iter} in _:_")
        proc = unroll_loop(proc, "for jm in _:_")
        return proc
    stride_1 = interleave_instructions(stride_1, "jm")
    stride_1 = interleave_instructions(stride_1, "jm")
    stride_1 = interleave_instructions(stride_1, "jm")
  
    stride_1 = divide_loop(stride_1, stride_1.find_loop("i"), ROW_FACTOR, ("io", "ii"), tail="cut")
    
    stride_1 = expand_dim(stride_1, "result : _", ROW_FACTOR, "ii")
    stride_1 = lift_alloc(stride_1, "result: _ ")
    stride_1 = expand_dim(stride_1, "resultReg : _", ROW_FACTOR, "ii")
    stride_1 = lift_alloc(stride_1, "resultReg : _ ")
    stride_1 = fission(stride_1, stride_1.find("result[_] = _").after())
    stride_1 = fission(stride_1, stride_1.find_loop(f"ji #{INTERLEAVE_FACTOR - 1}").after())
    stride_1 = fission(stride_1, stride_1.find_loop("jo").after())
    stride_1 = reorder_loops(stride_1, "ii jo")
    stride_1 = lift_alloc(stride_1, "xReg")
    
    for i in range(0, INTERLEAVE_FACTOR):
      io0_loop_1 = stride_1.find(f"for i0 in _:_ #{i}")
      stride_1 = reorder_stmts(stride_1, io0_loop_1.expand(-1))
      stride_1 = fission(stride_1, stride_1.forward(io0_loop_1).after())
      stride_1 = remove_loop(stride_1, stride_1.forward(io0_loop_1).parent())
    
    stride_1 = replace_all(stride_1, instructions)
    
    for i in range(4):
      stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    
    
    return simplify(stride_1)
   
# TODO: add to EXO's Neon library
@instr("*{result} += vaddvq_f32({x_data});")
def neon_assoc_reduce_add_instr_4xf32(result: f32 @ DRAM, x: [f32][4] @ Neon4f):
  for i in seq(0, 4):
      result += x[i]

@instr("*{result} += vaddvq_f32({x_data});")
def neon_assoc_reduce_add_instr_4xf32_buffer(result: f32[1] @ DRAM, x: [f32][4] @ Neon4f):
  for i in seq(0, 4):
      result[0] += x[i]

@instr(
    """
    {{
        __m256 tmp = _mm256_hadd_ps({x_data}, {x_data});
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        {result_data} += _mm256_cvtss_f32(tmp);
    }}
    """
)
def avx2_assoc_reduce_add_ps_buffer(x: [f32][8] @ AVX2, result: [f32][1]):
    # WARNING: This instruction assumes float addition associativity
    assert stride(x, 0) == 1
    assert stride(result, 0) == 1
    
    for i in seq(0, 8):
        result[0] += x[i]

f32_instructions = [C.Machine.load_instr_f32,
                     C.Machine.store_instr_f32,
                     C.Machine.set_zero_instr_f32,
                     C.Machine.fmadd_instr_f32,]
if C.Machine.name == "neon":
    f32_instructions.append(neon_assoc_reduce_add_instr_4xf32)
else:
    f32_instructions.append(C.Machine.assoc_reduce_add_instr_f32)

if C.Machine.name == "neon":
    f32_instructions.append(neon_assoc_reduce_add_instr_4xf32_buffer)
else:
    f32_instructions.append(avx2_assoc_reduce_add_ps_buffer)


sgemv_stride_1 = schedule_multiple_rows(C.Machine.vec_width, 4, 2, C.Machine.mem_type, f32_instructions)

  
__all__ = ["sgemv_stride_1"]