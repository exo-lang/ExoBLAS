from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

@proc
def tbsv_raw_major_Upper_NoneTrans_template(n: size, k: size, x: [R][n], A: [R][n, n], Diag: size):
    assert stride(A, 1) == 1
    assert k <= n - 1
    
    for i in seq(0, n):
        # Row (n - i - 1)
        
        pivot: R
        if Diag == 0:
            pivot = A[n - i - 1, 0]
        else:
            pivot = 1.0
        
        dot: R
        dot = 0.0
        
        for j in seq(0, k):
            if n - i + j < n:
               dot += A[n - i - 1, j + 1] * x[n - i + j]
        
        x[n - i - 1] = (x[n - i - 1] - dot) / pivot

@proc
def tbsv_raw_major_Lower_NoneTrans_template(n: size, k: size, x: [R][n], A: [R][n, n], Diag: size):
    assert stride(A, 1) == 1
    assert k <= n - 1
    
    for i in seq(0, n):
        # Row (i)
        
        dot: R
        dot = 0.0
        
        for j in seq(0, k):
            if i - j - 1 >= 0:
                dot += A[i, k - j - 1] * x[i - j - 1]
        
        pivot: R
        if Diag == 0:
            pivot = A[i, k]
        else:
            pivot = 1.0
            
        x[i] = (x[i] - dot) / pivot

@proc
def tbsv_raw_major_Upper_Trans_template(n: size, k: size, x: [R][n], A: [R][n, n], Diag: size):
    assert stride(A, 1) == 1
    assert k <= n - 1
    
    for i in seq(0, n):
        # Row (i)
        
        dot: R
        dot = 0.0
        
        for j in seq(0, k):
            if i - j - 1 >= 0:
                dot += A[i - j - 1, j + 1] * x[i - j - 1]
        
        pivot: R
        if Diag == 0:
            pivot = A[i, 0]
        else:
            pivot = 1.0
            
        x[i] = (x[i] - dot) / pivot

@proc
def tbsv_raw_major_Lower_Trans_template(n: size, k: size, x: [R][n], A: [R][n, n], Diag: size):
    assert stride(A, 1) == 1
    assert k <= n - 1
    
    for i in seq(0, n):
        # Row (n - i - 1)
        
        pivot: R
        if Diag == 0:
            pivot = A[n - i - 1, k]
        else:
            pivot = 1.0
        
        dot: R
        dot = 0.0
        
        for j in seq(0, k):
            if n - i + j < n:
               dot += A[n - i + j, k - j - 1] * x[n - i + j]
        
        x[n - i - 1] = (x[n - i - 1] - dot) / pivot

def specialize_tbsv(tbsv, precision):
    prefix = "s" if precision == "f32" else "d"
    name = tbsv.name()
    name = name.replace("_template", "")
    specialized = rename(tbsv, "exo_" + prefix + name)
    
    args = ["x", "A", "dot", "pivot"]
        
    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return specialized

def schedule_interleave_tbsv_raw_major_stride_1(tbsv, VEC_W, RAW_INTERLEAVE_FACTOR, memory, instructions, precision):
    stride_1 = specialize_tbsv(tbsv, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")
    
    return stride_1

#################################################
# Kernel Parameters
#################################################

RAW_INTERLEAVE_FACTOR = C.Machine.vec_units

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_stbsv_raw_major_Upper_NoneTrans_stride_any = specialize_tbsv(tbsv_raw_major_Upper_NoneTrans_template, "f32")
exo_stbsv_raw_major_Upper_NoneTrans_stride_any = rename(exo_stbsv_raw_major_Upper_NoneTrans_stride_any, 
                                                                 exo_stbsv_raw_major_Upper_NoneTrans_stride_any.name() + "_stride_any")
exo_stbsv_raw_major_Lower_NoneTrans_stride_any = specialize_tbsv(tbsv_raw_major_Lower_NoneTrans_template, "f32")
exo_stbsv_raw_major_Lower_NoneTrans_stride_any = rename(exo_stbsv_raw_major_Lower_NoneTrans_stride_any, 
                                                        exo_stbsv_raw_major_Lower_NoneTrans_stride_any.name() + "_stride_any")
exo_stbsv_raw_major_Upper_Trans_stride_any = specialize_tbsv(tbsv_raw_major_Upper_Trans_template, "f32")
exo_stbsv_raw_major_Upper_Trans_stride_any = rename(exo_stbsv_raw_major_Upper_Trans_stride_any, 
                                                    exo_stbsv_raw_major_Upper_Trans_stride_any.name() + "_stride_any")
exo_stbsv_raw_major_Lower_Trans_stride_any = specialize_tbsv(tbsv_raw_major_Lower_Trans_template, "f32")
exo_stbsv_raw_major_Lower_Trans_stride_any = rename(exo_stbsv_raw_major_Lower_Trans_stride_any, 
                                                    exo_stbsv_raw_major_Lower_Trans_stride_any.name() + "_stride_any")

f32_instructions = [C.Machine.load_instr_f32,
                     C.Machine.store_instr_f32,
                     C.Machine.mul_instr_f32,
                     C.Machine.fmadd_instr_f32,
                     C.Machine.broadcast_instr_f32,
                     C.Machine.broadcast_scalar_instr_f32,
                     ]

exo_stbsv_raw_major_Upper_NoneTrans_stride_1 = schedule_interleave_tbsv_raw_major_stride_1(tbsv_raw_major_Upper_NoneTrans_template,
                                                                                           C.Machine.vec_width, RAW_INTERLEAVE_FACTOR, C.Machine.mem_type, f32_instructions, "f32")
exo_stbsv_raw_major_Lower_NoneTrans_stride_1 = schedule_interleave_tbsv_raw_major_stride_1(tbsv_raw_major_Lower_NoneTrans_template,
                                                                                           C.Machine.vec_width, RAW_INTERLEAVE_FACTOR, C.Machine.mem_type, f32_instructions, "f32")

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dtbsv_raw_major_Upper_NoneTrans_stride_any = specialize_tbsv(tbsv_raw_major_Upper_NoneTrans_template, "f64")
exo_dtbsv_raw_major_Upper_NoneTrans_stride_any = rename(exo_dtbsv_raw_major_Upper_NoneTrans_stride_any,
                                                        exo_dtbsv_raw_major_Upper_NoneTrans_stride_any.name() + "_stride_any")
exo_dtbsv_raw_major_Lower_NoneTrans_stride_any = specialize_tbsv(tbsv_raw_major_Lower_NoneTrans_template, "f64")
exo_dtbsv_raw_major_Lower_NoneTrans_stride_any = rename(exo_dtbsv_raw_major_Lower_NoneTrans_stride_any,
                                                        exo_dtbsv_raw_major_Lower_NoneTrans_stride_any.name() + "_stride_any")
exo_dtbsv_raw_major_Upper_Trans_stride_any = specialize_tbsv(tbsv_raw_major_Upper_Trans_template, "f64")
exo_dtbsv_raw_major_Upper_Trans_stride_any = rename(exo_dtbsv_raw_major_Upper_Trans_stride_any,
                                                    exo_dtbsv_raw_major_Upper_Trans_stride_any.name() + "_stride_any")
exo_dtbsv_raw_major_Lower_Trans_stride_any = specialize_tbsv(tbsv_raw_major_Lower_Trans_template, "f64")
exo_dtbsv_raw_major_Lower_Trans_stride_any = rename(exo_dtbsv_raw_major_Lower_Trans_stride_any,
                                                    exo_dtbsv_raw_major_Lower_Trans_stride_any.name() + "_stride_any")

f64_instructions = [C.Machine.load_instr_f64,
                     C.Machine.store_instr_f64,
                     C.Machine.mul_instr_f64,
                     C.Machine.fmadd_instr_f64,
                     C.Machine.broadcast_instr_f64,
                     C.Machine.broadcast_scalar_instr_f64,
                     ]

exo_dtbsv_raw_major_Upper_NoneTrans_stride_1 = schedule_interleave_tbsv_raw_major_stride_1(tbsv_raw_major_Upper_NoneTrans_template,
                                                                                           C.Machine.vec_width // 2, RAW_INTERLEAVE_FACTOR, C.Machine.mem_type, f64_instructions, "f64")
exo_dtbsv_raw_major_Lower_NoneTrans_stride_1 = schedule_interleave_tbsv_raw_major_stride_1(tbsv_raw_major_Lower_NoneTrans_template,
                                                                                           C.Machine.vec_width // 2, RAW_INTERLEAVE_FACTOR, C.Machine.mem_type, f64_instructions, "f64")

entry_points = [
                exo_stbsv_raw_major_Upper_NoneTrans_stride_any, exo_stbsv_raw_major_Upper_NoneTrans_stride_1,
                exo_dtbsv_raw_major_Upper_NoneTrans_stride_any, exo_dtbsv_raw_major_Upper_NoneTrans_stride_1,
                
                exo_stbsv_raw_major_Lower_NoneTrans_stride_any, exo_stbsv_raw_major_Lower_NoneTrans_stride_1,
                exo_dtbsv_raw_major_Lower_NoneTrans_stride_any, exo_dtbsv_raw_major_Lower_NoneTrans_stride_1,
                
                exo_stbsv_raw_major_Upper_Trans_stride_any,
                exo_dtbsv_raw_major_Upper_Trans_stride_any,
                
                exo_stbsv_raw_major_Lower_Trans_stride_any,
                exo_dtbsv_raw_major_Lower_Trans_stride_any,
                ]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
