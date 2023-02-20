from __future__ import annotations

from exo.platforms.x86 import *

from .machine import MachineParameters

Machine = MachineParameters(
    name="avx2",
    mem_type=AVX2,
    n_vec_registers=16,
    vec_width=8,
    l1_cache=None,
    l2_cache=None,
    l3_cache=None,
    load_instr_f32=mm256_loadu_ps,
    load_instr_f32_str="mm256_loadu_ps(_)",
    store_instr_f32=mm256_storeu_ps,
    broadcast_instr_f32=mm256_broadcast_ss,
    broadcast_instr_f32_str="mm256_broadcast_ss(_)",
    broadcast_scalar_instr_f32=mm256_broadcast_ss_scalar,
    fmadd_instr_f32=mm256_fmadd_ps,
    zpad_ld_instr=None,
    zpad_fmadd_instr=None,
    zpad_broadcast_instr=None,
    zpad_store_instr=None,
    set_zero_instr_f32=mm256_setzero,
    assoc_reduce_add_instr_f32=avx2_assoc_reduce_add_ps,
    mul_instr_f32=mm256_mul_ps,
    add_instr_f32=mm256_add_ps,
    reduce_add_wide_instr_f32=avx2_reduce_add_wide_ps,
    reg_copy_instr_f32=mm256_reg_copy,
    sign_instr_f32=avx2_sign_ps,
    select_instr_f32=avx2_select_ps,
)
