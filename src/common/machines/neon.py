from __future__ import annotations

from exo.platforms.neon import *


from .machine import MachineParameters



Machine = MachineParameters(
    name="neon",
    mem_type=Neon4f,
    n_vec_registers=32,
    vec_width=4,
    l1_cache=None,
    l2_cache=None,
    l3_cache=None,
    load_instr_f32=neon_vld_4xf32,
    load_instr_f32_str="neon_vld_4xf32(_)",  # Instructions for matmul
    store_instr_f32=neon_vst_4xf32,
    broadcast_instr_f32=neon_broadcast_4xf32,
    broadcast_instr_f32_str="neon_broadcast_4xf32(_)",
    broadcast_scalar_instr_f32=None,
    fmadd_instr_f32=neon_vfmadd_4xf32_4xf32,
    zpad_ld_instr=None,
    zpad_fmadd_instr=None,
    zpad_broadcast_instr=None,
    zpad_store_instr=None,
    set_zero_instr_f32=neon_zero_4xf32,
    assoc_reduce_add_instr_f32=None,
    mul_instr_f32_hack=neon_vmul_4xf32_alias_hack,
    mul_instr_f32=neon_vmul_4xf32,
    add_instr_f32=None,
    reduce_add_wide_instr_f32=None,
    reg_copy_instr_f32=neon_reg_copy_4xf32,
    sign_instr_f32=None,
    select_instr_f32=None,
)
