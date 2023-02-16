from __future__ import annotations

from exo.platforms.x86 import *

from .machine import MachineParameters

Machine = MachineParameters(
    name="avx512",
    mem_type=AVX512,
    n_vec_registers=32,
    vec_width=16,
    l1_cache=None,
    l2_cache=None,
    l3_cache=None,
    load_instr_f32=mm512_loadu_ps,
    load_instr_f32_str="mm512_loadu_ps(_)",
    store_instr_f32=mm512_storeu_ps,
    broadcast_instr_f32=mm512_set1_ps,
    broadcast_instr_f32_str="mm512_set1_ps(_)",
    fmadd_instr_f32=mm512_fmadd_ps,
    zpad_ld_instr=None,
    zpad_fmadd_instr=None,
    zpad_broadcast_instr=None,
    zpad_store_instr=None,
)
