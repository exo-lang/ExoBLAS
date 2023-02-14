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
    load_instr=mm256_loadu_ps,
    load_instr_str="mm256_loadu_ps(_)",
    store_instr=mm256_storeu_ps,
    broadcast_instr=mm256_set1_ps,
    broadcast_instr_str="mm256_set1_ps(_)",
    fmadd_instr=mm256_fmadd_ps,
    zpad_ld_instr=None,
    zpad_fmadd_instr=None,
    zpad_broadcast_instr=None,
    zpad_store_instr=None,
)
