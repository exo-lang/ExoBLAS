from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MachineParameters:
    """
    Data class representing an abstract model of a machine
    """

    name: str

    mem_type: Any
    n_vec_registers: int
    vec_width: int
    vec_units: int

    # These are not currently used for anything
    l1_cache: int
    l2_cache: int
    l3_cache: int

    load_instr_f32: Any
    load_instr_f32_str: str
    store_instr_f32: Any
    broadcast_instr_f32: Any
    broadcast_instr_f32_str: str
    broadcast_scalar_instr_f32: Any
    fmadd_instr_f32: Any
    zpad_ld_instr: Any
    zpad_fmadd_instr: Any
    zpad_broadcast_instr: Any
    zpad_store_instr: Any
    set_zero_instr_f32: Any
    assoc_reduce_add_instr_f32: Any
    mul_instr_f32: Any
    add_instr_f32: Any
    reduce_add_wide_instr_f32: Any
    reg_copy_instr_f32: Any
    sign_instr_f32: Any
    select_instr_f32: Any
    assoc_reduce_add_f32_buffer: Any
    
    load_instr_f64: Any
    store_instr_f64: Any
    broadcast_instr_f64: Any
    broadcast_scalar_instr_f64: Any
    fmadd_instr_f64: Any
    set_zero_instr_f64: Any
    assoc_reduce_add_instr_f64: Any
    mul_instr_f64: Any
    add_instr_f64: Any
    reduce_add_wide_instr_f64: Any
    reg_copy_instr_f64: Any
    sign_instr_f64: Any
    select_instr_f64: Any
