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

    # These are not currently used for anything
    l1_cache: int
    l2_cache: int
    l3_cache: int

    load_instr_f32: Any
    load_instr_f32_str: str
    store_instr_f32: Any
    broadcast_instr_f32: Any
    broadcast_instr_f32_str: str
    fmadd_instr_f32: Any
    zpad_ld_instr: Any
    zpad_fmadd_instr: Any
    zpad_broadcast_instr: Any
    zpad_store_instr: Any
    set_zero_instr_f32: Any
    assoc_reduce_add_instr_f32: Any