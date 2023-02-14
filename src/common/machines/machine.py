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

    load_instr: Any
    load_instr_str: str
    store_instr: Any
    broadcast_instr: Any
    broadcast_instr_str: str
    fmadd_instr: Any
    zpad_ld_instr: Any
    zpad_fmadd_instr: Any
    zpad_broadcast_instr: Any
    zpad_store_instr: Any
