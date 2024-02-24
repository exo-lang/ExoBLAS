from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exo import *

from stdlib import *


@dataclass
class MachineParameters:
    """
    Data class representing an abstract model of a machine
    """

    name: str

    mem_type: Any
    n_vec_registers: int
    f32_vec_width: int
    vec_units: int
    supports_predication: bool

    # These are not currently used for anything
    l1_cache: int
    l2_cache: int
    l3_cache: int

    load_instr_f32: Any
    load_backwards_instr_f32: Any
    prefix_load_instr_f32: Any
    prefix_load_backwards_instr_f32: Any
    store_instr_f32: Any
    store_backwards_instr_f32: Any
    prefix_store_backwards_instr_f32: Any
    prefix_store_instr_f32: Any
    broadcast_instr_f32: Any
    broadcast_scalar_instr_f32: Any
    prefix_broadcast_instr_f32: Any
    prefix_broadcast_scalar_instr_f32: Any
    fmadd_instr_f32: Any
    prefix_fmadd_instr_f32: Any
    fmadd_reduce_instr_f32: Any
    prefix_fmadd_reduce_instr_f32: Any
    set_zero_instr_f32: Any
    prefix_set_zero_instr_f32: Any
    assoc_reduce_add_instr_f32: Any
    mul_instr_f32: Any
    prefix_mul_instr_f32: Any
    add_instr_f32: Any
    prefix_add_instr_f32: Any
    reduce_add_wide_instr_f32: Any
    prefix_reduce_add_wide_instr_f32: Any
    reg_copy_instr_f32: Any
    prefix_reg_copy_instr_f32: Any
    sign_instr_f32: Any
    prefix_sign_instr_f32: Any
    select_instr_f32: Any
    assoc_reduce_add_f32_buffer: Any
    abs_instr_f32: Any
    prefix_abs_instr_f32: Any

    load_instr_f64: Any
    load_backwards_instr_f64: Any
    prefix_load_instr_f64: Any
    prefix_load_backwards_instr_f64: Any
    prefix_store_instr_f64: Any
    prefix_store_backwards_instr_f64: Any
    store_instr_f64: Any
    store_backwards_instr_f64: Any
    broadcast_instr_f64: Any
    broadcast_scalar_instr_f64: Any
    prefix_broadcast_instr_f64: Any
    prefix_broadcast_scalar_instr_f64: Any
    fmadd_instr_f64: Any
    prefix_fmadd_instr_f64: Any
    fmadd_reduce_instr_f64: Any
    prefix_fmadd_reduce_instr_f64: Any
    set_zero_instr_f64: Any
    prefix_set_zero_instr_f64: Any
    assoc_reduce_add_instr_f64: Any
    mul_instr_f64: Any
    prefix_mul_instr_f64: Any
    add_instr_f64: Any
    prefix_add_instr_f64: Any
    reduce_add_wide_instr_f64: Any
    prefix_reduce_add_wide_instr_f64: Any
    assoc_reduce_add_f64_buffer: Any
    reg_copy_instr_f64: Any
    prefix_reg_copy_instr_f64: Any
    sign_instr_f64: Any
    prefix_sign_instr_f64: Any
    select_instr_f64: Any
    abs_instr_f64: Any
    prefix_abs_instr_f64: Any

    convert_f32_lower_to_f64: Any
    convert_f32_upper_to_f64: Any

    fused_load_cvt_f32_f64: Any
    prefix_fused_load_cvt_f32_f64: Any

    def __getitem__(self, item):
        return getattr(self, item)

    def get_fma_variants(self, precision):
        attr = f"fma_variants_{precision}"
        if not hasattr(self, attr):
            fmas = [
                i[1]
                for i in self.__dict__.items()
                if precision in i[0]
                and isinstance(i[1], Procedure)
                and "fmadd_instr" in i[0]
            ]

            def rewrite(fma):
                fma = commute_expr(fma, [fma.find(f"_ + _")])
                return rename(fma, f"{fma.name()}_commute")

            setattr(self, attr, [rewrite(fma) for fma in fmas])
        return getattr(self, attr)

    def get_instructions(self, precision):
        instrs = [
            i[1]
            for i in self.__dict__.items()
            if precision in i[0] and isinstance(i[1], Procedure)
        ]
        fma_variants = self.get_fma_variants(precision)
        return instrs + fma_variants

    def vec_width(self, precision):
        vec_width_map = {"f32": self.f32_vec_width, "f64": self.f32_vec_width // 2}
        if precision not in vec_width_map:
            raise TypeError(f"Unsupported precision {precision}")
        return vec_width_map[precision]
