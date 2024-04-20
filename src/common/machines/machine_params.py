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

    instrs: Any

    # fused_load_cvt_f32_f64: Any
    # prefix_fused_load_cvt_f32_f64: Any

    def __getitem__(self, item):
        return getattr(self, item)

    def get_instructions(self, precision):
        return list(filter(lambda p: precision in p.name(), self.instrs))

    def vec_width(self, precision):
        vec_width_map = {"f32": self.f32_vec_width, "f64": self.f32_vec_width // 2}
        if precision not in vec_width_map:
            raise TypeError(f"Unsupported precision {precision}")
        return vec_width_map[precision]
