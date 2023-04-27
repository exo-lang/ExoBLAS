from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from exo import *

import exo_blas_config as C


@dataclass
class BLAS_Params:
    """
    Base data class representing any BLAS kernel parameters
    """

    precision: str = "f32"
    instructions: Any = None
    mem_type: Any = C.Machine.mem_type

    def __post_init__(self):
        self.instructions = C.Machine.get_instructions(self.precision)


@dataclass
class Level_1_Params(BLAS_Params):
    """
    Data class representing parameters used in level 1 kernels
    """

    vec_width: int = C.Machine.vec_width
    interleave_factor: int = C.Machine.vec_units * 2
    accumulators_count: int = C.Machine.vec_units * 2

    def __post_init__(self):
        super().__post_init__()
        if self.precision == "f64":
            self.vec_width = C.Machine.vec_width // 2


@dataclass
class Level_2_Params(Level_1_Params):
    """
    Data class representing parameters used in level 2 kernels
    """

    rows_interleave_factor: int = 4

    def __post_init__(self):
        super().__post_init__()
        self.interleave_factor = 2
        self.accumulators_count = 2


@dataclass
class Level_3_Params(Level_2_Params):
    """
    Data class representing parameters used in level 3 kernels
    """

    def __post_init__(self):
        super().__post_init__()
