from __future__ import annotations

from exo import *
from exo.platforms.x86 import *

from blaslib import *
from codegen_helpers import *

### EXO_LOC ALGORITHM START ###
@proc
def asum(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
    result = 0.0
    for i in seq(0, n):
        result += select(0.0, x[i], x[i], -x[i])


### EXO_LOC ALGORITHM END ###

### EXO_LOC SCHEDULE START ###
def schedule_asum(asum, loop, precision, machine, interleave_factor):
    if machine.mem_type is not AVX2:
        return asum
    return optimize_level_1(asum, loop, precision, machine, interleave_factor)


variants_generator(schedule_asum)(asum, "i", 7, globals=globals())
### EXO_LOC SCHEDULE END ###
