from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

from composed_schedules import vectorize, BLAS_SchedulingError
from parameters import Level_1_Params


def blas_vectorize(proc, loop_cursor, params):
    if not isinstance(params, Level_1_Params):
        raise BLAS_SchedulingError("params must be Level_1_params")

    return vectorize(
        proc,
        loop_cursor,
        params.vec_width,
        params.interleave_factor,
        params.accumulators_count,
        params.mem_type,
        params.precision,
        params.instructions,
    )
