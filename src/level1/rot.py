from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *

### EXO_LOC ALGORITHM START ###
@proc
def rot(n: size, x: [R][n], y: [R][n], c: R, s: R):
    for i in seq(0, n):
        xReg: R
        xReg = x[i]
        x[i] = c * xReg + s * y[i]
        y[i] = -s * xReg + c * y[i]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
variants_generator(optimize_level_1)(rot, "i", 4, globals=globals())
### EXO_LOC SCHEDULE END ###
