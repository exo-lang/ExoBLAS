from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *

### EXO_LOC ALGORITHM START ###
@proc
def swap(n: size, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        tmp: R
        tmp = x[i]
        x[i] = y[i]
        y[i] = tmp


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
variants_generator(optimize_level_1)(swap, "i", 4, globals=globals())
### EXO_LOC SCHEDULE END ###
