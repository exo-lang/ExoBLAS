from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *


### EXO_LOC ALGORITHM START ###
@proc
def dot(n: size, x: [R][n], y: [R][n], result: R):
    result = 0.0
    for i in seq(0, n):
        result += x[i] * y[i]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
variants_generator(optimize_level_1)(dot, "i", 4, globals=globals())
### EXO_LOC SCHEDULE END ###
