from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *

### EXO_LOC ALGORITHM START ###
@proc
def scal(n: size, alpha: R, x: [R][n]):
    for i in seq(0, n):
        x[i] = alpha * x[i]


@proc
def scal_alpha_0(n: size, x: [R][n]):
    for i in seq(0, n):
        x[i] = 0.0


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
for proc in scal, scal_alpha_0:
    variants_generator(optimize_level_1)(proc, "i", 4, globals=globals())
### EXO_LOC SCHEDULE END ###
