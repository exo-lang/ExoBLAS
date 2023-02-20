from __future__ import annotations

import sys
import getopt 
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo import *
from exo.syntax import *

from exo.stdlib.scheduling import *


@proc
def f(A: f32[4], N: size):
    if 1<10:
        A[0] = 1.0
    else:
        A[0] = 2.0

assert_if(f, f.find("if _:_"), True)