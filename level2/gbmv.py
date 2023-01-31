from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

@proc
def gbmv():
    pass

if __name__ == "__main__":
    print(gbmv)

__all__ = ["gbmv"]
