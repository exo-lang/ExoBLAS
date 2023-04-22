from __future__ import annotations

from exo import *
from exo.libs.memories import *
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
import exo.API_cursors as pc


def generate_stride_any_proc(template_proc, specialize_func, precision):
    proc = specialize_func(template_proc, precision)
    proc = rename(proc, proc.name() + "_stride_any")
    return proc


def export_exo_proc(globals, proc):
    globals[proc.name()] = proc
    globals.setdefault("__all__", []).append(proc.name())
