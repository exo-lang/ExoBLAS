from pathlib import Path
import importlib.util

from inspection import *
from exo.stdlib.scheduling import *

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
CACHE_DIR = REPO_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def get_path(proc):
    return CACHE_DIR / f"{proc.name()}.py"


def filter_proc(proc):
    filtered_lines = []
    lines = proc.split("\n")
    # Loop through each line in the original file
    for line in lines:
        # Check if the line starts (considering leading whitespaces) with "assert"
        # and ends with "== True" or "== False"
        line_lstripped = line.lstrip()  # Remove leading whitespaces for the startswith check
        if not (
            line_lstripped.startswith("assert")
            and (line_lstripped.strip().endswith("== True") or line_lstripped.strip().endswith("== False"))
        ):
            # If it doesn't match the criteria, add it to the list of filtered lines
            filtered_lines.append(line)
    return "\n".join(filtered_lines)


def serialize_(proc):
    string = ""
    calls = filter_cursors(is_call)(proc, lrn_stmts(proc))
    for c in calls:
        subproc = c.subproc()
        if not subproc.is_instr():
            string += serialize_(subproc)
    proc_str = str(proc)
    proc_str = filter_proc(proc_str)
    string += f"@proc\n{proc_str}\n"
    return string


def serialize(proc):
    string = f"""
from __future__ import annotations
from exo import *
from exo.libs.memories import *
from exo_blas_config import *

{serialize_(proc)}
"""
    path = get_path(proc)
    with open(path, "w") as f:
        f.write(string)


def deserialize(proc):
    path = get_path(proc)
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(proc.name(), path)
    if spec is None:
        return spec
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proc_sched = module.__dict__[proc.name()]
    return proc_sched.unsafe_assert_eq(proc)
