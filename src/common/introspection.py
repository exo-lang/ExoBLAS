from __future__ import annotations

from exo import *
from exo.libs.memories import *
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *


def get_statemnts(proc):
    def get_statemnts_helper(body):
        for stmt in body:
            yield stmt
            if isinstance(stmt, IfCursor):
                yield from get_statemnts_helper(stmt.body())
                yield from get_statemnts_helper(stmt.orelse())
            elif isinstance(stmt, ForSeqCursor):
                yield from get_statemnts_helper(stmt.body())

    return get_statemnts_helper(proc.body())
