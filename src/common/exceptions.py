from exo.stdlib.scheduling import *
from exo.API_cursors import *
from exo.stdlib.scheduling import _UnificationError


class BLAS_SchedulingError(Exception):
    pass


exo_exceptions = {ValueError, TypeError, SchedulingError, BLAS_SchedulingError, InvalidCursorError, _UnificationError}

__all__ = ["BLAS_SchedulingError", "exo_exceptions"]
