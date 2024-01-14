from exceptions import *
from introspection import *


def attempt(op, errs=exo_exceptions, rc=False):
    errs = tuple(errs)

    def rewrite(p, *args, **kwargs):
        try:
            res = op(p, *args, **kwargs), True
        except errs:
            res = p, False
        if not rc:
            res = res[0]
        return res

    return rewrite


def apply(op):
    def rewrite(proc, cursors, *args, **kwargs):
        for c in cursors:
            proc = op(proc, c, *args, **kwargs)
        return proc

    return rewrite


def make_pass(op):
    def rewrite(proc, block=InvalidCursor(), *args, **kwargs):
        stmts = nlr_stmts(proc, block)
        return apply(op)(proc, stmts, *args, **kwargs)

    return rewrite


__all__ = ["apply", "attempt", "make_pass"]
