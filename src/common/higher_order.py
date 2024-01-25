from exceptions import *
from introspection import *


def attempt(op, errs=exo_exceptions):
    errs = tuple(errs)

    def rewrite(p, *args, rs=False, **kwargs):
        try:
            res = op(p, *args, **kwargs), True
        except errs:
            res = p, False
        if not rs:
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


def lift_rc(op, attr):
    def rewrite(*args, **kwargs):
        proc, cursors = op(*args, **kwargs, rc=True)
        c = getattr(cursors, attr)
        return proc, c

    return rewrite


__all__ = ["apply", "attempt", "make_pass", "lift_rc"]
