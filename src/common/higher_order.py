from exo.stdlib.scheduling import *

from exceptions import *


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


def predicate(op, pred):
    def rewrite(proc, *args, **kwargs):
        if pred(proc, *args, **kwargs):
            proc = op(proc, *args, **kwargs)
        return proc

    return rewrite


def make_pass(op, trav_start):
    def rewrite(proc, block=InvalidCursor(), *args, **kwargs):
        stmts = trav_start(proc, block)
        return apply(op)(proc, stmts, *args, **kwargs)

    return rewrite


def lift_rc(op, attr):
    def rewrite(*args, **kwargs):
        proc, cursors = op(*args, **kwargs, rc=True)
        c = getattr(cursors, attr)
        return proc, c

    return rewrite


def repeate(op):
    op_attempt = attempt(op)

    def rewrite(p, *args, **kwargs):
        success = True
        while success:
            p, success = op_attempt(p, *args, **kwargs, rs=True)
        return p

    return rewrite


def repeate_n(op):
    def rewrite(p, *args, n=1, **kwargs):
        for i in range(n):
            p = op(p, *args, **kwargs)
        return p

    return rewrite


def extract_and_schedule(op, include_asserts=True):
    def rewrite(proc, block, subproc_name, *args, rc=False, **kwargs):
        block = proc.forward(block)
        block = block.as_block()
        proc, subproc = extract_subproc(proc, block, subproc_name, include_asserts=include_asserts)
        subproc_sched = op(subproc, subproc.body()[0], *args, **kwargs)
        call = proc.forward(block)[0]
        proc = call_eqv(proc, call, subproc_sched)
        if not rc:
            return proc
        return proc, (subproc, subproc_sched)

    return rewrite


def filter_cursors(op):
    def filter_c(proc, cursors, *args, **kwargs):
        for c in cursors:
            if op(proc, c, *args, **kwargs):
                yield c

    return filter_c


__all__ = [
    "apply",
    "attempt",
    "make_pass",
    "lift_rc",
    "repeate",
    "predicate",
    "extract_and_schedule",
    "repeate_n",
]
