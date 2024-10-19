from __future__ import annotations

from exo import *
from exo.API_cursors import *
from exo.frontend.syntax import *
from exo.libs.externs import select

from stdlib import *


class VEC(Memory):
    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        assert strides[-1] == "1"
        idxs = indices[:-1] or ""
        if idxs:
            idxs = "[" + "][".join(idxs) + "]"
        return f"{baseptr}{idxs}"


@proc
def vec_load(n: size, m: size, dst: [R][n] @ VEC, src: [R][n] @ DRAM):
    for i in seq(0, n):
        if i < m:
            dst[i] = src[i]


@proc
def vec_load_bck(n: size, m: size, dst: [R][n] @ VEC, src: [R][n] @ DRAM):
    for i in seq(0, n):
        if i < m:
            dst[i] = src[n - 1 - i]


@proc
def vec_store(n: size, m: size, dst: [R][n] @ DRAM, src: [R][n] @ VEC):
    for i in seq(0, n):
        if i < m:
            dst[i] = src[i]


@proc
def vec_store_bck(n: size, m: size, dst: [R][n] @ DRAM, src: [R][n] @ VEC):
    for i in seq(0, n):
        if i < m:
            dst[n - i - 1] = src[i]


gen_vec_op = lambda p, name, op: rename(simplify(p.partial_eval(op=op)), f"vec_{name}")

COPY = 1
NEG = 2
ADD_RED = 3


@proc
def vec_unop(op: size, n: size, m: size, dst: [R][n] @ VEC, src: [R][n] @ VEC):
    for i in seq(0, n):
        if i < m:
            if op == COPY:
                dst[i] = src[i]
            if op == NEG:
                dst[i] = -src[i]
            if op == ADD_RED:
                dst[i] += src[i]


vec_copy = gen_vec_op(vec_unop, "copy", COPY)
vec_neg = gen_vec_op(vec_unop, "neg", NEG)
vec_add_red = gen_vec_op(vec_unop, "add_red", ADD_RED)

ADD = 1
SUB = 2
MUL = 3
DIV = 4
FMADD_RED = 5


@proc
def vec_binop(op: size, n: size, m: size, dst: [R][n] @ VEC, src1: [R][n] @ VEC, src2: [R][n] @ VEC):
    for i in seq(0, n):
        if i < m:
            if op == ADD:
                dst[i] = src1[i] + src2[i]
            if op == SUB:
                dst[i] = src1[i] - src2[i]
            if op == MUL:
                dst[i] = src1[i] * src2[i]
            if op == DIV:
                dst[i] = src1[i] / src2[i]
            if op == FMADD_RED:
                dst[i] += src1[i] * src2[i]


vec_add = gen_vec_op(vec_binop, "add", ADD)
vec_sub = gen_vec_op(vec_binop, "sub", SUB)
vec_mul = gen_vec_op(vec_binop, "mul", MUL)
vec_div = gen_vec_op(vec_binop, "div", DIV)
vec_fmadd_red = gen_vec_op(vec_binop, "fmadd_red", FMADD_RED)


@proc
def vec_brdcst_scl(n: size, m: size, dst: [R][n] @ VEC, src: R @ DRAM):
    for i in seq(0, n):
        if i < m:
            dst[i] = src


@proc
def vec_brdcst_buf(n: size, m: size, dst: [R][n] @ VEC, src: [R][1] @ DRAM):
    for i in seq(0, n):
        if i < m:
            dst[i] = src[0]


@proc
def vec_zero(n: size, m: size, dst: [R][n] @ VEC):
    for i in seq(0, n):
        if i < m:
            dst[i] = 0.0


@proc
def vec_fmadd1(n: size, m: size, dst: [R][n] @ VEC, src1: [R][n] @ VEC, src2: [R][n] @ VEC, src3: [R][n] @ VEC):
    for i in seq(0, n):
        if i < m:
            dst[i] = src1[i] * src2[i] + src3[i]


vec_fmadd2 = rename(commute_expr(vec_fmadd1, "_ + _"), "vec_fmadd2")


@proc
def vec_abs(n: size, m: size, dst: [R][n] @ VEC, src: [R][n] @ VEC):
    for i in seq(0, n):
        if i < m:
            dst[i] = select(0.0, src[i], src[i], -src[i])


@proc
def vec_reduce_add_scl(n: size, m: size, dst: R @ DRAM, src: [R][n] @ VEC):
    for i in seq(0, n):
        if i < m:
            dst += src[i]


@proc
def vec_reduce_add_buf(n: size, m: size, dst: [R][1] @ DRAM, src: [R][n] @ VEC):
    for i in seq(0, n):
        if i < m:
            dst[0] += src[i]


@proc
def load_f32_cvt(n: size, m: size, dst: [R][n] @ VEC, src: [f32][n] @ DRAM):
    for i in seq(0, n):
        if i < m:
            dst[i] = src[i]


def specialize_vec_op(proc, n, precision, mem):
    assert issubclass(mem, VEC)
    proc = proc.add_assertion("m <= n")

    proc = specialize_precision(proc, precision)
    for arg in proc.args():
        if arg.is_tensor():
            proc = proc.add_assertion(f"stride({arg.name()}, {len(arg.shape()) - 1}) == 1")
    for arg in proc.args():
        if arg.type().is_numeric() and arg.mem() is VEC:
            proc = set_memory(proc, arg, mem)

    proc = proc.partial_eval(n=n)

    proc = rename(proc, proc.name() + f"_{precision}x{n}")

    predicated_proc = rename(proc, proc.name() + "_pfx")
    unpredicate_proc = simplify(dce(proc.partial_eval(m=n)))

    return unpredicate_proc, predicated_proc
