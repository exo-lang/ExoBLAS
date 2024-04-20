from __future__ import annotations

from machines.abstract_vector import *
from machines.machine_params import MachineParameters

forceinline = """
// Portable macro adapted from https://en.wikipedia.org/wiki/Inline_function
#ifdef _MSC_VER
    #define forceinline __forceinline
#elif defined(__GNUC__)
    #define forceinline inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
    #if __has_attribute(__always_inline__)
        #define forceinline inline __attribute__((__always_inline__))
    #else
        #define forceinline inline
    #endif
#else
    #define forceinline inline
#endif
"""
get_mask_func = """
static forceinline __m256i mm256_prefix_mask_epi32(int32_t count) {
    __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i prefix = _mm256_set1_epi32(count);
    return _mm256_cmpgt_epi32(prefix, indices);
}

static forceinline __m256i mm256_prefix_mask_epi64x(int32_t count) {
    __m256i indices = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i prefix = _mm256_set1_epi64x(count);
    return _mm256_cmpgt_epi64(prefix, indices);
}

static forceinline __m128i mm_prefix_mask_epi32(int32_t count) {
    __m128i indices = _mm_setr_epi32(0, 1, 2, 3);
    __m128i prefix = _mm_set1_epi32(count);
    return _mm_cmpgt_epi32 (prefix, indices);
}
"""

predicate_func = """
static forceinline __m256 mm256_prefix_ps(__m256 dst, __m256 src, int32_t count) {
    __m256i mask = mm256_prefix_mask_epi32(count);
    return _mm256_blendv_ps(dst, src, mask);
}
static forceinline __m256 mm256_prefix_pd(__m256d dst, __m256d src, int32_t count) {
    __m256i mask = mm256_prefix_mask_epi64x(count);
    return _mm256_blendv_pd(dst, src, mask);
}
"""

reduce_add = """
static forceinline float mm256_reduce_add_ps(__m256 a) {
    __m128 lo = _mm256_extractf128_ps (a, 0);
    __m128 hi = _mm256_extractf128_ps (a, 1);
    __m128 add = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_shuffle_ps(add, add, 0b00001110);
    add = _mm_add_ps(add, shuf);
    __m128 dup = _mm_movehdup_ps(add);
    add = _mm_add_ps(add, dup);
    return add[0];
}
static forceinline double mm256_reduce_add_pd(__m256d a) {
    __m128d lo = _mm256_extractf128_pd (a, 0);
    __m128d hi = _mm256_extractf128_pd (a, 1);
    __m128d add = _mm_add_pd(lo, hi);
    __m128d shuf = _mm_shuffle_pd (add, add, 0b01);
    add = _mm_add_pd(add, shuf);
    return add[0];
}
"""


class VEC_AVX2(VEC):
    @classmethod
    def global_(cls):
        include = "#include <immintrin.h>"
        return "\n".join([include, forceinline, get_mask_func, predicate_func, reduce_add])

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            raise MemGenError(f"{srcinfo}: AVX2 vectors are not scalar values")

        vec_types = {
            "float": (8, "__m256"),
            "double": (4, "__m256d"),
            "uint16_t": (16, "__m256i"),
        }

        if not prim_type in vec_types.keys():
            raise MemGenError(f"{srcinfo}: AVX2 vectors must be f32/f64/ui16 (for now), got {prim_type}")

        reg_width, C_reg_type_name = vec_types[prim_type]
        if not (shape[-1].isdecimal() and int(shape[-1]) == reg_width):
            raise MemGenError(f"{srcinfo}: AVX2 vectors of type {prim_type} must be {reg_width}-wide, got {shape}")
        shape = shape[:-1]
        if shape:
            result = f'{C_reg_type_name} {new_name}[{"][".join(map(str, shape))}];'
        else:
            result = f"{C_reg_type_name} {new_name};"
        return result


def get_avx2_instrs(vw, precision):
    type_sfx = "ps" if precision == "f32" else "pd"
    itype_sfx = "epi32" if precision == "f32" else "epi64x"

    def make_stmt(instr, pfx=False):
        if not pfx:
            return f"{{dst_data}} = {instr};"
        else:
            return f"{{dst_data}} = mm256_prefix_{type_sfx}({{dst_data}}, {instr}, {{m}});"

    # Load Family
    avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_load, vw, precision, VEC_AVX2)
    yield make_instr(avx2_vec_op, f"{{dst_data}} = _mm256_loadu_{type_sfx}(&{{src_data}});")
    yield make_instr(
        avx2_vec_op_pfx, f"{{dst_data}} = _mm256_maskload_{type_sfx}(&{{src_data}}, mm256_prefix_mask_{itype_sfx}({{m}}));"
    )
    avx2_vec_op, _ = specialize_vec_op(vec_load_bck, vw, precision, VEC_AVX2)
    yield make_instr(avx2_vec_op, f"{{dst_data}} = _mm256_loadu_{type_sfx}(&{{src_data}});")

    # Store Family
    avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_store, vw, precision, VEC_AVX2)
    yield make_instr(avx2_vec_op, f"_mm256_storeu_{type_sfx}(&{{dst_data}}, {{src_data}});")
    yield make_instr(
        avx2_vec_op_pfx, f"_mm256_maskstore_{type_sfx}(&{{dst_data}}, mm256_prefix_mask_{itype_sfx}({{m}}), {{src_data}});"
    )
    avx2_vec_op, _ = specialize_vec_op(vec_store_bck, vw, precision, VEC_AVX2)
    yield make_instr(avx2_vec_op, f"_mm256_storeu_{type_sfx}(&{{dst_data}}, {{src_data}});")

    # Copy Family
    avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_copy, vw, precision, VEC_AVX2)
    yield make_instr(avx2_vec_op, "{dst_data} = {src_data};")
    yield make_instr(avx2_vec_op_pfx, "{dst_data} = {src_data};")

    # Neg Family
    avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_neg, vw, precision, VEC_AVX2)
    neg_mask = "0x8" + "0" * 7 if precision == "f32" else "0x8" + "0" * 15 + "LL"
    neg_mask = f"_mm256_castsi256_{type_sfx}(_mm256_set1_{itype_sfx}({neg_mask}))"
    neg_instr = f"_mm256_xor_{type_sfx}({{src_data}}, {neg_mask})"
    yield make_instr(avx2_vec_op, make_stmt(neg_instr))
    yield make_instr(avx2_vec_op_pfx, make_stmt(neg_instr, True))

    # Add_red Family
    avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_add_red, vw, precision, VEC_AVX2)
    add_red_instr = f"_mm256_add_{type_sfx}({{dst_data}}, {{src_data}})"
    yield make_instr(avx2_vec_op, make_stmt(add_red_instr))
    yield make_instr(avx2_vec_op_pfx, make_stmt(add_red_instr, True))

    # binop Family
    binops = [(vec_add, "add"), (vec_sub, "sub"), (vec_mul, "mul"), (vec_div, "div")]
    for instr, name in binops:
        avx2_vec_binop, avx2_vec_binop_pfx = specialize_vec_op(instr, vw, precision, VEC_AVX2)
        c_instr = f"_mm256_{name}_{type_sfx}({{src1_data}}, {{src2_data}})"
        yield make_instr(avx2_vec_binop, make_stmt(c_instr))
        yield make_instr(avx2_vec_binop_pfx, make_stmt(c_instr, True))

    # Broadcast Family
    for instr, c_pfx in (vec_brdcst_buf, ""), (vec_brdcst_scl, "*"):
        avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(instr, vw, precision, VEC_AVX2)
        brdcst_instr = f"_mm256_set1_{type_sfx}({c_pfx}{{src_data}})"
        yield make_instr(avx2_vec_op, make_stmt(brdcst_instr))
        yield make_instr(avx2_vec_op_pfx, make_stmt(brdcst_instr, True))

    # Set zero Family
    avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_zero, vw, precision, VEC_AVX2)
    zero_instr = f"_mm256_setzero_ps()"
    yield make_instr(avx2_vec_op, make_stmt(zero_instr))
    yield make_instr(avx2_vec_op_pfx, make_stmt(zero_instr, True))

    # Fmadd Family
    avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_fmadd_red, vw, precision, VEC_AVX2)
    fmadd_red_instr = f"_mm256_fmadd_{type_sfx}({{src1_data}}, {{src2_data}}, {{dst_data}})"
    yield make_instr(avx2_vec_op, make_stmt(fmadd_red_instr))
    yield make_instr(avx2_vec_op_pfx, make_stmt(fmadd_red_instr, True))

    for vec_fmadd in vec_fmadd1, vec_fmadd2:
        avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_fmadd, vw, precision, VEC_AVX2)
        fmadd_instr = f"_mm256_fmadd_{type_sfx}({{src1_data}}, {{src2_data}}, {{src3_data}})"
        yield make_instr(avx2_vec_op, make_stmt(fmadd_instr))
        yield make_instr(avx2_vec_op_pfx, make_stmt(fmadd_instr, True))

    # Abs Family
    avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_abs, vw, precision, VEC_AVX2)
    abs_mask = "0x7" + "F" * 7 if precision == "f32" else "0x7" + "F" * 15 + "LL"
    abs_mask = f"_mm256_castsi256_{type_sfx}(_mm256_set1_{itype_sfx}({abs_mask}))"
    abs_instr = f"_mm256_and_{type_sfx}({{src_data}}, {abs_mask})"
    yield make_instr(avx2_vec_op, make_stmt(abs_instr))
    yield make_instr(avx2_vec_op_pfx, make_stmt(abs_instr, True))

    # Reduce Add Family
    for instr, c_pfx in (vec_reduce_add_buf, ""), (vec_reduce_add_scl, "*"):
        avx2_vec_op, _ = specialize_vec_op(instr, vw, precision, VEC_AVX2)
        reduce_add_instr = f"mm256_reduce_add_{type_sfx}({{src_data}})"
        yield make_instr(avx2_vec_op, f"{c_pfx}{{dst_data}} = {reduce_add_instr};")

    # Load f32 cvt Family
    if precision == "f64":
        avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(load_f32_cvt, vw, precision, VEC_AVX2)
        yield make_instr(avx2_vec_op, f"{{dst_data}} = _mm256_cvtps_pd(_mm_loadu_ps(&{{src_data}}));")
        yield make_instr(
            avx2_vec_op_pfx, f"{{dst_data}} = _mm256_cvtps_pd(_mm_maskload_ps(&{{src_data}}, mm_prefix_mask_epi32({{m}})));"
        )


avx2_instrs = []

for vw, precision in ((8, "f32"), (4, "f64")):
    avx2_instrs += list(get_avx2_instrs(vw, precision))

Machine = MachineParameters(
    name="avx2",
    mem_type=VEC_AVX2,
    n_vec_registers=16,
    f32_vec_width=8,
    vec_units=2,
    supports_predication=True,
    l1_cache=None,
    l2_cache=None,
    l3_cache=None,
    instrs=avx2_instrs,
)
