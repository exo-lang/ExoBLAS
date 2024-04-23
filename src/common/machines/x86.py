from __future__ import annotations

from machines.abstract_vector import *
from machines.machine_params import MachineParameters

from stdlib import *

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
get_mask_func_256 = """
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

predicate_func_256 = """
static forceinline __m256 mm256_prefix_ps(__m256 dst, __m256 src, int32_t count) {
    __m256i mask = mm256_prefix_mask_epi32(count);
    return _mm256_blendv_ps(dst, src, mask);
}
static forceinline __m256 mm256_prefix_pd(__m256d dst, __m256d src, int32_t count) {
    __m256i mask = mm256_prefix_mask_epi64x(count);
    return _mm256_blendv_pd(dst, src, mask);
}
"""

reduce_add_256 = """
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
        return "\n".join([include, forceinline, get_mask_func_256, predicate_func_256, reduce_add_256])

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


class VEC_AVX512(VEC):
    @classmethod
    def global_(cls):
        include = "#include <immintrin.h>"
        pfx_mask_fun = "#define get_prefix_mask(count) ((1 << count) - 1)"
        return "\n".join([include, pfx_mask_fun])

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            raise MemGenError(f"{srcinfo}: AVX2 vectors are not scalar values")

        vec_types = {
            "float": (16, "__m512"),
            "double": (8, "__m512d"),
            "uint16_t": (32, "__m512i"),
        }

        if not prim_type in vec_types.keys():
            raise MemGenError(f"{srcinfo}: AVX512 vectors must be f32/f64/ui16 (for now), got {prim_type}")

        reg_width, C_reg_type_name = vec_types[prim_type]
        if not (shape[-1].isdecimal() and int(shape[-1]) == reg_width):
            raise MemGenError(f"{srcinfo}: AVX512 vectors of type {prim_type} must be {reg_width}-wide, got {shape}")
        shape = shape[:-1]
        if shape:
            result = f'{C_reg_type_name} {new_name}[{"][".join(map(str, shape))}];'
        else:
            result = f"{C_reg_type_name} {new_name};"
        return result


def get_avx_instrs(VEC_MEM):
    if VEC_MEM is VEC_AVX2:
        bit_l = 256
    elif VEC_MEM is VEC_AVX512:
        bit_l = 512
    else:
        assert False, f"Unkown type {VEC_MEM}"

    for precision_l in 32, 64:
        precision = f"f{precision_l}"
        vw = bit_l // precision_l
        type_sfx = "ps" if precision == "f32" else "pd"
        itype_sfx = "epi32" if precision == "f32" else ("epi64x" if bit_l == 256 else "epi64")
        mm_pfx = f"mm{bit_l}"

        def make_stmt(instr, pfx=False):
            # This in-principle can generate incorrect instructions, but in linear algebra it is okay since we are doing multiplies on masked loads
            DONT_PREDICATE = True

            if VEC_MEM is VEC_AVX2:
                if not pfx or DONT_PREDICATE:
                    return f"{{dst_data}} = {instr};"
                else:
                    return f"{{dst_data}} = {mm_pfx}_prefix_{type_sfx}({{dst_data}}, {instr}, {{m}});"
            elif VEC_MEM is VEC_AVX512:
                if pfx and "set1" not in instr and "setzero" not in instr and not DONT_PREDICATE:
                    pat = "mm512"
                    index = instr.find(pat)
                    instr = instr[: index + len(pat)] + "_mask" + instr[index + len(pat) :]
                    if "fmadd" not in instr:
                        pat = "("
                        index = instr.find(pat)
                        instr = instr[: index + len(pat)] + "{dst_data}, get_prefix_mask({m}), " + instr[index + len(pat) :]
                    else:
                        pat = ", "
                        index = instr.find(pat)
                        instr = instr[: index + len(pat)] + "get_prefix_mask({m}), " + instr[index + len(pat) :]
                return f"{{dst_data}} = {instr};"

        # Load Family
        for load_op in vec_load, vec_load_bck:
            avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(load_op, vw, precision, VEC_MEM)
            yield make_instr(avx2_vec_op, f"{{dst_data}} = _{mm_pfx}_loadu_{type_sfx}(&{{src_data}});")
            mload_op = "maskload" if VEC_MEM is VEC_AVX2 else "maskz_loadu"
            mask = f"{mm_pfx}_prefix_mask_{itype_sfx}({{m}})" if VEC_MEM is VEC_AVX2 else "get_prefix_mask({m})"
            args = f"&{{src_data}}, {mask}" if VEC_MEM is VEC_AVX2 else f"{mask}, &{{src_data}}"
            yield make_instr(avx2_vec_op_pfx, f"{{dst_data}} = _{mm_pfx}_{mload_op}_{type_sfx}({args});")

        # Store Family
        for store_op in vec_store, vec_store_bck:
            avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(store_op, vw, precision, VEC_MEM)
            yield make_instr(avx2_vec_op, f"_{mm_pfx}_storeu_{type_sfx}(&{{dst_data}}, {{src_data}});")
            mstore_op = "maskstore" if VEC_MEM is VEC_AVX2 else "mask_storeu"
            args = f"&{{dst_data}}, {mask}, {{src_data}}"
            yield make_instr(avx2_vec_op_pfx, f"_{mm_pfx}_{mstore_op}_{type_sfx}({args});")

        # Copy Family
        avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_copy, vw, precision, VEC_MEM)
        yield make_instr(avx2_vec_op, "{dst_data} = {src_data};")
        yield make_instr(avx2_vec_op_pfx, "{dst_data} = {src_data};")

        # Neg Family
        avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_neg, vw, precision, VEC_MEM)
        neg_mask = "0x8" + "0" * 7 if precision == "f32" else "0x8" + "0" * 15 + "LL"
        neg_mask = f"_{mm_pfx}_castsi{bit_l}_{type_sfx}(_{mm_pfx}_set1_{itype_sfx}({neg_mask}))"
        neg_instr = f"_{mm_pfx}_xor_{type_sfx}({{src_data}}, {neg_mask})"
        yield make_instr(avx2_vec_op, make_stmt(neg_instr))
        yield make_instr(avx2_vec_op_pfx, make_stmt(neg_instr, True))

        # Add_red Family
        avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_add_red, vw, precision, VEC_MEM)
        add_red_instr = f"_{mm_pfx}_add_{type_sfx}({{dst_data}}, {{src_data}})"
        yield make_instr(avx2_vec_op, make_stmt(add_red_instr))
        yield make_instr(avx2_vec_op_pfx, make_stmt(add_red_instr, True))

        # binop Family
        binops = [(vec_add, "add"), (vec_sub, "sub"), (vec_mul, "mul"), (vec_div, "div")]
        for instr, name in binops:
            avx2_vec_binop, avx2_vec_binop_pfx = specialize_vec_op(instr, vw, precision, VEC_MEM)
            c_instr = f"_{mm_pfx}_{name}_{type_sfx}({{src1_data}}, {{src2_data}})"
            yield make_instr(avx2_vec_binop, make_stmt(c_instr))
            yield make_instr(avx2_vec_binop_pfx, make_stmt(c_instr, True))

        # Broadcast Family
        for instr, c_pfx in (vec_brdcst_buf, ""), (vec_brdcst_scl, "*"):
            avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(instr, vw, precision, VEC_MEM)
            brdcst_instr = f"_{mm_pfx}_set1_{type_sfx}({c_pfx}{{src_data}})"
            yield make_instr(avx2_vec_op, make_stmt(brdcst_instr))
            yield make_instr(avx2_vec_op_pfx, make_stmt(brdcst_instr, True))

        # Set zero Family
        avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_zero, vw, precision, VEC_MEM)
        zero_instr = f"_{mm_pfx}_setzero_{type_sfx}()"
        yield make_instr(avx2_vec_op, make_stmt(zero_instr))
        yield make_instr(avx2_vec_op_pfx, make_stmt(zero_instr, True))

        # Fmadd Family
        avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_fmadd_red, vw, precision, VEC_MEM)
        fmadd_red_instr = f"_{mm_pfx}_fmadd_{type_sfx}({{src1_data}}, {{src2_data}}, {{dst_data}})"
        yield make_instr(avx2_vec_op, make_stmt(fmadd_red_instr))
        yield make_instr(avx2_vec_op_pfx, make_stmt(fmadd_red_instr, True))

        for vec_fmadd in vec_fmadd1, vec_fmadd2:
            avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_fmadd, vw, precision, VEC_MEM)
            fmadd_instr = f"_{mm_pfx}_fmadd_{type_sfx}({{src1_data}}, {{src2_data}}, {{src3_data}})"
            yield make_instr(avx2_vec_op, make_stmt(fmadd_instr))
            yield make_instr(avx2_vec_op_pfx, make_stmt(fmadd_instr, True))

        # Abs Family
        avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(vec_abs, vw, precision, VEC_MEM)
        if VEC_MEM is VEC_AVX2:
            abs_mask = "0x7" + "F" * 7 if precision == "f32" else "0x7" + "F" * 15 + "LL"
            abs_mask = f"_{mm_pfx}_castsi256_{type_sfx}(_{mm_pfx}_set1_{itype_sfx}({abs_mask}))"
            abs_instr = f"_{mm_pfx}_and_{type_sfx}({{src_data}}, {abs_mask})"
        else:
            abs_instr = f"_{mm_pfx}_abs_{type_sfx}({{src_data}})"
        yield make_instr(avx2_vec_op, make_stmt(abs_instr))
        yield make_instr(avx2_vec_op_pfx, make_stmt(abs_instr, True))

        # Reduce Add Family
        native = "_" if VEC_MEM is VEC_AVX512 else ""
        for instr, c_pfx in (vec_reduce_add_buf, ""), (vec_reduce_add_scl, "*"):
            avx2_vec_op, _ = specialize_vec_op(instr, vw, precision, VEC_MEM)
            reduce_add_instr = f"{native}{mm_pfx}_reduce_add_{type_sfx}({{src_data}})"
            yield make_instr(avx2_vec_op, f"{c_pfx}{{dst_data}} = {reduce_add_instr};")

        # Load f32 cvt Family
        if precision == "f64":
            avx2_vec_op, avx2_vec_op_pfx = specialize_vec_op(load_f32_cvt, vw, precision, VEC_MEM)
            if VEC_MEM is VEC_AVX2:
                yield make_instr(avx2_vec_op, f"{{dst_data}} = _{mm_pfx}_cvtps_pd(_mm_loadu_ps(&{{src_data}}));")
                yield make_instr(
                    avx2_vec_op_pfx,
                    f"{{dst_data}} = _{mm_pfx}_cvtps_pd(_mm_maskload_ps(&{{src_data}}, mm_prefix_mask_epi32({{m}})));",
                )
            else:
                yield make_instr(avx2_vec_op, f"{{dst_data}} = _{mm_pfx}_cvtps_pd(_mm256_loadu_ps(&{{src_data}}));")
                yield make_instr(
                    avx2_vec_op_pfx,
                    f"{{dst_data}} = _{mm_pfx}_cvtps_pd(_mm_maskz_load_ps(&{{src_data}}, get_prefix_mask({{m}})));",
                )


avx2_Machine = MachineParameters(
    name="avx2",
    mem_type=VEC_AVX2,
    n_vec_registers=16,
    f32_vec_width=8,
    vec_units=2,
    supports_predication=True,
    l1_cache=None,
    l2_cache=None,
    l3_cache=None,
    instrs=list(get_avx_instrs(VEC_AVX2)),
    patterns=[fma_rule, abs_rule],
)

avx512_Machine = MachineParameters(
    name="avx512",
    mem_type=VEC_AVX512,
    n_vec_registers=32,
    f32_vec_width=16,
    vec_units=2,
    supports_predication=True,
    l1_cache=None,
    l2_cache=None,
    l3_cache=None,
    instrs=list(get_avx_instrs(VEC_AVX512)),
    patterns=[fma_rule, abs_rule],
)
