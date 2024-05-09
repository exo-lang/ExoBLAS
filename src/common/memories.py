from exo.libs.memories import DRAM


class ALIGNED_DRAM_STATIC(DRAM):
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        # Error checking only
        for extent in shape:
            try:
                int(extent)
            except ValueError as e:
                raise MemGenError(f"DRAM_STATIC requires constant shapes. Saw: {extent}") from e

        return f'_Alignas(64) static {prim_type} {new_name}[{" * ".join(shape)}];'

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""
