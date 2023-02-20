def gepp_syrk_scheduled(N: size, A1: [f32][N, 64] @ DRAM,
                        A2: [f32][64, N] @ DRAM, C: [f32][N, N] @ DRAM):
    assert N >= 1
    assert stride(A1, 1) == 1
    assert stride(A2, 1) == 1
    assert stride(C, 1) == 1
    for io in seq(0, N / 64):
        for jo in seq(0, io):
            gebp_64x64_0(64, C[64 * io:64 + 64 * io, 1 + 64 * jo:65 + 64 * jo],
                         A1[64 * io:64 + 64 * io,
                            0:64], A2[0:64, 1 + 64 * jo:65 + 64 * jo])
        for ii in seq(0, 64):
            for j in seq(0, 1):
                for k in seq(0, 64):
                    C[ii + 64 * io, j] += A1[ii + 64 * io, k] * A2[k, j]
        for ii in seq(0, 1):
            if ii > 0:
                for ji in seq(0, ii):
                    for k in seq(0, 64):
                        C[ii + 64 * io,
                          1 + ji + 64 * io] += A1[ii + 64 * io,
                                                  k] * A2[k, 1 + ji + 64 * io]
        for ii in seq(0, 63):
            for ji in seq(0, 1 + ii):
                for k in seq(0, 64):
                    C[1 + ii + 64 * io,
                      1 + ji + 64 * io] += A1[1 + ii + 64 * io,
                                              k] * A2[k, 1 + ji + 64 * io]
    if N % 64 > 0:
        for ii in seq(0, N % 64):
            for j in seq(0, ii + N / 64 * 64 + 1):
                for k in seq(0, 64):
                    C[ii + N / 64 * 64,
                      j] += A1[ii + N / 64 * 64, k] * A2[k, j]