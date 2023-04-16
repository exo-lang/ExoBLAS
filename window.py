
import sys
import re

f = open("build/avx2/src/level3/exo_gemm.exo/exo_gemm.c", "r")
new_f = open("build/avx2/src/level3/exo_gemm.exo/exo_gemm_new.c", "w")

for line in f:
    new_line = line
    if "avx2_microkernel" in line:
        new_line = re.sub(r'([^a-zA-Z0-9_])(exo_win_2f32c?)([^a-zA-Z0-9_])' +
                r'(.*?[^a-zA-Z0-9_])(exo_win_2f32c?)([^a-zA-Z0-9_].*?)' +
                r'([^a-zA-Z0-9_])(exo_win_2f32c?)([^a-zA-Z0-9_])',
                r'\1exo_win_2f32_test\3\4\5\6\7exo_win_2f32_test\9', line) 
                
    new_f.write(new_line)

f.close()
new_f.close()
