import difflib
import sys
import os
from pathlib import Path


def check_diff(file1, file2):
    # Open and read the files
    with open(file1, "r") as f1:
        f1_text = f1.readlines()

    with open(file2, "r") as f2:
        f2_text = f2.readlines()

    if f1_text != f2_text:
        diff = difflib.unified_diff(
            f1_text, f2_text, fromfile=str(file1), tofile=str(file2), lineterm=""
        )
        diff = "\n".join(diff)
        exit(f"Error: files {file1} and {file2} have the following diff:\n{diff}")


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).parent.parent.parent.resolve()

    target_arch = sys.argv[1]
    level = sys.argv[2]
    kernel = sys.argv[3]

    KERNEL_BUILD_DIR = (
        REPO_ROOT / "build" / target_arch / "src" / level / f"{kernel}.exo"
    )
    EXPECTED_DIR = REPO_ROOT / "test" / "codegen" / "expected" / target_arch

    header = f"{kernel}.h"
    source = f"{kernel}.c"

    check_diff(KERNEL_BUILD_DIR / header, EXPECTED_DIR / header)
    check_diff(KERNEL_BUILD_DIR / source, EXPECTED_DIR / source)
