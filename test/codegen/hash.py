import difflib
import sys
import os
from pathlib import Path
import json
import hashlib
import shutil


REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
CODEGEN_DIR = REPO_ROOT / "test" / "codegen"
REF_DIR = CODEGEN_DIR / "reference"
REF_HASH_DIR = REF_DIR / "sha256"
REF_SRC_DIR = REF_DIR / "sources"
BLD_DIR = REPO_ROOT / "build"
VERBOSE = False


# Be very catious adding tests here. Ideally, we don't want any.
NON_DETERMINISTIC_TESTS = {"exo_trmv"}


def get_diff(file1, file2):
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
        return diff

    return ""


def get_sha256_of_file(filename):
    """Calculates the SHA-256 hash of a file.

    Args:
        filename: The path to the file.

    Returns:
        The SHA-256 hash of the file, as a hexadecimal string.
    """

    with open(filename, "rb") as f:
        sha256_hash = hashlib.sha256()
        # Read and update hash in chunks for efficiency
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_build_filename(target_arch, level, kernel):
    return BLD_DIR / target_arch / "src" / level / f"{kernel}.exo" / f"{kernel}.c"


def get_reference_source_filename(target_arch, kernel):
    return REF_SRC_DIR / target_arch / f"{kernel}.c"


def get_reference_hash_filename(target_arch):
    return REF_HASH_DIR / f"{target_arch}.json"


def get_reference_hash(target_arch, kernel):
    filename = get_reference_hash_filename(target_arch)
    with open(filename, "r") as f:
        sha256_dict = json.load(f)
    return sha256_dict.get(kernel)


def update_reference_hash(target_arch, kernel, new_hash):
    filename = get_reference_hash_filename(target_arch)
    with open(filename, "r") as f:
        sha256_dict = json.load(f)
    sha256_dict[kernel] = new_hash
    with open(filename, "w") as f:
        json.dump(sha256_dict, f, sort_keys=True, indent=4, separators=(",", ": "))


def get_build_hash(target_arch, level, kernel):
    filename = get_build_filename(target_arch, level, kernel)
    return get_sha256_of_file(filename)


def get_reference_source_hash(target_arch, kernel):
    filename = get_reference_source_filename(target_arch, kernel)
    return get_sha256_of_file(filename)


update_instructions = """If you want the reference sources to correspond to the main branch generated source, then
1) stash your changes 2) checkout main 3) rebuild the sources 4) run the ctests suffixed with `update_reference`.
"""


def check_sha256(target_arch, level, kernel):
    reference_hash = get_reference_hash(target_arch, kernel)
    build_hash = get_build_hash(target_arch, level, kernel)

    if reference_hash == build_hash:
        return

    err = f"Hash mismatch for kernel {kernel} and arch {target_arch}!\n Expected {reference_hash}, got {build_hash}\n"

    if VERBOSE:
        build_file = get_build_filename(target_arch, level, kernel)
        with open(build_file, "r") as f:
            build_result = f.read()
        err += f"Build hash was computed on the following file:\n### Beginning of file ###\n{build_result}\n### End of file ###\n"

    if kernel in NON_DETERMINISTIC_TESTS:
        print("Failed on a non-deterministic test. Failing Silently!")
        print(err)
        exit(0)

    reference_source = get_reference_source_filename(target_arch, kernel)

    if not os.path.exists(reference_source):
        exit(
            err
            + f"Reference source was not found at {reference_source} to show the diff.\n{update_instructions}."
        )

    reference_source_hash = get_reference_source_hash(target_arch, kernel)
    if reference_source_hash != reference_hash:
        exit(
            err
            + f"""Reference source was found at {reference_source}, but its hash does not correspond
            to the hash found in the reference hash in {get_reference_hash_filename(target_arch)}.\n{update_instructions}"""
        )

    build_source = get_build_filename(target_arch, level, kernel)
    diff = get_diff(reference_source, build_source)
    assert diff, "Hashes mistmatched, must have a diff!"

    exit(
        err
        + f"Reference source file at {reference_source} and build source file at {build_source} have the following diff:\n{diff}"
    )


def update(target_arch, level, kernel):
    reference_source_filename = get_reference_source_filename(target_arch, kernel)
    reference_source_filename.parent.mkdir(parents=True, exist_ok=True)

    build_source_filename = get_build_filename(target_arch, level, kernel)
    if not os.path.exists(build_source_filename):
        exit(
            f"Attempted an update for kernel {kernel} and arch {target_arch}, but build source file not found at {build_source_filename}"
        )
    shutil.copy(build_source_filename, reference_source_filename)

    build_source_hash = get_build_hash(target_arch, level, kernel)
    update_reference_hash(target_arch, kernel, build_source_hash)


def help():
    return """
    Usage: python3 hash.py [check | update] [target_arch] [level] [kernel] [Optional: -V]
    """


if __name__ == "__main__":

    if len(sys.argv) < 5 or len(sys.argv) > 6:
        help()

    command = sys.argv[1]
    target_arch = sys.argv[2]
    level = sys.argv[3]
    kernel = sys.argv[4]

    if len(sys.argv) == 6:
        if sys.argv[5] != "-V":
            help()
        VERBOSE = True

    if command == "check":
        check_sha256(target_arch, level, kernel)
    elif command == "update":
        update(target_arch, level, kernel)
    else:
        help()
