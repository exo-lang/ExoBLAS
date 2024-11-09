#!/bin/bash

# Check if the directory path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

DIR="$1"

# Create level1, level2, and level3 directories inside the provided directory
mkdir -p "$DIR/level1"
mkdir -p "$DIR/level2"
mkdir -p "$DIR/level3"

# List of subdirectories to copy
SUBDIRS=("exo" "FLAME" "Intel10_64lp_seq" "OpenBLAS")

# Copy the subdirectories into all levels
for SUBDIR in "${SUBDIRS[@]}"; do
    cp -r "$DIR/$SUBDIR" "$DIR/level1/"
    cp -r "$DIR/$SUBDIR" "$DIR/level2/"
    cp -r "$DIR/$SUBDIR" "$DIR/level3/"
done

# BLAS Level 1 operations
LEVEL1_OPS=(
    "asum.json" "axpy.json" "copy.json" "dot.json" "dsdot.json"
    "rot.json" "rotm.json" "sdsdot.json" "scal.json" "swap.json"
)

# Files to keep in level2
LEVEL2_KEEP=(
    "gemv.json" "ger.json" "symv.json" "syr2.json"
    "syr.json" "trmv.json" "trsv.json"
)

# Files to keep in level3
LEVEL3_KEEP=(
    "gemm.json" "symm.json" "syrk.json"
)

# Function to delete files not in a given list
delete_unwanted_files() {
    local TARGET_DIR="$1"
    shift
    local KEEP_FILES=("$@")

    for SUBDIR in "${SUBDIRS[@]}"; do
        for FILE in "$TARGET_DIR/$SUBDIR/"*.json; do
            BASENAME=$(basename "$FILE")
            if [[ ! " ${KEEP_FILES[@]} " =~ " $BASENAME " ]]; then
                rm -f "$FILE"
            fi
        done
    done
}

# Delete unwanted files in level1
delete_unwanted_files "$DIR/level1" "${LEVEL1_OPS[@]}"

# Delete unwanted files in level2
delete_unwanted_files "$DIR/level2" "${LEVEL2_KEEP[@]}"

# Delete unwanted files in level3
delete_unwanted_files "$DIR/level3" "${LEVEL3_KEEP[@]}"
