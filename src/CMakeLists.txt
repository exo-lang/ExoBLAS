set(COMMON "${PROJECT_SOURCE_DIR}/src/common" "${PROJECT_BINARY_DIR}/src/common")

function(add_blas_library kernel_name file)
    add_exo_library(
        NAME ${kernel_name}
        SOURCES ${file}
        PYTHONPATH ${COMMON})
endfunction()

add_subdirectory(common)
add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)
