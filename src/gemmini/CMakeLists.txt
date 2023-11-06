# Skip gemmini until we move composed scheudles into
# a standard library
if (OFF)
  add_exo_library(
    NAME exo_gemmini_matmul
    SOURCES exo_gemmini_matmul.py
    PYTHONPATH
      "${PROJECT_BINARY_DIR}/src/common"
      "${PROJECT_SOURCE_DIR}/src/common"
  )
endif()
