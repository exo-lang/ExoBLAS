add_exo_library(
  NAME exo_copy
  SOURCES exo_copy.py
  PYTHONPATH
    "${PROJECT_BINARY_DIR}/src/common"
    "${PROJECT_SOURCE_DIR}/src/common"
)

add_exo_library(
  NAME exo_dot
  SOURCES dot.py
  PYTHONPATH
    "${PROJECT_BINARY_DIR}/src/common"
    "${PROJECT_SOURCE_DIR}/src/common"
)
