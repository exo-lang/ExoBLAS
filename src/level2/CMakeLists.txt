set(PYTHONPATH
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/src/common"
  "${PROJECT_BINARY_DIR}/src/common"
)

add_exo_library(
NAME exo_gemv
SOURCES gemv.py
PYTHONPATH
    "${PROJECT_BINARY_DIR}/src/common"
    "${PROJECT_SOURCE_DIR}/src/common")

add_exo_library(
  NAME exo_ger
  SOURCES ger.py
  PYTHONPATH
      "${PROJECT_BINARY_DIR}/src/common"
      "${PROJECT_SOURCE_DIR}/src/common")
  