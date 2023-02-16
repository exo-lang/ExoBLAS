set(PYTHONPATH
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/src/common"
  "${PROJECT_BINARY_DIR}/src/common"
)

add_exo_library(
  NAME exo_syrk
  SOURCES syrk.py
  PYTHONPATH ${PYTHONPATH}
)

add_exo_library(
  NAME exo_gemm
  SOURCES gemm.py
  PYTHONPATH ${PYTHONPATH}
)
