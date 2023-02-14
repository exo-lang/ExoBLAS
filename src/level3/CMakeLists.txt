if (HAVE_NEON)
  add_exo_library(
    NAME syrk
    SOURCES syrk.py
    PYTHONPATH "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/src/common"
  )
  add_exo_library(
    NAME gemm
    SOURCES gemm.py
    PYTHONPATH "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/src/common"
  )
endif ()
