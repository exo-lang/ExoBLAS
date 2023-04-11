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
  NAME exo_syr2k
  SOURCES syr2k.py
  PYTHONPATH ${PYTHONPATH}
)

add_exo_library(
  NAME exo_gemm
  SOURCES gemm.py
  PYTHONPATH ${PYTHONPATH}
)

add_exo_library(
  NAME exo_trmm
  SOURCES trmm.py
  PYTHONPATH ${PYTHONPATH}
)

add_exo_library(
  NAME exo_symm
  SOURCES symm.py
  PYTHONPATH ${PYTHONPATH}
)

