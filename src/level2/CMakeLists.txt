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
NAME exo_gbmv
SOURCES gbmv.py
PYTHONPATH
    "${PROJECT_BINARY_DIR}/src/common"
    "${PROJECT_SOURCE_DIR}/src/common"
    "${PROJECT_SOURCE_DIR}/src/level1")

add_exo_library(
  NAME exo_ger
  SOURCES ger.py
  PYTHONPATH
      "${PROJECT_BINARY_DIR}/src/common"
      "${PROJECT_SOURCE_DIR}/src/common")

add_exo_library(
  NAME exo_trmv
  SOURCES trmv.py
  PYTHONPATH
      "${PROJECT_BINARY_DIR}/src/common"
      "${PROJECT_SOURCE_DIR}/src/common")

add_exo_library(
  NAME exo_tbmv
  SOURCES tbmv.py
  PYTHONPATH
      "${PROJECT_BINARY_DIR}/src/common"
      "${PROJECT_SOURCE_DIR}/src/common")

add_exo_library(
  NAME exo_symv
  SOURCES symv.py
  PYTHONPATH
      "${PROJECT_BINARY_DIR}/src/common"
      "${PROJECT_SOURCE_DIR}/src/common")

add_exo_library(
  NAME exo_trsv
  SOURCES trsv.py
  PYTHONPATH
      "${PROJECT_BINARY_DIR}/src/common"
      "${PROJECT_SOURCE_DIR}/src/common")

add_exo_library(
  NAME exo_tbsv
  SOURCES tbsv.py
  PYTHONPATH
      "${PROJECT_BINARY_DIR}/src/common"
      "${PROJECT_SOURCE_DIR}/src/common")

add_exo_library(
  NAME exo_sbmv
  SOURCES sbmv.py
  PYTHONPATH
      "${PROJECT_BINARY_DIR}/src/common"
      "${PROJECT_SOURCE_DIR}/src/common")

add_exo_library(
  NAME exo_syr
  SOURCES syr.py
  PYTHONPATH
      "${PROJECT_BINARY_DIR}/src/common"
      "${PROJECT_SOURCE_DIR}/src/common")

add_exo_library(
  NAME exo_syr2
  SOURCES syr2.py
  PYTHONPATH
      "${PROJECT_BINARY_DIR}/src/common"
      "${PROJECT_SOURCE_DIR}/src/common")
