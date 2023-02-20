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

add_exo_library(
  NAME exo_asum
  SOURCES asum.py
  PYTHONPATH
    "${PROJECT_BINARY_DIR}/src/common"
    "${PROJECT_SOURCE_DIR}/src/common"
)

add_exo_library(
  NAME exo_nrm2
  SOURCES nrm2.py
  PYTHONPATH
    "${PROJECT_BINARY_DIR}/src/common"
    "${PROJECT_SOURCE_DIR}/src/common"
)

add_exo_library(
  NAME exo_rot
  SOURCES rot.py
  PYTHONPATH
    "${PROJECT_BINARY_DIR}/src/common"
    "${PROJECT_SOURCE_DIR}/src/common"
)

add_exo_library(
  NAME exo_rotm
  SOURCES rotm.py
  PYTHONPATH
    "${PROJECT_BINARY_DIR}/src/common"
    "${PROJECT_SOURCE_DIR}/src/common"
)

add_exo_library(
  NAME exo_scal
  SOURCES scal.py
  PYTHONPATH
    "${PROJECT_BINARY_DIR}/src/common"
    "${PROJECT_SOURCE_DIR}/src/common"
)