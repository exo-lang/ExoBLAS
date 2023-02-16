set(PYTHONPATH
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/src/common"
  "${PROJECT_BINARY_DIR}/src/common"
)

if (HAVE_NEON)
  add_exo_library(
    NAME exo_gemv
    SOURCES gemv.py
    PYTHONPATH ${PYTHONPATH}
  )
endif ()
