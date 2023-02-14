if (HAVE_NEON)
  add_exo_library(
    NAME gemv
    SOURCES gemv.py
    PYTHONPATH "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif ()
