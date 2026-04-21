#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "rt_core::rt_core" for configuration "Release"
set_property(TARGET rt_core::rt_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(rt_core::rt_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librt_core.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS rt_core::rt_core )
list(APPEND _IMPORT_CHECK_FILES_FOR_rt_core::rt_core "${_IMPORT_PREFIX}/lib/librt_core.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
