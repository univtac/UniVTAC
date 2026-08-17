include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/mudaTargets.cmake")