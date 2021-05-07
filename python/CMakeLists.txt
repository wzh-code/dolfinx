cmake_minimum_required(VERSION 3.10)

PROJECT(dolfinx_pybind11)

find_package(Basix REQUIRED)
find_package(DOLFINX REQUIRED)

# Find dependecies
set(XTENSOR_PYTHON_MIN_VERSION 0.25.1)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
  find_package(xtensor-python ${XTENSOR_PYTHON_MIN_VERSION}...<0.26 QUIET)
else()
  find_package(xtensor-python ${XTENSOR_PYTHON_MIN_VERSION} QUIET)
<<<<<<< HEAD
=======
endif()
if(NOT xtensor-python_FOUND)
  include(FetchContent)
  message("downloading xtensor-python source...")
  FetchContent_Declare(
    xtensor-python
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-python.git
    GIT_TAG        0.25.1
  )
  FetchContent_MakeAvailable(xtensor-python)
else()
  message("found xtensor-python ${xtensor-python_VERSION}")
>>>>>>> garth/xtensor-pybind11-2
endif()
if(NOT xtensor-python_FOUND)
  include(FetchContent)
  message("downloading xtensor-python source...")
  FetchContent_Declare(
    xtensor-python
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-python.git
    GIT_TAG        0.25.1
  )
  FetchContent_MakeAvailable(xtensor-python)
else()
  message("found xtensor-python ${xtensor-python_VERSION}")
endif()



# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

# Create the binding library
pybind11_add_module(cpp SHARED
  dolfinx/wrappers/dolfin.cpp
  dolfinx/wrappers/common.cpp
  dolfinx/wrappers/fem.cpp
  dolfinx/wrappers/generation.cpp
  dolfinx/wrappers/geometry.cpp
  dolfinx/wrappers/graph.cpp
  dolfinx/wrappers/io.cpp
  dolfinx/wrappers/la.cpp
  dolfinx/wrappers/log.cpp
  dolfinx/wrappers/mesh.cpp
  dolfinx/wrappers/nls.cpp
  dolfinx/wrappers/refinement.cpp)

# Add srict compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -pedantic" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  target_compile_options(cpp PRIVATE -Wall;-Werror;-pedantic)
endif()

# Add DOLFINX libraries and other config
target_link_libraries(cpp PRIVATE pybind11::headers dolfinx xtensor-python)

# Add xsimd if the Basix C++ interface was compiled with it
if("XTENSOR_USE_XSIMD" IN_LIST BASIX_DEFN OR "XTENSOR_USE_XSIMD" IN_LIST DOLFINX_DEFN)
  target_link_libraries(cpp PRIVATE xsimd)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_target_properties(cpp PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF
                                       CXX_VISIBILITY_PRESET default
                                       VISIBLITY_INLINES_HIDDEN OFF)
else()
  set_target_properties(cpp PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON
                                       CXX_VISIBILITY_PRESET hidden
                                       VISIBLITY_INLINES_HIDDEN ON)
endif()

# Add to CMake search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Check for petsc4py
find_package(PETSc4py REQUIRED)
if (PETSC4PY_FOUND)
  target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIRS})
endif()

# Check for mpi4py
find_package(MPI4PY REQUIRED)
if (MPI4PY_FOUND)
  target_include_directories(cpp PRIVATE ${MPI4PY_INCLUDE_DIR})
  target_compile_definitions(cpp PRIVATE HAS_PYBIND11_MPI4PY)
endif()
