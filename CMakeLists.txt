cmake_minimum_required(VERSION 3.22)

project(graphot_core LANGUAGES CXX)

option(GRAPHOT_USE_CONAN "Install C++ dependencies via Conan during CMake configure" ON)
option(GRAPHOT_ENABLE_OPENMP "Enable OpenMP parallelism in the core solver" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

if(GRAPHOT_USE_CONAN)
  find_program(GRAPHOT_CONAN_EXECUTABLE conan)
  if(NOT GRAPHOT_CONAN_EXECUTABLE)
    message(FATAL_ERROR "GRAPHOT_USE_CONAN=ON but 'conan' was not found on PATH")
  endif()

  set(GRAPHOT_CONAN_HOME "${CMAKE_BINARY_DIR}/conan-home")
  set(GRAPHOT_CONAN_OUTPUT "${CMAKE_BINARY_DIR}/conan")
  set(GRAPHOT_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
  if(NOT GRAPHOT_BUILD_TYPE)
    set(GRAPHOT_BUILD_TYPE "Release")
  endif()
  file(MAKE_DIRECTORY "${GRAPHOT_CONAN_HOME}")
  file(MAKE_DIRECTORY "${GRAPHOT_CONAN_OUTPUT}")

  if(NOT EXISTS "${GRAPHOT_CONAN_HOME}/profiles/default")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env "CONAN_HOME=${GRAPHOT_CONAN_HOME}" "${GRAPHOT_CONAN_EXECUTABLE}" profile detect --force
      RESULT_VARIABLE graphot_conan_profile_result
      COMMAND_ECHO STDOUT
    )
    if(NOT graphot_conan_profile_result EQUAL 0)
      message(FATAL_ERROR "Conan profile detection failed with exit code ${graphot_conan_profile_result}")
    endif()
  endif()

  if(NOT EXISTS "${GRAPHOT_CONAN_OUTPUT}/pybind11-config.cmake")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env "CONAN_HOME=${GRAPHOT_CONAN_HOME}" "${GRAPHOT_CONAN_EXECUTABLE}" install "${CMAKE_SOURCE_DIR}" -of "${GRAPHOT_CONAN_OUTPUT}" --build=missing -s "build_type=${GRAPHOT_BUILD_TYPE}"
      RESULT_VARIABLE graphot_conan_result
      COMMAND_ECHO STDOUT
    )
    if(NOT graphot_conan_result EQUAL 0)
      message(FATAL_ERROR "Conan dependency resolution failed with exit code ${graphot_conan_result}")
    endif()
  endif()

  list(PREPEND CMAKE_PREFIX_PATH "${GRAPHOT_CONAN_OUTPUT}")
endif()

find_package(pybind11 CONFIG REQUIRED)

add_library(
  graphot_core_internal
  STATIC
    cpp/src/runtime.cpp
    cpp/src/mean_ops.cpp
    cpp/src/state_ops.cpp
    cpp/src/ceh_linear.cpp
    cpp/src/prox_operators.cpp
    cpp/src/solver.cpp
)
target_compile_features(graphot_core_internal PRIVATE cxx_std_20)
target_include_directories(graphot_core_internal PUBLIC "${CMAKE_SOURCE_DIR}/cpp/include")

pybind11_add_module(_core MODULE cpp/module.cpp cpp/src/pybind_io.cpp)
target_compile_features(_core PRIVATE cxx_std_20)
target_compile_definitions(_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)
target_include_directories(_core PRIVATE "${CMAKE_SOURCE_DIR}/cpp/include")
target_link_libraries(_core PRIVATE graphot_core_internal)

if(GRAPHOT_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(graphot_core_internal PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(graphot_core_internal PUBLIC GRAPHOT_USE_OPENMP=1)
  else()
    message(WARNING "GRAPHOT_ENABLE_OPENMP=ON but OpenMP was not found; building single-threaded core")
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(graphot_core_internal PRIVATE -O3 -Wall -Wextra -Wpedantic)
  target_compile_options(_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _core LIBRARY DESTINATION graphot)
