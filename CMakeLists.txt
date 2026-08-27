cmake_minimum_required(VERSION 3.24)

project(PrefillFabric
        VERSION 1.0.0
        DESCRIPTION "Vendor-neutral runtime for scheduling, packing, executing, measuring, and governing prefill work"
        LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ---- Options ----
option(PREFILLFABRIC_BUILD_TESTS "Build Prefill Fabric tests" ON)
option(PREFILLFABRIC_BUILD_EXAMPLES "Build Prefill Fabric examples" ON)
option(PREFILLFABRIC_BUILD_CLI "Build the Prefill Fabric CLI" ON)
option(PREFILLFABRIC_BUILD_BENCHMARKS "Build Prefill Fabric benchmarks" ON)
option(PREFILLFABRIC_ENABLE_CUDA "Enable the CUDA executor backend (auto-detected)" ON)
if(NOT DEFINED PREFILLFABRIC_CUDA_ARCH)
  set(PREFILLFABRIC_CUDA_ARCH "native" CACHE STRING "CUDA compute architecture (e.g. 120, 80, native)")
endif()

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ---- Strict warnings ----
if(MSVC)
  add_compile_options(/W4 /WX /permissive- /utf-8)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# ---- CUDA auto-detect ----
if(PREFILLFABRIC_ENABLE_CUDA)
  find_package(CUDAToolkit 12.0 QUIET)
  if(CUDAToolkit_FOUND)
    set(PREFILLFABRIC_HAS_CUDA TRUE)
    enable_language(CUDA)
    message(STATUS "PrefillFabric: CUDA toolkit ${CUDAToolkit_VERSION} detected")
  else()
    set(PREFILLFABRIC_HAS_CUDA FALSE)
    message(STATUS "PrefillFabric: CUDA toolkit not found; CPU-only build")
  endif()
else()
  set(PREFILLFABRIC_HAS_CUDA FALSE)
endif()

if(PREFILLFABRIC_HAS_CUDA)
  set(CUDA_ARCH_LIST ${PREFILLFABRIC_CUDA_ARCH})
  if(PREFILLFABRIC_CUDA_ARCH MATCHES "^native$")
    message(STATUS "PrefillFabric: CUDA architecture = native")
  else()
    message(STATUS "PrefillFabric: CUDA architecture = ${PREFILLFABRIC_CUDA_ARCH}")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${PREFILLFABRIC_CUDA_ARCH})
endif()

# ---- Core library sources ----
file(GLOB_RECURSE PREFILLFABRIC_CORE_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
file(GLOB PREFILLFABRIC_CPU_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/exec/cpu/*.cpp")
file(GLOB PREFILLFABRIC_CUDA_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/exec/cuda/*.cu")
file(GLOB PREFILLFABRIC_NET_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/net/*.cpp")

add_library(PrefillFabric STATIC
    ${PREFILLFABRIC_CORE_SOURCES}
    ${PREFILLFABRIC_CPU_SOURCES}
    ${PREFILLFABRIC_NET_SOURCES})

if(WIN32)
  target_link_libraries(PrefillFabric PRIVATE ws2_32)
endif()

if(PREFILLFABRIC_HAS_CUDA AND PREFILLFABRIC_CUDA_SOURCES)
  # CMake-generated nvcc commands on Ninja+MSVC emit -Xcompiler=-Fd...,-FS which
  # nvcc rejects ("single input file required"). Compile the CUDA executor with a
  # dedicated nvcc invocation and add the resulting object to the library.
  set(FF_CUDA_OBJ ${CMAKE_CURRENT_BINARY_DIR}/ff_cuda_executor.obj)
  if(PREFILLFABRIC_CUDA_ARCH MATCHES "^native$")
    set(FF_CUDA_GENCODE "-arch=native")
  else()
    set(FF_CUDA_GENCODE "--generate-code=arch=compute_${PREFILLFABRIC_CUDA_ARCH},code=[compute_${PREFILLFABRIC_CUDA_ARCH},sm_${PREFILLFABRIC_CUDA_ARCH}]")
  endif()
  add_custom_command(
    OUTPUT ${FF_CUDA_OBJ}
    COMMAND ${CMAKE_CUDA_COMPILER} -forward-unknown-to-host-compiler
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            -std=c++20
            ${FF_CUDA_GENCODE}
            -Xcompiler=/EHsc
            -Xcompiler=$<$<CONFIG:Debug>:/MDd>$<$<CONFIG:Release>:/MD>
            -Xcompiler=/W4
            -Xcompiler=/WX
            -c ${CMAKE_CURRENT_SOURCE_DIR}/src/exec/cuda/cuda_executor.cu
            -o ${FF_CUDA_OBJ}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/exec/cuda/cuda_executor.cu
    COMMENT "Compiling CUDA executor (nvcc)")
  add_custom_target(prefillfabric_cuda_obj DEPENDS ${FF_CUDA_OBJ})
  add_dependencies(PrefillFabric prefillfabric_cuda_obj)
  target_sources(PrefillFabric PRIVATE ${FF_CUDA_OBJ})
  target_link_libraries(PrefillFabric PUBLIC CUDA::cudart)
else()
  target_sources(PrefillFabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/exec/cuda/cuda_executor_stub.cpp)
endif()

target_include_directories(PrefillFabric PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(PrefillFabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(PrefillFabric PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
    COMPILE_PDB_NAME ""
    CUDA_PDB_NAME "")

if(MSVC)
  target_compile_options(PrefillFabric PRIVATE /W4 /WX)
endif()

# Ensure the vendor-neutral public headers are available to consumers.
target_compile_definitions(PrefillFabric PUBLIC PREFILLFABRIC_VERSION="1.0.0")

# ---- Subdirectories ----
if(PREFILLFABRIC_BUILD_CLI)
  add_subdirectory(src/cli)
endif()
if(PREFILLFABRIC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(PREFILLFABRIC_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(PREFILLFABRIC_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# ---- Install / export ----
install(TARGETS PrefillFabric EXPORT PrefillFabricTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/prefillfabric DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT PrefillFabricTargets
        FILE PrefillFabricTargets.cmake
        NAMESPACE PrefillFabric::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PrefillFabric)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PrefillFabricConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PrefillFabricConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PrefillFabric)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/PrefillFabricConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/PrefillFabricConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/PrefillFabricConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PrefillFabric)
