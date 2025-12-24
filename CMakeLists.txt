cmake_minimum_required(VERSION 3.26.4)

# Include RAPIDS CMake helpers from cuopt
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cuopt/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cuopt/cpp/cmake/thirdparty")

include(cuopt/cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(MINIMAL_DUAL_SIMPLEX)

project(minimal_dual_simplex 
    VERSION "${RAPIDS_VERSION}"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set default build type
rapids_cmake_build_type(Release)

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC FALSE)

# Find required packages
rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET dual-simplex-exports
    INSTALL_EXPORT_SET dual-simplex-exports
)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Try to find CUDSS (optional - for barrier/interior point methods)
# Note: CUDSS is required for barrier methods but not for dual-simplex LP solving
set(HAVE_CUDSS OFF)
if(DEFINED ENV{CUDSS_DIR})
    message(STATUS "CUDSS_DIR is set - attempting to find CUDSS")
    include(cuopt/cpp/cmake/thirdparty/FindCUDSS.cmake OPTIONAL RESULT_VARIABLE CUDSS_FOUND_FILE)
    if(CUDSS_FOUND_FILE AND CUDSS_FOUND)
        message(STATUS "CUDSS found - barrier/interior point methods available")
        set(HAVE_CUDSS ON)
    else()
        message(WARNING "CUDSS_DIR set but CUDSS not found")
    endif()
else()
    message(STATUS "CUDSS not available (CUDSS_DIR not set)")
    message(STATUS "  Dual-simplex LP solver will build successfully")
    message(STATUS "  Barrier/interior point methods require CUDSS")
    message(STATUS "  To enable: set CUDSS_DIR environment variable and reconfigure")
endif()

# Get dependencies (same as cuopt)
rapids_cpm_init()

include(cuopt/cpp/cmake/thirdparty/get_cccl.cmake)
include(cuopt/cpp/cmake/thirdparty/get_rmm.cmake)
include(cuopt/cpp/cmake/thirdparty/get_raft.cmake)

# Get rapids logger and generate logger macros
include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
rapids_cpm_rapids_logger(BUILD_EXPORT_SET dual-simplex-exports INSTALL_EXPORT_SET dual-simplex-exports)
create_logger_macros(CUOPT "cuopt::default_logger()" cuopt/cpp/include/cuopt)

# Write version file for cuopt
rapids_cmake_write_version_file(cuopt/cpp/include/cuopt/version_config.hpp)

# MPS Parser library
add_library(mps_parser STATIC
    libmps_parser/src/parser.cpp
    libmps_parser/src/mps_parser.cpp
    libmps_parser/src/mps_data_model.cpp
    libmps_parser/src/data_model_view.cpp
    libmps_parser/src/writer.cpp
    libmps_parser/src/mps_writer.cpp
)
target_include_directories(mps_parser 
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/libmps_parser/src
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/libmps_parser/include
)

# Utilities library (logger only - version_info not needed for minimal solver)
add_library(utilities STATIC
    cuopt/cpp/src/utilities/logger.cpp
)
target_include_directories(utilities PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuopt/cpp/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cuopt/cpp/include
    ${CMAKE_CURRENT_BINARY_DIR}/cuopt/cpp/include
)
target_link_libraries(utilities PUBLIC
    rapids_logger::rapids_logger
)

# Dual Simplex solver library (LP only, exclude MIP/B&B and barrier/IPM files)
file(GLOB DUAL_SIMPLEX_SOURCES
    "src/dual_simplex/*.cpp"
    "src/dual_simplex/*.cu"
)

# Exclude MIP-specific files (branch_and_bound, mip_node, diving, pseudo_costs)
# and interior point method files (barrier, sparse_cholesky which need CUDSS)
list(FILTER DUAL_SIMPLEX_SOURCES EXCLUDE REGEX "branch_and_bound|mip_node|diving|pseudo_costs|barrier")

add_library(dual_simplex STATIC
    ${DUAL_SIMPLEX_SOURCES}
)

target_include_directories(dual_simplex PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cuopt/cpp/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cuopt/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libmps_parser/include
)

# Add CUDSS include directories if available
if(HAVE_CUDSS)
    target_include_directories(dual_simplex PUBLIC ${CUDSS_INCLUDE})
endif()

target_link_libraries(dual_simplex PUBLIC
    CUDA::cublas
    CUDA::cusparse
    CUDA::cudart
    utilities
    raft::raft
    rmm::rmm
    OpenMP::OpenMP_CXX
)

# Add CUDSS libraries if available
if(HAVE_CUDSS)
    target_link_libraries(dual_simplex PUBLIC ${CUDSS_LIBRARIES} ${CUDSS_MT_LIB_FILE})
endif()

# Linear programming glue layer  
add_library(linear_programming STATIC
    src/linear_programming/optimization_problem.cu
    src/linear_programming/solve.cu
)

target_include_directories(linear_programming PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cuopt/cpp/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cuopt/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libmps_parser/include
)

# Add CUDSS include directories if available
if(HAVE_CUDSS)
    target_include_directories(linear_programming PUBLIC ${CUDSS_INCLUDE})
    # Pass CUDSS_MT_LIB_FILE_NAME as a compile definition
    get_filename_component(CUDSS_MT_LIB_FILE_NAME "${CUDSS_MT_LIB_FILE}" NAME)
    target_compile_definitions(linear_programming PRIVATE CUDSS_MT_LIB_FILE_NAME="${CUDSS_MT_LIB_FILE_NAME}")
    target_compile_definitions(dual_simplex PRIVATE CUDSS_MT_LIB_FILE_NAME="${CUDSS_MT_LIB_FILE_NAME}")
endif()

target_link_libraries(linear_programming PUBLIC
    dual_simplex
    mps_parser
    utilities
    raft::raft
    rmm::rmm
    CUDA::cudart
)

# Add CUDSS libraries if available
if(HAVE_CUDSS)
    target_link_libraries(linear_programming PUBLIC ${CUDSS_LIBRARIES} ${CUDSS_MT_LIB_FILE})
endif()

# Main executable
add_executable(dual_simplex_solver
    main.cpp
)

target_include_directories(dual_simplex_solver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cuopt/cpp/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cuopt/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libmps_parser/include
)

target_link_libraries(dual_simplex_solver PRIVATE
    dual_simplex
    linear_programming
    mps_parser
    utilities
    CUDA::cudart
)

# Set CUDA architectures
set_target_properties(dual_simplex linear_programming dual_simplex_solver
    PROPERTIES
    CUDA_ARCHITECTURES "80;86;89;90"
    CUDA_SEPARABLE_COMPILATION ON
)

# Compiler options
target_compile_options(dual_simplex PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_compile_options(linear_programming PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Set log level
target_compile_definitions(dual_simplex PUBLIC "CUOPT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_INFO")
target_compile_definitions(linear_programming PUBLIC "CUOPT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_INFO")
target_compile_definitions(dual_simplex_solver PUBLIC "CUOPT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_INFO")

message(STATUS "Minimal Dual Simplex Solver Configuration Complete")
message(STATUS "Build with: cmake -B build && cmake --build build")

