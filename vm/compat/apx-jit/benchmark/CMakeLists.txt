# APX Benchmark Integration
# Builds benchmark workloads and LD_PRELOAD interceptor

cmake_minimum_required(VERSION 3.16)

# Find required packages
find_package(LLVM REQUIRED CONFIG)

# Get LLVM libraries
llvm_map_components_to_libnames(llvm_libs
  Core
  Support
  Target
  X86CodeGen
  X86AsmParser
  X86Desc
  X86Info
  OrcJIT
  MCJIT
  native
  nativecodegen
)

# ============================================================================
# Benchmark workload (standalone application)
# ============================================================================
add_executable(apx_workload
    workload.cpp
)

target_compile_features(apx_workload PRIVATE cxx_std_17)
target_link_libraries(apx_workload PRIVATE pthread)

# ============================================================================
# APX interceptor library (LD_PRELOAD)
# ============================================================================
add_library(apx_interceptor SHARED
    apx_interceptor.cpp
)

target_include_directories(apx_interceptor
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(apx_interceptor
    PRIVATE
        bpftime_apx_jit
        ${llvm_libs}
        spdlog::spdlog
        pthread
        dl
)

target_compile_features(apx_interceptor PRIVATE cxx_std_20)
target_compile_definitions(apx_interceptor PRIVATE ${LLVM_DEFINITIONS})

# ============================================================================
# Standalone benchmark (no LD_PRELOAD)
# ============================================================================
add_executable(apx_benchmark_standalone
    benchmark_standalone.cpp
)

target_include_directories(apx_benchmark_standalone
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(apx_benchmark_standalone
    PRIVATE
        bpftime_apx_jit
        ${llvm_libs}
        spdlog::spdlog
        pthread
)

target_compile_features(apx_benchmark_standalone PRIVATE cxx_std_20)
target_compile_definitions(apx_benchmark_standalone PRIVATE ${LLVM_DEFINITIONS})

# ============================================================================
# Install
# ============================================================================
install(TARGETS apx_workload apx_interceptor apx_benchmark_standalone
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# ============================================================================
# Run targets
# ============================================================================
add_custom_target(run_apx_benchmark
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/apx_benchmark_standalone
    DEPENDS apx_benchmark_standalone
    COMMENT "Running APX standalone benchmark"
)

add_custom_target(run_apx_workload_native
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/apx_workload
    DEPENDS apx_workload
    COMMENT "Running workload without APX interception"
)

add_custom_target(run_apx_workload_intercepted
    COMMAND ${CMAKE_COMMAND} -E env
        LD_PRELOAD=${CMAKE_CURRENT_BINARY_DIR}/libapx_interceptor.so
        ${CMAKE_CURRENT_BINARY_DIR}/apx_workload
    DEPENDS apx_workload apx_interceptor
    COMMENT "Running workload with APX interception"
)
