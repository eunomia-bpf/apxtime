# APX-aware JIT backend for bpftime
# This module provides APX (Advanced Performance Extensions) optimizations
# for the LLVM JIT when running on APX-capable Intel CPUs.

cmake_minimum_required(VERSION 3.16)

project(bpftime_apx_jit
  VERSION 1.0.0
  DESCRIPTION "APX-aware LLVM JIT backend for bpftime"
  LANGUAGES CXX
)

# Option to enable/disable APX JIT
option(BPFTIME_ENABLE_APX_JIT "Enable APX-aware JIT backend" ON)

if(NOT BPFTIME_ENABLE_APX_JIT)
  message(STATUS "APX JIT backend disabled")
  return()
endif()

# Find required packages
find_package(LLVM REQUIRED CONFIG)

message(STATUS "APX JIT: Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "APX JIT: Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Check LLVM version - APX requires LLVM 18+ for full support
if(LLVM_VERSION_MAJOR LESS 15)
  message(WARNING "APX JIT: LLVM ${LLVM_VERSION_MAJOR} may have limited APX support. LLVM 18+ recommended.")
endif()

# Option to enable Rellume-based x86-64 lifter
option(BPFTIME_APX_ENABLE_RELLUME "Enable Rellume-based x86-64 to LLVM IR lifter" ON)

# Try to find Rellume
find_package(rellume QUIET)

if(rellume_FOUND)
  message(STATUS "APX JIT: Found Rellume for x86-64 lifting")
  set(HAVE_RELLUME TRUE)
else()
  message(STATUS "APX JIT: Rellume not found, using fallback lifter")
  set(HAVE_RELLUME FALSE)
endif()

# APX JIT library sources
set(APX_JIT_SOURCES
  apx_cpu_features.cpp
  compat_apx_llvm.cpp
  apx_hotpath_manager.cpp
)

# Add lifter sources
set(APX_LIFTER_SOURCES
  lifter/rellume_lifter.cpp
)

# Create APX JIT library
add_library(bpftime_apx_jit STATIC ${APX_JIT_SOURCES} ${APX_LIFTER_SOURCES})

# Include directories
target_include_directories(bpftime_apx_jit
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/lifter
    ${LLVM_INCLUDE_DIRS}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Rellume integration
if(HAVE_RELLUME)
  target_link_libraries(bpftime_apx_jit PUBLIC rellume::rellume)
  target_compile_definitions(bpftime_apx_jit PRIVATE HAVE_RELLUME=1)
endif()

# LLVM definitions
target_compile_definitions(bpftime_apx_jit PRIVATE ${LLVM_DEFINITIONS})

# Link against LLVM
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

target_link_libraries(bpftime_apx_jit
  PUBLIC
    ${llvm_libs}
  PRIVATE
    spdlog::spdlog
)

# Compiler features
target_compile_features(bpftime_apx_jit PUBLIC cxx_std_20)

# Platform-specific settings
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  message(STATUS "APX JIT: Building for x86_64 with CPUID support")
  target_compile_definitions(bpftime_apx_jit PRIVATE BPFTIME_HAS_X86_CPUID=1)
else()
  message(STATUS "APX JIT: Building for ${CMAKE_SYSTEM_PROCESSOR} - APX features unavailable")
endif()

# Compiler-specific flags for APX support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "14.0")
    message(STATUS "APX JIT: GCC ${CMAKE_CXX_COMPILER_VERSION} has APX inline asm support")
    target_compile_definitions(bpftime_apx_jit PRIVATE BPFTIME_COMPILER_SUPPORTS_APX_ASM=1)
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.0")
    message(STATUS "APX JIT: Clang ${CMAKE_CXX_COMPILER_VERSION} has APX inline asm support")
    target_compile_definitions(bpftime_apx_jit PRIVATE BPFTIME_COMPILER_SUPPORTS_APX_ASM=1)
  endif()
endif()

# Install targets
install(TARGETS bpftime_apx_jit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(FILES
  apx_cpu_features.hpp
  compat_apx_llvm.hpp
  apx_hotpath_manager.hpp
  lifter/rellume_lifter.hpp
  DESTINATION include/bpftime/apx
)

# Unit tests
if(BPFTIME_ENABLE_UNIT_TESTING)
  add_subdirectory(test)
endif()

# Benchmarks
add_subdirectory(benchmark)
