cmake_minimum_required(VERSION 3.15)

#
# Set project options
#
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/StandardSettings.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CompilerWarnings.cmake)


add_library(bpftime_vm_compat INTERFACE)

target_include_directories(bpftime_vm_compat INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include ${SPDLOG_INCLUDE})

if(${BPFTIME_LLVM_JIT})
  message(STATUS "Supporting llvm-jit")
  add_subdirectory(llvm-vm)
endif()

if(${BPFTIME_UBPF_JIT})
  message(STATUS "Supporting ubpf-jit")
  add_subdirectory(ubpf-vm)
endif()

# APX-aware JIT backend (requires LLVM JIT)
option(BPFTIME_APX_JIT "Enable APX-aware JIT backend for Intel APX CPUs" ON)
if(${BPFTIME_APX_JIT} AND ${BPFTIME_LLVM_JIT})
  message(STATUS "Supporting APX-aware JIT backend")
  add_subdirectory(apx-jit)
endif()
