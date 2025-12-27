# Unit tests for APX JIT module

# Find GTest
find_package(GTest REQUIRED)

# Find LLVM
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

# APX Feature Detection Tests
add_executable(apx_feature_test apx_feature_test.cpp)

target_link_libraries(apx_feature_test
    PRIVATE
        bpftime_apx_jit
        ${llvm_libs}
        GTest::gtest
        GTest::gtest_main
        spdlog::spdlog
)

target_include_directories(apx_feature_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${LLVM_INCLUDE_DIRS}
)

target_compile_definitions(apx_feature_test PRIVATE ${LLVM_DEFINITIONS})

# Register with CTest
include(GoogleTest)
gtest_discover_tests(apx_feature_test)
