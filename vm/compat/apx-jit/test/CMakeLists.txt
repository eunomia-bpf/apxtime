# Unit tests for APX JIT module

# Find GTest
find_package(GTest REQUIRED)

# APX Feature Detection Tests
add_executable(apx_feature_test apx_feature_test.cpp)

target_link_libraries(apx_feature_test
    PRIVATE
        bpftime_apx_jit
        GTest::gtest
        GTest::gtest_main
        spdlog::spdlog
)

target_include_directories(apx_feature_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Register with CTest
include(GoogleTest)
gtest_discover_tests(apx_feature_test)
