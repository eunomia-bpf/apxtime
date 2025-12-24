/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, eunomia-bpf org
 * All rights reserved.
 *
 * Unit tests for APX CPU feature detection
 */

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include "apx_cpu_features.hpp"

using namespace bpftime::vm::apx;

class APXFeatureTest : public ::testing::Test {
protected:
    void SetUp() override {
        spdlog::set_level(spdlog::level::debug);
    }
};

TEST_F(APXFeatureTest, DetectAPXFeatures) {
    APXFeatures features = detect_apx_features();

    // Log what we found
    spdlog::info("APX_F detected: {}", features.has_apx_f);
    spdlog::info("AVX10 detected: {}", features.has_avx10);
    spdlog::info("AVX512 detected: {}", features.has_avx512);
    spdlog::info("XCR0 value: {:#x}", features.xcr0);

    if (features.has_apx_f) {
        spdlog::info("APX sub-features:");
        spdlog::info("  EGPR (R16-R31): {}", features.has_egpr);
        spdlog::info("  PUSH2/POP2: {}", features.has_push2pop2);
        spdlog::info("  PPX: {}", features.has_ppx);
        spdlog::info("  NDD (3-operand): {}", features.has_ndd);
        spdlog::info("  CCMP: {}", features.has_ccmp);
        spdlog::info("  CF: {}", features.has_cf);
        spdlog::info("  NF (flag suppression): {}", features.has_nf);
    }

    // Basic sanity checks
    // If APX_F is present, all sub-features should be present
    if (features.has_apx_f) {
        EXPECT_TRUE(features.has_egpr);
        EXPECT_TRUE(features.has_ndd);
        EXPECT_TRUE(features.can_use_apx());
    }

    // Test helper methods
    if (features.has_apx_f) {
        EXPECT_TRUE(features.can_use_3operand());
        EXPECT_TRUE(features.can_use_flag_suppression());
    }
}

TEST_F(APXFeatureTest, GetCachedFeatures) {
    // First call should detect and cache
    const APXFeatures& features1 = get_cached_apx_features();

    // Second call should return same cached results
    const APXFeatures& features2 = get_cached_apx_features();

    // Should be the same object (cached)
    EXPECT_EQ(&features1, &features2);

    // Values should match
    EXPECT_EQ(features1.has_apx_f, features2.has_apx_f);
    EXPECT_EQ(features1.has_avx512, features2.has_avx512);
    EXPECT_EQ(features1.xcr0, features2.xcr0);
}

TEST_F(APXFeatureTest, GetLLVMFeatureString) {
    APXFeatures features = detect_apx_features();

    std::string llvm_features = get_llvm_apx_features(features);

    spdlog::info("LLVM feature string: {}", llvm_features);

    if (features.can_use_apx()) {
        // Should contain APX-related features
        EXPECT_TRUE(llvm_features.find("+egpr") != std::string::npos);
        EXPECT_TRUE(llvm_features.find("+ndd") != std::string::npos);
    } else {
        // Should be empty if APX not available
        EXPECT_TRUE(llvm_features.empty());
    }
}

TEST_F(APXFeatureTest, GetLLVMCPUTarget) {
    APXFeatures features = detect_apx_features();

    std::string cpu_target = get_llvm_cpu_target(features);

    spdlog::info("LLVM CPU target: {}", cpu_target);

    // Should always return a valid target
    EXPECT_FALSE(cpu_target.empty());

    // Should be a known target
    EXPECT_TRUE(cpu_target == "generic" ||
                cpu_target == "skylake-avx512" ||
                cpu_target.find("x86-64") != std::string::npos);
}

TEST_F(APXFeatureTest, EnvironmentVariableControl) {
    // Save original value
    const char* original = std::getenv("BPFTIME_APX_ENABLED");

    // Test with disabled
    setenv("BPFTIME_APX_ENABLED", "0", 1);
    EXPECT_FALSE(is_apx_enabled_by_env());

    setenv("BPFTIME_APX_ENABLED", "false", 1);
    EXPECT_FALSE(is_apx_enabled_by_env());

    setenv("BPFTIME_APX_ENABLED", "no", 1);
    EXPECT_FALSE(is_apx_enabled_by_env());

    // Test with enabled
    setenv("BPFTIME_APX_ENABLED", "1", 1);
    EXPECT_TRUE(is_apx_enabled_by_env());

    setenv("BPFTIME_APX_ENABLED", "yes", 1);
    EXPECT_TRUE(is_apx_enabled_by_env());

    // Restore original
    if (original) {
        setenv("BPFTIME_APX_ENABLED", original, 1);
    } else {
        unsetenv("BPFTIME_APX_ENABLED");
    }

    // Default should be enabled
    EXPECT_TRUE(is_apx_enabled_by_env());
}

TEST_F(APXFeatureTest, WarmupRegisters) {
    // This should not crash regardless of APX availability
    EXPECT_NO_THROW(warmup_apx_registers());
}

// Test APX feature structure default values
TEST_F(APXFeatureTest, DefaultFeatureValues) {
    APXFeatures features;

    // Default constructed should have everything false/zero
    EXPECT_FALSE(features.has_apx_f);
    EXPECT_FALSE(features.has_egpr);
    EXPECT_FALSE(features.has_push2pop2);
    EXPECT_FALSE(features.has_ppx);
    EXPECT_FALSE(features.has_ndd);
    EXPECT_FALSE(features.has_ccmp);
    EXPECT_FALSE(features.has_cf);
    EXPECT_FALSE(features.has_nf);
    EXPECT_FALSE(features.has_avx10);
    EXPECT_FALSE(features.has_avx512);
    EXPECT_EQ(features.xcr0, 0u);

    EXPECT_FALSE(features.can_use_apx());
    EXPECT_FALSE(features.can_use_3operand());
    EXPECT_FALSE(features.can_use_flag_suppression());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
