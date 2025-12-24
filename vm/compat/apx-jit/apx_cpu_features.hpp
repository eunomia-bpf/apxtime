/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, eunomia-bpf org
 * All rights reserved.
 *
 * APX CPU Feature Detection for bpftime JIT backend
 * Leverages Intel APX (Advanced Performance Extensions) when available
 */

#ifndef _BPFTIME_APX_CPU_FEATURES_HPP
#define _BPFTIME_APX_CPU_FEATURES_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace bpftime::vm::apx {

// CPUID leaf/subleaf constants for APX detection
constexpr uint32_t CPUID_LEAF_FEATURES = 7;
constexpr uint32_t CPUID_SUBLEAF_APX = 1;

// APX feature bits in CPUID Leaf 7, Subleaf 1
constexpr uint32_t APX_F_BIT = 21;    // EDX[21] - APX_F (APX Foundation)
constexpr uint32_t AVX10_BIT = 19;    // EDX[19] - AVX10 support

// XFEATURE masks (from Intel SDM)
constexpr uint64_t XFEATURE_MASK_X87 = (1ULL << 0);
constexpr uint64_t XFEATURE_MASK_SSE = (1ULL << 1);
constexpr uint64_t XFEATURE_MASK_AVX = (1ULL << 2);
constexpr uint64_t XFEATURE_MASK_OPMASK = (1ULL << 5);     // AVX-512 k0-k7
constexpr uint64_t XFEATURE_MASK_ZMM_HI256 = (1ULL << 6);  // ZMM0-15 Hi256
constexpr uint64_t XFEATURE_MASK_HI16_ZMM = (1ULL << 7);   // ZMM16-31 Full
constexpr uint64_t XFEATURE_MASK_APX = (1ULL << 19);

/**
 * APX feature set that can be enabled/detected
 */
struct APXFeatures {
    bool has_apx_f = false;       // APX Foundation (R16-R31 registers)
    bool has_egpr = false;        // Extended GPRs
    bool has_push2pop2 = false;   // PUSH2/POP2 instructions
    bool has_ppx = false;         // Push-Pop Acceleration
    bool has_ndd = false;         // New Data Destination (3-operand forms)
    bool has_ccmp = false;        // Conditional Compare
    bool has_cf = false;          // Conditional Faults
    bool has_nf = false;          // No-Flags (flag suppression)
    bool has_avx10 = false;       // AVX10 support
    bool has_avx512 = false;      // AVX-512 support

    // XCR0 state
    uint64_t xcr0 = 0;

    /**
     * @brief Check if APX optimizations can be applied
     */
    bool can_use_apx() const { return has_apx_f && has_egpr; }

    /**
     * @brief Check if flag suppression forms can be used
     */
    bool can_use_flag_suppression() const { return has_nf; }

    /**
     * @brief Check if 3-operand forms can be used
     */
    bool can_use_3operand() const { return has_ndd; }
};

/**
 * @brief Detect APX features on the current CPU
 *
 * Uses CPUID to detect APX support, following the patterns from xsave-utils.
 * This includes:
 * - APX_F foundation (extended GPRs R16-R31)
 * - APX sub-features (NDD, PPX, CCMP, CF, NF)
 * - AVX-512 and AVX10 support for context
 *
 * @return APXFeatures structure with detected capabilities
 */
APXFeatures detect_apx_features();

/**
 * @brief Get LLVM feature string for APX-capable target
 *
 * Generates an LLVM feature string (e.g., "+egpr,+push2pop2,+ndd")
 * based on detected APX capabilities. This is used when creating
 * the LLVM TargetMachine.
 *
 * @param features Previously detected APX features
 * @return LLVM-compatible feature string
 */
std::string get_llvm_apx_features(const APXFeatures& features);

/**
 * @brief Get the best CPU target name for LLVM
 *
 * Returns an appropriate LLVM CPU target name based on capabilities.
 * For APX-capable systems, returns architectures that support APX.
 * Falls back to "generic" or appropriate targets for older CPUs.
 *
 * @param features Previously detected APX features
 * @return LLVM CPU target name
 */
std::string get_llvm_cpu_target(const APXFeatures& features);

/**
 * @brief Check if APX is enabled via environment variable
 *
 * Reads BPFTIME_APX_ENABLED environment variable.
 * Values: "1", "true", "yes" enable APX (default if hardware supports)
 *         "0", "false", "no" disable APX
 *
 * @return true if APX should be enabled, false otherwise
 */
bool is_apx_enabled_by_env();

/**
 * @brief Get cached APX features (singleton)
 *
 * Returns cached APX features, detecting on first call.
 * Thread-safe.
 *
 * @return Reference to cached APXFeatures
 */
const APXFeatures& get_cached_apx_features();

/**
 * @brief Warm up APX registers (useful before JIT execution)
 *
 * If APX is available, performs dummy operations on R16-R31
 * to ensure the extended register state is initialized.
 * This can help avoid cold-start penalties.
 */
void warmup_apx_registers();

} // namespace bpftime::vm::apx

#endif // _BPFTIME_APX_CPU_FEATURES_HPP
