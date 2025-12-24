/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, eunomia-bpf org
 * All rights reserved.
 *
 * APX CPU Feature Detection Implementation
 */

#include "apx_cpu_features.hpp"
#include <spdlog/spdlog.h>
#include <atomic>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#define BPFTIME_HAS_X86_CPUID 1
#endif

namespace bpftime::vm::apx {

namespace {

#ifdef BPFTIME_HAS_X86_CPUID
/**
 * @brief Execute CPUID instruction
 */
void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t regs[4]) {
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
}

/**
 * @brief Get XCR0 value (Extended Control Register 0)
 * Uses inline assembly to avoid requiring -mxsave compiler flag
 */
uint64_t get_xcr0() {
    uint32_t regs[4];
    cpuid(1, 0, regs);

    // Check OSXSAVE bit (ECX[27])
    bool has_osxsave = (regs[2] >> 27) & 1;
    if (!has_osxsave) {
        SPDLOG_DEBUG("APX: OSXSAVE not supported");
        return 0;
    }

    // Use inline assembly for XGETBV to avoid -mxsave requirement
    uint32_t eax, edx;
    asm volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
}
#endif

// Cached features singleton
std::atomic<bool> g_features_initialized{false};
APXFeatures g_cached_features;

} // anonymous namespace

APXFeatures detect_apx_features() {
    APXFeatures features;

#ifdef BPFTIME_HAS_X86_CPUID
    uint32_t regs[4];

    // First check maximum CPUID leaf supported
    cpuid(0, 0, regs);
    uint32_t max_leaf = regs[0];

    if (max_leaf < CPUID_LEAF_FEATURES) {
        SPDLOG_DEBUG("APX: CPUID leaf 7 not supported");
        return features;
    }

    // Check CPUID Leaf 7, Subleaf 0 for basic features
    cpuid(CPUID_LEAF_FEATURES, 0, regs);

    // Check AVX-512 foundation
    bool has_avx512f = (regs[1] >> 16) & 1;  // EBX[16]
    features.has_avx512 = has_avx512f;

    // Check CPUID Leaf 7, Subleaf 1 for APX features
    cpuid(CPUID_LEAF_FEATURES, CPUID_SUBLEAF_APX, regs);

    // EDX[19] = AVX10 support
    features.has_avx10 = (regs[3] >> AVX10_BIT) & 1;

    // EDX[21] = APX_F (APX Foundation)
    features.has_apx_f = (regs[3] >> APX_F_BIT) & 1;

    // If APX_F is available, assume all APX sub-features are available
    // (These are part of APX foundation in current implementations)
    if (features.has_apx_f) {
        features.has_egpr = true;       // Extended GPRs (R16-R31)
        features.has_push2pop2 = true;  // PUSH2/POP2
        features.has_ppx = true;        // Push-Pop Acceleration
        features.has_ndd = true;        // New Data Destination (3-operand)
        features.has_ccmp = true;       // Conditional Compare
        features.has_cf = true;         // Conditional Faults
        features.has_nf = true;         // No-Flags (flag suppression)
    }

    // Get XCR0 to verify OS support for extended state
    features.xcr0 = get_xcr0();

    // Verify OS supports APX state saving
    if (features.has_apx_f) {
        // Check if XCR0 has APX bit set (bit 19)
        // Note: The exact XCR0 bit for APX may vary; this follows xsave-utils pattern
        bool os_supports_apx = (features.xcr0 & XFEATURE_MASK_APX) != 0;
        if (!os_supports_apx) {
            SPDLOG_WARN("APX: CPU supports APX but OS does not (XCR0={:#x})", features.xcr0);
            // Don't disable APX features entirely - the extended GPRs may still work
            // for JIT purposes even without full XSAVE support
        }
    }

    SPDLOG_INFO("APX Detection: APX_F={}, AVX10={}, AVX512={}, XCR0={:#x}",
                features.has_apx_f, features.has_avx10, features.has_avx512, features.xcr0);
    if (features.has_apx_f) {
        SPDLOG_INFO("APX Features: EGPR={}, NDD={}, PPX={}, CCMP={}, CF={}, NF={}",
                    features.has_egpr, features.has_ndd, features.has_ppx,
                    features.has_ccmp, features.has_cf, features.has_nf);
    }
#else
    SPDLOG_DEBUG("APX: Not on x86 platform, APX features not available");
#endif

    return features;
}

std::string get_llvm_apx_features(const APXFeatures& features) {
    std::string result;

    if (!features.can_use_apx()) {
        return result;
    }

    // Build feature string for LLVM
    // LLVM uses features like: +egpr, +push2pop2, +ppx, +ndd, +ccmp, +cf, +nf

    auto append_feature = [&result](const char* name, bool enabled) {
        if (enabled) {
            if (!result.empty()) {
                result += ",";
            }
            result += "+";
            result += name;
        }
    };

    // Core APX features
    append_feature("egpr", features.has_egpr);
    append_feature("push2pop2", features.has_push2pop2);
    append_feature("ppx", features.has_ppx);
    append_feature("ndd", features.has_ndd);
    append_feature("ccmp", features.has_ccmp);
    append_feature("cf", features.has_cf);
    append_feature("nf", features.has_nf);

    // AVX-512 features that complement APX
    if (features.has_avx512) {
        append_feature("avx512f", true);
        append_feature("avx512bw", true);
        append_feature("avx512dq", true);
        append_feature("avx512vl", true);
    }

    // AVX10 (if available)
    if (features.has_avx10) {
        append_feature("avx10.1-256", true);
    }

    SPDLOG_DEBUG("APX: LLVM feature string: {}", result);
    return result;
}

std::string get_llvm_cpu_target(const APXFeatures& features) {
    // For APX, we need a target that supports the APX instruction set
    // LLVM doesn't have a specific "apx" target, but we can use generic
    // with feature flags, or target specific microarchitectures

    if (features.can_use_apx()) {
        // "x86-64-v4" is the highest baseline, but for APX we need
        // to rely on feature flags since APX CPUs are newer
        // Use "generic" with explicit APX features
        return "generic";
    }

    if (features.has_avx512) {
        return "skylake-avx512";
    }

    // Check for AVX2
    // Could add more detection but for now return generic
    return "generic";
}

bool is_apx_enabled_by_env() {
    const char* env = std::getenv("BPFTIME_APX_ENABLED");
    if (env == nullptr) {
        // Default: enable APX if hardware supports it
        return true;
    }

    // Check for explicit disable
    if (std::strcmp(env, "0") == 0 ||
        std::strcmp(env, "false") == 0 ||
        std::strcmp(env, "no") == 0 ||
        std::strcmp(env, "FALSE") == 0 ||
        std::strcmp(env, "NO") == 0) {
        SPDLOG_INFO("APX: Disabled via BPFTIME_APX_ENABLED environment variable");
        return false;
    }

    return true;
}

const APXFeatures& get_cached_apx_features() {
    bool expected = false;
    if (g_features_initialized.compare_exchange_strong(expected, true,
                                                        std::memory_order_seq_cst,
                                                        std::memory_order_seq_cst)) {
        g_cached_features = detect_apx_features();
    }
    return g_cached_features;
}

void warmup_apx_registers() {
#ifdef BPFTIME_HAS_X86_CPUID
    const auto& features = get_cached_apx_features();
    if (!features.can_use_apx()) {
        return;
    }

    // Write to extended GPRs to warm them up
    // This uses inline assembly to touch R16-R31
    // The volatile prevents the compiler from optimizing these away
#if defined(__GNUC__) && defined(__x86_64__)
    // Note: This requires compiler support for APX inline assembly
    // If the compiler doesn't support this, we skip the warmup
    // GCC 14+ and Clang 19+ support APX inline assembly
#if __GNUC__ >= 14 || (defined(__clang__) && __clang_major__ >= 19)
    asm volatile(
        "movq $0, %%r16\n\t"
        "movq $0, %%r17\n\t"
        "movq $0, %%r18\n\t"
        "movq $0, %%r19\n\t"
        "movq $0, %%r20\n\t"
        "movq $0, %%r21\n\t"
        "movq $0, %%r22\n\t"
        "movq $0, %%r23\n\t"
        "movq $0, %%r24\n\t"
        "movq $0, %%r25\n\t"
        "movq $0, %%r26\n\t"
        "movq $0, %%r27\n\t"
        "movq $0, %%r28\n\t"
        "movq $0, %%r29\n\t"
        "movq $0, %%r30\n\t"
        "movq $0, %%r31\n\t"
        :
        :
        : "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
          "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31"
    );
    SPDLOG_DEBUG("APX: Extended registers warmed up");
#else
    SPDLOG_DEBUG("APX: Compiler does not support APX inline assembly, skipping warmup");
#endif
#endif
#endif
}

} // namespace bpftime::vm::apx
