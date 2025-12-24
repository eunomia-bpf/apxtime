/* SPDX-License-Identifier: MIT
 *
 * Simple test program for APX CPU feature detection
 * Can be compiled standalone without full bpftime
 */

#include <iostream>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#include <immintrin.h>
#include <x86intrin.h>
#endif

// CPUID constants
constexpr uint32_t CPUID_LEAF_FEATURES = 7;
constexpr uint32_t CPUID_SUBLEAF_APX = 1;
constexpr uint32_t APX_F_BIT = 21;
constexpr uint32_t AVX10_BIT = 19;

struct APXFeatures {
    bool has_apx_f = false;
    bool has_avx10 = false;
    bool has_avx512 = false;
    uint64_t xcr0 = 0;
};

#if defined(__x86_64__) || defined(_M_X64)
void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t regs[4]) {
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
}

uint64_t get_xcr0() {
    uint32_t regs[4];
    cpuid(1, 0, regs);
    bool has_osxsave = (regs[2] >> 27) & 1;
    if (!has_osxsave) return 0;
    return _xgetbv(0);
}
#endif

APXFeatures detect_apx() {
    APXFeatures features;

#if defined(__x86_64__) || defined(_M_X64)
    uint32_t regs[4];

    // Check max CPUID leaf
    cpuid(0, 0, regs);
    uint32_t max_leaf = regs[0];

    if (max_leaf >= CPUID_LEAF_FEATURES) {
        // Check AVX-512 in leaf 7, subleaf 0
        cpuid(CPUID_LEAF_FEATURES, 0, regs);
        features.has_avx512 = (regs[1] >> 16) & 1;

        // Check APX in leaf 7, subleaf 1
        cpuid(CPUID_LEAF_FEATURES, CPUID_SUBLEAF_APX, regs);
        features.has_avx10 = (regs[3] >> AVX10_BIT) & 1;
        features.has_apx_f = (regs[3] >> APX_F_BIT) & 1;
    }

    features.xcr0 = get_xcr0();
#endif

    return features;
}

void print_xcr0_features(uint64_t xcr0) {
    std::cout << "  XCR0 Components:\n";
    if (xcr0 & (1ULL << 0)) std::cout << "    [0] x87 FPU\n";
    if (xcr0 & (1ULL << 1)) std::cout << "    [1] SSE (XMM)\n";
    if (xcr0 & (1ULL << 2)) std::cout << "    [2] AVX (YMM)\n";
    if (xcr0 & (1ULL << 3)) std::cout << "    [3] MPX BNDREGS\n";
    if (xcr0 & (1ULL << 4)) std::cout << "    [4] MPX BNDCSR\n";
    if (xcr0 & (1ULL << 5)) std::cout << "    [5] AVX-512 opmask (k0-k7)\n";
    if (xcr0 & (1ULL << 6)) std::cout << "    [6] AVX-512 ZMM_Hi256\n";
    if (xcr0 & (1ULL << 7)) std::cout << "    [7] AVX-512 Hi16_ZMM\n";
    if (xcr0 & (1ULL << 9)) std::cout << "    [9] PKRU\n";
    if (xcr0 & (1ULL << 17)) std::cout << "    [17] AMX TILECFG\n";
    if (xcr0 & (1ULL << 18)) std::cout << "    [18] AMX TILEDATA\n";
    if (xcr0 & (1ULL << 19)) std::cout << "    [19] APX Extended GPRs (R16-R31)\n";
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "     APX CPU Feature Detection Test\n";
    std::cout << "===========================================\n\n";

    auto features = detect_apx();

    std::cout << "CPU Features:\n";
    std::cout << "  APX_F (Advanced Performance Extensions): "
              << (features.has_apx_f ? "YES" : "NO") << "\n";
    std::cout << "  AVX10: " << (features.has_avx10 ? "YES" : "NO") << "\n";
    std::cout << "  AVX-512: " << (features.has_avx512 ? "YES" : "NO") << "\n";

    std::cout << "\nXCR0 Register: 0x" << std::hex << features.xcr0 << std::dec << "\n";
    print_xcr0_features(features.xcr0);

    if (features.has_apx_f) {
        std::cout << "\n*** APX IS AVAILABLE! ***\n";
        std::cout << "Extended GPRs R16-R31 can be used for:\n";
        std::cout << "  - Reducing register spills\n";
        std::cout << "  - 3-operand NDD forms\n";
        std::cout << "  - Flag suppression (NF)\n";
        std::cout << "  - PUSH2/POP2 stack operations\n";

        // Check if OS supports APX state saving
        bool os_apx = (features.xcr0 & (1ULL << 19)) != 0;
        std::cout << "\nOS APX Support (XCR0[19]): "
                  << (os_apx ? "YES" : "NO") << "\n";
    } else {
        std::cout << "\nAPX not available on this CPU.\n";
        std::cout << "This is expected on CPUs before Granite Rapids/Arrow Lake.\n";
    }

    std::cout << "\n===========================================\n";
    return 0;
}
